# imports
import os
import sys
import time
import pandas as pd
import pickle
import numpy as np
from fractions import Fraction
import scipy
from scipy.signal import resample_poly, iirnotch, filtfilt
import mne
from mne_bids import BIDSPath, write_raw_bids
import pyedflib
from tqdm import tqdm
from pqdm.processes import pqdm
import getpass
import argparse
import matplotlib.pyplot as plt
import gzip
import shutil
from utils import get_iEEG_data
import datetime
import ast

#from config import CONFIG

# definitions
sleep_stage_dict = {
    "R": 1,
    "W": 2,
    "N1": 3,
    "N2": 4,
    "N3": 5
}

def padId(id):
    if len(id) == 5:
        id = id[0:3] + "0" + id[-2:]
    return id

def convertTimeStamp(time):
    secs = time.second
    mins = time.minute
    hours = time.hour
    return hours*60*60 + mins*60 + secs

def getStartTime(fst,x):
    id = padId(x.split("_")[0])
    if len(x.split("_")) > 2:
        ans = fst.loc[fst['name'] == id][int(x.split("_")[2][-1:])]
    else:
        ans = fst.loc[fst['name'] == id][1]
    return convertTimeStamp(ans.values[0])

def convert_hup_to_rid(rh_table,hup_id):
    # get number from hup_id
    hup_num = int(hup_id[3:6])
    # get rid from rid_hup_table
    rid = rh_table.loc[rh_table["hupsubjno"] == hup_num]["record_id"].values[0]
    return rid

def smart_match_label(label,full_rid,this_pt_dict,ch_harmonize):
    # get rows of ch_harmonize where rid == full_rid and old = label
    # get channel name from bipolar label
    ch_name = ''.join([i for i in label.split("-")[0] if not i.isdigit()])
    
    # strip label of alpha characters to get electrode numbers
    numbers = ''.join([i for i in label if not i.isalpha()]).split("-")
    matches = ch_harmonize[(ch_harmonize["rid"] == full_rid) & (ch_harmonize["old"] == ch_name)]
    # if matches is not an empty dataframe
    if not matches.empty:
        new_label = ''.join([matches["new"].values[0],numbers[0],"-",matches["new"].values[0],numbers[1]])
    else:
        new_label = label
    
    # if label is a key in this_pt_dict
    if label in this_pt_dict.keys():
        #print(f"{full_rid}: {label} -> no change")
        return this_pt_dict[label]
    elif new_label in this_pt_dict.keys():
        print(f"{full_rid}: {label} -> {new_label}")
        return this_pt_dict[new_label]
    else:
        # for each row in ch_harmonize where old == ch_name
        new_chs = ch_harmonize[ch_harmonize["old"] == ch_name]
        for k in range(len(new_chs)):
            # if new is a key in this_pt_dict
            new_ch = new_chs["new"].iloc[k]
            new_label = ''.join([new_ch,numbers[0],"-",new_ch,numbers[1]])
            if new_label in this_pt_dict.keys():
                print(f"{full_rid}: {label} -> {new_label} (not in ch_harmonize!)")
                return this_pt_dict[new_label]
        # if no matches are found, raise an exception
        raise Exception(f"{full_rid}: no localization found for {label}.")

def get_time_delta_for_stage(result_path, sleep_stage):
    # result_path: path to sleepSEEG summary csv
    # sleep_stage: "R", "W", "N1", "N2", "N3"
    # convert sleep_stage to sleepSEEG number
    period = 30  # each index is 30 secs
    sleep_stage_num = sleep_stage_dict[sleep_stage]
    # read result_path
    result_data = pd.read_csv(result_path)
    # get column values of sleep_stage where file_index = 1
    sleep_stage_results = result_data.loc[result_data["file_index"] == 1]["sleep_stage"]
    # find start index of longest segment spent in sleep_stage_num
    longest_len = 0
    curr_len = 0
    longest_index = 0
    for k in range(len(sleep_stage_results)):
        if (sleep_stage_results.iloc[k] == sleep_stage_num):
            curr_len += 1
        # if the current segment is longer than the longest segment but shorter than one hour (to ignore faulty data), update longest_len and longest_index
        elif ((curr_len > longest_len) and ((curr_len*period < 3600) or (sleep_stage_num == 2))):
            longest_len = curr_len
            longest_index = (k-curr_len)//2 # middle index of longest segment
            curr_len = 0
        else:
            curr_len = 0
    time_delta_sec = longest_index*period
    print(f"Longest length in {sleep_stage} = {longest_len*period} seconds. Midpoint of segment occurs {longest_index*30} seconds after start of recording.")
    return time_delta_sec
    
def get_coherence(pkl, bands, interval_length = 1, fs = 200):
    # pkl: saved iEEG data
    # interval_length: length of interval in seconds
    # fs: sampling rate (Hz)
    # returns: median coherence of pkl

    # read pickle file
    pickle_data = pd.read_pickle(pkl)
    signals = np.transpose(pickle_data[0].to_numpy())
    # if signals is all nan, warn
    if np.isnan(signals).all():
        print("WARNING: signals contains only NaN values!")
    # replace NaN values with interpolated values
    signals = pd.DataFrame(signals).interpolate(axis=1).fillna(method="ffill",axis=1).fillna(method="bfill",axis=1).to_numpy()
    # replace NaN values with 0
    #signals = np.nan_to_num(signals)
    #print(signals)
    print("Shape of signals: ", signals.shape)

    # get column names from pickle_data
    channel_names = pickle_data[0].columns.values.tolist()
    print("Channel names: ", channel_names)

    indices_to_delete = []

    # generate bipolar montage
    for elec_name in channel_names:
        # strip numbers
        label = ''.join([i for i in elec_name if not i.isdigit()])
        # strip letters
        number = ''.join([i for i in elec_name if i.isdigit()])
        if number != '' and f"{label}{str(int(number)+1).zfill(2)}" in channel_names:
            # if the next electrode in the series exists
            next_elec = f"{label}{str(int(number)+1).zfill(2)}"
            # subtract the next electrode in the series from the current electrode
            signals[channel_names.index(elec_name),:] = signals[channel_names.index(elec_name),:] - signals[channel_names.index(next_elec),:]
            # rename the current electrode
            channel_names[channel_names.index(elec_name)] = f"{elec_name}-{next_elec}"
        else:
            # mark the channel for deletion
            indices_to_delete.append(channel_names.index(elec_name))
    
    # delete indices_to_delete from signals and channel_names
    signals = np.delete(signals,indices_to_delete,0)
    channel_names = [i for j, i in enumerate(channel_names) if j not in indices_to_delete]

    print("Bipolar montage constructed.")
    print("Shape of bipolar signals: ", signals.shape)
    #print("Channel names: ", channel_names)

    # powerline noise
    # apply a low-pass antialiasing filter at 80 Hz using scipy.signal at 180, 120, 60
    if fs > 360:
        notches = [180, 120, 60]
    else:
        notches = [120, 60]

    for notch in notches:
        # estimate phase and amplitude at notch
        b, a = iirnotch(2*(notch/fs), 100, fs)
        signals = filtfilt(b, a, signals, axis=1)

    # resample to 200 Hz
    new_fs = 200
    frac = Fraction(new_fs, int(fs))
    signals = resample_poly(
        signals.T, up=frac.numerator, down=frac.denominator
    ).T  # (n_samples, n_channels)
    fs = new_fs

    # subtract mean value from each channel
    signals = signals - np.mean(signals, axis=1, keepdims=True)

    print("Calculating coherence...")
    
    # initialize list of len(channel_names) by len(channel_names) numpy array of coherences
    coherences_by_band = [np.array([[0.0 for k in range(len(channel_names))] for j in range(len(channel_names))]) for _ in range(len(bands))]

    # for each unique pair of channel names, calculate coherence
    for k in range(len(channel_names)):
        for j in range(k+1,len(channel_names)):
            # get coherence between channels k and j
            f, Cxy = scipy.signal.coherence(signals[k,:],signals[j,:],nperseg = 2*fs)
            for band in bands:
                band_start_hz = band[0]
                band_end_hz = band[1]
                # take the median coherence over the frequency band of interest
                # find the indices of the start and end of the band
                ind_start = np.argmax(f*fs >= band_start_hz)
                ind_end = np.argmax(f*fs >= band_end_hz)
                coherences_by_band[bands.index(band)][k,j] = np.median(Cxy[ind_start:ind_end])

    for k in range(len(coherences_by_band)):
        coherences = coherences_by_band[k]
        # symmetrize coherences
        coherences = coherences + coherences.T - np.diag(np.diag(coherences))
        # self-coherence is 1
        coherences = coherences + np.identity(len(channel_names))
        coherences_by_band[k] = pd.DataFrame(coherences, columns = channel_names, index = channel_names)

    return coherences_by_band

# MAIN
parser = argparse.ArgumentParser()
parser.add_argument("username", help="iEEG.org username")
parser.add_argument("password", help="path to iEEG.org password bin file")
parser.add_argument("metadata_path", help="path to combined_atlas_metadata.csv")
parser.add_argument("-r", "--reverse", help="run patients in reverse sorted order", action="store_true", default=False)
parser.add_argument("-t", "--plot", help="plot adjacency matrices and save as a .png", action="store_true", default=False)
parser.add_argument("-c", "--construct", help="construct normative atlases and save as .csv files", action="store_true", default=False)
parser.add_argument("-z", "--zscore", help="calculate abnormality z-scores for all patient feature matrices", action="store_true", default=False)
parser.add_argument("-o", "--overwrite", help="overwrite existing feature matrices", action="store_true", default=False)
args = parser.parse_args()

# read metadata csv
metadata_csv = pd.read_csv(args.metadata_path)

# read xlsx
xls = pd.ExcelFile("manual_validation.xlsx")
# get actual start times
fst = pd.read_excel(xls, "FileStartTimes")
# get patient names from fst where a value exists in the 3rd column
all_pats = fst.loc[fst[1].notnull()]["name"]
# add "_phaseII" to each patient name
all_pats = [x+"_phaseII" for x in all_pats]
# remove the zero before the first number in each patient name if the number is less than or equal to 89
all_pats = [x[0:3]+x[4:] if int(x[3:6]) <= 89 else x for x in all_pats]

# add "_D01" to each patient name if fst[2] is not null
all_pats = [x+"_D01" if isinstance(y, datetime.time) else x for x,y in zip(all_pats,fst[2])]

if args.reverse:
    print("Running patients in reverse sorted order.")
all_pats = sorted([x for x in all_pats if str(x) != 'nan'], reverse=args.reverse)
# get list of rids
rids = [padId(x.split("_")[0]) for x in all_pats]
# get list of real_offset_usec
start_times = [getStartTime(fst,x) for x in all_pats]

# load nights_classified_final.csv
nights_classified = pd.read_csv("nights_classified_final.csv")
# generate dictionary from each entry of the columns of csv file
nights_classified_dict = {x:y for x,y in zip(nights_classified["patient"],nights_classified["night_number"])}
# get list of nights
nights = [nights_classified_dict[x] for x in all_pats]

# read channel_harmonize.xlsx
ch_xls = pd.ExcelFile("channel_harmonize.xlsx")
# read using first row as labels
ch_harmonize = pd.read_excel(ch_xls, "Sheet1", header=0)

amf_xls = pd.ExcelFile("atlas_metadata_final.xlsx")
# read using first row as labels
amf = pd.read_excel(amf_xls, "Sheet1", header=0)

# skip N1
sleep_stages_to_run = ["W","N2","N3","R"]
# band cutoffs in Hz
bands = {
    "delta": [1, 4],
    "theta": [4, 8],
    "alpha": [8, 12],
    "beta": [12, 30],
    "gamma": [30, 80],
    "broad": [1, 80]
}
band_names = ["delta", "theta", "alpha", "beta", "gamma", "broad"]

parent_directory = os.path.join(os.path.dirname(os.path.abspath(os.getcwd())),'data')

# calculate the atlas
# first, get coherence matrices
# for each rid
for k in range(len(rids)):
    rid = rids[k]
    night = nights[k]
    print(f">>> Calculating coherences for {rid}... (patient {k+1} of {len(rids)})")
    iEEG_filename = all_pats[k]
    patient_directory = os.path.join(parent_directory,"sub-{}".format(rid))
    eeg_directory = os.path.join(patient_directory,'eeg')
    func_directory = os.path.join(patient_directory,'func')
    for root, dirs, files in os.walk(patient_directory, topdown=True):
        for fi in files:
            # if the file contains results from the correct night
            if (padId(fi.split("_")[0]) == rid) and (f"night{night}" in fi) and ("sleepstage" in fi):
                ssr_path = os.path.join(root,fi)
                break
    # check if parent directory of ssr_path is the same as the patient directory
    if os.path.dirname(ssr_path) != patient_directory:
        print("SleepSEEG results not found, skipping...")
        continue
    # print ssr_path
    print("SleepSEEG results path = {}".format(ssr_path))
    # for each sleep stage
    for stage in sleep_stages_to_run:
        print(f">> Sleep stage: {stage} in {rid}.")
        # get the seconds since clip start for the middle of the longest segment spent in the sleep stage
        if stage != "W":
            td = get_time_delta_for_stage(ssr_path, stage)
        else:
            # if wake, use times from the original iEEG Atlas paper
            # get ieeg filename from ssr_path
            filename_parts = ssr_path.split("/")[-1].split("_")
            # if starts with "D"
            if filename_parts[2][0] == "D":
                iEEG_filename = "_".join(filename_parts[0:3])
            else:
                iEEG_filename = "_".join(filename_parts[0:2])
            #print(ssr_path)
            #print(iEEG_filename)
            try:
                td = amf.loc[amf["portal_ID"] == iEEG_filename]["clip1_awake"].values[0]
            except IndexError:
                print(f"Could not find {iEEG_filename} in atlas_metadata_final.xlsx. Using SleepSEEG results...")
                td = get_time_delta_for_stage(ssr_path, stage)
            #print(f"td={td}")
        # for each band
        # for band_name in band_names:
        #     print(f"> Band: {band_name} in {rid}.")
        #     band_cutoffs = bands[band_name]
            # check if the coherence is already calculated
        broad_path = os.path.join(func_directory,f"sub-{rid}_{iEEG_filename}_night{night}_{stage}_broad_coherence.csv")
        if (not os.path.isfile(broad_path)) or args.overwrite:
            # if contains letters
            st = ssr_path.split("_")[2]
            if "D" in ssr_path.split("_")[2]:
                st = ssr_path.split("_")[3]
            this_start_us = int(float(st)) + td*1e6 # start time in ssr file name + time delta for longest segment
            this_end_us = this_start_us + 30e6 #30 secs
            # .pickle output path
            ieeg_output_file = os.path.join(eeg_directory,"sub-{}_{}_{}_{}_EEG.pickle".format(rid,iEEG_filename,this_start_us,this_end_us))

            # download a 30-second clip for that patient and save to output_file
            try:
                get_iEEG_data(args.username, args.password, iEEG_filename, this_start_us, this_end_us, outputfile = ieeg_output_file)
            except Exception as e:
                print(e)
                print("Skipping...")
                continue

            # calculate coherence and save matrix
            # 30 sec window with 1 sec intervals
            print("Pickle saved.")
            # get list of values from bands
            band_cutoffs = [bands[x] for x in band_names]
            coherence_result = get_coherence(ieeg_output_file,band_cutoffs)
            # save coherence_result to .npy file, create directory if it doesn't exist
            if not os.path.exists(func_directory):
                os.makedirs(func_directory)
            # save pandas dataframe as csv
            [x.to_csv(os.path.join(func_directory,f"sub-{rid}_{iEEG_filename}_night{night}_{stage}_{band}_coherence.csv")) for x,band in zip(coherence_result,band_names)]
            #coherence_result.to_csv(coherence_path)
            print(f"Coherence matrices saved for {rid}|night{night}|{stage}.")
        else:
            print(f"Coherence matrices already exist for {rid}|night{night}|{stage}. Skipping...")

# localize edges and construct atlases
if (args.construct == True):
    print("Constructing atlases...")
    atlas_directory = os.path.join(parent_directory,"atlas")
    # load rid_hup_table.csv
    rid_hup_table = pd.read_csv("/mnt/leif/littlab/users/ianzyong/sleep-atlas/util/rid_hup_table.csv")
    
    # get set of unique values in "reg" column from metadata_csv
    regions = set(metadata_csv["reg"])
    # remove nans
    regions = [x for x in regions if str(x) != 'nan']
    # sort and convert to list
    regions = sorted(list(regions))
    print(f"Regions: {regions}")
    # for each sleep stage

    for stage in sleep_stages_to_run:
        print(f">> Sleep stage: {stage}")
        # for each band
        for band_name in band_names:
            print(f"> Band: {band_name}")
            
            # initialize pandas dataframe with shape (num_regions, num_regions) and regions as column and row names and dtype=object
            this_atlas = pd.DataFrame(np.zeros((len(regions),len(regions))),columns=regions,index=regions,dtype=object)
            # set every value to an empty list
            for i in range(len(regions)):
                for j in range(len(regions)):
                    this_atlas.iloc[i,j] = []
            
            coherence_paths = []
            # find all coherence matrices (one for each patient) matching this stage and band name
            rids_for_lookup = []
            for k in range(len(rids)):
                rid = rids[k]
                iEEG_filename = all_pats[k]
                patient_directory = os.path.join(parent_directory,"sub-{}".format(rid))
                func_directory = os.path.join(patient_directory,'func')
                this_night = nights_classified_dict[iEEG_filename]
                coherence_path = os.path.join(func_directory,f"sub-{rid}_{iEEG_filename}_night{this_night}_{stage}_{band_name}_coherence.csv")
                if os.path.isfile(coherence_path):
                    coherence_paths.append(coherence_path)
                    rids_for_lookup.append(convert_hup_to_rid(rid_hup_table,rid))
            print(f"Number of coherence matrices found: {len(coherence_paths)}")
            # load csvs into list of pandas dataframes
            coherences = [pd.read_csv(x,index_col=0,header=0) for x in coherence_paths]
            # for each coherence matrix, get a list of regions and corresponding coherences
            for k in range(len(coherences)):
                matrix = coherences[k]
                this_rid = rids_for_lookup[k] 
                # get data from metadata_csv where name matches this_rid and normative is True
                full_rid = f"sub-RID{this_rid:04d}"
                this_pt_metadata = metadata_csv[(metadata_csv["pt"] == full_rid) & (metadata_csv["normative"] == True)]
                # construct a dictionary from this_pt_metadata using the name column as keys and the reg column as values
                this_pt_dict = dict(zip(this_pt_metadata["name"],this_pt_metadata["reg"]))
                print(f"RID: {full_rid}")
                # print number of keys in dict
                print(f"Number of channels in localization file: {len(this_pt_dict.keys())}")
                # print number of channels in the feature matrix
                print(f"Number of channels in feature matrix: {len(matrix)}")
                # compare the keys in this_pt_dict to the row labels of feature matrix, print number of matches
                print(f"Number of matches: {len(set(this_pt_dict.keys()).intersection(set(matrix.index)))}")
                # for each row and column in the feature matrix
                missing_localization = 0
                for i in range(len(matrix)):
                    for j in range(len(matrix)):
                        # get the row and column labels
                        row = matrix.index[i]
                        col = matrix.columns[j]
                        # get the region names from the metadata dictionary
                        try:
                            row_reg = smart_match_label(row,full_rid,this_pt_dict,ch_harmonize)
                            col_reg = smart_match_label(col,full_rid,this_pt_dict,ch_harmonize)
                        except Exception as e:
                            missing_localization += 1
                            continue
                        # add the value to the corresponding list in this_atlas
                        this_atlas.loc[row_reg,col_reg].append(matrix.iloc[i,j])
                print(f"Missing localization (or not normative) count for {coherence_paths[k]}: {missing_localization}")

            # save this_atlas as a .csv file
            this_atlas.to_csv(os.path.join(atlas_directory,f"{stage}_{band_name}_atlas.csv"))
            print(f"Atlas saved for {stage}|{band_name}.")

# generate z-scores for feature matrices using atlas distributions as reference
if (args.zscore == True):
    print("Generating z-scores...")
    atlas_directory = os.path.join(parent_directory,"atlas")
    # load rid_hup_table.csv
    rid_hup_table = pd.read_csv("/mnt/leif/littlab/users/ianzyong/sleep-atlas/util/rid_hup_table.csv")
    # for each sleep stage
    for stage in sleep_stages_to_run:
        print(f">> Sleep stage: {stage}")
        # for each band
        for band_name in band_names:
            print(f"> Band: {band_name}")

            # load atlas
            this_atlas = pd.read_csv(os.path.join(atlas_directory,f"{stage}_{band_name}_atlas.csv"),index_col=0,header=0)

            # find all coherence matrices (one for each patient) matching this stage and band name
            for k in range(len(rids)):
                rid = rids[k]
                iEEG_filename = all_pats[k]
                patient_directory = os.path.join(parent_directory,"sub-{}".format(rid))
                func_directory = os.path.join(patient_directory,'func')
                this_night = nights_classified_dict[iEEG_filename]
                coherence_path = os.path.join(func_directory,f"sub-{rid}_{iEEG_filename}_night{this_night}_{stage}_{band_name}_coherence.csv")
                if os.path.isfile(coherence_path):
                    print(f"Generating z-scores for {coherence_path}...")
                    this_rid = convert_hup_to_rid(rid_hup_table,rid)
                    coherence_df = pd.read_csv(coherence_path,index_col=0,header=0)
                    # initialize pandas dataframe with shape (num_electrodes, num_electrodes) and electrodes as column and row names
                    this_scores = pd.DataFrame(np.zeros((len(coherence_df),len(coherence_df))),columns=coherence_df.columns,index=coherence_df.index)
                    # get data from metadata_csv where name matches this_rid
                    full_rid = f"sub-RID{this_rid:04d}"
                    this_pt_metadata = metadata_csv[(metadata_csv["pt"] == full_rid)]
                    # construct a dictionary from this_pt_metadata using the name column as keys and the reg column as values
                    this_pt_dict = dict(zip(this_pt_metadata["name"],this_pt_metadata["reg"]))
                    # for each entry in the coherence matrix
                    for i in range(len(coherence_df)):
                        # get the row label
                        row = coherence_df.index[i]
                        try:
                            row_reg = smart_match_label(row,full_rid,this_pt_dict,ch_harmonize)
                        except Exception as e:
                            #print(e)
                            #print(f"Missing localization for {row}.")
                            this_scores.iloc[i,:] = np.nan
                            continue
                        for j in range(i+1,len(coherence_df)):
                            # get the value
                            feature_val = coherence_df.iloc[i,j]
                            # get the column label
                            col = coherence_df.columns[j]
                            # get the region names from the metadata dictionary
                            try:
                                col_reg = smart_match_label(col,full_rid,this_pt_dict,ch_harmonize)
                            except Exception as e:
                                #print(e)
                                #print(f"Missing localization for {col}.")
                                this_scores.iloc[i,j] = np.nan
                                continue
                            #print(f"Connection: {row_reg} to {col_reg}.")
                            # get the list of values from this_atlas
                            atlas_conns = this_atlas.loc[row_reg,col_reg]
                            #print(f"atlas_conns: {atlas_conns}")
                            if atlas_conns == "[]":
                                #print(f"{stage}, {band_name}: no atlas distribution for {row_reg} to {col_reg}.")
                                this_scores.iloc[i,j] = np.nan
                                continue
                            # convert string of list to list
                            atlas_conns = [float(s.strip()) for s in atlas_conns[1:-1].split(',')]
                            # if the feature_val is in atlas_conns, remove it (leave this normative connection for this patient out of the atlas distribution)
                            if np.count_nonzero(atlas_conns == feature_val) > 0:
                                #print("Connection from this patient removed from atlas distribution before scoring.")
                                atlas_conns = np.delete(atlas_conns, np.where(atlas_conns == feature_val)[0])
                            # calculate the absolute value z-score for this value
                            # ignore warnings for nanmean and nanstd
                            with np.errstate(all='ignore'):
                                this_scores.iloc[i,j] = abs((feature_val - np.nanmean(atlas_conns)) / np.nanstd(atlas_conns))
                    # if this_scores is all nan, warn
                    if np.all(np.isnan(this_scores)):
                        print(f"!!! All nan scores for {rid}({full_rid})|{stage}|{band_name}.")
                    # symmetrize scores
                    this_scores = this_scores + this_scores.T - np.diag(np.diag(this_scores))
                    # set z-scores along diagonal to nan
                    this_scores = this_scores.where(~np.eye(this_scores.shape[0],dtype=bool),np.nan)
                    this_scores.to_csv(os.path.join(func_directory,f"sub-{rid}_{iEEG_filename}_night{this_night}_{stage}_{band_name}_z-scores.csv"))
                    print(f"Scores saved for {rid}|{stage}|{band_name}.")
                else:
                    print(f"!!! Missing coherence matrix for {rid}|{stage}|{band_name}.")

print("Done.")




