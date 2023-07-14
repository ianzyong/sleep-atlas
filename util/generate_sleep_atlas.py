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
    
def get_coherence(pkl, band_start_hz, band_end_hz, interval_length = 1, fs = 200):
    # pkl: saved iEEG data
    # interval_length: length of interval in seconds
    # fs: sampling rate (Hz)
    # returns: median coherence of pkl

    # read pickle file
    pickle_data = pd.read_pickle(pkl)
    signals = np.transpose(pickle_data[0].to_numpy())
    # replace NaN values with interpolated values
    signals = pd.DataFrame(signals).interpolate().to_numpy()
    # replace NaN values with 0
    signals = np.nan_to_num(signals)
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
    
    # initialize len(channel_names) by len(channel_names) numpy array of coherences
    coherences = np.array([[0.0 for k in range(len(channel_names))] for j in range(len(channel_names))])
    # for each unique pair of channel names, calculate coherence
    for k in range(len(channel_names)):
        for j in range(k+1,len(channel_names)):
            # get coherence between channels k and j
            #print(f"Calculating coherence between {channel_names[k]} and {channel_names[j]}...")
            f, Cxy = scipy.signal.coherence(signals[k,:],signals[j,:],nperseg = 2*fs)
            # take the median coherence over the frequency band of interest
            # find the indices of the start and end of the band
            ind_start = np.argmax(f*fs >= band_start_hz)
            ind_end = np.argmax(f*fs >= band_end_hz)
            #print(ind_start, ind_end)
            #print(f"Cxy: {Cxy}")
            coherences[k,j] = np.median(Cxy[ind_start:ind_end])
            #print(coherences[k,j])
    # symmetrize coherences
    coherences = coherences + coherences.T - np.diag(np.diag(coherences))
    # self-coherence is 1
    coherences = coherences + np.identity(len(channel_names))
    # return a pandas dataframe with coherences and channel_names as the column and row names
    return pd.DataFrame(coherences, columns = channel_names, index = channel_names)

# MAIN
parser = argparse.ArgumentParser()
parser.add_argument("username", help="iEEG.org username")
parser.add_argument("password", help="path to iEEG.org password bin file")
parser.add_argument("metadata_path", help="path to combined_atlas_metadata.csv")
parser.add_argument("-r", "--reverse", help="run patients in reverse sorted order", action="store_true", default=False)
parser.add_argument("-t", "--plot", help="plot adjacency matrices and save as a .png", action="store_true", default=False)
parser.add_argument("-z", "--score", help="get z-scores and normative values", action="store_true", default=False)
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

# skip N1
sleep_stages_to_run = ["N2","N3","R"]
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
        td = get_time_delta_for_stage(ssr_path, stage)
        # for each band
        for band_name in band_names:
            print(f"> Band: {band_name} in {rid}.")
            band_cutoffs = bands[band_name]
            # check if the coherence is already calculated
            coherence_path = os.path.join(func_directory,f"sub-{rid}_{iEEG_filename}_night{night}_{stage}_{band_name}_coherence.csv")
            if not os.path.isfile(coherence_path):
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
                coherence_result = get_coherence(ieeg_output_file,band_cutoffs[0],band_cutoffs[1])
                # save coherence_result to .npy file, create directory if it doesn't exist
                if not os.path.exists(func_directory):
                    os.makedirs(func_directory)
                # save pandas dataframe as csv
                coherence_result.to_csv(coherence_path)
                print(f"Coherence matrix saved for {rid}|night{night}|{stage}|{band_name}.")
            else:
                print(f"Coherence matrix already exists for {rid}|night{night}|{stage}|{band_name}. Skipping...")

# localize edges and assign z-scores
if (args.score == True):
    # for each sleep stage
    for stage in sleep_stages_to_run:
        # for each band
        for band_name in band_names:
            coherence_paths = []
            # find all coherence matrices (one for each patient) matching this stage and band name
            for k in range(len(rids)):
                rid = rids[k]
                iEEG_filename = all_pats[k]
                patient_directory = os.path.join(parent_directory,"sub-{}".format(rid))
                func_directory = os.path.join(patient_directory,'func')
                coherence_path = os.path.join(func_directory,f"sub-{rid}_{iEEG_filename}_{stage}_{band_name}_coherence.npy")
                if os.path.isfile(coherence_path):
                    coherence_paths.append(coherence_path)
            # load numpy arrays in coherence_paths
            coherences = [np.load(x) for x in coherence_paths]
            # for each coherence matrix, get a list of regions and corresponding coherences

            # calculate z-scores

            # take the 75th percentile z-score for each region

            # save cutoff z-score as matrix
            #np.save(score_output_file) # TODO

print("Done.")




