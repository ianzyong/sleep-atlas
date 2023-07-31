import os
import re
import sys
import time
from download_iEEG_data import get_iEEG_data
sys.path.append('../../ieegpy/ieeg')
import ieeg
from ieeg.auth import Session
import pandas as pd
import pickle
import numpy as np
import matlab
import matlab.engine
from fractions import Fraction
import scipy
import mne
from mne_bids import BIDSPath, write_raw_bids
import pyedflib
from tqdm import tqdm
from pqdm.processes import pqdm
import getpass
import argparse
import matplotlib.pyplot as plt
import matplotlib.dates as md
import gzip
import shutil
import datetime
import csv

# DEFINITIONS
# convert number of seconds to hh:mm:ss
def convertSeconds(time): 
    seconds = time % 3600 
    hours = time // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    if seconds < 10:
        seconds = "0" + str(seconds)
    if minutes < 10:
        minutes = "0" + str(minutes)
    if hours < 10:
        hours = "0" + str(hours)
    return ":".join(str(n) for n in [hours,minutes,seconds])

def convertTimeStamp(time):
    secs = time.second
    mins = time.minute
    hours = time.hour
    return hours*60*60 + mins*60 + secs

def padId(id):
    if len(id) == 5:
        id = id[0:3] + "0" + id[-2:]
    return id

def getStartTime(fst,x):
    id = padId(x.split("_")[0])
    if len(x.split("_")) > 2:
        ans = fst.loc[fst['name'] == id][int(x.split("_")[2][-1:])]
    else:
        ans = fst.loc[fst['name'] == id][1]
    return convertTimeStamp(ans.values[0])

def getNightNumber(ntc,x):
    id = padId(x.split("_")[0])
    ans = ntc.loc[ntc['patient'] == id]["night_to_classify"].values
    # if there is no night number, return 2
    if len(ans) == 0:
        return 2
    else:
        return ans[0]

def getNightSeizureStartTimes(dn,x,buffer=2*3600):
    id = padId(x.split("_")[0])
    # output list
    times = []
    # get entries of real_start that match id
    #night_seizure_df = dn.loc[(dn['Patient'] == id) & (dn["day_night_number"] % 2 == 0)]
    # for now, make sure the recording matches exactly
    night_seizure_df = dn.loc[(dn['IEEGname'] == x) & (dn["day_night_number"] % 2 == 0)]
    # for each unique string in night_seizure_df["IEEGname"] 
    for iEEGname in night_seizure_df["IEEGname"].unique():
        this_file_df = night_seizure_df.loc[night_seizure_df["IEEGname"] == iEEGname]
        names = this_file_df["IEEGname"].values
        starts = this_file_df["start"].values
        ends = this_file_df["end"].values
        # for each start time, check if the end time in the previous row is within 3600 seconds
        for k in range(len(starts)):
            # if not, add it to the list of night seizure cluster start times
            if (k == 0) or (starts[k] - ends[k-1] > buffer):
                times.append([names[k],starts[k]])
    return times

def get_edf(username, password, iEEG_filename, rid, start_time_usec, stop_time_usec, real_offset_sec, parent_directory, compress = False, overwrite = False, removed_channels = [], downsample = 200):
    # define paths
    patient_directory = os.path.join(parent_directory,"sub-{}".format(rid))
    eeg_directory = os.path.join(patient_directory,'eeg')
    output_file = os.path.join(eeg_directory,"sub-{}_{}_{}_{}_EEG.pickle".format(rid,iEEG_filename,start_time_usec,stop_time_usec))
    # .edf output path
    interval_name = f"{iEEG_filename}_{start_time_usec}_{stop_time_usec}"
    edf_file = os.path.join(eeg_directory,"{}.edf".format(interval_name))
    final_edf = edf_file

    # create necessary directories if they do not exist
    if not os.path.exists(patient_directory):
        os.makedirs(eeg_directory)

    if compress:
        final_edf = final_edf + ".gz"
    
    num_attempts = 5
    # is overwrite is False, check if .pickle file already exists
    if not overwrite:
        if os.path.exists(edf_file):
            print(f".edf file already exists for {interval_name}")
            return edf_file
        elif os.path.exists(final_edf):
            print(f".edf file already exists for {interval_name}")
            return final_edf
        elif os.path.exists(output_file) and os.path.getsize(output_file) > 0:
            print(f".pickle file already exists for {interval_name}")
        else:
            if os.path.exists(output_file) and os.path.getsize(output_file) == 0:
                print(f".pickle file already exists for {interval_name}, but is empty. Downloading again.")
                # delete empty file
                os.remove(output_file)
            # download iEEG data
            for attempt in range(num_attempts):
                try:
                    get_iEEG_data(username, password, iEEG_filename, start_time_usec, stop_time_usec, removed_channels, output_file)
                except ieeg.ieeg_api.IeegServiceError as e:
                    print(e)
                    #print("No iEEG data exists for this night, skipping.")
                    return
                except ieeg.ieeg_api.IeegConnectionError as e:
                    print(e)
                    print(repr(e))
                    print(f"Error encountered while downloading {iEEG_filename}, {start_time_usec} to {stop_time_usec}, retrying... ({attempt+1}/{num_attempts})")
                else:
                    break
            else:
                print(f"Error downloading {iEEG_filename}, {start_time_usec} to {stop_time_usec}, skipping.")
                # throw exception
                raise ieeg.ieeg_api.IeegConnectionError("Error downloading iEEG data.")
                
    else:
        # download iEEG data
        for attempt in range(num_attempts):
            try:
                get_iEEG_data(username, password, iEEG_filename, start_time_usec, stop_time_usec, removed_channels, output_file)
            except ieeg.ieeg_api.IeegServiceError:
                print("No iEEG data exists for this night, skipping.")
                return
            except ieeg.ieeg_api.IeegConnectionError:
                print(f"Error downloading {iEEG_filename}, {start_time_usec} to {stop_time_usec}, retrying... (attempt {attempt+1}/{num_attempts})")
            else:
                break
        else:
            print(f"Error downloading {iEEG_filename}, {start_time_usec} to {stop_time_usec}, skipping.")
            # throw exception
            raise ieeg.ieeg_api.IeegConnectionError("Error downloading iEEG data.")
    
    # read pickle and save as a .edf file
    try:
        pickle_data = pd.read_pickle(output_file)
    except EOFError:
        print("Empty pickle file, removing and skipping.")
        # delete output file
        os.remove(output_file)
        return
    signals = np.transpose(pickle_data[0].to_numpy())

    # get column names from pickle_data
    channel_names = pickle_data[0].columns.values.tolist()
    #print("Channel names: {}".format(channel_names))

    # change nan to zero
    signals = np.nan_to_num(signals)

    # get sample rate
    s = Session(username, password)
    ds = s.open_dataset(iEEG_filename)
    fs = ds.get_time_series_details(ds.ch_labels[0]).sample_rate

    print(f"Sample rate = {fs} hz, downsampling to {downsample} hz.")

    # downsample to 200 Hz
    new_fs = downsample
    frac = Fraction(new_fs, int(fs))
    signals = scipy.signal.resample_poly(
    signals, up=frac.numerator, down=frac.denominator, axis=1
    ) # (n_samples, n_channels)
    fs = new_fs 
    
    # write interval to an edf file
    signal_headers = pyedflib.highlevel.make_signal_headers(channel_names, physical_min=-50000, physical_max=50000, sample_rate=fs)
    #sample_rate = ds.sample_rate
    # determine datetime object, assume a start date of Jan 1st
    start_secs = start_time_usec/1e6 + real_offset_sec
    # declare timedelta object
    start_delta = datetime.timedelta(seconds=start_secs)
    # get datetime object assuming start date of Jan 1, 2020
    start_datetime = datetime.datetime(2020,1,1) + start_delta
    print(f"Start datetime = {start_datetime}")

    # write edf header
    header = pyedflib.highlevel.make_header(patientname=rid, startdate=start_datetime)
    
    # write .edf
    pyedflib.highlevel.write_edf(edf_file, signals, signal_headers, header)
    print(".edf saved.")

    if compress:
        # compress .edf file
        with open(edf_file, 'rb') as f_in:
            with gzip.open(final_edf, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        # remove .edf file
        os.remove(edf_file)
        print(".edf compressed.")
    
    # remove .pickle file
    os.remove(output_file)
    print(".pickle removed.")

    return final_edf

def classify_single_patient(username,password,iEEG_filename,rid,real_offset_sec,make_plot,night,extra_starts=[],preictal_sec=3600): 
    # username: iEEG.org username
    # password: iEEG.org password
    # iEEG_filename: iEEG.org filename
    # rid: patient number
    # real_offset_sec: offset in seconds
    # make_plot: boolean, whether or not to make a plot
    # night: night number
    # extra_starts: list of extra start times formatted as [iEEG.org filename, start time in seconds]
    # preictal_sec: number of seconds before each extra start time to include in extra files

    # download eeg data
    removed_channels = []
    parent_directory = os.path.join(os.path.dirname(os.path.abspath(os.getcwd())),'data')
    patient_directory = os.path.join(parent_directory,"sub-{}".format(rid))
    eeg_directory = os.path.join(patient_directory,'eeg')

    # get start and stop times 
    this_start = (20*60*60) - (real_offset_sec % (20*60*60))
    this_end = this_start + 12*60*60

    this_start += (night-1)*24*60*60
    this_end += (night-1)*24*60*60
    
    start_time_usec = this_start*1e6
    stop_time_usec = this_end*1e6

    print(f"\n{iEEG_filename}, night {night}:")

    # prefix for SleepSEEG output save path
    sleep_result_prefix = patient_directory+f"/{iEEG_filename}_{start_time_usec}_{stop_time_usec}_night{night}_"

    # if plot exists, skip
    #plot_file = os.path.join(sleep_result_prefix+"plot.png")
    if make_plot:
        regex = f"{iEEG_filename}_{start_time_usec}_{stop_time_usec}_night{night}_.*plot.png"
        if len([file for file in os.listdir(patient_directory) if re.search(regex, file)]):
            print("Plot already exists, skipping.")
            return

    # create necessary directories if they do not exist
    if not os.path.exists(patient_directory):
        os.makedirs(eeg_directory)

    # .pickle ouput path
    output_file = os.path.join(eeg_directory,"sub-{}_{}_{}_{}_EEG.pickle".format(rid,iEEG_filename,start_time_usec,stop_time_usec))
    # .edf output path
    interval_name = f"{iEEG_filename}_{start_time_usec}_{stop_time_usec}"

    # download main .edf file
    try:
        edf_file = get_edf(username, password, iEEG_filename, rid, start_time_usec, stop_time_usec, real_offset_sec, parent_directory, compress = True, overwrite = False)
    except ieeg.ieeg_api.IeegConnectionError:
        print(f"Error downloading {iEEG_filename}, {start_time_usec} to {stop_time_usec}, skipping patient.")
        return   

    # if edf_file is None, skip
    if edf_file is None:
        print(f"Error downloading {iEEG_filename}, {start_time_usec} to {stop_time_usec}, skipping patient.")
        return

    # get number of channels in main file
    # if edf_file ends with ".gz", decompress .edf
    if edf_file.endswith(".gz"):
        print("Decompressing .edf file...")
        with gzip.open(edf_file, 'r') as f_in, open(edf_file[0:-3], 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
        f = pyedflib.EdfReader(edf_file[0:-3])
        os.remove(edf_file[0:-3])
        print("Decompressed .edf removed.")
    else:
        f = pyedflib.EdfReader(edf_file)
    n_channels = len(f.getSignalLabels())
    print(f"Number of channels: {n_channels}")
    
    # download .edfs for each extra start time
    extra_files = []
    print(f"extra_starts = {extra_starts}")
    for extra_start in extra_starts:
        # print progress with number done
        print(f"Downloading extra file for {extra_start}... ({extra_starts.index(extra_start)+1}/{len(extra_starts)})")
        extra_filename = extra_start[0]
        stop = extra_start[1]*1e6 # convert to usec
        start = stop - (preictal_sec)*1e6 # convert to usec
        try:
            if start >= 0:
                extra_edf_path = get_edf(username, password, extra_filename, rid, start, stop, real_offset_sec, parent_directory, compress = False, overwrite = False)
            else:
                print("Start would be less than zero, skipping...")
                continue
        except ieeg.ieeg_api.IeegConnectionError:
            print(f"Error downloading {iEEG_filename}, {start_time_usec} to {stop_time_usec}, skipping patient.")
            return
        
        if extra_edf_path is None:
            print(f"Skipping extra file.")
        # if the number of channels in the extra file is the same as the main file, append it to extra_files
        elif len(pyedflib.EdfReader(extra_edf_path).getSignalLabels()) == n_channels:
                extra_files.append(extra_edf_path)
        else:
            print(f"Extra file {extra_filename} has a different number of channels than the main file, skipping... ({len(pyedflib.EdfReader(extra_edf_path).getSignalLabels())} channels vs {n_channels} channels in main file)")
        
    # add the number of extra hours to the prefix
    if len(extra_files) > 0:
        sleep_result_prefix += f"{len(extra_files)}preictalhrs_"

    # run SleepSEEG if results do not already exist
    if not os.path.isfile(sleep_result_prefix + "summary.csv"):
        # if edf_file ends with ".gz", decompress .edf
        if edf_file.endswith(".gz"):
            print("Decompressing .edf file...")
            with gzip.open(edf_file, 'r') as f_in, open(edf_file[0:-3], 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
            classify_edf = edf_file[0:-3]
        else:
            classify_edf = edf_file
        # run SleepSEEG in MATLAB
        eng = matlab.engine.start_matlab()
        try:
            eng.SleepSEEG(classify_edf,sleep_result_prefix,extra_files,nargout=0)
        except Exception as e:
            print(e)
            print("MATLAB error encountered, skipping to next night.")
            return
        print(f"Results saved to {patient_directory}.")

        if edf_file.endswith(".gz"):
            # delete decompressed .edf file
            os.remove(edf_file[0:-3])
            print("Decompressed .edf removed.")  
            
    else:
        print(f"Sleep stage classifications for night {night} already exist.")

    # if make_plot is True and the plot does not exist, make a plot
    plot_file = os.path.join(sleep_result_prefix+"plot.png")
    if make_plot:
        if not os.path.isfile(plot_file):
            # generate a plot of sleep stages for this night
            # read SleepSEEG output .csv
            sleepSEEG_output = pd.read_csv(sleep_result_prefix+"sleepstage.csv")
            # get x vals (time)
            # 86400 seconds in a day
            # 719529 is epoch time in seconds
            x = pd.to_datetime(sleepSEEG_output["time"].to_numpy()-719529, unit='D')
            #x = [datetime.datetime.fromtimestamp(ts*86400) for ts in sleepSEEG_output["time"]]
            # get y vals (sleep stage)
            y = sleepSEEG_output["sleep_stage"]
            # line plot
            fig = plt.figure()
            ax = fig.gca()
            # format dates
            xfmt = md.DateFormatter('%H:%M:%S')
            ax.xaxis.set_major_formatter(xfmt)  
            ax.plot(x,y)
            ax.set_title(f"Sleep stages for {iEEG_filename}, night {night}")
            ax.set_ylabel('sleep stage')
            ax.set_xlabel('time')
            ax.set_yticks([1,2,3,4,5])
            ax.set_yticklabels(["R","W","N1","N2","N3"])
            # save to .png
            plt.savefig(plot_file, bbox_inches='tight')
            print(f"Saved plot.")
        else:
            print("Plot already exists.")

# MAIN
parser = argparse.ArgumentParser()
parser.add_argument("username", help="iEEG.org username")
parser.add_argument("validation_path", help="path to manual validation .xlsx")
parser.add_argument("-r", "--reverse", help="run patients in reverse sorted order", action="store_true", default=False)
parser.add_argument("-p", "--plot", help="plot sleep stages for each night and save as a .png", action="store_true", default=False)
args = parser.parse_args()

username = args.username
validation_path = args.validation_path

# get password
password = getpass.getpass(prompt='IEEG.org password: ', stream=None)

if args.plot:
    print("Generating sleep stage plots for each night.")

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
# specify night number
# load nights_to_clasify.csv
ntc = pd.read_csv("nights_to_classify.csv")
# get night number
night_numbers = [getNightNumber(ntc,x) for x in all_pats]

# save all_pats and night_numbers to a .csv
with open("nights_classified_final.csv", "w") as f:
    writer = csv.writer(f)
    # write header
    writer.writerow(["patient","night_number"])
    writer.writerows(zip(all_pats,night_numbers))

# read excel file as dataframe
dn = pd.read_excel("day_night_seizure_data.xlsx")
# buffer for determining nighttime seizure clusters
buffer = 2*3600 # 2 hours
# get start times of nighttime seizure clusters for each patient
night_start_times = [getNightSeizureStartTimes(dn,x,buffer) for x in all_pats]

# preictal duration
preictal_dur = 3600 # 1 hour

# format argument list
args = [[username,password,all_pats[x],rids[x],start_times[x],args.plot,night_numbers[x],night_start_times[x],preictal_dur] for x in range(len(all_pats))]

# remove items in args that are not in hup_list.csv
hup_list = pd.read_csv("hup_list.csv", header=None)[0].tolist()
args = [x for x in args if (x[3] in hup_list)]

print(f"total number of patients = {len(args)}")
# print 3rd item in each arg
print([x[2] for x in args])

total_start = time.time()

# run patient classifications
for arg in args:
    classify_single_patient(*arg)

total_end = time.time()

print(f"Time elapsed = {convertSeconds(int(total_end - total_start))}")
print("Done.")
