import os
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
from pqdm.processes import pqdm
import getpass

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

def getStartTime(fst,x):
    if len(x.split("_")) > 2:
        ans = convertTimeStamp(str(fst.loc[fst['name'] == x.split("_")[0]][x.split("_")[2][-1:]]).split(" ")[1])
    else:
        ans = convertTimeStamp(str(fst.loc[fst['name'] == x.split("_")[0]][1]).split(" ")[1])
    return ans

def classify_single_patient(username,password,iEEG_filename,rid,real_offset_sec): 
    # download eeg data
    removed_channels = []
    parent_directory = os.path.join(os.path.dirname(os.path.abspath(os.getcwd())),'data')
    patient_directory = os.path.join(parent_directory,"sub-{}".format(rid))
    eeg_directory = os.path.join(patient_directory,'eeg')

    # get dataset metadata
    s = Session(username, password)
    ds = s.open_dataset(iEEG_filename)
    channel_names = ds.get_channel_labels()
    fs = ds.get_time_series_details(ds.ch_labels[0]).sample_rate # get sample rate

    # get duration in seconds
    duration = ds.get_time_series_details(ds.ch_labels[0]).number_of_samples/fs

    # get list of start and stop times 
    this_start = (20*60*60) - (real_offset_sec % (20*60*60))
    this_end = this_start + 12*60*60
    nights = []
    while (this_start < duration) and (this_end < duration):
        nights.append([this_start,this_end])
        this_start += 24*60*60
        this_end += 24*60*60

    for night in nights:
        start = night[0]
        end = night[1]
        start_time_usec = start*1e6
        stop_time_usec = end*1e6

        # create necessary directories if they do not exist
        if not os.path.exists(patient_directory):
            os.makedirs(eeg_directory)
        output_file = os.path.join(eeg_directory,"sub-{}_{}_{}_{}_EEG.pickle".format(rid,iEEG_filename,start_time_usec,stop_time_usec))
        
        # download data if the file does not already exist
        if not os.path.isfile(output_file):
            get_iEEG_data(username, password, iEEG_filename, start_time_usec, stop_time_usec, removed_channels, output_file)
        else:
            pass
            #print("{} exists, skipping...".format(output_file))
        
        pickle_data = pd.read_pickle(output_file)
        signals = np.transpose(pickle_data[0].to_numpy())

        # change nan to zero
        signals = np.nan_to_num(signals)
    
        # downsample to 200 Hz
        new_fs = 200
        frac = Fraction(new_fs, int(fs))
        signals = scipy.signal.resample_poly(
        signals, up=frac.numerator, down=frac.denominator, axis=1
        ) # (n_samples, n_channels)
        fs = new_fs 
        
        # save the data
        # run is the iEEG file number
        # task is ictal with the start time in seconds appended
        data_info = mne.create_info(ch_names=list(channel_names), sfreq=fs, ch_types="eeg", verbose=False)
        raw = mne.io.RawArray(signals / 1e6, data_info, verbose=False)
        
        # write interval to an edf file
        signal_headers = pyedflib.highlevel.make_signal_headers(channel_names, physical_min=-50000, physical_max=50000)
        #sample_rate = ds.sample_rate
        header = pyedflib.highlevel.make_header(patientname=rid)

        # edf_file = os.path.join(patient_directory,"sub-{}_{}_{}_{}_EEG.edf".format(rid,iEEG_filename,start_time_usec,stop_time_usec))
        interval_name = f"{rid}_{start_time_usec}_{stop_time_usec}"

        edf_file = os.path.join(patient_directory,"{}.edf".format(interval_name))
        
        pyedflib.highlevel.write_edf(edf_file, signals, signal_headers, header)

        # run SleepSEEG in MATLAB
        eng = matlab.engine.start_matlab()
        eng.SleepSEEG(edf_file,patient_directory+"/",nargout=0)

# MAIN
if len(sys.argv) == 3:
    # get the filename of the list of patients to run
    username = sys.argv[1]
    validation_path = sys.argv[2]
else:
    # get path to manual validation
    validation_path = input('Path to manual validation .xlsx: ')
    # get username
    username = input('IEEG.org username: ')

# get password
password = getpass.getpass(prompt='IEEG.org password: ', stream=None)

# read xlsx
xls = pd.ExcelFile("manual_validation.xlsx")
# read file containing patients
ast = pd.read_excel(xls, "AllSeizureTimes")
# list of iEEG_filename
all_pats = list(set(ast["IEEGname"]))
# get list of rids
rids = [x.split("_")[0] for x in all_pats]
# get actual start times
fst = pd.read_excel(xls, "FileStartTimes")
# get list of real_offset_usec
start_times = [getStartTime(fst,x) for x in all_pats]
# format argument list to pass to pqdm
args = [[username,password,all_pats[x],rids[x],start_times[x]] for x in range(len(all_pats))]

total_start = time.time()

# run patient classifications in parallel
result = pqdm(args, classify_single_patient, n_jobs=2, argument_type='args')

total_end = time.time()

print(f"Time elapsed = {convertSeconds(int(total_end - total_start))}")
print("Done.")

        

