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
from tqdm import tqdm
from pqdm.processes import pqdm
import getpass
import argparse
import matplotlib.pyplot as plt

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

def classify_single_patient(username,password,iEEG_filename,rid,real_offset_sec,make_plot): 
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
    
    print(" ")
    print(f"number of nights for {iEEG_filename} = {len(nights)}")
    for k in range(len(nights)):
        night = nights[k]
        start = night[0]
        end = night[1]
        start_time_usec = start*1e6
        stop_time_usec = end*1e6

        print(f"Night {k+1}:")

        # create necessary directories if they do not exist
        if not os.path.exists(patient_directory):
            os.makedirs(eeg_directory)

        # .pickle ouput path
        output_file = os.path.join(eeg_directory,"sub-{}_{}_{}_{}_EEG.pickle".format(rid,iEEG_filename,start_time_usec,stop_time_usec))
        # .edf output path
        interval_name = f"{iEEG_filename}_{start_time_usec}_{stop_time_usec}"
        edf_file = os.path.join(patient_directory,"{}.edf".format(interval_name))

        # download .pickle and write to .edf if the .edf file does not already exist
        if not os.path.isfile(edf_file):
            # check to see if the .edf exists under an old file name
            old_edf = os.path.join(patient_directory,f"{rid}_{start_time_usec}_{stop_time_usec}.edf")
            if os.path.isfile(old_edf):
                # rename the old file
                os.rename(old_edf,edf_file)
                print("old .edf file name detected and renamed.")
            else:
                # download .pickle file
                get_iEEG_data(username, password, iEEG_filename, start_time_usec, stop_time_usec, removed_channels, output_file)
                
                # if the .pickle failed to download, skip this night
                if not os.path.isfile(output_file):
                    print(f"Skipping {iEEG_filename}, {start_time_usec} to {stop_time_usec}.")
                    continue

                # read pickle and save as a .edf file
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
                #data_info = mne.create_info(ch_names=list(channel_names), sfreq=fs, ch_types="eeg", verbose=False)
                #raw = mne.io.RawArray(signals / 1e6, data_info, verbose=False)
                
                # write interval to an edf file
                signal_headers = pyedflib.highlevel.make_signal_headers(channel_names, physical_min=-50000, physical_max=50000, sample_rate=fs)
                #sample_rate = ds.sample_rate
                header = pyedflib.highlevel.make_header(patientname=rid)
                
                # write .edf
                pyedflib.highlevel.write_edf(edf_file, signals, signal_headers, header)
                print(".edf saved.")

        else:
            print(f".edf for {rid} exists, skipping download.")
        
        if os.path.isfile(output_file):
            # delete .pickle file to save space
            os.remove(output_file)
            print(".pickle file removed.")
        
        # run SleepSEEG if results do not already exist
        if not os.path.isfile(patient_directory+f"/{iEEG_filename}_{start_time_usec}_{stop_time_usec}_night{k+1}_summary.csv"):
            # run SleepSEEG in MATLAB
            eng = matlab.engine.start_matlab()
            try:
                eng.SleepSEEG(edf_file,patient_directory+f"/{iEEG_filename}_{start_time_usec}_{stop_time_usec}_night{k+1}_",nargout=0)
            except Exception as e:
                print(e)
                print("MATLAB error encountered, skipping to next night.")
                continue
            print(f"Results saved to {patient_directory}.")    
        else:
            print(f"Sleep stage classifications for night {k+1} already exist.")

        # if make_plot is True and the plot does not exist, make a plot
        plot_file = os.path.join(patient_directory,f"{iEEG_filename}_{start_time_usec}_{stop_time_usec}_night{k+1}_plot.png")
        if make_plot:
            if not os.path.isfile(plot_file):
                # generate a plot of sleep stages for this night
                # read SleepSEEG output .csv
                sleepSEEG_output = pd.read_csv(patient_directory+f"/{iEEG_filename}_{start_time_usec}_{stop_time_usec}_night{k+1}_sleepstage.csv")
                # get x vals (time)
                x = sleepSEEG_output["time"]
                # get y vals (sleep stage)
                y = sleepSEEG_output["sleep_stage"]
                # line plot
                fig = plt.figure()
                ax = fig.gca()
                ax.plot(x,y)
                ax.set_title(f"Sleep stages for {iEEG_filename}, night {k+1}")
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
# read file containing patients
ast = pd.read_excel(xls, "AllSeizureTimes")
# list of iEEG_filename
all_pats = list(set(ast["IEEGname"]))
if args.reverse:
    print("Running patients in reverse sorted order.")
all_pats = sorted([x for x in all_pats if str(x) != 'nan'], reverse=args.reverse)
# get list of rids
rids = [padId(x.split("_")[0]) for x in all_pats]
# get actual start times
fst = pd.read_excel(xls, "FileStartTimes")
# get list of real_offset_usec
start_times = [getStartTime(fst,x) for x in all_pats]
# format argument list to pass to pqdm
args = [[username,password,all_pats[x],rids[x],start_times[x],args.plot] for x in range(len(all_pats))]

print(f"total number of patients = {len(args)}")

total_start = time.time()

# run patient classifications
for arg in tqdm(args):
    classify_single_patient(*arg)

#result = pqdm(args, classify_single_patient, n_jobs=2, argument_type='args')

total_end = time.time()

print(f"Time elapsed = {convertSeconds(int(total_end - total_start))}")
print("Done.")

        

