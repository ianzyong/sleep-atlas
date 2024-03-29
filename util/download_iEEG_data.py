""""
2020.04.06. Python 3.7
Andy Revell
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Purpose:
    To get iEEG data from iEEG.org. Note, you must download iEEG python package from GitHub - instructions are below
    1. Gets time series data and sampling frequency information. Specified electrodes are removed.
    2. Saves as a pickle format
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Input
    username: your iEEG.org username
    password: your iEEG.org password
    iEEG_filename: The file name on iEEG.org you want to download from
    start_time_usec: the start time in the iEEG_filename. In microseconds
    stop_time_usec: the stop time in the iEEG_filename. In microseconds.
        iEEG.org needs a duration input: this is calculated by stop_time_usec - start_time_usec
    ignore_electrodes: the electrode/channel names you want to exclude. EXACT MATCH on iEEG.org. Caution: some may be LA08 or LA8
    outputfile: the path and filename you want to save.
        PLEASE INCLUDE EXTENSION .pickle.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Output:
    Saves file outputfile as a pickel. For more info on pickeling, see https://docs.python.org/3/library/pickle.html
    Briefly: it is a way to save + compress data. it is useful for saving lists, as in a list of time series data and sampling frequency together along with channel names

    List index 0: Pandas dataframe. T x C (rows x columns). T is time. C is channels.
    List index 1: float. Sampling frequency. Single number
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Example usage:

username = 'username'
password = 'password'
iEEG_filename='HUP138_phaseII'
start_time_usec = 248432340000
stop_time_usec = 248525740000
removed_channels = ['EKG1', 'EKG2', 'CZ', 'C3', 'C4', 'F3', 'F7', 'FZ', 'F4', 'F8', 'LF04', 'RC03', 'RE07', 'RC05', 'RF01', 'RF03', 'RB07', 'RG03', 'RF11', 'RF12']
outputfile = '/Users/andyrevell/mount/DATA/Human_Data/BIDS_processed/sub-RID0278/eeg/sub-RID0278_HUP138_phaseII_248432340000_248525740000_EEG.pickle'
get_iEEG_data(username, password, iEEG_filename, start_time_usec, stop_time_usec, removed_channels, outputfile)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
To run from command line:
python3.6 -c 'import get_iEEG_data; get_iEEG_data.get_iEEG_data("arevell", "password", "HUP138_phaseII", 248432340000, 248525740000, ["EKG1", "EKG2", "CZ", "C3", "C4", "F3", "F7", "FZ", "F4", "F8", "LF04", "RC03", "RE07", "RC05", "RF01", "RF03", "RB07", "RG03", "RF11", "RF12"], "/gdrive/public/DATA/Human_Data/BIDS_processed/sub-RID0278/eeg/sub-RID0278_HUP138_phaseII_D01_248432340000_248525740000_EEG.pickle")'

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#How to extract pickled files
with open(outputfile, 'rb') as f: data, fs = pickle.load(f)
"""
import sys
import os
sys.path.append('../../ieegpy/ieeg')
import ieeg
from ieeg.auth import Session
import pandas as pd
import pickle
import numpy as np
import tqdm

def get_iEEG_data(username, password, iEEG_filename, start_time_usec, stop_time_usec, ignore_electrodes, outputfile, get_all_channels = False, redownload = False):
    print("\nGetting data from iEEG.org:")
    print("iEEG_filename: {0}".format(iEEG_filename))
    print("start_time_usec: {0}".format(start_time_usec))
    print("stop_time_usec: {0}".format(stop_time_usec))
    #print("ignore_electrodes: {0}".format(ignore_electrodes))

    if not os.path.isfile(outputfile) or redownload == True:
        # data has not yet been downloaded for this interval
        start_time_usec = int(start_time_usec)
        stop_time_usec = int(stop_time_usec)
        duration = stop_time_usec - start_time_usec
        s = Session(username, password)
        ds = s.open_dataset(iEEG_filename)
        channels = list(range(len(ds.ch_labels)))
        fs = ds.get_time_series_details(ds.ch_labels[0]).sample_rate #get sample rate
        
        #if duration is greater than ~10 minutes, then break up the API request to iEEG.org. 
        #The server blocks large requests, so the below code breaks up the request and 
        #concatenates the data
        server_limit_minutes = 5

        if duration < server_limit_minutes*60*1e6:
            data = ds.get_data(start_time_usec, duration, channels)
        if duration >= server_limit_minutes*60*1e6:
            break_times = np.ceil(np.linspace(start_time_usec, stop_time_usec, num=int(np.ceil(duration/(server_limit_minutes*60*1e6))+1), endpoint=True))
            #break_data = np.zeros(shape = (int(np.ceil(duration/1e6*fs)), len(channels)))#initialize
            break_data = np.empty(shape=(0,len(channels)), dtype=float)
            #print("breaking up data request from server because length is too long")
            for i in tqdm.trange(len(break_times)-1):
                #print("{0}/{1}".format(i+1, len(break_times)-1))
                break_data = np.append(break_data, ds.get_data(break_times[i], break_times[i+1]-break_times[i], channels), axis=0)
                #try:
                #    break_data[range(int( np.floor((break_times[i]-break_times[0])/1e6*fs) ), int(  np.floor((break_times[i+1]- break_times[0])/1e6*fs) )  ),:] = ds.get_data(break_times[i], break_times[i+1]-break_times[i], channels)
                #except ValueError as e:
                #    print(e)
                #    print("ValueError encountered in breaking data up for download, arrays are likely mishaped. Skipping...")
                #    return
            data = break_data

        df = pd.DataFrame(data, columns=ds.ch_labels)
        true_ignore_electrodes = []
        if ignore_electrodes != ['']:
            true_ignore_electrodes = get_true_ignore_electrodes(ds.ch_labels, ignore_electrodes)
        
        if not get_all_channels:
            df = pd.DataFrame.drop(df, true_ignore_electrodes, axis=1)

        print("Saving to: {0}".format(outputfile))
        with open(outputfile, 'wb') as f: pickle.dump([df, fs], f, protocol=4)
        #print("...done\n")
    else:
        s = Session(username, password)
        ds = s.open_dataset(iEEG_filename)

        true_ignore_electrodes = []
        if ignore_electrodes != ['']:
            true_ignore_electrodes = get_true_ignore_electrodes(ds.ch_labels, ignore_electrodes)
        # data has already been downloaded for this interval
        print("{} exists, skipping...".format(outputfile))

    s.close_dataset(iEEG_filename)
    # return ignore_electrodes as they are called on ieeg.org
    return fs, true_ignore_electrodes

def get_true_ignore_electrodes(labels, ignore_electrodes):
    true_ignore_electrodes = []
    for electrode in ignore_electrodes:
        if electrode in labels:
            true_ignore_electrodes.append(electrode)
        else:
            for i, c in enumerate(electrode):
                if c.isdigit():
                    break
            padded_num = electrode[i:].zfill(2)
            padded_name = electrode[0:i] + padded_num
            if padded_name in labels:
                true_ignore_electrodes.append(padded_name)
            elif "EEG {} {}-Ref".format(electrode[0:i],padded_num) in labels:
                true_ignore_electrodes.append("EEG {} {}-Ref".format(electrode[0:i],padded_num))
            elif "EEG {}-Ref".format(electrode) in labels:
                true_ignore_electrodes.append("EEG {}-Ref".format(electrode))
            else:
                print("Could not resolve electrode name {}, it will not be ignored.".format(electrode))
    return true_ignore_electrodes

""""
Download and install iEEG python package - ieegpy
GitHub repository: https://github.com/ieeg-portal/ieegpy

If you downloaded this code from https://github.com/andyrevell/paper001.git then skip to step 2
1. Download/clone ieepy. 
    git clone https://github.com/ieeg-portal/ieegpy.git
2. Change directory to the GitHub repo
3. Install libraries to your python Path. If you are using a virtual environment (ex. conda), make sure you are in it
    a. Run:
        python setup.py build
    b. Run: 
        python setup.py install
              
"""
