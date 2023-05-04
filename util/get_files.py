import os
import sys
import time
from download_iEEG_data import get_iEEG_data
import numpy as np
import pandas as pd
sys.path.append('../../ieegpy/ieeg')
import ieeg
from ieeg.auth import Session
import pyedflib

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

if __name__ == '__main__':
    # input parameters
    username = input('IEEG.org username: ')
    password = input('IEEG.org password: ')
    #window_length = int(input('Input window_length (integer number of seconds): '))
    # initilize list to hold tuples corresponding to each patient
    patient_list = []
    # if there are arguments
    if len(sys.argv) > 1:
        # get the filename of the list of patients to run
        patient_metadata = sys.argv[1]
        with open(patient_metadata) as f:
            lines = [line.rstrip() for line in f]
            for k in range(0,len(lines)-4,5):
                # add patient tuple to the list
                patient_list.append(tuple(lines[k:k+5]))
                
    else: # if no arguments are supplied, run just one patient
        iEEG_filename = input('Input iEEG_filename: ')
        rid = input('Input RID: ')
        start_time_usec = int(input('Input start_time_usec: '))
        stop_time_usec = int(input('Input stop_time_usec: '))
        removed_channels = input('Input removed_channels: ')
        #removed_channels = removed_channels.split(",")
        patient_list.append(tuple([iEEG_filename,rid,start_time_usec,stop_time_usec,removed_channels]))

    total_size = 0
    for patient in patient_list:
        start_time_usec = int(patient[2])
        stop_time_usec = int(patient[3])
        # calculate duration requested an estimate the space required on disk for the eeg data
        total_time = (stop_time_usec - start_time_usec)//1000000
        eeg_file_size = round((total_time/60)*(48962/1024**2),4)
        total_size += eeg_file_size
        print("{}:".format(patient[1]))
        print("Duration requested = {}. Estimated space required = {} GB.".format(convertSeconds(total_time),eeg_file_size))
    print("Total number of intervals requested = {}. Total estimated space required = {} GB.".format(len(patient_list),total_size))
    answer = input("Proceed? (y/n) ")

    # download eeg data
    if answer == 'y' or answer == 'Y':
        process_count = 0
        # start timer
        total_start = time.time()
        for patient in patient_list:
            iEEG_filename = patient[0]
            rid = patient[1]
            start_time_usec = patient[2]
            stop_time_usec = patient[3]
            removed_channels = patient[4].split(",")
            parent_directory = os.path.join(os.path.dirname(os.path.abspath(os.getcwd())),'data')
            patient_directory = os.path.join(parent_directory,"sub-{}".format(rid))
            eeg_directory = os.path.join(patient_directory,'eeg')
            
            # create necessary directories if they do not exist
            if not os.path.exists(patient_directory):
                os.makedirs(eeg_directory)
            output_file = os.path.join(eeg_directory,"sub-{}_{}_{}_{}_EEG.pickle".format(rid,iEEG_filename,start_time_usec,stop_time_usec))
            
            # download data if the file does not already exist
            if not os.path.isfile(output_file):
                get_iEEG_data(username, password, iEEG_filename, start_time_usec, stop_time_usec, removed_channels, output_file)
                process_count += 1
            else:
                print("{} exists, skipping...".format(output_file))
            
            pickle_data = pd.read_pickle(output_file)
            signals = np.transpose(pickle_data[0].to_numpy())

            # change nan to zero
            signals = np.nan_to_num(signals)

            s = Session(username, password)
            ds = s.open_dataset(iEEG_filename)
            channel_names = ds.get_channel_labels()

            signal_headers = pyedflib.highlevel.make_signal_headers(channel_names, physical_min=-50000, physical_max=50000, transducer="seeg")
            header = pyedflib.highlevel.make_header(patientname=rid)
            edf_file = os.path.join(patient_directory,"{}_{}_{}.edf".format(iEEG_filename,start_time_usec,stop_time_usec))
            pyedflib.highlevel.write_edf(edf_file, signals, signal_headers, header)

        total_end = time.time()
        print("{}/{} intervals(s) processed in {}.".format(process_count,len(patient_list),convertSeconds(int(total_end - total_start))))
        print("Done.")
