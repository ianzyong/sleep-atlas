
from utils import *
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from fooof import FOOOF
import time
import os
import openpyxl

def convertSeconds(time): 
    days = time // (3600*24)
    time -= days*3600*24
    hours = time // 3600
    time -= hours*3600
    minutes = time // 60
    time -= minutes*60
    seconds = time
    return ":".join(str(int(n)).rjust(2,'0') for n in [days,hours,minutes,seconds])

def convertTimeStamp(time):
    secs = time.second
    mins = time.minute
    hours = time.hour
    return hours*60*60 + mins*60 + secs

parser = argparse.ArgumentParser()
parser.add_argument('interval_spacing_usec', type=int)
parser.add_argument('interval_duration_usec', type=int)
args = parser.parse_args()

# read file with start times
start_times = pd.read_excel("start_times.xlsx")

dir_list = os.listdir("../data/")

ratios_list = [x for x in dir_list if x.endswith("_ratios.npy")]

print(f"{len(ratios_list)} ratio files found:")
print(ratios_list)

input("Press enter to continue...")

rows_to_write = []

for file_name in ratios_list:
    dataset_name = file_name[0:-11]
    print(f"Finding sleep periods for {dataset_name}...")
    
    # read file with start times
    start_times = pd.read_excel("start_times.xlsx")

    # get real start time in seconds
    name_parts = dataset_name.split("_")
    if (len(name_parts) < 3):
        timestamp = start_times.loc[start_times['name'] == name_parts[0],1].values[0]
    else:
        timestamp = start_times.loc[start_times['name'] == name_parts[0],int(name_parts[2][-1])].values[0]

    offset_seconds = convertTimeStamp(timestamp)

    print(f"Actual start time = {timestamp} = {offset_seconds} seconds")
    
    # determine number of sleep periods

    # read in ratio values

    all_ratios = np.load(f"../data/{file_name}")

    print("all_ratios:")
    print(all_ratios)

    # replae NaN values
    all_ratios = np.nan_to_num(all_ratios)

    # determine sleep/wake
    # average across channels
    ad = np.nanmean(all_ratios, axis=0)
    # normalize across times
    norm_ad = np.divide((ad-np.nanmedian(ad)),scipy.stats.iqr(ad))
    disc = -0.4054
    # classify
    is_awake = norm_ad > disc
    print("is_awake:")
    print(is_awake)
    # flatten array if necessary
    is_awake = is_awake.flatten()
    # bool to int
    is_awake = [int(x) for x in is_awake]
    
    # filter classifier output
    N = 8
    filter_class = scipy.ndimage.uniform_filter1d(is_awake, size=N)

    print("Filtered classifier data:")
    print(filter_class)
    zero_crossings = np.where(np.diff(np.signbit(np.subtract(filter_class,0.5))))[0]
    #k = 0
    night_counter = 1
    this_pat_rows = []
    
    period_starts = [0] + zero_crossings
    for m in range(len(period_starts)):
        p_start = period_starts[m]
        if filter_class[p_start+1] < 0.5: # if this is a sleep period
            row_to_write = []
            row_to_write.append(name_parts[0]) # patient ID
            row_to_write.append(night_counter) # night number
            night_counter += 1
            this_start = int((args.interval_spacing_usec//2)+(p_start-1)*(args.interval_spacing_usec))
            row_to_write.append(this_start + offset_seconds*1e6) # real start time
            if (m < len(period_starts)-1): # if this is not the last period
                p_end = period_starts[m+1]
                this_end = int((args.interval_spacing_usec//2)+(args.interval_spacing_usec*(p_end)))
            else:
                p_end = len(filter_class)
                this_end = int(args.interval_spacing_usec*(p_end-1) + args.interval_duration_usec) 
            row_to_write.append(this_end + offset_seconds*1e6) # real end time
            row_to_write.append(this_start) # iEEG.org start time
            row_to_write.append(this_end) # iEEG.org end time
            row_to_write.append(convertSeconds(offset_seconds+(this_start//1e6))) # start timestamp
            row_to_write.append(convertSeconds(offset_seconds+(this_end//1e6))) # end timestamp
            row_to_write.append(convertSeconds((this_end-this_start)//1e6)) # duration

            this_pat_rows.append(row_to_write)
    
    print("Rows for this patient:")
    print(this_pat_rows)
    rows_to_write = rows_to_write + this_pat_rows
    
# initialize dataframe
period_df = pd.DataFrame(columns=["ID","sleep_num","real_start_us","real_end_us","ieeg_start_us","ieeg_end_us","start_timestamp","end_timestamp","duration"], data=rows_to_write)
print(period_df)
period_df.to_excel("../data/sleep_periods.xlsx")
print("Sleep periods saved to disk.")

