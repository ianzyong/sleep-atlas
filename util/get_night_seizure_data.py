
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

# read file containing seizure occurences
#xls = pd.ExcelFile(r"sleep-atlas\util\manual_validation.xlsx")
xls = pd.ExcelFile("manual_validation.xlsx")
seizure_times = pd.read_excel(xls, "AllSeizureTimes")
seizure_data = seizure_times[["Patient", "IEEGID", "IEEGname", "start", "end"]]
seizure_data = seizure_data.dropna(axis=0)

start_times = pd.read_excel("start_times.xlsx")
#start_times = pd.read_excel(r"sleep-atlas\util\start_times.xlsx")

#print(seizure_data.iloc[range(0,20)])

real_start = []
real_end = []
day_night_number = []
day_night_label = []

for index, row in seizure_data.iterrows():

    # get real start time in seconds
    name_parts = row['IEEGname'].split("_")
    # pad number
    if (len(name_parts[0]) < 6):
        name_parts[0] = f"{name_parts[0][0:3]}0{name_parts[0][3:5]}"
    try:
        if (len(name_parts) < 3):
            timestamp = start_times.loc[start_times['name'] == name_parts[0],1].values[0]
        else:
            timestamp = start_times.loc[start_times['name'] == name_parts[0],int(name_parts[2][-1])].values[0]
    except:
        print(f"start time data not found for {name_parts}, skipping...")
        continue

    offset_seconds = convertTimeStamp(timestamp)

    rs = row['start']+offset_seconds
    re = row['end']+offset_seconds
    real_start.append(rs)
    real_end.append(re)

    dn_num = int((rs+(4*60*60))//(12*60*60))
    day_night_number.append(dn_num)

    cycle_num = dn_num // 2
    if (dn_num % 2 == 0):
        day_night_label.append(f"night {cycle_num+1}")
    else:
        day_night_label.append(f"day {cycle_num+1}")

# add columns to dataframe
seizure_data['real_start'] = real_start
seizure_data['real_end'] = real_end
seizure_data['day_night_number'] = day_night_number
seizure_data['day_night_label'] = day_night_label
print(seizure_data)
seizure_data.to_excel("../data/day_night_seizure_data.xlsx")
#seizure_data.to_excel("day_night_seizure_data.xlsx")
print("Day/night seizure data saved to disk.")

