from utils import *
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from fooof import FOOOF

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

figure_saved = False

parser = argparse.ArgumentParser()
parser.add_argument('-u', '--user', required=True, help='username')
parser.add_argument('-p', '--password', required=True, help='password')
parser.add_argument('iEEG_filename', type=str)
parser.add_argument('interval_spacing_usec', type=int)
parser.add_argument('interval_duration_usec', type=int)

args = parser.parse_args()

with open(args.password, 'r') as f:
    pwd = f.read()

dataset_name = args.iEEG_filename

# get info from patient recording
with Session(args.user, pwd) as session:
    ds = session.open_dataset(dataset_name)
    duration = ds.get_time_series_details(ds.ch_labels[0]).duration
    labels = ds.ch_labels
    #print(duration)
    session.close_dataset(dataset_name)
    
all_ratios = []

# iterate through time clips
for start in range(0,int(duration),int(args.interval_spacing_usec)):

    print(f"Start = {start} out of {int(duration)}")
    
    # params
    params = {
        "username": args.user,
        "password_bin_file": args.password,
        "iEEG_filename": args.iEEG_filename,
        "start_time_usec": start,
        "stop_time_usec": start + args.interval_duration_usec
    }
    
    #print(params)

    data, fs = get_iEEG_data(**params)
    #print(data.shape)

    # calculate power spectrum
    freqs, ps = scipy.signal.welch(data.T, fs=fs)

    #print(freqs.shape)
    #print(freqs)
    #print(ps.shape)
    #print(ps)
    
    if (figure_saved == False):
        fm = FOOOF()
        freq_range = [2,80]
        #fm.fit(freqs, ps[0], freq_range)
        #fm.print_results()
        figure_saved = True

    # calculate bandpower
    alpha_f = [8,12]
    delta_f = [0.5, 4]

    alpha_ind_min = np.argmax(freqs > alpha_f[0]) - 1
    alpha_ind_max = np.argmax(freqs > alpha_f[1]) - 1
    alpha_band_power = np.trapz(ps[:,alpha_ind_min:alpha_ind_max], freqs[alpha_ind_min:alpha_ind_max])

    delta_ind_min = np.argmax(freqs > delta_f[0]) - 1
    delta_ind_max = np.argmax(freqs > delta_f[1]) - 1
    delta_band_power = np.trapz(ps[:,delta_ind_min:delta_ind_max], freqs[delta_ind_min:delta_ind_max])

    ad_ratios = np.reshape(np.divide(alpha_band_power,delta_band_power), (-1,1))
    all_ratios.append(ad_ratios)

all_ratios = np.hstack(all_ratios)
print(all_ratios.shape)
print(all_ratios)

# plot result
plt.imshow(all_ratios, extent=[0,all_ratios.shape[1],0,all_ratios.shape[0]], cmap='hot', interpolation='nearest')
plt.xlabel("Window Start (s)")
plt.ylabel("Channel")
plt.xticks(range(int(duration)//int(args.interval_spacing_usec)+1),labels=[convertSeconds(x) for x in range(int(0),int(duration/1000000),int(args.interval_spacing_usec/1000000))], fontsize=4, rotation = "vertical")
plt.yticks(range(len(labels)),labels=labels, fontsize=4)
plt.title(f"Patient = {args.iEEG_filename}")
cbar = plt.colorbar()
cbar.set_label("Alpha to Delta Band Power Ratio")
#plt.tight_layout()
fig = plt.gcf()
fig.set_size_inches(40, 8)
plt.savefig("heatmap.png",bbox_inches="tight",dpi=600)
print("Figure saved to file.")
