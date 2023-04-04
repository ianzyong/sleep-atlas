from utils import *
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from fooof import FOOOF
import time

def convertSeconds(time): 
    days = time // (3600*24)
    time -= days*3600*24
    hours = time // 3600
    time -= hours*3600
    minutes = time // 60
    time -= minutes*60
    seconds = time
    return ":".join(str(n).rjust(2,'0') for n in [days,hours,minutes,seconds])

def convertTimeStamp(time):
    secs = time.second
    mins = time.minute
    hours = time.hour
    return hours*60*60 + mins*60 + secs

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

# get info from patient recording
with Session(args.user, pwd) as session:
    ds = session.open_dataset(dataset_name)
    duration = ds.get_time_series_details(ds.ch_labels[0]).duration
    labels = ds.ch_labels
    #print(duration)
    session.close_dataset(dataset_name)
    
all_ratios = []

# start timer
total_start = time.time()

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

    # remove NaNs
    #xi = np.arange(data.shape[1])
    #mask = np.isfinite(data)
    #data_filtered = np.vstack([np.interp(xi,xi[mask[k]], data.iloc[[k]][mask[k]]) for k in range(data.shape[0])])
    data.fillna(0)

    # calculate power spectrum
    freqs, ps = scipy.signal.welch(data.T, fs=fs)
    #freqs, ps = scipy.signal.welch(data_filtered.T, fs=fs)

    # take the log of the power spectrum
    #ps = np.log10(ps)

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
all_ratios[np.isnan(all_ratios)] = 0

# remove EKG channels
not_EKG = ["EKG" not in x for x in labels]
#print(not_EKG)
filter_inds = np.ravel(np.argwhere(not_EKG))
#print(filter_inds)
all_ratios = np.squeeze(all_ratios[filter_inds,:])
print(f"{np.sum(['EKG' in x for x in labels])} EKG channels removed.")
labels = [label for label in labels if "EKG" not in label]
print(labels)

print("Calculated ratios:")
print(all_ratios.shape)
print(all_ratios)

np.save(f"../data/{dataset_name}_ratios",all_ratios)
total_end = time.time()

print("Ratios saved to file. Intervals processed in {}.".format(convertSeconds(int(total_end - total_start))))

# determine sleep/wake
# average across channels
ad = np.nanmean(all_ratios, axis=0)

# normalize across times
norm_ad = np.divide((ad-np.nanmedian(ad)),scipy.stats.iqr(ad))
disc = -0.4054
# classify
is_awake = norm_ad > disc

#norm_ad histogram
plt.hist(norm_ad[~np.isnan(norm_ad)])
plt.axvline(x=disc,color='r',linestyle='dashed')
plt.xlabel("norm_ad")
plt.ylabel("Counts")
plt.savefig(f"{dataset_name}_norm_ad.png",bbox_inches="tight",dpi=600)
print("norm_ad figure saved to file.")
plt.close()

print("Sleep/wake classification:")
print(is_awake)

# sleep/wake colors
swc = ["#f5e642" if x else "#0005a1" for x in is_awake]

# plot result
plt.imshow(all_ratios, extent=[0,all_ratios.shape[1],0,all_ratios.shape[0]], cmap='hot', interpolation='nearest')
cbar = plt.colorbar()
cbar.set_label("Alpha to Delta Band Power Ratio")
plt.scatter(range(len(is_awake)),np.zeros((1,len(is_awake))),c=swc)
plt.xlabel("Time of day (s)")
plt.ylabel("Channel")
plt.xticks(range(int(duration)//int(args.interval_spacing_usec)+1),labels=[convertSeconds(offset_seconds+x) for x in range(int(0),int(duration/1000000),int(args.interval_spacing_usec/1000000))], fontsize=3, rotation = "vertical")
print(len(labels))
plt.yticks(range(len(labels)),labels=labels, fontsize=3)
plt.title(f"Patient = {args.iEEG_filename}")
fig = plt.gcf()
fig.set_size_inches(40, 8)
plt.savefig(f"{dataset_name}_heatmap.png",bbox_inches="tight",dpi=600)
print("Figure saved to file.")
plt.close()
