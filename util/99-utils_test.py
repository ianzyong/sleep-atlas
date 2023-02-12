#%%
%load_ext autoreload
%autoreload 2

import json

from utils import *

# %%
# paths
with open("config.json", "rb") as f:
    config = json.load(f)
# %%
# params
params = {
    "username": config['usr'],
    "password_bin_file": config['pwd'],
    "iEEG_filename": "HUP172_phaseII",
    "start_time_usec": 200000 * 1e6,
    "stop_time_usec": 200010 * 1e6,
    "select_electrodes": ["LB1", "LB2"]
}

# %%
data, fs = get_iEEG_data(**params)
t = np.linspace(0, 10, data.shape[0])

# %%
artifacts = artifact_removal(data, fs)
# set channels with more than 20% artifacts to nan
remv_ch_idx = artifacts.sum(axis=0) / artifacts.shape[0] > 0.2
data_clean = data.copy()
remv_ch = data.columns[remv_ch_idx]
data_clean[remv_ch] = np.nan

# calculate spectral features on unpreprocessed data
# having nan causes an error with the spectral features, instead, remove them
spec_features = spectral_features(data.iloc[:, ~remv_ch_idx], fs)

for idx, name in zip(np.where(remv_ch_idx)[0], remv_ch):
    spec_features.insert(loc = int(idx), column = name, value = np.nan)

# remove 60Hz noise and bandpass
data_filt = notch_filter(data_clean, fs)
data_filt = bandpass(data_filt, fs)
data_preprocessed = pd.DataFrame(data_filt, columns=data_clean.columns)

# calculate coherence
coher_bands = coherence_bands(data_preprocessed, fs)

# %%
plot_iEEG_data(data, t)
plot_iEEG_data(data_preprocessed, t)
# %%
