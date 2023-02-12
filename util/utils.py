# %%
# pylint: disable-msg=C0103
# pylint: disable-msg=W0703

# standard imports
from os.path import join as ospj
import pickle
from numbers import Number
import time
import re
from typing import Union
import itertools
from glob import glob

# nonstandard imports
from ieeg.auth import Session
import pandas as pd
import numpy as np
from scipy.signal import iirnotch, filtfilt, butter, welch, coherence
from scipy.integrate import simpson
import matplotlib.pyplot as plt
import seaborn as sns
from fooof import FOOOFGroup

################################################ Data I/O ################################################
def _pull_iEEG(ds, start_usec, duration_usec, channel_ids):
    '''
    Pull data while handling iEEGConnectionError
    '''
    i = 0
    while True:
        if i == 100:
            return None
        try:
            data = ds.get_data(start_usec, duration_usec, channel_ids)
            return data
        except Exception as _:
            time.sleep(1)
            i += 1

def get_iEEG_data(
    username: str, password_bin_file: str, iEEG_filename: str,
    start_time_usec: float, stop_time_usec: float,
    select_electrodes=None, ignore_electrodes=None, outputfile=None):
    """_summary_

    Args:
        username (str): _description_
        password_bin_file (str): _description_
        iEEG_filename (str): _description_
        start_time_usec (float): _description_
        stop_time_usec (float): _description_
        select_electrodes (_type_, optional): _description_. Defaults to None.
        ignore_electrodes (_type_, optional): _description_. Defaults to None.
        outputfile (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """


    start_time_usec = int(start_time_usec)
    stop_time_usec = int(stop_time_usec)
    duration = stop_time_usec - start_time_usec

    with open(password_bin_file, 'r') as f:
        pwd = f.read()

    while True:
        try:
            s = Session(username, pwd)
            ds = s.open_dataset(iEEG_filename)
            all_channel_labels = ds.get_channel_labels()
            break
        except Exception as e:
            time.sleep(1)

    all_channel_labels = clean_labels(all_channel_labels, iEEG_filename)

    if select_electrodes is not None:
        if isinstance(select_electrodes[0], Number):
            channel_ids = select_electrodes
            channel_names = [all_channel_labels[e] for e in channel_ids]
        elif isinstance(select_electrodes[0], str):
            select_electrodes = clean_labels(select_electrodes, iEEG_filename)
            if any([i not in all_channel_labels for i in select_electrodes]):
                raise ValueError('Channel not in iEEG')

            channel_ids = [i for i, e in enumerate(all_channel_labels) if e in select_electrodes]
            channel_names = select_electrodes
        else:
            print("Electrodes not given as a list of ints or strings")

    elif ignore_electrodes is not None:
        if isinstance(ignore_electrodes[0], int):
            channel_ids = [i for i in np.arange(len(all_channel_labels))
                if i not in ignore_electrodes]
            channel_names = [all_channel_labels[e] for e in channel_ids]
        elif isinstance(ignore_electrodes[0], str):
            ignore_electrodes = clean_labels(ignore_electrodes)
            channel_ids = [i for i, e in enumerate(all_channel_labels)
                if e not in ignore_electrodes]
            channel_names = [e for e in all_channel_labels if e not in ignore_electrodes]
        else:
            print("Electrodes not given as a list of ints or strings")

    else:
        channel_ids = np.arange(len(all_channel_labels))
        channel_names = all_channel_labels

    try:
        data =  _pull_iEEG(ds, start_time_usec, duration, channel_ids)
    except Exception as e:
        # clip is probably too big, pull chunks and concatenate
        clip_size = 60 * 1e6

        clip_start = start_time_usec
        data = None
        while clip_start + clip_size < stop_time_usec:
            if data is None:
                data = _pull_iEEG(ds, clip_start, clip_size, channel_ids)
            else:
                data = np.concatenate(
                    (data, _pull_iEEG(ds, clip_start, clip_size, channel_ids)),
                    axis=0)
            clip_start = clip_start + clip_size

    df = pd.DataFrame(data, columns=channel_names)
    fs = ds.get_time_series_details(ds.ch_labels[0]).sample_rate #get sample rate

    if outputfile:
        with open(outputfile, 'wb') as f:
            pickle.dump([df, fs], f)
    else:
        return df, fs

def clean_labels(channel_li: list, pt: str) -> list:
    """This function cleans a list of channels and returns the new channels

    Args:
        channel_li (list): _description_

    Returns:
        list: _description_
    """

    new_channels = []
    for i in channel_li:
        # standardizes channel names
        regex_match = re.match(r"(\D+)(\d+)", i)
        if regex_match is None:
            new_channels.append(i)
            continue
        lead = regex_match.group(1).replace("EEG", "").strip()
        contact = int(regex_match.group(2))

        if pt == "HUP89_phaseII":
            if lead == "GRID":
                lead = "RG"
            if lead == "AST":
                lead = "AS"
            if lead == "MST":
                lead = "MS"
        if pt == "HUP112_phaseII":
            if "-" in i:
                new_channels.append(f"{lead}{contact:02d}-{i.strip().split('-')[-1]}")
                continue
        if pt == "HUP116_phaseII":
                new_channels.append(f"{lead}{contact:02d}".replace("-", ""))
                continue
        if pt == "HUP123_phaseII_D02":
            if lead == "RS":
                lead = "RSO"
        new_channels.append(f"{lead}{contact:02d}")

    return new_channels

######################## BIDS ########################
BIDS_DIR = "/mnt/leif/littlab/data/Human_Data/CNT_iEEG_BIDS"
BIDS_INVENTORY = "/mnt/leif/littlab/users/pattnaik/ieeg_recon/migrate/cnt_ieeg_bids.csv"
def get_cnt_inventory(bids_inventory=BIDS_INVENTORY):
    inventory = pd.read_csv(bids_inventory, index_col=0)
    inventory = inventory == 'yes'
    return inventory


def get_pt_coords(pt):
    coords_path = glob(ospj(BIDS_DIR, pt, 'derivatives', 'ieeg_recon', 'module3', '*DKTantspynet*csv'))[0]
    return pd.read_csv(coords_path, index_col=0)
################################################ Plotting and Visualization ################################################
def plot_iEEG_data(data: Union[pd.DataFrame, np.ndarray], t: np.ndarray, colors=None, dr=None):
    """_summary_

    Args:
        data (Union[pd.DataFrame, np.ndarray]): _description_
        t (np.ndarray): _description_
        colors (_type_, optional): _description_. Defaults to None.
        dr (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    if data.shape[0] != np.size(t):
        data = data.T
    n_rows = data.shape[1]
    duration = t[-1] - t[0]

    fig, ax = plt.subplots(figsize=(duration / 7.5, n_rows / 5))
    sns.despine()

    ticklocs = []
    ax.set_xlim(t[0], t[-1])
    dmin = data.min().min()
    dmax = data.max().min()

    if dr is None:
        dr = (dmax - dmin) * 0.8 # Crowd them a bit.

    y0 = dmin
    y1 = (n_rows - 1) * dr + dmax
    ax.set_ylim(y0, y1)

    segs = []
    for i in range(n_rows):
        if isinstance(data, pd.DataFrame):
            segs.append(np.column_stack((t, data.iloc[:,i])))
        elif isinstance(data, np.ndarray):
            segs.append(np.column_stack((t, data[:,i])))
        else:
            print("Data is not in valid format")

    for i in reversed(range(n_rows)):
        ticklocs.append(i * dr)

    offsets = np.zeros((n_rows, 2), dtype=float)
    offsets[:, 1] = ticklocs

    # # Set the yticks to use axes coordinates on the y axis
    ax.set_yticks(ticklocs)
    if isinstance(data, pd.DataFrame):
        ax.set_yticklabels(data.columns)

    if colors:
        for col, lab in zip(colors, ax.get_yticklabels()):
            print(col, lab)
            lab.set_color(col)

    ax.set_xlabel('Time (s)')
    ax.plot(t, data +ticklocs, color='k', lw=0.4)

    return fig, ax

def plot_pi_features(features: np.ndarray, names: list):
    pass
################################################ Preprocessing ################################################
def notch_filter(data: np.ndarray, fs: float) -> np.array:
    """_summary_

    Args:
        data (np.ndarray): _description_
        fs (float): _description_

    Returns:
        np.array: _description_
    """
    # remove 60Hz noise
    b, a = iirnotch(60, 30, fs)
    data_filt = filtfilt(b, a, data, axis=0)
    # TODO: add option for causal filter
    # TODO: add optional argument for order

    return data_filt

def bandpass(data: np.ndarray, fs: float, order=3, lo=1, hi=120) -> np.array:
    """_summary_

    Args:
        data (np.ndarray): _description_
        fs (float): _description_
        order (int, optional): _description_. Defaults to 3.
        lo (int, optional): _description_. Defaults to 1.
        hi (int, optional): _description_. Defaults to 120.

    Returns:
        np.array: _description_
    """
    # TODO: add causal function argument
    # TODO: add optional argument for order
    bandpass_b, bandpass_a = butter(order, [lo, hi], btype='bandpass', fs=fs)
    data_filt = filtfilt(bandpass_b, bandpass_a, data, axis=0)

    return data_filt

def artifact_removal(data: np.ndarray, fs:float,
    discon=1/12, noise=15000, win_size=1) -> np.ndarray:
    """_summary_

    Args:
        data (np.ndarray): _description_
        fs (float): _description_
        discon (_type_, optional): _description_. Defaults to 1/12.
        noise (int, optional): _description_. Defaults to 15000.
        win_size (int, optional): _description_. Defaults to 1.

    Returns:
        np.ndarray: _description_
    """
    win_size = int(win_size * fs)
    ind_overlap = np.reshape(np.arange(data.shape[0]), (-1, int(win_size)))

    artifacts = np.empty_like(data)
    # mask indices with nan values
    artifacts = np.isnan(data).values
    for win_inds in ind_overlap:
        is_disconnected = np.sum(np.abs(data.iloc[win_inds]), axis=0) < discon

        is_noise = np.sqrt(np.sum(np.power(np.diff(data.iloc[win_inds], axis=0), 2), axis=0)) \
            > noise
        
        artifacts[win_inds, :] = np.logical_or(artifacts[win_inds, :].any(axis=0),
                                    np.logical_or(is_disconnected, is_noise))

    return artifacts


################################################ Feature Extraction ################################################

######################## Univariate, Time Domain ########################
def _timeseries_to_wins(data: pd.DataFrame, fs: float, win_size=2, win_stride=1) -> np.ndarray:
    """_summary_

    Args:
        data (pd.DataFrame): _description_
        fs (float): _description_
        win_size (int, optional): _description_. Defaults to 2.
        win_stride (int, optional): _description_. Defaults to 1.

    Returns:
        np.ndarray: _description_
    """
    n_samples = data.shape[-1]

    idx = np.arange(win_size*fs, dtype=int)[None, :] + \
        np.arange(n_samples - win_size*fs + 1, dtype=int)[::int(win_stride*fs), None]
    return data[:, idx]

def ft_extract(data: np.ndarray, fs: float, ft: str, win_size=2, win_stride=1) -> np.ndarray:
    """_summary_

    Args:
        data (np.ndarray): n_ch x n_wins
        fs (float): _description_
        ft (str): must be 'line_length'
        win_size (int, optional): _description_. Defaults to 2.
        win_stride (int, optional): _description_. Defaults to 1.

    Raises:
        ValueError: _description_

    Returns:
        np.ndarray: _description_
    """

    # in most cases, samples > ch
    assert data.shape[0] < data.shape[1], 'Reshape data to n_ch x n_wins'

    wins = _timeseries_to_wins(data, fs, win_size, win_stride)
    wins = np.transpose(wins, (1, 0, 2))
    (n_wins, n_ch, _) = wins.shape

    if ft == 'line_length':
        ft_array = np.empty((n_ch, n_wins))
        for i, win in enumerate(wins):
            ft_array[:, i] = _ll(win)
    else:
        raise ValueError("Incorrect feature argument given")

    return ft_array

def _ll(x):
    return np.sum(np.abs(np.diff(x)), axis=-1)


######################## Univariate, Spectral Domain ########################
bands = [
    [1, 4], # delta
    [4, 8], # theta
    [8, 12], # alpha
    [12, 30], # beta
    [30, 80], # gamma
    [1, 80] # broad
]
band_names = ["delta", "theta", "alpha", "beta", "gamma", "broad"]
N_BANDS = len(bands)

def _one_over_f(f: np.ndarray, b0: float, b1: float) -> np.ndarray:
    """_summary_

    Args:
        f (np.ndarray): _description_
        b0 (float): _description_
        b1 (float): _description_

    Returns:
        np.ndarray: _description_
    """
    return b0 - np.log10(f ** b1)

def spectral_features(data: np.ndarray, fs: float, win_size=2, win_stride=1) -> pd.DataFrame:
    """_summary_

    Args:
        data (np.ndarray): _description_
        fs (float): _description_

    Returns:
        pd.DataFrame: _description_
    """
    feature_names = [f"{i} power" for i in band_names] + ['b0', 'b1']

    freq, pxx = welch(
        x=data,
        fs=fs,
        window='hamming',
        nperseg=int(fs*win_size),
        noverlap=int(fs*win_stride),
        axis=0
    )

    # Initialize a FOOOF object
    fg = FOOOFGroup(verbose=False)

    # Set the frequency range to fit the model
    freq_range = [0.5, 80]

    # Report: fit the model, print the resulting parameters, and plot the reconstruction
    fg.fit(freq, pxx.T, freq_range)
    fres = fg.get_results()

    idx = np.logical_and(freq >= freq_range[0], freq <= freq_range[1])
    one_over_f_curves = np.array([_one_over_f(freq[idx], *i.aperiodic_params) for i in fres])

    residual = np.log10(pxx[idx]).T - one_over_f_curves
    freq = freq[idx]

    bandpowers = np.zeros((len(bands), pxx.shape[-1]))
    for i_band, (lo, hi) in enumerate(bands):
        if np.logical_and(60 >= lo, 60 <= hi):
            idx1 = np.logical_and(freq >= lo, freq <= 55)
            idx2 = np.logical_and(freq >= 65, freq <= hi)
            bp1 = simpson(
                y=residual[:, idx1],
                x=freq[idx1],
                dx=freq[1] - freq[0]
            )
            bp2 = simpson(
                y=residual[:, idx2],
                x=freq[idx2],
                dx=freq[1] - freq[0]
            )
            bandpowers[i_band] = bp1 + bp2
        else:
            idx = np.logical_and(freq >= lo, freq <= hi)
            bandpowers[i_band] = simpson(
                y=residual[:, idx],
                x=freq[idx],
                dx=freq[1] - freq[0]
            )
    aperiodic_params = np.array([i.aperiodic_params for i in fres])
    clip_features = np.row_stack((bandpowers, aperiodic_params.T))


    return pd.DataFrame(clip_features, index=feature_names, columns=data.columns)


def coherence_bands(data: Union[pd.DataFrame, np.ndarray], fs: float, win_size=2, win_stride=1) -> np.ndarray:
    """_summary_

    Args:
        data (Union[pd.DataFrame, np.ndarray]): _description_
        fs (float): _description_

    Returns:
        np.ndarray: _description_
    """
    _, n_channels = data.shape
    n_edges = sum(1 for i in itertools.combinations(range(n_channels), 2))
    n_freq = int(fs) + 1

    cohers = np.zeros((n_freq, n_edges))

    for i_pair, (ch1, ch2) in enumerate(itertools.combinations(range(n_channels), 2)):
        freq, pair_coher = coherence(
            data.iloc[:, ch1],
            data.iloc[:, ch2],
            fs=fs,
            window='hamming',
            nperseg=int(fs*win_size),
            noverlap=int(fs*win_stride))

        cohers[:, i_pair] = pair_coher

    # keep only between originally filtered range
    filter_idx = np.logical_and(freq >= 0.5, freq <= 80)
    freq = freq[filter_idx]
    cohers = cohers[filter_idx]

    coher_bands = np.empty((N_BANDS, n_edges))
    coher_bands[-1] = np.mean(cohers, axis=0)

    # format all frequency bands
    for i_band, (lower, upper) in enumerate(bands[:-1]):
        filter_idx = np.logical_and(freq >= lower, freq <= upper)
        coher_bands[i_band] = np.mean(cohers[filter_idx], axis=0)

    return coher_bands
