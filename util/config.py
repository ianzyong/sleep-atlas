import sys
from os.path import join as ospj
import pandas as pd
import numpy as np

from utils import *

class CONFIG:
    """Configuration class for the project."""
    # Device configuration
    is_interactive = hasattr(sys, "ps1")


    # paths
    bids_dir = "/mnt/leif/littlab/data/Human_Data/CNT_iEEG_BIDS"
    spikes_dir = "/mnt/leif/littlab/users/aguilac/Projects/Spike_Rates/gen_w_chlabels"
    data_dir = "/mnt/leif/littlab/users/pattnaik/ictal_patterns/data"
    fig_dir = "/mnt/leif/littlab/users/pattnaik/ictal_patterns/figures"
    log_dir = "/mnt/leif/littlab/users/pattnaik/ictal_patterns/code/logs/"

    # plotting and colors
    plotting_kwargs = dict(
        scalings=dict(eeg=2e-4),
        show_scrollbars=False,
        show=True,
        block=True,
        verbose=False,
        n_channels=20,
        duration=30,
        )
    stage_colors = {
        "N2": "#20c5e3",
        "N3": "#31839e",
        "R": "#2a4858",
        "W": "#e37720",
    }
    elec_type_colors = {
        'norm MNI': 'limegreen',
        'norm HUP': 'darkgreen',
        'irritative': 'darkorange',
        'soz': 'firebrick',
    }

    # ieeg.org login
    usr = "pattnaik"
    pwd = "/mnt/leif/littlab/users/pattnaik/pat_ieeglogin.bin"

    # constants
    ##############################
    # interictal clip size
    clip_size = 60  # seconds

    # spike threshold
    spike_thresh = 2

    # metadata I/O
    ##############################
    rid_hup_musc_table = pd.read_csv(
        ospj(data_dir, "metadata/rid_hup_table.csv"), index_col=0
    )
    rid_hup_table = rid_hup_musc_table.dropna(subset=["hupsubjno"])
    for ind, row in rid_hup_table.iterrows():
        rid_hup_table.loc[ind, "hupsubjno"] = int(row["hupsubjno"][:3])
    del ind, row
    rid_hup_table.index = [f"sub-RID{x:04d}" for x in rid_hup_table.index]
    rid_hup_table['hupsubjno'] = [f"HUP{x:03d}" for x in rid_hup_table['hupsubjno']]

    rid_to_hup = rid_hup_table['hupsubjno']
    rid_to_hup = rid_to_hup.to_dict()
    # invert dict
    hup_to_rid = {v: k for k, v in rid_to_hup.items()}

    rid_musc_table = rid_hup_musc_table.dropna(subset=["muscsubjno"])
    rid_musc_table.index = [f"sub-RID{x:04d}" for x in rid_musc_table.index]

    rid_to_musc = rid_musc_table['muscsubjno']
    rid_to_musc = rid_to_musc.to_dict()
    # invert dict
    musc_to_rid = {v: k for k, v in rid_to_musc.items()}


    del rid_hup_table
    
    ##############################
    interictal_metadata = pd.read_excel(
        ospj(data_dir, "metadata/atlas_metadata_final_updated_interictal.xlsx")
    )

    ##############################
    sz_times = pd.read_excel(
        ospj(data_dir, "metadata/Manual validation.xlsx"),
        sheet_name="AllSeizureTimes",
        index_col=0,
    )
    # drop nan rows
    sz_times.dropna(inplace=True, subset=["IEEGname"])

    ##############################
    soz_metadata = pd.read_excel(
        ospj(data_dir, "metadata/Manual validation.xlsx"), sheet_name="SOZ"
    )

    ##############################
    # patient table
    patient_tab = pd.read_excel(
        ospj(data_dir, "metadata/master_pt_table_manual.xlsx"),
        index_col=0)

    ##############################
    # mni files
    # iterate through the mni_reg for MNI rows and populate all_signals
    ch_info = pd.read_csv(ospj(data_dir, 'mni', 'Information', 'ChannelInformation.csv'))
    # remove the apostrophes at beginning and end of "Channel name" "Electrode type" and "Hemisphere" in ch_info
    ch_info['Channel name'] = ch_info['Channel name'].str[1:-1]
    ch_info['Electrode type'] = ch_info['Electrode type'].str[1:-1]
    ch_info['Hemisphere'] = ch_info['Hemisphere'].str[1:-1]

    reg_info = pd.read_csv(ospj(data_dir, 'mni', 'Information', 'RegionInformation.csv'))
    # remove the apostrophes at beginning and end of "Region name" in reg_info
    reg_info['Region name'] = reg_info['Region name'].str[1:-1]

    pt_info = pd.read_csv(ospj(data_dir, 'mni', 'Information', 'PatientInformation.csv'))

    ##############################
    dkt_mni_parcs = pd.read_excel(ospj(data_dir, 'metadata', 'dkt_mni_parcs_RG.xlsx'), header=None, sheet_name='Sheet1')
    dkt_custom = dkt_mni_parcs.iloc[:, [0, 2]]
    mni_custom = dkt_mni_parcs.iloc[:, [7, 9]]

    dkt_custom.columns = ['dkt', 'custom']
    mni_custom.columns = ['mni', 'custom']


    # # drop rows that don't start with Label
    dkt_custom = dkt_custom[dkt_custom['dkt'].str.startswith('Label')]

    # # reformat dkt column so that Label i: xxx becomes xxx and keep i in a separate column
    dkt_custom['dkt_id'] = dkt_custom['dkt'].str.split(':').str[0].str.strip()
    dkt_custom['dkt_id'] = dkt_custom['dkt_id'].str.split(' ').str[1].str.strip().astype(int)
    dkt_custom['dkt'] = dkt_custom['dkt'].str.split(':').str[1].str.strip()

    # # reformat mni1 and mni2 so that there aren't apostrophes at beginning and end, only if they exist
    mni_custom['mni'] = mni_custom['mni'].str.strip()

    mni_custom['mni'] = mni_custom['mni'].str.strip("'")

    dkt_custom.dropna(inplace=True)
    mni_custom.dropna(inplace=True)

    dkt_to_custom = dict(zip(dkt_custom['dkt'], dkt_custom['custom']))
    mni_to_custom = dict(zip(mni_custom['mni'], mni_custom['custom']))

    # display(dkt_to_custom)
    del dkt_mni_parcs

    ##############################
    bands = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 12), "beta": (12, 30), "gamma": (30, 80)}

    # make a json for the feature options
    feature_options = {
        'bandpower': {
            'func': bandpower,
            'kwargs': {'fs': 200},
            'name': 'bandpower',
            'ft_names': list(bands.keys()),
            'ft_units': 'uV^2',
            'ft_type': 'spectral',
            'n_features': len(bands)
        },
        'coherence': {
            'func': coherence_bands,
            'kwargs': {'fs': 200},
            'name': 'coherence',
            'ft_names': list(bands.keys()),
            'ft_units': 'uV^2',
            'ft_type': 'spectral',
            'n_features': len(bands)
        },
        # 'line_length': {
        #     'func': line_length,
        #     'kwargs': {},
        #     'name': 'line_length',
        #     'ft_names': ['line_length'],
        #     'ft_units': 'uV',
        #     'ft_type': 'temporal',
        #     'n_features': 1
        # },
    }