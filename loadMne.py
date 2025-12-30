import os
import mne
from mne.datasets import eegbci
from mne.io import read_raw_edf
import numpy as np
import pandas as pd
from plotMne import visualize_group_raw


def load_raw_eeg(subject, runs):
    files = eegbci.load_data(subject, runs)
    raws = [read_raw_edf(f, preload=True, verbose=False) for f in files]
    raw = mne.concatenate_raws(raws)
    raw.rename_channels(lambda x: x.strip('.'))
    raw.set_montage('standard_1020', on_missing='ignore')
    return raw









