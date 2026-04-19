import numpy as np
import pandas as pd
from mne.filter import filter_data
from mne.time_frequency import psd_array_welch

EEG_CONFIG = {
    "channels": ["C1", "C2", "C3", "C4", "Cz"],
    "bands": {
        "mu": (8, 12),
        "beta_low": (13, 20),
        "beta_high": (20, 30),
    },
}

TIME_STATS = ["mean", "std", "max", "min", "range"]
EPSILON = 1e-10


def compute_features_from_epoch(data, channels, sfreq):
    """Compute spectral and time-domain features for a single epoch.

    The function extracts log band power, relative band power, and basic
    time-domain statistics from band-pass filtered signals for each channel
    and frequency band.
    """
    psd, freqs = psd_array_welch(
        data,
        sfreq=sfreq,
        fmin=1,
        fmax=30,
        n_fft=256,
        average="mean",
    )

    features = {}
    total_power = psd.sum(axis=1)

    for band_name, (fmin, fmax) in EEG_CONFIG["bands"].items():
        band_mask = (freqs >= fmin) & (freqs <= fmax)
        band_power = psd[:, band_mask].sum(axis=1)

        band_filtered = filter_data(
            data,
            sfreq=sfreq,
            l_freq=fmin,
            h_freq=fmax,
            verbose=False,
        )

        for ch_idx, power in enumerate(band_power):
            ch_name = channels[ch_idx]
            band_signal = band_filtered[ch_idx, :]

            features[f"{ch_name}_{band_name}_power"] = np.log(power + EPSILON)
            features[f"{ch_name}_{band_name}_rel"] = (
                power / (total_power[ch_idx] + EPSILON)
            )
            features[f"{ch_name}_{band_name}_mean"] = band_signal.mean()
            features[f"{ch_name}_{band_name}_std"] = band_signal.std()
            features[f"{ch_name}_{band_name}_max"] = band_signal.max()
            features[f"{ch_name}_{band_name}_min"] = band_signal.min()
            features[f"{ch_name}_{band_name}_range"] = (
                band_signal.max() - band_signal.min()
            )

    return features


def calculate_differences_and_ratios(df):
    """Add pairwise channel differences and ratios to the feature table.

    The function creates derived features for power, relative power, and
    time-domain statistics across predefined channel pairs for each band.
    """
    channels = EEG_CONFIG["channels"]
    bands = list(EEG_CONFIG["bands"].keys())

    channel_pairs = [
        (channels[0], channels[1]),
        (channels[0], channels[2]),
        (channels[1], channels[2]),
    ]

    new_cols = {}

    for band in bands:
        for ch_a, ch_b in channel_pairs:
            new_cols[f"{ch_a}_{ch_b}_{band}_diff_power"] = (
                df[f"{ch_a}_{band}_power"] - df[f"{ch_b}_{band}_power"]
            )
            new_cols[f"{ch_a}_{ch_b}_{band}_diff_rel"] = (
                df[f"{ch_a}_{band}_rel"] - df[f"{ch_b}_{band}_rel"]
            )
            new_cols[f"{ch_a}_{ch_b}_{band}_ratio_rel"] = (
                df[f"{ch_a}_{band}_rel"] / (df[f"{ch_b}_{band}_rel"] + EPSILON)
            )

            for stat in TIME_STATS:
                new_cols[f"{ch_a}_{ch_b}_{band}_diff_{stat}"] = (
                    df[f"{ch_a}_{band}_{stat}"] - df[f"{ch_b}_{band}_{stat}"]
                )
                new_cols[f"{ch_a}_{ch_b}_{band}_ratio_{stat}"] = (
                    df[f"{ch_a}_{band}_{stat}"]
                    / (df[f"{ch_b}_{band}_{stat}"] + EPSILON)
                )

    new_cols_df = pd.DataFrame(new_cols, index=df.index)
    return pd.concat([df, new_cols_df], axis=1)
