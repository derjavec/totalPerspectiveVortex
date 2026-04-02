import numpy as np
import pandas as pd
from mne.time_frequency import psd_array_welch

# Configuración global
EEG_CONFIG = {
    "channels": ["C1", "C2", "C3", "C4", "Cz"],
    "bands": {
        "mu": (8, 12),
        "beta_low": (13, 20),
        "beta_high": (20, 30),
    }
}

def compute_features_from_row(row, sfreq):
    """Compute band-power features and simple statistics for one epoch row."""
    eeg_cols = [c for c in row.index if "_t" in c]

    channels_all = sorted({c.split("_t")[0] for c in eeg_cols})
    times = sorted({int(c.split("_t")[1]) for c in eeg_cols})
    channels = [ch for ch in channels_all if ch in EEG_CONFIG["channels"]]
    ch_index = {ch: i for i, ch in enumerate(channels)}
    t_index = {t: i for i, t in enumerate(times)}

    n_channels = len(channels)
    n_times = len(times)
    data = np.zeros((n_channels, n_times))

    for col in eeg_cols:
        ch, t = col.split("_t")
        t = int(t)
        if ch in channels:
            data[ch_index[ch], t_index[t]] = row[col]

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
        idx = (freqs >= fmin) & (freqs <= fmax)
        band_power = psd[:, idx].sum(axis=1)

        for ch_idx, power in enumerate(band_power):
            features[f"{channels[ch_idx]}_{band_name}_power"] = np.log(power + 1e-10)
            features[f"{channels[ch_idx]}_{band_name}_rel"] = power / (total_power[ch_idx] + 1e-10)
            band_data = data[ch_idx, :]
            features[f"{channels[ch_idx]}_{band_name}_mean"] = band_data.mean()
            features[f"{channels[ch_idx]}_{band_name}_std"] = band_data.std()
            features[f"{channels[ch_idx]}_{band_name}_max"] = band_data.max()
            features[f"{channels[ch_idx]}_{band_name}_min"] = band_data.min()
            features[f"{channels[ch_idx]}_{band_name}_range"] = band_data.max() - band_data.min()

    return features


def extract_features_from_raw_dataset(df_raw, sfreq=160):
    """Transform raw epoch samples into a tabular feature set."""
    feature_rows = []

    for _, row in df_raw.iterrows():
        features = compute_features_from_row(row, sfreq)
        features["subject"] = row["subject"]
        features["label"] = row["label"]
        features["epoch_id"] = row["epoch_id"]
        feature_rows.append(features)

    return pd.DataFrame(feature_rows)


def calculate_differences_and_ratios(df):
    """Add pairwise differences and ratios between channels and bands."""
    channels = EEG_CONFIG["channels"]
    bands = list(EEG_CONFIG["bands"].keys())
    stats = ["mean", "std", "max", "min", "range"]

    for band in bands:
        # diferencias de power
        df[f"{channels[0]}_{channels[1]}_{band}_diff_power"] = df[f"{channels[0]}_{band}_power"] - df[f"{channels[1]}_{band}_power"]
        df[f"{channels[0]}_{channels[2]}_{band}_diff_power"] = df[f"{channels[0]}_{band}_power"] - df[f"{channels[2]}_{band}_power"]
        df[f"{channels[1]}_{channels[2]}_{band}_diff_power"] = df[f"{channels[1]}_{band}_power"] - df[f"{channels[2]}_{band}_power"]

        # diferencias de relative power
        df[f"{channels[0]}_{channels[1]}_{band}_diff_rel"] = df[f"{channels[0]}_{band}_rel"] - df[f"{channels[1]}_{band}_rel"]
        df[f"{channels[0]}_{channels[2]}_{band}_diff_rel"] = df[f"{channels[0]}_{band}_rel"] - df[f"{channels[2]}_{band}_rel"]
        df[f"{channels[1]}_{channels[2]}_{band}_diff_rel"] = df[f"{channels[1]}_{band}_rel"] - df[f"{channels[2]}_{band}_rel"]

        # ratios de relative power
        df[f"{channels[0]}_{channels[1]}_{band}_ratio_rel"] = df[f"{channels[0]}_{band}_rel"] / (df[f"{channels[1]}_{band}_rel"] + 1e-10)
        df[f"{channels[0]}_{channels[2]}_{band}_ratio_rel"] = df[f"{channels[0]}_{band}_rel"] / (df[f"{channels[2]}_{band}_rel"] + 1e-10)
        df[f"{channels[1]}_{channels[2]}_{band}_ratio_rel"] = df[f"{channels[1]}_{band}_rel"] / (df[f"{channels[2]}_{band}_rel"] + 1e-10)

        # ratios de power (en log, se usa diff para interpretación correcta)
        df[f"{channels[0]}_{channels[1]}_{band}_ratio_power"] = df[f"{channels[0]}_{band}_power"] - df[f"{channels[1]}_{band}_power"]
        df[f"{channels[0]}_{channels[2]}_{band}_ratio_power"] = df[f"{channels[0]}_{band}_power"] - df[f"{channels[2]}_{band}_power"]
        df[f"{channels[1]}_{channels[2]}_{band}_ratio_power"] = df[f"{channels[1]}_{band}_power"] - df[f"{channels[2]}_{band}_power"]
        for stat in stats:
            df[f"{channels[0]}_{channels[1]}_{band}_diff_{stat}"] = df[f"{channels[0]}_{band}_{stat}"] - df[f"{channels[1]}_{band}_{stat}"]
            df[f"{channels[0]}_{channels[2]}_{band}_diff_{stat}"] = df[f"{channels[0]}_{band}_{stat}"] - df[f"{channels[2]}_{band}_{stat}"]
            df[f"{channels[1]}_{channels[2]}_{band}_diff_{stat}"] = df[f"{channels[1]}_{band}_{stat}"] - df[f"{channels[2]}_{band}_{stat}"]

            df[f"{channels[0]}_{channels[1]}_{band}_ratio_{stat}"] = df[f"{channels[0]}_{band}_{stat}"] / (df[f"{channels[1]}_{band}_{stat}"] + 1e-10)
            df[f"{channels[0]}_{channels[2]}_{band}_ratio_{stat}"] = df[f"{channels[0]}_{band}_{stat}"] / (df[f"{channels[2]}_{band}_{stat}"] + 1e-10)
            df[f"{channels[1]}_{channels[2]}_{band}_ratio_{stat}"] = df[f"{channels[1]}_{band}_{stat}"] / (df[f"{channels[2]}_{band}_{stat}"] + 1e-10)

    # De-fragment the DataFrame to avoid PerformanceWarning.
    return df.copy()
