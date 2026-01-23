import numpy as np
import pandas as pd
from mne.time_frequency import psd_array_welch

BANDS = {
    # "delta": (1, 4),
    # "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
}


def compute_features_from_row(row, sfreq):
    # columnas EEG (Fc5_t000, C3_t123, etc.)
    eeg_cols = [c for c in row.index if "_t" in c]

    # extraemos canales y tiempos explícitamente
    channels_all = sorted({c.split("_t")[0] for c in eeg_cols})
    times = sorted({int(c.split("_t")[1]) for c in eeg_cols})
    channels = [ch for ch in channels_all if ch in ["C3", "C4", "Cz"]]
    ch_index = {ch: i for i, ch in enumerate(channels)}
    t_index = {t: i for i, t in enumerate(times)}

    n_channels = len(channels)
    n_times = len(times)

    # matriz (canal × tiempo)
    data = np.zeros((n_channels, n_times))

    for col in eeg_cols:
        ch, t = col.split("_t")
        t = int(t)
        if ch in channels:
            data[ch_index[ch], t_index[t]] = row[col]

    # PSD
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

    for band_name, (fmin, fmax) in BANDS.items():
        idx = (freqs >= fmin) & (freqs <= fmax)
        band_power = psd[:, idx].sum(axis=1)

        for ch_idx, power in enumerate(band_power):
            features[f"{channels[ch_idx]}_{band_name}_power"] = np.log(
                power + 1e-10
            )
            features[f"{channels[ch_idx]}_{band_name}_rel"] = (
                power / (total_power[ch_idx] + 1e-10)
            )

    return features


def extract_features_from_raw_dataset(df_raw, sfreq=160):
    feature_rows = []

    for _, row in df_raw.iterrows():
        features = compute_features_from_row(row, sfreq)
        features["subject"] = row["subject"]
        features["label"] = row["label"]
        features["epoch_id"] = row["epoch_id"]
        feature_rows.append(features)

    return pd.DataFrame(feature_rows)
