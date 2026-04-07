import numpy as np
import pandas as pd
from mne.time_frequency import psd_array_welch

# Global configuration for feature extraction.
EEG_CONFIG = {
    "channels": ["C1", "C2", "C3", "C4", "Cz"],
    "bands": {
        "mu": (8, 12),
        "beta_low": (13, 20),
        "beta_high": (20, 30),
    },
}


def compute_features_from_epoch(data, channels, sfreq):
    """Compute band-power features and simple statistics for one epoch."""
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
            power_key = f"{channels[ch_idx]}_{band_name}_power"
            features[power_key] = np.log(power + 1e-10)
            features[f"{channels[ch_idx]}_{band_name}_rel"] = (
                power / (total_power[ch_idx] + 1e-10)
            )
            band_data = data[ch_idx, :]
            features[f"{channels[ch_idx]}_{band_name}_mean"] = band_data.mean()
            features[f"{channels[ch_idx]}_{band_name}_std"] = band_data.std()
            features[f"{channels[ch_idx]}_{band_name}_max"] = band_data.max()
            features[f"{channels[ch_idx]}_{band_name}_min"] = band_data.min()
            features[f"{channels[ch_idx]}_{band_name}_range"] = (
                band_data.max() - band_data.min()
            )

    return features


def extract_features_from_prepared_data(prepared_data, sfreq=160):
    """Transform prepared epoch tensors into a tabular feature set."""
    X = prepared_data["X"]
    y = prepared_data["y"]
    subject_vec = prepared_data["subject_vec"]
    epoch_ids = prepared_data["epoch_ids"]
    channels = [
        ch for ch in prepared_data["ch_names"] if ch in EEG_CONFIG["channels"]
    ]

    if not channels:
        return pd.DataFrame(
            columns=[
                "subject",
                "label",
                "epoch_id",
            ]
        )

    channel_indices = [prepared_data["ch_names"].index(ch) for ch in channels]
    feature_rows = []

    for epoch, subject, label, epoch_id in zip(X, subject_vec, y, epoch_ids):
        features = compute_features_from_epoch(
            epoch[channel_indices, :], channels, sfreq
        )
        features["subject"] = int(subject)
        features["label"] = int(label)
        features["epoch_id"] = int(epoch_id)
        feature_rows.append(features)

    return pd.DataFrame(feature_rows)


def calculate_differences_and_ratios(df):
    """Add pairwise differences and ratios between channels and bands."""
    channels = EEG_CONFIG["channels"]
    bands = list(EEG_CONFIG["bands"].keys())
    stats = ["mean", "std", "max", "min", "range"]

    for band in bands:
        # Power differences.
        df[f"{channels[0]}_{channels[1]}_{band}_diff_power"] = (
            df[f"{channels[0]}_{band}_power"]
            - df[f"{channels[1]}_{band}_power"]
        )
        df[f"{channels[0]}_{channels[2]}_{band}_diff_power"] = (
            df[f"{channels[0]}_{band}_power"]
            - df[f"{channels[2]}_{band}_power"]
        )
        df[f"{channels[1]}_{channels[2]}_{band}_diff_power"] = (
            df[f"{channels[1]}_{band}_power"]
            - df[f"{channels[2]}_{band}_power"]
        )

        # Relative-power differences.
        df[f"{channels[0]}_{channels[1]}_{band}_diff_rel"] = (
            df[f"{channels[0]}_{band}_rel"] - df[f"{channels[1]}_{band}_rel"]
        )
        df[f"{channels[0]}_{channels[2]}_{band}_diff_rel"] = (
            df[f"{channels[0]}_{band}_rel"] - df[f"{channels[2]}_{band}_rel"]
        )
        df[f"{channels[1]}_{channels[2]}_{band}_diff_rel"] = (
            df[f"{channels[1]}_{band}_rel"] - df[f"{channels[2]}_{band}_rel"]
        )

        # Relative-power ratios.
        df[f"{channels[0]}_{channels[1]}_{band}_ratio_rel"] = (
            df[f"{channels[0]}_{band}_rel"]
            / (df[f"{channels[1]}_{band}_rel"] + 1e-10)
        )
        df[f"{channels[0]}_{channels[2]}_{band}_ratio_rel"] = (
            df[f"{channels[0]}_{band}_rel"]
            / (df[f"{channels[2]}_{band}_rel"] + 1e-10)
        )
        df[f"{channels[1]}_{channels[2]}_{band}_ratio_rel"] = (
            df[f"{channels[1]}_{band}_rel"]
            / (df[f"{channels[2]}_{band}_rel"] + 1e-10)
        )

        # Log-power ratios are represented as differences.
        df[f"{channels[0]}_{channels[1]}_{band}_ratio_power"] = (
            df[f"{channels[0]}_{band}_power"]
            - df[f"{channels[1]}_{band}_power"]
        )
        df[f"{channels[0]}_{channels[2]}_{band}_ratio_power"] = (
            df[f"{channels[0]}_{band}_power"]
            - df[f"{channels[2]}_{band}_power"]
        )
        df[f"{channels[1]}_{channels[2]}_{band}_ratio_power"] = (
            df[f"{channels[1]}_{band}_power"]
            - df[f"{channels[2]}_{band}_power"]
        )
        for stat in stats:
            df[f"{channels[0]}_{channels[1]}_{band}_diff_{stat}"] = (
                df[f"{channels[0]}_{band}_{stat}"]
                - df[f"{channels[1]}_{band}_{stat}"]
            )
            df[f"{channels[0]}_{channels[2]}_{band}_diff_{stat}"] = (
                df[f"{channels[0]}_{band}_{stat}"]
                - df[f"{channels[2]}_{band}_{stat}"]
            )
            df[f"{channels[1]}_{channels[2]}_{band}_diff_{stat}"] = (
                df[f"{channels[1]}_{band}_{stat}"]
                - df[f"{channels[2]}_{band}_{stat}"]
            )

            df[f"{channels[0]}_{channels[1]}_{band}_ratio_{stat}"] = (
                df[f"{channels[0]}_{band}_{stat}"]
                / (df[f"{channels[1]}_{band}_{stat}"] + 1e-10)
            )
            df[f"{channels[0]}_{channels[2]}_{band}_ratio_{stat}"] = (
                df[f"{channels[0]}_{band}_{stat}"]
                / (df[f"{channels[2]}_{band}_{stat}"] + 1e-10)
            )
            df[f"{channels[1]}_{channels[2]}_{band}_ratio_{stat}"] = (
                df[f"{channels[1]}_{band}_{stat}"]
                / (df[f"{channels[2]}_{band}_{stat}"] + 1e-10)
            )

    # De-fragment the DataFrame to avoid PerformanceWarning.
    return df.copy()
