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


def compute_features_from_epoch(data, channels, sfreq):
    """Compute band-power and band-filtered time-domain features for one epoch."""
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

            features[f"{ch_name}_{band_name}_power"] = np.log(power + 1e-10)
            features[f"{ch_name}_{band_name}_rel"] = (
                power / (total_power[ch_idx] + 1e-10)
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
    """Add pairwise differences and ratios between channels and bands."""
    channels = EEG_CONFIG["channels"]
    bands = list(EEG_CONFIG["bands"].keys())
    stats = ["mean", "std", "max", "min", "range"]
    eps = 1e-10

    new_cols = {}

    for band in bands:
        # Power differences
        new_cols[f"{channels[0]}_{channels[1]}_{band}_diff_power"] = (
            df[f"{channels[0]}_{band}_power"] - df[f"{channels[1]}_{band}_power"]
        )
        new_cols[f"{channels[0]}_{channels[2]}_{band}_diff_power"] = (
            df[f"{channels[0]}_{band}_power"] - df[f"{channels[2]}_{band}_power"]
        )
        new_cols[f"{channels[1]}_{channels[2]}_{band}_diff_power"] = (
            df[f"{channels[1]}_{band}_power"] - df[f"{channels[2]}_{band}_power"]
        )

        # Relative-power differences
        new_cols[f"{channels[0]}_{channels[1]}_{band}_diff_rel"] = (
            df[f"{channels[0]}_{band}_rel"] - df[f"{channels[1]}_{band}_rel"]
        )
        new_cols[f"{channels[0]}_{channels[2]}_{band}_diff_rel"] = (
            df[f"{channels[0]}_{band}_rel"] - df[f"{channels[2]}_{band}_rel"]
        )
        new_cols[f"{channels[1]}_{channels[2]}_{band}_diff_rel"] = (
            df[f"{channels[1]}_{band}_rel"] - df[f"{channels[2]}_{band}_rel"]
        )

        # Relative-power ratios
        new_cols[f"{channels[0]}_{channels[1]}_{band}_ratio_rel"] = (
            df[f"{channels[0]}_{band}_rel"] / (df[f"{channels[1]}_{band}_rel"] + eps)
        )
        new_cols[f"{channels[0]}_{channels[2]}_{band}_ratio_rel"] = (
            df[f"{channels[0]}_{band}_rel"] / (df[f"{channels[2]}_{band}_rel"] + eps)
        )
        new_cols[f"{channels[1]}_{channels[2]}_{band}_ratio_rel"] = (
            df[f"{channels[1]}_{band}_rel"] / (df[f"{channels[2]}_{band}_rel"] + eps)
        )


        for stat in stats:
            new_cols[f"{channels[0]}_{channels[1]}_{band}_diff_{stat}"] = (
                df[f"{channels[0]}_{band}_{stat}"] - df[f"{channels[1]}_{band}_{stat}"]
            )
            new_cols[f"{channels[0]}_{channels[2]}_{band}_diff_{stat}"] = (
                df[f"{channels[0]}_{band}_{stat}"] - df[f"{channels[2]}_{band}_{stat}"]
            )
            new_cols[f"{channels[1]}_{channels[2]}_{band}_diff_{stat}"] = (
                df[f"{channels[1]}_{band}_{stat}"] - df[f"{channels[2]}_{band}_{stat}"]
            )

            new_cols[f"{channels[0]}_{channels[1]}_{band}_ratio_{stat}"] = (
                df[f"{channels[0]}_{band}_{stat}"]
                / (df[f"{channels[1]}_{band}_{stat}"] + eps)
            )
            new_cols[f"{channels[0]}_{channels[2]}_{band}_ratio_{stat}"] = (
                df[f"{channels[0]}_{band}_{stat}"]
                / (df[f"{channels[2]}_{band}_{stat}"] + eps)
            )
            new_cols[f"{channels[1]}_{channels[2]}_{band}_ratio_{stat}"] = (
                df[f"{channels[1]}_{band}_{stat}"]
                / (df[f"{channels[2]}_{band}_{stat}"] + eps)
            )

    new_cols_df = pd.DataFrame(new_cols, index=df.index)
    return pd.concat([df, new_cols_df], axis=1)
