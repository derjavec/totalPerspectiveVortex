import pandas as pd
from .calculate import compute_features_from_epoch

EEG_CONFIG = {
    "channels": ["C1", "C2", "C3", "C4", "Cz"],
    "bands": {
        "mu": (8, 12),
        "beta_low": (13, 20),
        "beta_high": (20, 30),
    },
}

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

