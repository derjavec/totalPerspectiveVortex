"""Feature extraction from prepared EEG data."""

import pandas as pd

from .calculate import compute_features_from_epoch, EEG_CONFIG


def extract_features_from_prepared_data(prepared_data, sfreq=160):
    """Convert prepared EEG epochs into a tabular feature dataset.

    The function extracts features from each epoch, keeps only the configured
    EEG channels, and returns the result as a pandas DataFrame with metadata.
    """
    X = prepared_data["X"]
    y = prepared_data["y"]
    subject_vec = prepared_data["subject_vec"]
    epoch_ids = prepared_data["epoch_ids"]

    channels = [
        ch for ch in prepared_data["ch_names"]
        if ch in EEG_CONFIG["channels"]
    ]

    if not channels:
        return pd.DataFrame(columns=["subject", "label", "epoch_id"])

    channel_indices = [prepared_data["ch_names"].index(ch) for ch in channels]
    feature_rows = []

    for epoch, subject, label, epoch_id in zip(X, subject_vec, y, epoch_ids):
        features = compute_features_from_epoch(
            epoch[channel_indices, :],
            channels,
            sfreq,
        )
        features["subject"] = int(subject)
        features["label"] = int(label)
        features["epoch_id"] = int(epoch_id)
        feature_rows.append(features)

    return pd.DataFrame(feature_rows)
