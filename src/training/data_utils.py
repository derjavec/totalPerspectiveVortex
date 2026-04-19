"""Utilities for subject selection and feature data preparation."""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

METADATA_COLUMNS = ["subject", "label", "epoch_id"]


def get_feature_columns(df, metadata_columns=None):
    """Return feature columns excluding metadata columns."""
    if metadata_columns is None:
        metadata_columns = METADATA_COLUMNS

    return [col for col in df.columns if col not in metadata_columns]


def resolve_subjects(subject, available_subjects):
    """Normalize the subject selection into a list of subject IDs."""
    if subject is None:
        return list(available_subjects)

    if isinstance(subject, int):
        return [subject]

    return list(subject)


def split_subject_data(X, y, test_size=0.2, random_state=42):
    """Split subject data while preserving class balance."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def standard_train_test(X_train, X_test):
    """Scale train and test data using statistics from the training set."""
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled


def get_feature_data_for_subject(df, subject):
    """Extract features and labels for a single subject."""
    subject_df = df[df["subject"] == subject]
    y = subject_df["label"]
    X = subject_df[get_feature_columns(subject_df)].values

    return X, y
