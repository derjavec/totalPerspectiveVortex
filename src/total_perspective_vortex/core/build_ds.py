import os
import logging
from .preparation import create_dataset

logger = logging.getLogger(__name__)


def save_features_dataset(df, task_name, dataset_dir):
    """Write the feature table to disk for later reuse."""

    dataset_path = os.path.join(
        dataset_dir, f"eeg_features_{task_name}.csv"
    )
    df.to_csv(dataset_path, index=False)

    logger.info(
        "Dataset saved to %s (%d epochs)",
        dataset_path,
        len(df),
    )
