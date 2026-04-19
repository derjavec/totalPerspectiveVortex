import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def save_features_dataset(df, task_name, dataset_dir):
    """Save the feature dataset to disk for reuse.

    The dataset is written as a CSV file inside the target directory.
    """
    dataset_dir = Path(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = dataset_dir / f"eeg_features_{task_name}.csv"
    df.to_csv(dataset_path, index=False)

    logger.info(
        "Dataset saved to %s (%d rows)",
        dataset_path,
        len(df),
    )
