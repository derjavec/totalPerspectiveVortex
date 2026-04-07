import os
import logging
from mne_manager import load_raw_eeg, filter_raw

logger = logging.getLogger(__name__)


def prepare_directories(base_dir: str) -> tuple[str, str]:
    """Ensure dataset and plot directories exist."""
    dataset_dir = os.path.join(base_dir, "dataset")
    plots_dir = os.path.join(base_dir, "plots")

    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    logger.debug("Directories prepared: %s, %s", dataset_dir, plots_dir)
    return dataset_dir, plots_dir


def load_subjects_raw(subjects, runs):
    """Load raw EEG recordings for each subject."""
    logger.info("Loading raw EEG data")

    raws = []
    failed = []

    for subject in subjects:
        try:
            raw = load_raw_eeg(subject, runs)
            if raw is not None:
                raws.append(raw)
                logger.debug("Loaded subject %s", subject)
            else:
                logger.debug("Empty raw for subject %s", subject)
                failed.append(subject)

        except Exception as e:
            logger.debug("Failed subject %s | error: %s", subject, str(e))
            failed.append(subject)

    logger.info("Loaded %d raw recordings", len(raws))

    if failed:
        logger.warning("Skipped %d subjects: %s", len(failed), failed)

    return raws


def apply_filter(raws):
    """Bandpass-filter all raw recordings."""
    logger.info("Applying bandpass filter 7–30 Hz")
    return [filter_raw(raw) for raw in raws]
