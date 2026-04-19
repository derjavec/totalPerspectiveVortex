import logging
import os

from mne_manager import filter_raw, load_raw_eeg

logger = logging.getLogger(__name__)


def prepare_directories(base_dir: str) -> tuple[str, str]:
    """Create dataset and plot directories if they do not exist."""
    dataset_dir = os.path.join(base_dir, "dataset")
    plots_dir = os.path.join(base_dir, "plots")

    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    logger.debug(
        "Prepared directories: dataset=%s, plots=%s",
        dataset_dir,
        plots_dir,
    )
    return dataset_dir, plots_dir


def load_subjects_raw(subjects, runs):
    """Load raw EEG recordings for the requested subjects.

    Subjects that cannot be loaded are skipped and reported in the logs.
    """
    logger.info("Loading raw EEG data")

    raws = []
    failed_subjects = []

    for subject in subjects:
        try:
            raw = load_raw_eeg(subject, runs)

            if raw is not None:
                raws.append(raw)
                logger.debug("Loaded subject %s", subject)
            else:
                logger.debug("Empty raw for subject %s", subject)
                failed_subjects.append(subject)

        except Exception as exc:
            logger.debug("Failed subject %s | error: %s", subject, str(exc))
            failed_subjects.append(subject)

    logger.info("Loaded %d raw recordings", len(raws))

    if failed_subjects:
        logger.warning(
            "Skipped %d subjects: %s",
            len(failed_subjects),
            failed_subjects,
        )

    return raws


def apply_filter(raws):
    """Apply band-pass filtering to all raw EEG recordings."""
    logger.info("Applying band-pass filter from 7 to 30 Hz")
    return [filter_raw(raw) for raw in raws]
