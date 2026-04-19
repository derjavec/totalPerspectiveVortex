import logging
from pathlib import Path

from mne_manager import visualize_group_raw

logger = logging.getLogger(__name__)


def visualize_raws(raws, plots_dir, task_name, stage):
    """Save a summary plot of raw EEG signals for a preprocessing stage."""
    logger.info("Visualizing EEG %s filtering", stage.lower())

    plot_path = Path(plots_dir) / f"EEG_{stage.lower()}_filter_{task_name}.png"

    visualize_group_raw(
        raws,
        title=f"{stage} filtering",
        filename=plot_path,
    )
