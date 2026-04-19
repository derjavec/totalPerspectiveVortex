import logging
from pathlib import Path

from mne_manager import visualize_filter_comparison, visualize_group_raw

logger = logging.getLogger(__name__)


def visualize_raws(raws, plots_dir, task_name, stage):
    """Save a PSD plot of raw EEG signals for a preprocessing stage."""
    logger.info("Visualizing EEG %s filtering", stage.lower())

    plot_path = Path(plots_dir) / f"EEG_{stage.lower()}_filter_{task_name}.png"

    visualize_group_raw(
        raws,
        title=f"{stage} filtering",
        filename=plot_path,
    )


def visualize_filter_effect(raws_before, raws_after, plots_dir, task_name):
    """Save a shared-axis before/after PSD comparison plot."""
    logger.info("Visualizing EEG filter effect")

    plot_path = Path(plots_dir) / f"EEG_filter_comparison_{task_name}.png"

    visualize_filter_comparison(
        raws_before,
        raws_after,
        title="Before vs after filtering",
        filename=plot_path,
    )
