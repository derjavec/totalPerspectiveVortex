import os
import logging

from mne_manager import visualize_group_raw

logger = logging.getLogger(__name__)

def visualize_raws(raws, plots_dir: str, task_name: str, stage: str):
    """Save a summary plot of raw signals for a preprocessing stage."""
    logger.info("Visualizing EEG %s filtering", stage.lower())
    visualize_group_raw(
        raws,
        title=f"{stage} filtering",
        filename=os.path.join(
            plots_dir, f"EEG_{stage.lower()}_filter_{task_name}.png"
        ),
    )
