import logging

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def visualize_group_raw(
    raws,
    title="EEG signals",
    filename=None,
    n_channels=10,
    duration=10,
):
    """Plot the mean signal and variability across multiple raw recordings.

    The function extracts the first channels and time window from each raw
    recording, computes the mean and standard deviation across recordings,
    and plots one curve per channel with a variability band.
    """
    if not raws:
        raise ValueError("The raws list must not be empty.")

    sfreq = raws[0].info["sfreq"]
    max_samples = int(sfreq * duration)
    n_available_channels = min(n_channels, len(raws[0].ch_names))

    all_data = []
    times = None

    for raw in raws:
        data, times = raw[:n_available_channels, :max_samples]
        all_data.append(data)

    all_data = np.stack(all_data)
    mean_data = all_data.mean(axis=0)
    std_data = all_data.std(axis=0)

    fig, ax = plt.subplots(figsize=(12, 6))

    for ch_idx in range(n_available_channels):
        ax.plot(times, mean_data[ch_idx], label=f"Ch {ch_idx + 1}")
        ax.fill_between(
            times,
            mean_data[ch_idx] - std_data[ch_idx],
            mean_data[ch_idx] + std_data[ch_idx],
            alpha=0.2,
        )

    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (µV)")
    ax.legend(loc="upper right")

    if filename:
        fig.savefig(filename)
        logger.info("Plot saved to %s", filename)

    plt.close(fig)
