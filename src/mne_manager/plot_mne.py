import matplotlib.pyplot as plt
import numpy as np
import logging

logger = logging.getLogger(__name__)

def visualize_group_raw(raws, title="EEG signals", filename=None, n_channels=10, duration=10):
    """Plot mean and variability across multiple raw recordings."""
    all_data = []
    for raw in raws:
        data, times = raw[:n_channels, :int(raw.info['sfreq']*duration)]
        all_data.append(data)
    all_data = np.stack(all_data)
    mean_data = all_data.mean(axis=0)
    std_data = all_data.std(axis=0)

    plt.figure(figsize=(12, 6))
    for ch in range(n_channels):
        plt.plot(times, mean_data[ch], label=f"Ch {ch+1}")
        plt.fill_between(times, mean_data[ch]-std_data[ch], mean_data[ch]+std_data[ch], alpha=0.2)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (µV)")
    plt.legend(loc="upper right")
    if filename:
        plt.savefig(filename)
        logger.info(f"Plot saved to {filename}")
    plt.close()
