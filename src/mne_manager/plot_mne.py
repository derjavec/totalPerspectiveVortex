import logging

import matplotlib.pyplot as plt
import numpy as np
from mne.time_frequency import psd_array_welch

logger = logging.getLogger(__name__)


def _extract_psd_window(raw, n_channels, start_time, duration):
    """Return the requested channel window, avoiding filter edge transients."""
    sfreq = raw.info["sfreq"]
    start_sample = int(sfreq * start_time)
    window_samples = int(sfreq * duration)
    stop_sample = start_sample + window_samples

    if stop_sample > raw.n_times:
        start_sample = 0
        stop_sample = min(window_samples, raw.n_times)

    data, _times = raw[:n_channels, start_sample:stop_sample]
    return data, sfreq


def _compute_group_psd(
    raws,
    n_channels=10,
    duration=10,
    start_time=2,
    fmin=1,
    fmax=40,
):
    """Compute mean and standard deviation PSD for a group of raw objects."""
    if not raws:
        raise ValueError("The raws list must not be empty.")

    n_available_channels = min(n_channels, len(raws[0].ch_names))
    channel_names = raws[0].ch_names[:n_available_channels]

    all_psd = []
    freqs = None

    for raw in raws:
        data, sfreq = _extract_psd_window(
            raw,
            n_available_channels,
            start_time,
            duration,
        )
        psd, freqs = psd_array_welch(
            data,
            sfreq=sfreq,
            fmin=fmin,
            fmax=fmax,
            n_fft=min(256, data.shape[1]),
            average="mean",
            verbose=False,
        )
        all_psd.append(10 * np.log10(psd + np.finfo(float).eps))

    all_psd = np.stack(all_psd)

    return {
        "freqs": freqs,
        "mean": all_psd.mean(axis=0),
        "std": all_psd.std(axis=0),
        "channel_names": channel_names,
    }


def _plot_psd(ax, psd_data):
    """Plot one PSD curve per channel on the provided axes."""
    freqs = psd_data["freqs"]
    mean_psd = psd_data["mean"]
    std_psd = psd_data["std"]
    channel_names = psd_data["channel_names"]

    for ch_idx, channel_name in enumerate(channel_names):
        ax.plot(freqs, mean_psd[ch_idx], label=channel_name)
        ax.fill_between(
            freqs,
            mean_psd[ch_idx] - std_psd[ch_idx],
            mean_psd[ch_idx] + std_psd[ch_idx],
            alpha=0.2,
        )


def visualize_group_raw(
    raws,
    title="EEG signals",
    filename=None,
    n_channels=10,
    duration=10,
    start_time=2,
    fmin=1,
    fmax=40,
):
    """Plot the mean frequency spectrum across multiple raw recordings.

    The function extracts the first channels and time window from each raw
    recording, computes the power spectral density for each channel, and plots
    one curve per channel with a variability band across recordings.
    """
    psd_data = _compute_group_psd(
        raws,
        n_channels=n_channels,
        duration=duration,
        start_time=start_time,
        fmin=fmin,
        fmax=fmax,
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    _plot_psd(ax, psd_data)
    ax.set_title(title)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power spectral density (dB)")
    ax.legend(loc="upper right")

    if filename:
        fig.savefig(filename)
        logger.info("Plot saved to %s", filename)

    plt.close(fig)


def visualize_filter_comparison(
    raws_before,
    raws_after,
    title="EEG filtering comparison",
    filename=None,
    n_channels=10,
    duration=10,
    start_time=2,
    fmin=1,
    fmax=40,
):
    """Plot before/after PSDs with shared axes for direct comparison."""
    before_psd = _compute_group_psd(
        raws_before,
        n_channels=n_channels,
        duration=duration,
        start_time=start_time,
        fmin=fmin,
        fmax=fmax,
    )
    after_psd = _compute_group_psd(
        raws_after,
        n_channels=n_channels,
        duration=duration,
        start_time=start_time,
        fmin=fmin,
        fmax=fmax,
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 6),
        sharex=True,
        sharey=True,
    )

    _plot_psd(axes[0], before_psd)
    _plot_psd(axes[1], after_psd)

    axes[0].set_title("Before filtering")
    axes[1].set_title("After filtering")

    for ax in axes:
        ax.set_xlabel("Frequency (Hz)")

    axes[0].set_ylabel("Power spectral density (dB)")
    axes[1].legend(loc="upper right")
    fig.suptitle(title)
    fig.tight_layout()

    if filename:
        fig.savefig(filename)
        logger.info("Plot saved to %s", filename)

    plt.close(fig)
