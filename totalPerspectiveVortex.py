import os
import mne
from mne.datasets import eegbci
from mne.io import read_raw_edf
import numpy as np
import pandas as pd
from plotMne import visualize_group_raw
from selectTask import select_task
from loadMne import load_raw_eeg
from filterMne import filter_raw
from createDs import create_dataset


def main():
    subjects = range(1, 6)

    base_dir = "generated_files"
    dataset_dir = os.path.join(base_dir, "dataset")
    plots_dir = os.path.join(base_dir, "plots")
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    task_name, runs = select_task()

    print("Loading raw EEG data...")
    raws = [load_raw_eeg(s, runs) for s in subjects]

    print("Visualizing EEG before filtering...")
    visualize_group_raw(raws, title="Before filtering",
                        filename=os.path.join(plots_dir, f"EEG_before_filter_{task_name}.png"))

    print("Applying bandpass filter 7-30 Hz...")
    raws_filtered = [filter_raw(raw) for raw in raws]

    print("Visualizing EEG after filtering...")
    visualize_group_raw(raws_filtered, title="After filtering",
                        filename=os.path.join(plots_dir, f"EEG_after_filter_{task_name}.png"))

    print("Creating structured dataset...")
    df = create_dataset(raws_filtered, subjects, task_name)
    dataset_path = os.path.join(dataset_dir, f"eeg_dataset_structured_{task_name}.csv")
    df.to_csv(dataset_path, index=False)
    print(f"Dataset saved to {dataset_path} with {len(df)} epochs.")


if __name__ == "__main__":
    main()