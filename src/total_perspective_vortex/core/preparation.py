import numpy as np
from mne_manager import get_events_and_labels
from mne import Epochs


def get_labels(task_name):
    """Return task-specific annotation labels to keep."""
    if task_name in ["hands_vs_feet", "left_vs_right"]:
        keep_labels = ["T1", "T2"]
    elif task_name == "rest_vs_movement":
        keep_labels = ["T0", "T1", "T2"]
    else:
        keep_labels = None
    return keep_labels


def prepare_global_data(raws, subjects, task_name, tmin=0.5, tmax=3):
    """Build a single epoch-based representation for all transformers."""

    keep_labels = get_labels(task_name)
    X_list = []
    y_list = []
    subject_vec = []
    epoch_ids = []
    ch_names = None
    next_epoch_id = 0

    for subject, raw in zip(subjects, raws):
        events, keep_events, mapping = get_events_and_labels(
            raw, keep_labels=keep_labels
        )
        epochs = Epochs(
            raw,
            events,
            event_id=keep_events,
            tmin=tmin,
            tmax=tmax,
            baseline=None,
            preload=True,
            verbose=False,
        )
        X = epochs.get_data()
        y_raw = epochs.events[:, -1]
        y = np.array([mapping[val] for val in y_raw], dtype=int)

        if ch_names is None:
            ch_names = list(epochs.ch_names)

        X_list.append(X)
        y_list.append(y)
        subject_vec.append(np.full(len(y), subject, dtype=int))
        epoch_ids.append(np.arange(next_epoch_id, next_epoch_id + len(y)))
        next_epoch_id += len(y)

    if not X_list:
        return {
            "X": np.empty((0, 0, 0)),
            "y": np.empty((0,), dtype=int),
            "subject_vec": np.empty((0,), dtype=int),
            "epoch_ids": np.empty((0,), dtype=int),
            "ch_names": [],
        }

    return {
        "X": np.concatenate(X_list, axis=0),
        "y": np.concatenate(y_list, axis=0),
        "subject_vec": np.concatenate(subject_vec, axis=0),
        "epoch_ids": np.concatenate(epoch_ids, axis=0),
        "ch_names": ch_names or [],
    }
