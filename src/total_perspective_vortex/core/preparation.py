import numpy as np
from mne import Epochs

from mne_manager import get_events_and_labels

EMPTY_PREPARED_DATA = {
    "X": np.empty((0, 0, 0)),
    "y": np.empty((0,), dtype=int),
    "subject_vec": np.empty((0,), dtype=int),
    "epoch_ids": np.empty((0,), dtype=int),
    "ch_names": [],
}


def get_task_labels(task_name):
    """Return the annotation labels to keep for the selected task."""
    if task_name in {"hands_vs_feet", "left_vs_right"}:
        return ["T1", "T2"]

    if task_name == "rest_vs_movement":
        return ["T0", "T1", "T2"]

    return None


def prepare_global_data(raws, subjects, task_name, tmin=0.5, tmax=3):
    """Create a unified epoch-based dataset across all selected subjects.

    The function extracts epochs from each raw recording, maps annotation
    events to class labels, and concatenates all subjects into a single
    structure used by downstream pipelines.
    """
    keep_labels = get_task_labels(task_name)

    X_list = []
    y_list = []
    subject_list = []
    epoch_id_list = []
    ch_names = None
    next_epoch_id = 0

    for subject, raw in zip(subjects, raws):
        events, event_id, mapping = get_events_and_labels(
            raw,
            keep_labels=keep_labels,
        )

        epochs = Epochs(
            raw,
            events,
            event_id=event_id,
            tmin=tmin,
            tmax=tmax,
            baseline=None,
            preload=True,
            verbose=False,
        )

        X = epochs.get_data()
        y_raw = epochs.events[:, -1]
        y = np.array([mapping[value] for value in y_raw], dtype=int)

        if ch_names is None:
            ch_names = list(epochs.ch_names)

        X_list.append(X)
        y_list.append(y)
        subject_list.append(np.full(len(y), subject, dtype=int))
        epoch_id_list.append(
            np.arange(next_epoch_id, next_epoch_id + len(y), dtype=int)
        )
        next_epoch_id += len(y)

    if not X_list:
        return EMPTY_PREPARED_DATA.copy()

    return {
        "X": np.concatenate(X_list, axis=0),
        "y": np.concatenate(y_list, axis=0),
        "subject_vec": np.concatenate(subject_list, axis=0),
        "epoch_ids": np.concatenate(epoch_id_list, axis=0),
        "ch_names": ch_names or [],
    }
