import numpy as np
from mne_manager import get_events_and_labels

def get_labels(task_name):

    if task_name in ['hands_vs_feet', 'left_vs_right']:
        keep_labels = ['T1', 'T2']
    elif task_name == 'rest_vs_movement':
        keep_labels = ['T0', 'T1', 'T2']
    else:
        keep_labels = None
    return keep_labels


def create_dataset(raws, subjects, task_name, tmin=0, tmax=4):
    import pandas as pd
    from mne import Epochs
    
    keep_labels = get_labels(task_name)
    
    rows = []
    epoch_id = 0

    for subject, raw in zip(subjects, raws):
        events, keep_events, mapping = get_events_and_labels(raw, keep_labels=keep_labels)
        epochs = Epochs(raw, events, event_id=keep_events, tmin=tmin, tmax=tmax,
                        baseline=None, preload=True, verbose=False)
        X = epochs.get_data()
        y_raw = epochs.events[:, -1]
        y = np.array([mapping[val] for val in y_raw])
        ch_names = epochs.ch_names
        n_times = X.shape[2]

        for i in range(X.shape[0]):
            row = {"epoch_id": epoch_id, "subject": subject, "label": int(y[i])}
            for ch_idx, ch_name in enumerate(ch_names):
                for t in range(n_times):
                    row[f"{ch_name}_t{t:03d}"] = X[i, ch_idx, t]
            rows.append(row)
            epoch_id += 1

    df = pd.DataFrame(rows)
    return df

def cov_epoch(X):
    return X @ X.T / X.shape[1]


def _mean_covariance(X_epochs):
    covs = [cov_epoch(X) for X in X_epochs]
    return np.mean(covs, axis=0)


def _inv_sqrtm(C):
    eigvals, eigvecs = np.linalg.eigh(C)
    eigvals[eigvals < 1e-12] = 1e-12
    return eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T


def euclidean_alignment(X_epochs):
    C = _mean_covariance(X_epochs)
    W = _inv_sqrtm(C)
    return np.array([W @ X for X in X_epochs])


def prepare_data_for_csp(raws, subjects, task_name, tmin=0, tmax=4, align=True):
    from mne import Epochs
    keep_labels = get_labels(task_name)

    X_list = []
    y_list = []
    subject_vec = []
    for subject, raw in zip(subjects, raws):
        events, keep_events, mapping = get_events_and_labels(raw, keep_labels=keep_labels)
        epochs = Epochs(raw, events, event_id=keep_events, tmin=tmin, tmax=tmax,
                        baseline=None, preload=True, verbose=False)
        X = epochs.get_data()
        if align:
            X = euclidean_alignment(X)
        y_raw = epochs.events[:, -1]
        y = np.array([mapping[val] for val in y_raw])
        X_list.append(X)
        y_list.append(y)
        subject_vec.append(np.full(len(y), subject))

    X = np.concatenate(X_list, axis = 0)
    y = np.concatenate(y_list, axis = 0)
    subject_vec = np.concatenate(subject_vec, axis=0)
    return X, y, subject_vec
