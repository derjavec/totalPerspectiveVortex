"""EEG preprocessing and event handling utilities."""

import mne

NOTCH_FREQ = 60.0


def filter_raw(raw, l_freq=7.0, h_freq=30.0):
    """Apply notch and band-pass filtering to a raw EEG recording.

    The signal is first filtered to remove power-line noise and then
    band-pass filtered within the specified frequency range.
    """
    raw_filtered = raw.copy()

    raw_filtered.notch_filter(freqs=NOTCH_FREQ, verbose=False)
    raw_filtered.filter(
        l_freq=l_freq,
        h_freq=h_freq,
        fir_design="firwin",
        verbose=False,
    )

    return raw_filtered


def get_events_and_labels(raw, keep_labels=None):
    """Extract events and create a label mapping for classification.

    The function filters annotation labels and assigns integer class
    indices depending on the number of classes detected.
    """
    events, event_id = mne.events_from_annotations(raw)

    if keep_labels is None:
        keep_labels = list(event_id.keys())

    filtered_event_id = {
        label: code for label, code in event_id.items()
        if label in keep_labels
    }

    labels = list(filtered_event_id.keys())

    if len(labels) == 2:
        mapping = {
            filtered_event_id[labels[0]]: 0,
            filtered_event_id[labels[1]]: 1,
        }
    elif len(labels) == 3:
        mapping = {
            filtered_event_id[labels[0]]: 0,
            filtered_event_id[labels[1]]: 1,
            filtered_event_id[labels[2]]: 1,
        }
    else:
        mapping = {
            filtered_event_id[label]: idx
            for idx, label in enumerate(labels)
        }

    return events, filtered_event_id, mapping
