def filter_raw(raw, l_freq=7., h_freq=30.):
    """Apply notch and bandpass filters to a raw EEG recording."""
    raw_filtered = raw.copy()
    raw_filtered.notch_filter(freqs=50, verbose=False)
    raw_filtered.filter(l_freq, h_freq, fir_design="firwin", verbose=False)
    return raw_filtered


def get_events_and_labels(raw, keep_labels=None):
    """Extract events and map annotation labels to class indices."""
    import mne
    events, event_id = mne.events_from_annotations(raw)
    if keep_labels is None:
        keep_labels = list(event_id.keys())

    keep_events = {k: v for k, v in event_id.items() if k in keep_labels}

    labels = list(keep_events.keys())
    if len(labels) == 2:
        mapping = {keep_events[labels[0]]: 0, keep_events[labels[1]]: 1}
    elif len(labels) == 3:
        mapping = {
            keep_events[labels[0]]: 0,
            keep_events[labels[1]]: 1,
            keep_events[labels[2]]: 1,
        }
    else:
        mapping = {keep_events[lab]: i for i, lab in enumerate(labels)}
    return events, keep_events, mapping
