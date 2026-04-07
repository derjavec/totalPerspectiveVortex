import os


def load_raw_eeg(subject, runs):
    """Load and concatenate EEGBCI runs with a consistent MNE setup."""
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    assets = os.path.join(root, "assets")
    mne_config_dir = os.path.join(assets, "mne_config")
    default_mne_data_dir = os.path.join(assets, "mne_data")
    external_mne_data_dir = os.path.expanduser("~/mne_data")
    mne_data_dir = (
        external_mne_data_dir
        if os.path.isdir(external_mne_data_dir)
        else default_mne_data_dir
    )
    mpl_config_dir = os.path.join(assets, "mpl_config")
    mne_home_dir = os.path.join(assets, "mne_home")
    os.makedirs(mne_config_dir, exist_ok=True)
    os.makedirs(mne_data_dir, exist_ok=True)
    os.makedirs(mpl_config_dir, exist_ok=True)
    os.makedirs(mne_home_dir, exist_ok=True)
    os.environ["MNE_CONFIG_DIR"] = mne_config_dir
    os.environ["MNE_DATA"] = mne_data_dir
    os.environ["MNE_DATASETS_EEGBCI_PATH"] = mne_data_dir
    os.environ["MPLCONFIGDIR"] = mpl_config_dir
    os.environ["_MNE_FAKE_HOME_DIR"] = mne_home_dir
    os.environ["MNE_DONTWRITE_HOME"] = "true"

    import mne
    from mne.datasets import eegbci
    from mne.io import read_raw_edf

    files = eegbci.load_data(
        subject,
        runs,
        path=mne_data_dir,
        update_path=False,
    )
    raws = [read_raw_edf(f, preload=True, verbose=False) for f in files]
    raw = mne.concatenate_raws(raws)
    raw.rename_channels(lambda x: x.strip("."))
    raw.set_montage("standard_1020", on_missing="ignore")
    return raw
