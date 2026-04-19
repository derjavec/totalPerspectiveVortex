import os

import mne
from mne.datasets import eegbci
from mne.io import read_raw_edf


def _get_mne_directories():
    """Return local directories used to configure MNE data and caches."""
    root_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )
    assets_dir = os.path.join(root_dir, "assets")

    default_mne_data_dir = os.path.join(assets_dir, "mne_data")
    external_mne_data_dir = os.path.expanduser("~/mne_data")

    mne_data_dir = (
        external_mne_data_dir
        if os.path.isdir(external_mne_data_dir)
        else default_mne_data_dir
    )

    return {
        "mne_config_dir": os.path.join(assets_dir, "mne_config"),
        "mne_data_dir": mne_data_dir,
        "mpl_config_dir": os.path.join(assets_dir, "mpl_config"),
        "mne_home_dir": os.path.join(assets_dir, "mne_home"),
    }


def _setup_mne_environment():
    """Create local MNE directories and configure environment variables."""
    directories = _get_mne_directories()

    for path in directories.values():
        os.makedirs(path, exist_ok=True)

    os.environ["MNE_CONFIG_DIR"] = directories["mne_config_dir"]
    os.environ["MNE_DATA"] = directories["mne_data_dir"]
    os.environ["MNE_DATASETS_EEGBCI_PATH"] = directories["mne_data_dir"]
    os.environ["MPLCONFIGDIR"] = directories["mpl_config_dir"]
    os.environ["_MNE_FAKE_HOME_DIR"] = directories["mne_home_dir"]
    os.environ["MNE_DONTWRITE_HOME"] = "true"

    return directories["mne_data_dir"]


def load_raw_eeg(subject, runs):
    """Load and concatenate EEGBCI runs with a consistent MNE setup."""
    mne_data_dir = _setup_mne_environment()

    files = eegbci.load_data(
        subject,
        runs,
        path=mne_data_dir,
        update_path=False,
    )
    raws = [
        read_raw_edf(file_path, preload=True, verbose=False)
        for file_path in files
    ]

    raw = mne.concatenate_raws(raws)
    raw.rename_channels(lambda name: name.strip("."))
    raw.set_montage("standard_1020", on_missing="ignore")

    return raw
