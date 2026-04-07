import os


def _prepare_runtime_directories():
    """Ensure MNE and Matplotlib use writable directories."""
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


# Ensure MNE/Matplotlib config dirs are writable before any MNE import.
if __name__ == "__main__":
    _prepare_runtime_directories()
    from .main import main

    main()
