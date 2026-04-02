import os

# Ensure MNE/Matplotlib config dirs are writable before any MNE import.
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_assets = os.path.join(_root, "assets")
_mne_config_dir = os.path.join(_assets, "mne_config")
_default_mne_data_dir = os.path.join(_assets, "mne_data")
_external_mne_data_dir = os.path.expanduser("~/mne_data")
_mne_data_dir = _external_mne_data_dir if os.path.isdir(_external_mne_data_dir) else _default_mne_data_dir
_mpl_config_dir = os.path.join(_assets, "mpl_config")
_mne_home_dir = os.path.join(_assets, "mne_home")
os.makedirs(_mne_config_dir, exist_ok=True)
os.makedirs(_mne_data_dir, exist_ok=True)
os.makedirs(_mpl_config_dir, exist_ok=True)
os.makedirs(_mne_home_dir, exist_ok=True)
os.environ["MNE_CONFIG_DIR"] = _mne_config_dir
os.environ["MNE_DATA"] = _mne_data_dir
os.environ["MNE_DATASETS_EEGBCI_PATH"] = _mne_data_dir
os.environ["MPLCONFIGDIR"] = _mpl_config_dir
os.environ["_MNE_FAKE_HOME_DIR"] = _mne_home_dir
os.environ["MNE_DONTWRITE_HOME"] = "true"

from .main import main

if __name__ == "__main__":
    main()
