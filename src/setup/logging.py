import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logging(
    level=logging.INFO,
    log_dir="logs",
    log_file="tpv.log",
):
    """Configure console and rotating file logging for the application.

    The root logger is reset, and both console and file handlers are added.
    External libraries such as matplotlib and MNE are silenced to warnings.
    """
    Path(log_dir).mkdir(exist_ok=True)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    root_logger.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    file_handler = RotatingFileHandler(
        Path(log_dir) / log_file,
        maxBytes=5_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("mne").setLevel(logging.WARNING)
