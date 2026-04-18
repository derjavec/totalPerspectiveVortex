import logging

from setup import parse_args, setup_logging
from total_perspective_vortex.core import run_pipeline

logger = logging.getLogger(__name__)


def main():
    """Entry point for the EEG pipeline."""
    args = parse_args()
    setup_logging(level=args.level)
    run_pipeline(args)


if __name__ == "__main__":
    main()