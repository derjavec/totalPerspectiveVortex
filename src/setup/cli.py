"""Command-line argument parsing for the application."""

import argparse


def parse_args():
    """Parse CLI arguments and return the configuration namespace."""
    parser = argparse.ArgumentParser(
        description="Total Perspective Vortex"
    )

    parser.add_argument(
        "--level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level.",
    )

    parser.add_argument(
        "--subject",
        type=int,
        default=None,
        help="Subject ID to analyze. If not set, all subjects are used.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="logistic",
        choices=["logistic", "randomforest"],
        help="Classifier to use.",
    )

    parser.add_argument(
        "--transformer",
        type=str,
        default="none",
        choices=["none", "pca", "my_pca", "csp"],
        help="Feature transformer to apply.",
    )

    parser.add_argument(
        "--anova",
        type=int,
        default=None,
        help=(
            "Apply ANOVA feature selection with k best features "
            "(e.g. --anova 30). Ignored when using CSP."
        ),
    )

    return parser.parse_args()
