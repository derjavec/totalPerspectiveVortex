import argparse


def parse_args():
    """Parse command-line arguments for the CLI."""
    parser = argparse.ArgumentParser(description="Total Perspective Vortex")

    parser.add_argument(
        "--level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level.",
    )
    parser.add_argument(
        "--subject",
        type=int,
        default=None,
        help="ID of the subject to analyze (default: all subjects).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="logistic",
        choices=["logistic", "randomforest"],
        help="Choose the classifier.",
    )
    parser.add_argument(
        "--transformer",
        type=str,
        default="none",
        choices=["none", "pca", "csp"],
        help="Transformer choice (none, pca or csp).",
    )

    parser.add_argument(
        "--anova",
        type=int,
        default=None,
        help="Apply ANOVA feature selection with k best features (e.g. --anova 30). Works only with transformer 'pca' or 'none' and is ignored for CSP.",    )

    return parser.parse_args()
