import logging

from features_manager import (
    calculate_differences_and_ratios,
    extract_features_from_prepared_data,
)
from setup import (
    apply_filter,
    load_subjects_raw,
    prepare_directories,
)
from total_perspective_vortex.core import (
    prepare_global_data,
    save_features_dataset,
    select_task,
    train_and_validate,
    train_and_validate_csp,
    visualize_raws,
)

logger = logging.getLogger(__name__)


def get_subjects(args):
    """Return the subject list based on CLI arguments."""
    if args.subject is not None:
        if args.subject < 1 or args.subject > 109:
            raise ValueError("Subject must be between 1 and 109")
        return [args.subject]

    return range(1, 6)


def initialize_pipeline():
    """Prepare directories and select the task."""
    base_dir = "assets"
    dataset_dir, plots_dir = prepare_directories(base_dir)
    task_name, runs = select_task()

    logger.info("Selected task: %s | runs: %s", task_name, runs)
    return dataset_dir, plots_dir, task_name, runs


def load_and_filter_raws(subjects, runs, plots_dir, task_name):
    """Load EEG recordings, visualize them, apply filtering, and visualize again."""
    raws = load_subjects_raw(subjects, runs)
    if not raws:
        logger.error(
            "Could not load any EEG recordings. Check PhysioNet data access."
        )
        return None

    visualize_raws(raws, plots_dir, task_name, stage="Before")

    raws_filtered = apply_filter(raws)
    visualize_raws(raws_filtered, plots_dir, task_name, stage="After")

    return raws_filtered


def prepare_dataset(raws_filtered, subjects, task_name):
    """Create structured data from filtered raw EEG recordings."""
    logger.info("Creating structured dataset")
    return prepare_global_data(
        raws_filtered,
        subjects,
        task_name,
        tmin=0.5,
        tmax=2.5,
    )


def run_csp_pipeline(prepared_data, args):
    """Train and validate the CSP-based pipeline."""
    train_and_validate_csp(
        prepared_data["X"],
        prepared_data["y"],
        prepared_data["subject_vec"],
        args.subject,
    )


def run_feature_pipeline(prepared_data, args, dataset_dir, task_name):
    """Extract handcrafted features, save them, and train the model."""
    df_principal_features = extract_features_from_prepared_data(prepared_data)
    df_features = calculate_differences_and_ratios(df_principal_features)
    save_features_dataset(df_features, task_name, dataset_dir)

    train_and_validate(
        df_features,
        args.subject,
        transformer=args.transformer,
    )


def run_pipeline(args):
    """Run the full EEG pipeline."""
    subjects = get_subjects(args)
    dataset_dir, plots_dir, task_name, runs = initialize_pipeline()

    raws_filtered = load_and_filter_raws(subjects, runs, plots_dir, task_name)
    if raws_filtered is None:
        return

    prepared_data = prepare_dataset(raws_filtered, subjects, task_name)

    if args.transformer == "csp":
        run_csp_pipeline(prepared_data, args)
    else:
        run_feature_pipeline(prepared_data, args, dataset_dir, task_name)