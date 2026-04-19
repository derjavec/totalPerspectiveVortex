import logging

from features_manager import (
    calculate_differences_and_ratios,
    extract_features_from_prepared_data,
    select_best_features_anova,
)
from setup import (
    apply_filter,
    load_subjects_raw,
    prepare_directories,
)

from .build_ds import save_features_dataset
from .pipeline_csp import train_and_validate_csp
from .pipeline_pca import train_and_validate
from .preparation import prepare_global_data
from .selection import select_task
from .visualize import visualize_filter_effect, visualize_raws

logger = logging.getLogger(__name__)

DEFAULT_SUBJECTS = range(1, 6)
MIN_SUBJECT_ID = 1
MAX_SUBJECT_ID = 109


def get_subjects(args):
    """Return the list of subjects requested through the CLI."""
    if args.subject is None:
        return list(DEFAULT_SUBJECTS)

    if not MIN_SUBJECT_ID <= args.subject <= MAX_SUBJECT_ID:
        raise ValueError(
            f"Subject must be between {MIN_SUBJECT_ID} and {MAX_SUBJECT_ID}"
        )

    return [args.subject]


def initialize_pipeline():
    """Prepare output directories and resolve the selected task."""
    base_dir = "assets"
    dataset_dir, plots_dir = prepare_directories(base_dir)
    task_name, runs = select_task()

    logger.info("Selected task: %s | runs: %s", task_name, runs)

    return dataset_dir, plots_dir, task_name, runs


def load_and_filter_raws(subjects, runs, plots_dir, task_name):
    """Load, visualize, filter, and re-visualize EEG recordings."""
    raws = load_subjects_raw(subjects, runs)

    if not raws:
        logger.error(
            "Could not load any EEG recordings. "
            "Check the EEGBCI data availability."
        )
        return None

    visualize_raws(raws, plots_dir, task_name, stage="Before")
    filtered_raws = apply_filter(raws)
    visualize_raws(filtered_raws, plots_dir, task_name, stage="After")
    visualize_filter_effect(raws, filtered_raws, plots_dir, task_name)

    return filtered_raws


def prepare_dataset(filtered_raws, subjects, task_name):
    """Build the structured dataset from filtered EEG recordings."""
    logger.info("Creating structured dataset")

    return prepare_global_data(
        filtered_raws,
        subjects,
        task_name,
        tmin=0.5,
        tmax=3,
    )


def run_csp_pipeline(prepared_data, subject):
    """Train and evaluate the CSP-based pipeline."""
    return train_and_validate_csp(
        prepared_data["X"],
        prepared_data["y"],
        prepared_data["subject_vec"],
        subject,
    )


def extract_feature_table(prepared_data):
    """Extract handcrafted EEG features and derived channel relations."""
    principal_features_df = extract_features_from_prepared_data(prepared_data)
    return calculate_differences_and_ratios(principal_features_df)


def maybe_apply_anova(df_features, k_best):
    """Apply ANOVA feature selection when requested."""
    if not k_best:
        return df_features

    selected_df, _ranking_df = select_best_features_anova(
        df_features,
        k=k_best,
    )
    return selected_df


def run_feature_pipeline(
    prepared_data,
    subject,
    transformer,
    anova,
    dataset_dir,
    task_name,
):
    """Train and evaluate the feature-based pipeline."""
    features_df = extract_feature_table(prepared_data)
    features_df = maybe_apply_anova(features_df, anova)

    save_features_dataset(features_df, task_name, dataset_dir)

    return train_and_validate(
        features_df,
        subject=subject,
        transformer=transformer,
    )


def run_pipeline(args):
    """Run the full EEG processing and training pipeline."""
    subjects = get_subjects(args)
    dataset_dir, plots_dir, task_name, runs = initialize_pipeline()

    filtered_raws = load_and_filter_raws(
        subjects,
        runs,
        plots_dir,
        task_name,
    )
    if filtered_raws is None:
        return None

    prepared_data = prepare_dataset(filtered_raws, subjects, task_name)

    if args.transformer == "csp":
        return run_csp_pipeline(prepared_data, args.subject)

    return run_feature_pipeline(
        prepared_data=prepared_data,
        subject=args.subject,
        transformer=args.transformer,
        anova=args.anova,
        dataset_dir=dataset_dir,
        task_name=task_name,
    )
