import logging
from features_manager import (
    calculate_differences_and_ratios,
    extract_features_from_prepared_data,
)
from setup import (
    apply_filter,
    load_subjects_raw,
    parse_args,
    prepare_directories,
    setup_logging,
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


def main():
    """Run the full pipeline from data loading to training."""
    args = parse_args()
    setup_logging(level=args.level)
    if args.subject is not None:
        if args.subject < 1 or args.subject > 109:
            raise ValueError("Subject must be between 1 and 109")

        subjects = [args.subject]

    else:
        subjects = range(1, 6)
    base_dir = "assets"

    dataset_dir, plots_dir = prepare_directories(base_dir)
    task_name, runs = select_task()
    logger.info("Selected task: %s | runs: %s", task_name, runs)

    raws = load_subjects_raw(subjects, runs)
    if not raws:
        logger.error(
            "Could not load any EEG recordings. Check PhysioNet data access."
        )
        return
    visualize_raws(raws, plots_dir, task_name, stage="Before")

    raws_filtered = apply_filter(raws)
    visualize_raws(raws_filtered, plots_dir, task_name, stage="After")

    logger.info("Creating structured dataset")
    prepared_data = prepare_global_data(
        raws_filtered, subjects, task_name, tmin=0.5, tmax=2.5
    )

    if args.transformer == "csp":
        train_and_validate_csp(
            prepared_data["X"],
            prepared_data["y"],
            prepared_data["subject_vec"],
            args.subject,
        )
    else:
        df_principal_features = extract_features_from_prepared_data(
            prepared_data
        )
        df_features = calculate_differences_and_ratios(df_principal_features)
        save_features_dataset(df_features, task_name, dataset_dir)

        train_and_validate(
            df_features,
            args.subject,
            transformer=args.transformer,
        )
