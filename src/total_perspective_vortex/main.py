import logging
import argparse
from setup import setup_logging, prepare_directories, load_subjects_raw, apply_filter, parse_args
from total_perspective_vortex.core import select_task, visualize_raws, create_dataset, prepare_data_for_csp, save_features_dataset, train_and_validate, train_and_validate_csp
from features_manager import extract_features_from_raw_dataset, calculate_differences_and_ratios

logger = logging.getLogger(__name__)

def main():
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
    visualize_raws(raws, plots_dir, task_name, stage="Before")

    raws_filtered = apply_filter(raws)

    visualize_raws(raws_filtered, plots_dir, task_name, stage="After")
    
    logger.info("Creating structured dataset")
    if args.transformer == 'csp':
        X, y, subject_vec = prepare_data_for_csp(raws_filtered, subjects, task_name, tmin=0.5, tmax=2.5)
        train_and_validate_csp(X, y, subject_vec, args.subject)
    else:
        df_raw = create_dataset(raws_filtered, subjects, task_name, tmin=0.5, tmax=2.5)
    
        df_principal_features = extract_features_from_raw_dataset(df_raw)
        df_features = calculate_differences_and_ratios(df_principal_features)

        save_features_dataset(df_features, task_name, dataset_dir)
    
        train_and_validate(df_features, args.subject, transformer=args.transformer)
    
