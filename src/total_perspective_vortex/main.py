import logging

from setup import setup_logging, prepare_directories, load_subjects_raw, apply_filter
from total_perspective_vortex.core import select_task, visualize_raws, build_and_save_dataset
from features_manager import extract_features_from_raw_dataset

logger = logging.getLogger(__name__)

def main():
    
    setup_logging(level=logging.INFO)

    subjects = range(1, 6)
    base_dir = "generated_files"

    dataset_dir, plots_dir = prepare_directories(base_dir)

    task_name, runs = select_task()
    logger.info("Selected task: %s | runs: %s", task_name, runs)

    raws = load_subjects_raw(subjects, runs)
    

    visualize_raws(raws, plots_dir, task_name, stage="Before")

    raws_filtered = apply_filter(raws)

    visualize_raws(raws_filtered, plots_dir, task_name, stage="After")
    df = build_and_save_dataset(
        raws_filtered,
        subjects,
        task_name,
        dataset_dir,
    )
    extract_features_from_raw_dataset(df)


if __name__ == "__main__":
    main()
