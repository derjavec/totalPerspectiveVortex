import logging

from setup import setup_logging, prepare_directories, load_subjects_raw, apply_filter
from total_perspective_vortex.core import select_task, visualize_raws, create_dataset, save_features_dataset
from features_manager import extract_features_from_raw_dataset, calculate_differences_and_ratios
from pca_manager import apply_pca

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
    
    logger.info("Creating structured dataset")
    df_raw = create_dataset(raws_filtered, subjects, task_name)
    
    df_principal_features = extract_features_from_raw_dataset(df_raw)
    df_features = calculate_differences_and_ratios(df_principal_features)
    save_features_dataset(df_features, task_name, dataset_dir)
    df_pca, x_pca = apply_pca(df_features)
    save_features_dataset(df_pca, "f{task_name}_PCA", dataset_dir)


if __name__ == "__main__":
    main()
