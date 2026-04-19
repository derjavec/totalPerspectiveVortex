# Total Perspective Vortex

Total Perspective Vortex is an EEG processing and classification pipeline built
with [MNE-Python](https://mne.tools/), NumPy, pandas, matplotlib, and
scikit-learn.

The project loads PhysioNet EEGBCI motor imagery recordings, preprocesses the
signals, builds an epoch-based dataset, and evaluates machine-learning
pipelines for EEG task classification.

## What the project does

The pipeline can classify three EEGBCI task setups:

| Task | EEGBCI runs | Classes |
| --- | --- | --- |
| `hands_vs_feet` | `6`, `10`, `14` | imagined hands vs imagined feet |
| `left_vs_right` | `4`, `8`, `12` | imagined left hand vs imagined right hand |
| `rest_vs_movement` | `1`, `2`, `3`, `4`, `5` | rest vs movement imagery |

At runtime, the task is selected interactively in the terminal.

## Requirements

- Python `>=3.10,<3.13`
- Internet access on first run, unless the EEGBCI dataset is already cached
- Python dependencies listed in `pyproject.toml`:
  - `mne`
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `scikit-learn`

## Installation

From the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Python 3.10 also works. Python 3.13 is not supported by this project because
`pyproject.toml` requires Python lower than 3.13.

After installation, the command-line entrypoint is:

```bash
tpv
```

You can also run the package module directly:

```bash
python -m total_perspective_vortex
```

## Basic usage

Run the full pipeline with default settings:

```bash
tpv
```

The program will ask you to choose one of the available EEG tasks:

```text
Task options for EEG analysis:
1. hands_vs_feet - Differentiate between imagining hand movements and foot movements.
2. left_vs_right - Differentiate between imagining left-hand vs right-hand movements.
3. rest_vs_movement - Differentiate between rest (no movement) and imagining movement.
Please choose the task (enter the number):
```

By default, the pipeline uses subjects `1` through `5`, extracts handcrafted
features, trains both registered classifiers, and reports the best mean
cross-validation result in the logs.

## Command-line arguments

```bash
tpv [--level LEVEL] [--subject SUBJECT] [--model MODEL] [--transformer TRANSFORMER] [--anova K]
```

| Argument | Default | Choices / type | Description |
| --- | --- | --- | --- |
| `--level` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` | Logging level for console output and the effective minimum log level for the run. Use `DEBUG` to include debug details in `logs/tpv.log`. |
| `--subject` | all default subjects | integer from `1` to `109` | Run the pipeline for one EEGBCI subject. If omitted, subjects `1` to `5` are used. |
| `--model` | `logistic` | `logistic`, `randomforest` | Parsed by the CLI, but the current training code evaluates both registered models: `LogisticRegression` and `RandomForest`. |
| `--transformer` | `none` | `none`, `pca`, `my_pca`, `csp` | Selects the feature transformation path. `csp` uses epoched EEG data directly; the other options use handcrafted tabular features. |
| `--anova` | disabled | integer | Applies ANOVA `SelectKBest` feature selection with the requested number of features. Ignored when `--transformer csp` is used. |

## Examples

Run the default feature-based pipeline:

```bash
tpv
```

Run only subject 12:

```bash
tpv --subject 12
```

Run with detailed logs:

```bash
tpv --level DEBUG
```

Run feature extraction followed by scikit-learn PCA:

```bash
tpv --transformer pca
```

Run feature extraction followed by the custom PCA implementation:

```bash
tpv --transformer my_pca
```

Run ANOVA feature selection before model training:

```bash
tpv --anova 30
```

Run the CSP pipeline:

```bash
tpv --transformer csp
```

Combine options:

```bash
tpv --subject 7 --transformer pca --anova 30 --level DEBUG
```

## Pipeline flow

The application entrypoint is `total_perspective_vortex.__main__`, which calls
`total_perspective_vortex.main.main()`.

The high-level flow is:

1. Parse CLI arguments.
2. Configure logging.
3. Create output directories under `assets/`.
4. Ask the user to select the EEG task.
5. Resolve the subjects to process.
6. Load EEGBCI raw EDF files with MNE.
7. Save a before-filtering frequency spectrum plot.
8. Apply preprocessing filters.
9. Save an after-filtering frequency spectrum plot.
10. Save a shared-axis before/after frequency comparison plot.
11. Convert annotations into class labels.
12. Build epochs from `0.5s` to `3.0s`.
13. Run either the CSP pipeline or the feature-based pipeline.
14. Train and evaluate models.
15. Log the best model configuration and per-subject predictions.

## Data loading

Data is loaded with `mne.datasets.eegbci.load_data()`.

The project configures MNE to store data and cache files inside the repository
when possible:

```text
assets/mne_data/
assets/mne_config/
assets/mne_home/
assets/mpl_config/
```

If `~/mne_data` already exists, that external directory is preferred for the
EEGBCI dataset.

Each requested subject is loaded independently. If a subject cannot be loaded,
the pipeline skips it and logs a warning.

## Preprocessing

For every loaded raw recording, the pipeline:

1. Concatenates the requested EEGBCI runs.
2. Strips trailing dots from channel names.
3. Sets the `standard_1020` montage.
4. Applies a `60 Hz` notch filter.
5. Applies a `7-30 Hz` band-pass filter.

The filtered data is then converted into MNE epochs with:

```text
tmin = 0.5
tmax = 3.0
baseline = None
```

## Feature-based pipeline

This path is used when `--transformer` is `none`, `pca`, or `my_pca`.

The pipeline extracts handcrafted features from these channels:

```text
C1, C2, C3, C4, Cz
```

For each channel, it computes features over these frequency bands:

| Band | Frequency range |
| --- | --- |
| `mu` | `8-12 Hz` |
| `beta_low` | `13-20 Hz` |
| `beta_high` | `20-30 Hz` |

For each channel and band, the extracted features include:

- log band power
- relative band power
- mean
- standard deviation
- max
- min
- range

The project then adds derived pairwise features for selected channel pairs:

- differences
- relative-power ratios
- time-statistic ratios

The resulting CSV is saved to:

```text
assets/dataset/eeg_features_<task_name>.csv
```

Depending on the selected transformer, training uses:

| Transformer | Behavior |
| --- | --- |
| `none` | Train directly on standardized handcrafted features. |
| `pca` | Apply scikit-learn `PCA` before classification. |
| `my_pca` | Apply the local custom PCA transformer before classification. |

For `pca` and `my_pca`, the pipeline evaluates these component counts:

```text
2, 5, 10, 20, 30
```

The effective number of components is capped so it never exceeds the available
training samples or features.

## CSP pipeline

This path is used when:

```bash
tpv --transformer csp
```

Instead of using handcrafted tabular features, the CSP pipeline trains directly
on epoched EEG tensors with `mne.decoding.CSP`.

The CSP configuration evaluates:

```text
n_components = 2, 4, 6
reg = 0.1
log = True
```

When CSP is selected, `--anova` is ignored.

## Models and evaluation

The registered classifiers are:

| Model name in results | Implementation |
| --- | --- |
| `LogisticRegression` | `sklearn.linear_model.LogisticRegression(max_iter=2000, class_weight="balanced")` |
| `RandomForest` | `sklearn.ensemble.RandomForestClassifier(n_estimators=200, random_state=42)` |

For each subject and configuration, the pipeline:

1. Splits the subject data with an `80/20` stratified train/test split.
2. Standardizes feature data when using the feature-based pipeline.
3. Trains the classifier pipeline.
4. Computes train accuracy.
5. Computes test accuracy.
6. Computes mean `5`-fold stratified cross-validation accuracy.

After all configurations are evaluated, the pipeline groups results by model,
transformer, and component count, then logs the configuration with the highest
mean cross-validation score.

## Outputs

The pipeline writes the following generated files:

```text
assets/dataset/eeg_features_<task_name>.csv
assets/plots/EEG_before_filter_<task_name>.png
assets/plots/EEG_after_filter_<task_name>.png
assets/plots/EEG_filter_comparison_<task_name>.png
logs/tpv.log
```

The plot files show power spectral density by frequency in Hertz, with one
curve per displayed EEG channel and a variability band across loaded subjects.
The comparison plot uses shared axes for the before/after panels and skips the
first two seconds of signal when estimating PSD to reduce filter-edge effects.

The log file uses rotation:

```text
maxBytes = 5,000,000
backupCount = 3
```

## Project structure

```text
src/
  total_perspective_vortex/
    __main__.py
    main.py
    core/
      pipeline.py
      pipeline_pca.py
      pipeline_csp.py
      preparation.py
      selection.py
      visualize.py
      build_ds.py
  setup/
    cli.py
    loading.py
    logging.py
  mne_manager/
    load_mne.py
    filter_mne.py
    plot_mne.py
  features_manager/
    extract.py
    calculate.py
    anova.py
  training/
    data_utils.py
    evaluate.py
    model_registry.py
  my_pca/
    my_pca.py
```

## Troubleshooting

If `tpv` is not found, make sure the package was installed in editable mode and
that the virtual environment is active:

```bash
source .venv/bin/activate
python -m pip install -e .
```

If Python 3.13 is selected by default, create the virtual environment with
Python 3.10 or 3.11:

```bash
python3.11 -m venv .venv
```

If the first run is slow, it is likely downloading EEGBCI data through MNE.
Later runs reuse the local MNE cache.

If no recordings are loaded, check your internet connection, the MNE data
directory, and whether the requested subject ID is valid for EEGBCI.
