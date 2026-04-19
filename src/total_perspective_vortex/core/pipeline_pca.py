import logging
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

from training import (
    get_feature_data_for_subject,
    resolve_subjects,
    split_subject_data,
    standard_train_test,
    cv_mean_accuracy,
    evaluate_best_by_subject,
    log_best_result,
    results_dataframe,
    get_models
)

from my_pca import MyPCA

logger = logging.getLogger(__name__)

PCA_COMPONENTS = [2, 5, 10, 20, 30]


def safe_pca_components(requested_components, X_train):
    """Return a valid number of PCA components for the training split."""
    max_components = min(X_train.shape[0], X_train.shape[1])
    return min(requested_components, max_components)


def build_reduction_pipeline(model_name, transformer, n_components=None):
    """Build a feature-based pipeline with
        optional dimensionality reduction."""
    models = get_models()

    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}")

    classifier = models[model_name]

    if transformer == "pca":
        return Pipeline(
            [
                ("pca", PCA(n_components=n_components)),
                ("clf", classifier),
            ]
        )

    if transformer == "my_pca":
        return Pipeline(
            [
                ("my_pca", MyPCA(n_components=n_components)),
                ("clf", classifier),
            ]
        )

    if transformer in (None, "none"):
        return Pipeline([("clf", classifier)])

    raise ValueError(f"Unknown transformer: {transformer}")


def resolve_components(transformer, requested_components, X_train):
    """Resolve the effective number of components for the transformer."""
    if transformer in ("pca", "my_pca"):
        return safe_pca_components(requested_components, X_train)

    return None


def train_single_config(
    X,
    y,
    model_name,
    transformer,
    requested_components=None,
):
    """Train and evaluate one model configuration."""
    X_train, X_test, y_train, y_test = split_subject_data(X, y)
    X_train_scaled, X_test_scaled = standard_train_test(
        X_train,
        X_test,
    )

    n_components = resolve_components(
        transformer,
        requested_components,
        X_train_scaled,
    )

    pipeline = build_reduction_pipeline(
        model_name=model_name,
        transformer=transformer,
        n_components=n_components,
    )

    pipeline.fit(X_train_scaled, y_train)

    train_acc = pipeline.score(X_train_scaled, y_train)
    test_acc = pipeline.score(X_test_scaled, y_test)
    cv_mean = cv_mean_accuracy(X, y, pipeline, scale=True)

    return n_components, train_acc, test_acc, cv_mean


def get_component_values(transformer):
    """Return the list of component values to evaluate."""
    if transformer in ("pca", "my_pca"):
        return PCA_COMPONENTS

    return [None]


def collect_subject_results(df, subject, model_names, transformer):
    """Train all model configurations for one subject."""
    X, y = get_feature_data_for_subject(df, subject)
    subject_results = []

    for model_name in model_names:
        for requested_components in get_component_values(transformer):
            n_components, train_acc, test_acc, cv_mean = train_single_config(
                X=X,
                y=y,
                model_name=model_name,
                transformer=transformer,
                requested_components=requested_components,
            )

            logger.debug(
                (
                    "Subject: %s | Model: %s | Transformer: %s | "
                    "Components: %s | Train: %.3f | Test: %.3f | CV: %.3f"
                ),
                subject,
                model_name,
                transformer if transformer is not None else "none",
                n_components,
                train_acc,
                test_acc,
                cv_mean,
            )

            subject_results.append(
                (
                    subject,
                    model_name,
                    transformer if transformer is not None else "none",
                    n_components,
                    train_acc,
                    test_acc,
                    cv_mean,
                )
            )

    return subject_results


def log_best_predictions(df, subjects, best_result):
    """Refit the best global configuration and log per-subject predictions."""
    best_model = best_result["Model"]
    best_transformer = best_result["Transformer"]
    best_components = best_result["Components"]

    def get_subject_data(subject):
        return get_feature_data_for_subject(df, subject)

    def fit_transform_data(X_train, X_test):
        return standard_train_test(X_train, X_test)

    def build_best_pipeline(X_train_ready):
        requested_components = (
                None if pd.isna(best_components) else int(best_components)
            )

        n_components = resolve_components(
            best_transformer,
            requested_components,
            X_train_ready,
        )
        return build_reduction_pipeline(
            model_name=best_model,
            transformer=best_transformer,
            n_components=n_components,
        )

    evaluate_best_by_subject(
        subjects=subjects,
        get_subject_data=get_subject_data,
        fit_transform_data=fit_transform_data,
        build_best_pipeline=build_best_pipeline,
    )


def train_and_validate(df, subject=None, transformer=None):
    """Train and evaluate feature-based models."""
    resolved_transformer = (
        "none"
        if transformer in (None, "none")
        else transformer
    )
    subjects = resolve_subjects(subject, df["subject"].unique())
    model_names = list(get_models().keys())

    results = []

    for subject_id in subjects:
        results.extend(
            collect_subject_results(
                df=df,
                subject=subject_id,
                model_names=model_names,
                transformer=resolved_transformer,
            )
        )

    results_df = results_dataframe(results)

    best_result = log_best_result(
        results_df,
        ["Model", "Transformer", "Components"],
    )

    if best_result is not None:
        log_best_predictions(df, subjects, best_result)

    return results_df, best_result
