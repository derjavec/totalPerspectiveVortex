"""Training and evaluation utilities for CSP-based models."""

import logging

import numpy as np
from mne.decoding import CSP
from sklearn.pipeline import Pipeline

from training import (
    resolve_subjects,
    split_subject_data,
    cv_mean_accuracy,
    evaluate_best_by_subject,
    log_best_result,
    results_dataframe,
    get_models)

logger = logging.getLogger(__name__)

CSP_COMPONENTS = [2, 4, 6]
CSP_REG = 0.1


def get_epoch_data_for_subject(X, y, subject_vec, subject):
    """Extract epoch tensors and labels for a single subject."""
    subject_mask = subject_vec == subject
    return X[subject_mask], y[subject_mask]


def build_csp_pipeline(model_name, n_components):
    """Build a CSP pipeline with the selected classifier."""
    models = get_models()

    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}")

    return Pipeline(
        [
            (
                "csp",
                CSP(
                    n_components=n_components,
                    reg=CSP_REG,
                    log=True,
                ),
            ),
            ("clf", models[model_name]),
        ]
    )


def train_single_csp_config(X, y, model_name, n_components):
    """Train and evaluate one CSP configuration."""
    X_train, X_test, y_train, y_test = split_subject_data(X, y)
    pipeline = build_csp_pipeline(model_name, n_components)

    pipeline.fit(X_train, y_train)

    train_acc = pipeline.score(X_train, y_train)
    test_acc = pipeline.score(X_test, y_test)
    cv_mean = cv_mean_accuracy(X, y, pipeline, scale=False)

    return train_acc, test_acc, cv_mean


def collect_subject_results(
    X,
    y,
    subject_vec,
    subject,
    model_names,
    csp_components_list,
):
    """Train all CSP configurations for one subject."""
    X_subject, y_subject = get_epoch_data_for_subject(
        X,
        y,
        subject_vec,
        subject,
    )
    subject_results = []

    for model_name in model_names:
        for n_components in csp_components_list:
            train_acc, test_acc, cv_mean = train_single_csp_config(
                X_subject,
                y_subject,
                model_name,
                n_components,
            )

            logger.debug(
                (
                    "Model: %s | CSP components: %s | "
                    "Train: %.3f | Test: %.3f | CV: %.3f"
                ),
                model_name,
                n_components,
                train_acc,
                test_acc,
                cv_mean,
            )

            subject_results.append(
                (
                    subject,
                    model_name,
                    "CSP",
                    n_components,
                    train_acc,
                    test_acc,
                    cv_mean,
                )
            )

    return subject_results


def log_best_predictions(X, y, subject_vec, subjects, best_result):
    """Refit the best CSP configuration for
        each subject and log predictions."""
    best_model = best_result["Model"]
    best_components = int(best_result["Components"])

    def get_subject_data(subject):
        return get_epoch_data_for_subject(X, y, subject_vec, subject)

    def fit_transform_data(X_train, X_test):
        return X_train, X_test

    def build_best_pipeline(_X_train_ready):
        return build_csp_pipeline(best_model, best_components)

    evaluate_best_by_subject(
        subjects=subjects,
        get_subject_data=get_subject_data,
        fit_transform_data=fit_transform_data,
        build_best_pipeline=build_best_pipeline,
    )


def train_and_validate_csp(X, y, subject_vec, subject=None):
    """Train and evaluate CSP-based pipelines on epoched EEG data."""
    subjects = resolve_subjects(subject, np.unique(subject_vec))
    model_names = list(get_models().keys())

    results = []

    for subject_id in subjects:
        results.extend(
            collect_subject_results(
                X=X,
                y=y,
                subject_vec=subject_vec,
                subject=subject_id,
                model_names=model_names,
                csp_components_list=CSP_COMPONENTS,
            )
        )

    results_df = results_dataframe(results)

    best_result = log_best_result(
        results_df,
        ["Model", "Transformer", "Components"],
    )

    if best_result is not None:
        log_best_predictions(X, y, subject_vec, subjects, best_result)

    return results_df, best_result
