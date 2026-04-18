import logging

import numpy as np
from mne.decoding import CSP
from sklearn.pipeline import Pipeline

from .helpers import (
    cv_mean_accuracy,
    evaluate_best_by_subject,
    get_models,
    log_best_result,
    resolve_subjects,
    results_dataframe,
    split_subject_data,
)

logger = logging.getLogger(__name__)


def get_epoch_data_for_subject(X, y, subject_vec, subject):
    """Extract epoch tensors and labels for one subject."""
    mask = subject_vec == subject
    return X[mask], y[mask]


def build_csp_pipeline(model_name, components):
    """Build a CSP + classifier pipeline."""
    models = get_models()
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}")

    clf = models[model_name]
    return Pipeline(
        [
            ("csp", CSP(n_components=components, reg=0.1, log=True)),
            ("clf", clf),
        ]
    )


def train_single_csp_config(X, y, clf, n_comp):
    """Train and evaluate one CSP + classifier configuration."""
    X_train, X_test, y_train, y_test = split_subject_data(X, y)

    pipeline = Pipeline(
        [
            ("csp", CSP(n_components=n_comp, reg=0.1, log=True)),
            ("clf", clf),
        ]
    )
    pipeline.fit(X_train, y_train)

    train_acc = pipeline.score(X_train, y_train)
    test_acc = pipeline.score(X_test, y_test)
    cv_mean = cv_mean_accuracy(X, y, pipeline, scale=False)

    return train_acc, test_acc, cv_mean


def collect_subject_results(X, y, subject_vec, subject, models, csp_components_list):
    """Train all CSP model/config combinations for one subject."""
    X_s, y_s = get_epoch_data_for_subject(X, y, subject_vec, subject)
    subject_results = []

    for model_name, clf in models.items():
        for n_comp in csp_components_list:
            train_acc, test_acc, cv_mean = train_single_csp_config(
                X_s, y_s, clf, n_comp
            )
            logger.debug(
                "Model: %s | CSP components: %s | Train: %.3f | Test: %.3f | CV: %.3f",
                model_name,
                n_comp,
                train_acc,
                test_acc,
                cv_mean,
            )
            subject_results.append(
                (
                    subject,
                    model_name,
                    "CSP",
                    n_comp,
                    train_acc,
                    test_acc,
                    cv_mean,
                )
            )

    return subject_results


def log_best_predictions(X, y, subject_vec, subjects, best_result):
    """Refit best global CSP configuration and log per-subject predictions."""
    best_model = best_result["Model"]
    best_components = int(best_result["Components"])

    def get_subject_data(subject):
        return get_epoch_data_for_subject(X, y, subject_vec, subject)

    def fit_transform_data(X_train, X_test):
        return X_train, X_test

    def build_best_pipeline(_X_train_ready):
        return build_csp_pipeline(best_model, best_components)

    evaluate_best_by_subject(
        df=df,
        subjects=subjects,
        get_subject_data=get_subject_data,
        fit_transform_data=fit_transform_data,
        build_best_pipeline=build_best_pipeline,
    )


def train_and_validate_csp(X, y, subject_vec, subject=None):
    """Train and evaluate CSP-based models on epoch tensors."""
    models = get_models()
    csp_components_list = [2, 4, 6]
    subjects = resolve_subjects(subject, np.unique(subject_vec))

    results = []
    for s in subjects:
        results.extend(
            collect_subject_results(
                X=X,
                y=y,
                subject_vec=subject_vec,
                subject=s,
                models=models,
                csp_components_list=csp_components_list,
            )
        )

    res_df = results_dataframe(results)
    best_result = log_best_result(
        res_df,
        ["Model", "Transformer", "Components"],
    )

    if best_result is not None:
        log_best_predictions(X, y, subject_vec, subjects, best_result)

    return res_df, best_result