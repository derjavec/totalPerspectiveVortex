"""Evaluation utilities for model comparison and prediction logging."""

import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .data_utils import split_subject_data, standard_train_test

logger = logging.getLogger(__name__)


def cv_mean_accuracy(X, y, pipeline, scale=True):
    """Compute mean cross-validated accuracy."""
    y = np.asarray(y)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []

    for train_idx, test_idx in splitter.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if scale:
            X_train, X_test = standard_train_test(X_train, X_test)

        pipeline.fit(X_train, y_train)
        scores.append(pipeline.score(X_test, y_test))

    return float(np.mean(scores))


def results_dataframe(results):
    """Build the standard results dataframe."""
    return pd.DataFrame(
        results,
        columns=[
            "Subject",
            "Model",
            "Transformer",
            "Components",
            "Train",
            "Test",
            "CV",
        ],
    )


def best_mean_result(results_df, group_cols):
    """Aggregate metrics by configuration and return the best row."""
    grouped_df = (
        results_df.groupby(group_cols, as_index=False, dropna=False)[
            ["Train", "Test", "CV"]
        ]
        .mean()
        .sort_values("CV", ascending=False)
    )

    if grouped_df.empty:
        return grouped_df, None

    best_result = grouped_df.iloc[0]
    return grouped_df, best_result


def log_best_result(results_df, group_cols):
    """Compute and log the best mean result across subjects."""
    if results_df.empty:
        logger.warning("Could not train any model successfully.")
        return None

    _, best_result = best_mean_result(results_df, group_cols)

    if best_result is None:
        logger.warning("Could not calculate the mean CV score per model.")
        return None

    logger.info("Best mean combination: %s", best_result.to_dict())
    return best_result


def log_predictions(y_true, y_pred):
    """Log predictions and ground-truth labels for each epoch."""
    for idx, (pred, truth) in enumerate(zip(y_pred, y_true)):
        logger.debug(
            "epoch %02d: [%s] [%s] %s",
            idx,
            pred,
            truth,
            pred == truth,
        )


def evaluate_best_by_subject(
    subjects,
    get_subject_data,
    fit_transform_data,
    build_best_pipeline,
):
    """Refit the best configuration for each subject and log predictions."""
    for subject in subjects:
        X_subject, y_subject = get_subject_data(subject)

        X_train, X_test, y_train, y_test = split_subject_data(
            X_subject,
            y_subject,
        )
        X_train_ready, X_test_ready = fit_transform_data(X_train, X_test)

        pipeline = build_best_pipeline(X_train_ready)
        pipeline.fit(X_train_ready, y_train)
        y_pred = pipeline.predict(X_test_ready)

        logger.info("Predictions for subject %s:", subject)
        log_predictions(y_test, y_pred)
        logger.info(
            "Accuracy: %.4f",
            float(np.mean(y_pred == np.asarray(y_test))),
        )
