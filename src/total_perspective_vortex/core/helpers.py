import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def resolve_subjects(subject, available_subjects):
    """Normalize the subject argument into a list of subjects."""
    if subject is None:
        return list(available_subjects)
    if isinstance(subject, int):
        return [subject]
    return list(subject)


def split_subject_data(X, y, test_size=0.2, random_state=42):
    """Split subject data preserving class balance."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def standard_train_test(X_train, X_test):
    """Scale train and test sets using statistics from the training data."""
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


def cv_mean_accuracy(X, y, pipeline, scale=True):
    """Compute mean cross-validated accuracy with optional scaling."""
    y = np.asarray(y)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []

    for train_idx, test_idx in splitter.split(X, y):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        if scale:
            X_tr, X_te = standard_train_test(X_tr, X_te)

        pipeline.fit(X_tr, y_tr)
        scores.append(pipeline.score(X_te, y_te))

    return float(np.mean(scores))


def get_models():
    """Return the dictionary of classifiers to evaluate."""
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
        ),

        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
        ),
    }


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


def best_mean_result(res_df, group_cols):
    """Compute mean metrics by model and return the best row."""
    res_cv = (
        res_df.groupby(group_cols, as_index=False, dropna=False)[
            ["Train", "Test", "CV"]
        ]
        .mean()
        .sort_values("CV", ascending=False)
    )
    if res_cv.empty:
        return res_cv, None
    best_result = res_cv.iloc[0]
    return res_cv, best_result


def log_best_result(res_df, group_cols):
    """Compute and log the best mean result across subjects."""
    if res_df.empty:
        logger.warning("Could not train any model successfully.")
        return None

    _, best_result = best_mean_result(res_df, group_cols)
    if best_result is None:
        logger.warning("Could not calculate the mean CV score per model.")
        return None

    logger.info("Best mean combination: %s", best_result.to_dict())
    return best_result


def log_predictions(y_true, y_pred):
    """Log per-epoch predictions against ground-truth labels."""
    for i, (pred, truth) in enumerate(zip(y_pred, y_true)):
        logger.debug("epoch %02d: [%s] [%s] %s", i, pred, truth, pred == truth)

def get_feature_data_for_subject(df, subject):
    """Extract tabular features and labels for one subject."""
    df_s = df[df["subject"] == subject]
    y = df_s["label"]
    X = df_s[feature_columns(df_s)].values
    return X, y


def evaluate_best_by_subject(
    subjects,
    get_subject_data,
    fit_transform_data,
    build_best_pipeline,
):
    """
    Refit the best configuration per subject and log predictions.
    """
    for s in subjects:
        X_s, y_s = get_subject_data(s)
        X_train, X_test, y_train, y_test = split_subject_data(X_s, y_s)
        X_train_ready, X_test_ready = fit_transform_data(X_train, X_test)

        pipeline = build_best_pipeline(X_train_ready)
        pipeline.fit(X_train_ready, y_train)
        y_pred = pipeline.predict(X_test_ready)

        logger.info("Predictions for subject %s:", s)
        log_predictions(y_test, y_pred)
        logger.info(
            "Accuracy: %.4f",
            float(np.mean(y_pred == np.asarray(y_test))),
        )