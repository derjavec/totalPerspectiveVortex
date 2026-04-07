import logging

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

logger = logging.getLogger(__name__)


def standard_train_test(X_train, X_test):
    """Scale train and test sets using statistics from the training data."""
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


def _cv_mean_accuracy(X, y, pipeline, scale=True):
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


def _get_models():
    """Return the dictionary of classifiers to evaluate."""
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
        ),
        "LinearSVM": SVC(kernel="linear", C=1.0, class_weight="balanced"),
        "RBF_SVM": SVC(
            kernel="rbf",
            C=2.0,
            gamma="scale",
            class_weight="balanced",
        ),
        "LDA": LinearDiscriminantAnalysis(),
        "GaussianNB": GaussianNB(),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
        ),
    }


def _build_pipeline(model_name, transformer, components):
    """Create a sklearn pipeline for the requested model and transformer."""
    models = _get_models()
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}")

    clf = models[model_name]
    if transformer == "PCA":
        return Pipeline([("pca", PCA(n_components=components)), ("clf", clf)])

    if transformer == "CSP":
        from mne.decoding import CSP

        return Pipeline(
            [
                ("csp", CSP(n_components=components, reg=0.1, log=True)),
                ("clf", clf),
            ]
        )

    return Pipeline([("clf", clf)])


def _log_predictions(y_true, y_pred):
    """Log per-epoch predictions against ground-truth labels."""
    for i, (pred, truth) in enumerate(zip(y_pred, y_true)):
        logger.info("epoch %02d: [%s] [%s] %s", i, pred, truth, pred == truth)


def _best_mean_result(res_df, group_cols):
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


def _feature_columns(df):
    """Return the model feature columns."""
    return [
        column
        for column in df.columns
        if column not in ["subject", "label", "epoch_id"]
    ]


def train_and_validate(df, subject=None, transformer=None):
    """Train and evaluate feature-based models with optional PCA."""
    models = _get_models()
    pca_components_list = [2, 5, 10, 20, 30]

    if subject is None:
        subjects = df["subject"].unique()
    elif isinstance(subject, int):
        subjects = [subject]
    else:
        subjects = subject

    results = []
    for s in subjects:
        df_s = df[df["subject"] == s]
        y = df_s["label"]
        feature_cols = _feature_columns(df_s)
        X = df_s[feature_cols].values
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )
        X_train_scaled, X_test_scaled = standard_train_test(X_train, X_test)

        for model_name, clf in models.items():
            if transformer == "pca":
                for n_comp in pca_components_list:
                    pipeline = Pipeline(
                        [
                            (
                                "pca",
                                PCA(
                                    n_components=min(
                                        n_comp,
                                        X_train_scaled.shape[1],
                                    )
                                ),
                            ),
                            ("clf", clf),
                        ]
                    )
                    pipeline.fit(X_train_scaled, y_train)
                    train_acc = pipeline.score(X_train_scaled, y_train)
                    test_acc = pipeline.score(X_test_scaled, y_test)
                    cv_mean = _cv_mean_accuracy(X, y, pipeline, scale=True)
                    logger.debug(
                        (
                            "Model: %s | PCA components: %s | Train: %.3f | "
                            "Test: %.3f | CV: %.3f"
                        ),
                        model_name,
                        n_comp,
                        train_acc,
                        test_acc,
                        cv_mean,
                    )
                    results.append(
                        (
                            s,
                            model_name,
                            "PCA",
                            n_comp,
                            train_acc,
                            test_acc,
                            cv_mean,
                        )
                    )
            else:
                pipeline = Pipeline([("clf", clf)])
                pipeline.fit(X_train_scaled, y_train)
                train_acc = pipeline.score(X_train_scaled, y_train)
                test_acc = pipeline.score(X_test_scaled, y_test)
                cv_mean = _cv_mean_accuracy(X, y, pipeline, scale=True)
                logger.debug(
                    "Model: %s | NO PCA | Train: %.3f | Test: %.3f | CV: %.3f",
                    model_name,
                    train_acc,
                    test_acc,
                    cv_mean,
                )
                results.append(
                    (
                        s,
                        model_name,
                        "None",
                        None,
                        train_acc,
                        test_acc,
                        cv_mean,
                    )
                )

        # CSP is not used here because it requires raw epoch tensors rather
        # than tabular features.

    res_df = pd.DataFrame(
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

    if not res_df.empty:
        _, best_result = _best_mean_result(
            res_df,
            ["Model", "Transformer", "Components"],
        )
        if best_result is not None:
            logger.info("Best mean combination: %s", best_result.to_dict())
            best_model = best_result["Model"]
            best_transformer = best_result["Transformer"]
            best_components = best_result["Components"]

            for s in subjects:
                df_s = df[df["subject"] == s]
                y = df_s["label"]
                feature_cols = _feature_columns(df_s)
                X = df_s[feature_cols].values
                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=0.2,
                    random_state=42,
                    stratify=y,
                )
                X_train_scaled, X_test_scaled = standard_train_test(
                    X_train,
                    X_test,
                )
                if best_transformer == "PCA":
                    n_comp = min(int(best_components), X_train_scaled.shape[1])
                    pipeline = _build_pipeline(best_model, "PCA", n_comp)
                else:
                    pipeline = _build_pipeline(best_model, "None", None)

                pipeline.fit(X_train_scaled, y_train)
                y_pred = pipeline.predict(X_test_scaled)
                logger.info("Predictions for subject %s:", s)
                _log_predictions(y_test, y_pred)
                logger.info(
                    "Accuracy: %.4f",
                    float(np.mean(y_pred == np.asarray(y_test))),
                )
        else:
            logger.warning("Could not calculate the mean CV score per model.")
    else:
        best_result = None
        logger.warning("Could not train any model successfully.")

    return res_df, best_result


def train_and_validate_csp(X, y, subject_vec, subject=None):
    """Train and evaluate CSP-based models on epoch tensors."""
    from mne.decoding import CSP

    models = _get_models()
    csp_components_list = [2, 4, 6]

    if subject is None:
        subjects = np.unique(subject_vec)
    elif isinstance(subject, int):
        subjects = [subject]
    else:
        subjects = subject

    results = []
    for s in subjects:
        mask = subject_vec == s
        X_s = X[mask]
        y_s = y[mask]
        X_train, X_test, y_train, y_test = train_test_split(
            X_s,
            y_s,
            test_size=0.2,
            random_state=42,
            stratify=y_s,
        )

        for model_name, clf in models.items():
            for n_comp in csp_components_list:
                pipeline = Pipeline(
                    [
                        ("csp", CSP(n_components=n_comp, reg=0.1, log=True)),
                        ("clf", clf),
                    ]
                )
                pipeline.fit(X_train, y_train)
                train_acc = pipeline.score(X_train, y_train)
                test_acc = pipeline.score(X_test, y_test)
                cv_mean = _cv_mean_accuracy(X_s, y_s, pipeline, scale=False)
                logger.debug(
                    (
                        "Model: %s | CSP components: %s | Train: %.3f | "
                        "Test: %.3f | CV: %.3f"
                    ),
                    model_name,
                    n_comp,
                    train_acc,
                    test_acc,
                    cv_mean,
                )
                results.append(
                    (
                        s,
                        model_name,
                        "CSP",
                        n_comp,
                        train_acc,
                        test_acc,
                        cv_mean,
                    )
                )

    res_df = pd.DataFrame(
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

    if not res_df.empty:
        _, best_result = _best_mean_result(
            res_df,
            ["Model", "Transformer", "Components"],
        )
        if best_result is not None:
            logger.info("Best mean combination: %s", best_result.to_dict())
            best_model = best_result["Model"]
            best_transformer = best_result["Transformer"]
            best_components = best_result["Components"]

            for s in subjects:
                mask = subject_vec == s
                X_s = X[mask]
                y_s = y[mask]
                X_train, X_test, y_train, y_test = train_test_split(
                    X_s,
                    y_s,
                    test_size=0.2,
                    random_state=42,
                    stratify=y_s,
                )
                pipeline = _build_pipeline(
                    best_model,
                    best_transformer,
                    int(best_components),
                )
                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)
                logger.info("Predictions for subject %s:", s)
                _log_predictions(y_test, y_pred)
                logger.info(
                    "Accuracy: %.4f",
                    float(np.mean(y_pred == np.asarray(y_test))),
                )
        else:
            logger.warning("Could not calculate the mean CV score per model.")
    else:
        best_result = None
        logger.warning("Could not train any model successfully.")

    return res_df, best_result
