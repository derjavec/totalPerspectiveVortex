import logging

from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

from .helpers import (
    cv_mean_accuracy,
    evaluate_best_by_subject,
    get_models,
    log_best_result,
    resolve_subjects,
    results_dataframe,
    split_subject_data,
    standard_train_test,
)

logger = logging.getLogger(__name__)


def feature_columns(df):
    """Return the model feature columns for tabular data."""
    return [
        column
        for column in df.columns
        if column not in ["subject", "label", "epoch_id"]
    ]


def safe_pca_components(requested_components, X_train):
    """Return a PCA component count valid for the current train split."""
    max_components = min(X_train.shape[0], X_train.shape[1])
    return min(requested_components, max_components)


def get_feature_data_for_subject(df, subject):
    """Extract tabular features and labels for one subject."""
    df_s = df[df["subject"] == subject]
    y = df_s["label"]
    X = df_s[feature_columns(df_s)].values
    return X, y


def build_pca_pipeline(model_name, components):
    """Build a PCA + classifier pipeline."""
    models = get_models()
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}")

    clf = models[model_name]
    return Pipeline(
        [
            ("pca", PCA(n_components=components)),
            ("clf", clf),
        ]
    )


def build_plain_pipeline(model_name):
    """Build a classifier-only pipeline."""
    models = get_models()
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}")

    clf = models[model_name]
    return Pipeline([("clf", clf)])


def train_single_pca_config(X, y, clf, n_comp):
    """Train and evaluate one PCA + classifier configuration."""
    X_train, X_test, y_train, y_test = split_subject_data(X, y)
    X_train_scaled, X_test_scaled = standard_train_test(X_train, X_test)

    valid_n_comp = safe_pca_components(n_comp, X_train_scaled)
    pipeline = Pipeline(
        [
            ("pca", PCA(n_components=valid_n_comp)),
            ("clf", clf),
        ]
    )
    pipeline.fit(X_train_scaled, y_train)

    train_acc = pipeline.score(X_train_scaled, y_train)
    test_acc = pipeline.score(X_test_scaled, y_test)
    cv_mean = cv_mean_accuracy(X, y, pipeline, scale=True)

    return valid_n_comp, train_acc, test_acc, cv_mean


def train_single_plain_config(X, y, clf):
    """Train and evaluate one classifier without transformer."""
    X_train, X_test, y_train, y_test = split_subject_data(X, y)
    X_train_scaled, X_test_scaled = standard_train_test(X_train, X_test)

    pipeline = Pipeline([("clf", clf)])
    pipeline.fit(X_train_scaled, y_train)

    train_acc = pipeline.score(X_train_scaled, y_train)
    test_acc = pipeline.score(X_test_scaled, y_test)
    cv_mean = cv_mean_accuracy(X, y, pipeline, scale=True)

    return train_acc, test_acc, cv_mean


def _collect_pca_results(X, y, subject, model_name, clf, pca_components_list):
    """Collect results for PCA configurations."""
    results = []

    for n_comp in pca_components_list:
        valid_n_comp, train_acc, test_acc, cv_mean = train_single_pca_config(
            X, y, clf, n_comp
        )

        logger.debug(
            "Model: %s | PCA components: %s | Train: %.3f | Test: %.3f | CV: %.3f",
            model_name,
            valid_n_comp,
            train_acc,
            test_acc,
            cv_mean,
        )

        results.append(
            (
                subject,
                model_name,
                "PCA",
                valid_n_comp,
                train_acc,
                test_acc,
                cv_mean,
            )
        )

    return results


def _collect_plain_results(X, y, subject, model_name, clf):
    """Collect results for models without transformer."""
    train_acc, test_acc, cv_mean = train_single_plain_config(X, y, clf)

    logger.debug(
        "Model: %s | NO PCA | Train: %.3f | Test: %.3f | CV: %.3f",
        model_name,
        train_acc,
        test_acc,
        cv_mean,
    )

    return [
        (
            subject,
            model_name,
            "None",
            None,
            train_acc,
            test_acc,
            cv_mean,
        )
    ]


def collect_subject_results(df, subject, models, transformer, pca_components_list):
    """Train all model/config combinations for one subject."""
    X, y = get_feature_data_for_subject(df, subject)
    subject_results = []

    for model_name, clf in models.items():
        if transformer == "pca":
            subject_results.extend(
                _collect_pca_results(
                    X, y, subject, model_name, clf, pca_components_list
                )
            )
        else:
            subject_results.extend(
                _collect_plain_results(
                    X, y, subject, model_name, clf
                )
            )

    return subject_results


def log_best_predictions(df, subjects, best_result):
    """Refit best global configuration and log per-subject predictions."""
    best_model = best_result["Model"]
    best_transformer = best_result["Transformer"]
    best_components = best_result["Components"]

    def get_subject_data(subject):
        return get_feature_data_for_subject(df, subject)

    def fit_transform_data(X_train, X_test):
        return standard_train_test(X_train, X_test)

    def build_best_pipeline(X_train_ready):
        if best_transformer == "PCA":
            n_comp = safe_pca_components(int(best_components), X_train_ready)
            return build_pca_pipeline(best_model, n_comp)
        return build_plain_pipeline(best_model)

    evaluate_best_by_subject(
        subjects=subjects,
        get_subject_data=get_subject_data,
        fit_transform_data=fit_transform_data,
        build_best_pipeline=build_best_pipeline,
    )


def train_and_validate(df, subject=None, transformer=None):
    """Train and evaluate feature-based models with optional PCA."""
    pca_components_list = [2, 5, 10, 20, 30]
    models = get_models()
    subjects = resolve_subjects(subject, df["subject"].unique())

    results = []
    for s in subjects:
        results.extend(
            collect_subject_results(
                df=df,
                subject=s,
                models=models,
                transformer=transformer,
                pca_components_list=pca_components_list,
            )
        )
    res_df = results_dataframe(results)
    best_result = log_best_result(
        res_df,
        ["Model", "Transformer", "Components"],
    )

    if best_result is not None:
        log_best_predictions(df, subjects, best_result)

    return res_df, best_result