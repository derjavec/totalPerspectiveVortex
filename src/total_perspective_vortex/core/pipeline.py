import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger(__name__)

def standard_train_test(X_train, X_test):
    """Scale train and test sets using statistics from the training data."""
    X_train_scaled = np.zeros_like(X_train)
    X_test_scaled = np.zeros_like(X_test)


    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)

    X_test_scaled = scaler.transform(X_test)
 
    return X_train_scaled, X_test_scaled


def _cv_mean_accuracy(X, y, pipeline, scale=True):
    """Compute mean cross-validated accuracy with optional scaling."""
    y = np.asarray(y)

    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    split_iter = splitter.split(X, y)

    scores = []
    for train_idx, test_idx in split_iter:
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
        "LogisticRegression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "LinearSVM": SVC(kernel="linear", C=1.0, class_weight="balanced"),
        "RBF_SVM": SVC(kernel="rbf", C=2.0, gamma="scale", class_weight="balanced"),
        "LDA": LinearDiscriminantAnalysis(),
        "GaussianNB": GaussianNB(),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
    }


def _best_mean_result(res_df, group_cols):
    """Compute mean metrics by model and return the best row."""
    res_cv = (
        res_df
        .groupby(group_cols, as_index=False, dropna=False)[["Train", "Test", "CV"]]
        .mean()
        .sort_values("CV", ascending=False)
    )
    if res_cv.empty:
        return res_cv, None
    return res_cv, res_cv.iloc[0]


def train_and_validate(df, subject=None, transformer = None):
    """Train and evaluate feature-based models with optional PCA."""
    models = _get_models()

    pca_components_list = [2, 5, 10, 20, 30]

    if subject is None:
        subjects = df['subject'].unique()
    elif isinstance(subject, int):
        subjects = [subject]
    else:
        subjects = subject


    results = []
    for s in subjects:
        df_s = df[df['subject'] == s]
        y = df_s['label']
        feature_cols = [c for c in df_s.columns if c not in ["subject", "label", "epoch_id"]]
        X = df_s[feature_cols].values
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X_train_scaled, X_test_scaled = standard_train_test(X_train, X_test)
        for m_name, clf in models.items():
            if transformer == "pca":
                for n_comp in pca_components_list:
                    pipeline = Pipeline([
                        ('pca', PCA(n_components=min(n_comp, X_train_scaled.shape[1]))),
                        ('clf', clf)
                    ])
                    pipeline.fit(X_train_scaled, y_train)
                    train_acc = pipeline.score(X_train_scaled, y_train)
                    test_acc = pipeline.score(X_test_scaled, y_test)
                    cv_mean = _cv_mean_accuracy(X, y, pipeline, scale=True)
                    logger.debug("Model: %s | PCA components: %s | Train: %.3f | Test: %.3f | CV: %.3f",
                                m_name, n_comp, train_acc, test_acc, cv_mean)
                    results.append((s, m_name, 'PCA', n_comp, train_acc, test_acc, cv_mean))
            else:
                pipeline = Pipeline([
                    ('clf', clf)
                ])
                pipeline.fit(X_train_scaled, y_train)
                train_acc = pipeline.score(X_train_scaled, y_train)
                test_acc = pipeline.score(X_test_scaled, y_test)
                cv_mean = _cv_mean_accuracy(X, y, pipeline, scale=True)
                logger.debug("Model: %s | NO PCA | Train: %.3f | Test: %.3f | CV: %.3f",
                            m_name, train_acc, test_acc, cv_mean)
                results.append((s, m_name, 'None', None, train_acc, test_acc, cv_mean))

        # CSP removido: requiere epochs crudos (n_epochs, n_channels, n_times) y no features tabulares.

    res_df = pd.DataFrame(results, columns=['Subject', 'Model', 'Transformer', 'Components', 'Train', 'Test', 'CV'])

    # Mejor combinación (promedio entre sujetos)
    if not res_df.empty:
        res_cv, best_result = _best_mean_result(res_df, ["Model", "Transformer", "Components"])
        if best_result is not None:
            logger.info("Best mean combination: %s", best_result.to_dict())
        else:
            logger.warning("No se pudo calcular el promedio de CV por modelo.")
    else:
        best_result = None
        logger.warning("No se pudo entrenar ningún modelo correctamente.")

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
            X_s, y_s, test_size=0.2, random_state=42, stratify=y_s
        )
        for m_name, clf in models.items():
            for n_comp in csp_components_list:
                pipeline = Pipeline([
                    ('csp', CSP(n_components=n_comp, reg=0.1, log=True)),
                    ('clf', clf)
                ])
                pipeline.fit(X_train, y_train)
                train_acc = pipeline.score(X_train, y_train)
                test_acc = pipeline.score(X_test, y_test)
                cv_mean = _cv_mean_accuracy(X_s, y_s, pipeline, scale=False)
                logger.debug("Model: %s | CSP components: %s | Train: %.3f | Test: %.3f | CV: %.3f",
                            m_name, n_comp, train_acc, test_acc, cv_mean)
                results.append((s, m_name, 'CSP', n_comp, train_acc, test_acc, cv_mean))


    res_df = pd.DataFrame(results, columns=['Subject', 'Model', 'Transformer', 'Components', 'Train', 'Test', 'CV'])

    if not res_df.empty:
        res_cv, best_result = _best_mean_result(res_df, ["Model", "Transformer", "Components"])
        if best_result is not None:
            logger.info("Best mean combination: %s", best_result.to_dict())
        else:
            logger.warning("No se pudo calcular el promedio de CV por modelo.")
    else:
        best_result = None
        logger.warning("No se pudo entrenar ningún modelo correctamente.")

    return res_df, best_result
