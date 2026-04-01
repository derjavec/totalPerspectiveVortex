import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger(__name__)

def standard_by_subject_train_test(X_train, subj_train, X_test, subj_test):
    X_train_scaled = np.zeros_like(X_train)
    X_test_scaled = np.zeros_like(X_test)
    scalers = {}

    for s in np.unique(subj_train):
        idx_tr = subj_train == s
        scaler = StandardScaler().fit(X_train[idx_tr])
        X_train_scaled[idx_tr] = scaler.transform(X_train[idx_tr])
        scalers[s] = scaler

    for s in np.unique(subj_test):
        idx_te = subj_test == s
        if s in scalers:
            X_test_scaled[idx_te] = scalers[s].transform(X_test[idx_te])
        else:
            # sujeto solo en test: fallback global
            X_test_scaled[idx_te] = X_test[idx_te]

    return X_train_scaled, X_test_scaled


def _cv_mean_accuracy(X, y, subjects_vec, pipeline, scale_by_subject=True):
    unique_subjects = np.unique(subjects_vec)
    if len(unique_subjects) > 1:
        n_splits = min(5, len(unique_subjects))
        splitter = GroupKFold(n_splits=n_splits)
        split_iter = splitter.split(X, y, groups=subjects_vec)
    else:
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        split_iter = splitter.split(X, y)

    scores = []
    for train_idx, test_idx in split_iter:
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        subj_tr, subj_te = subjects_vec[train_idx], subjects_vec[test_idx]

        if scale_by_subject:
            # Escalado por sujeto usando solo train del fold (evita leakage)
            X_tr, X_te = standard_by_subject_train_test(X_tr, subj_tr, X_te, subj_te)

        pipeline.fit(X_tr, y_tr)
        scores.append(pipeline.score(X_te, y_te))

    return float(np.mean(scores))


def train_and_validate(df, subject=None, transformer = None):
    """
    Entrena modelos y evalúa PCA y CSP con distintos componentes.
    Devuelve un DataFrame con todos los resultados y la mejor combinación según CV.
    """
    models = {
        "LogisticRegression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "LinearSVM": SVC(kernel="linear", C=1.0, class_weight="balanced"),
        "RBF_SVM": SVC(kernel="rbf", C=2.0, gamma="scale", class_weight="balanced"),
        "LDA": LinearDiscriminantAnalysis(),
        "GaussianNB": GaussianNB(),
        "MLP": MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-4, max_iter=500, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42)
    }

    pca_components_list = [2, 5, 10, 20, 30]

    if subject is None:
        subjects = df['subject'].unique()
    elif isinstance(subject, int):
        subjects = [subject]
    else:
        subjects = subject

    df = df[df['subject'].isin(subjects)]
    y = df['label']
    subjects_vec = df['subject'].values
    feature_cols = [c for c in df.columns if c not in ["subject", "label", "epoch_id"]]
    X = df[feature_cols].values
    X_train, X_test, y_train, y_test, subj_train, subj_test = train_test_split(
        X, y, subjects_vec, test_size=0.2, random_state=42, stratify=y
    )

    X_train_scaled, X_test_scaled = standard_by_subject_train_test(X_train, subj_train, X_test, subj_test)
    results = []

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
                cv_mean = _cv_mean_accuracy(X, y, subjects_vec, pipeline, scale_by_subject=True)
                logger.debug("Model: %s | PCA components: %s | Train: %.3f | Test: %.3f | CV: %.3f",
                             m_name, n_comp, train_acc, test_acc, cv_mean)
                results.append((m_name, 'PCA', n_comp, train_acc, test_acc, cv_mean))
        else:
            pipeline = Pipeline([
                ('clf', clf)
            ])
            pipeline.fit(X_train_scaled, y_train)
            train_acc = pipeline.score(X_train_scaled, y_train)
            test_acc = pipeline.score(X_test_scaled, y_test)
            cv_mean = _cv_mean_accuracy(X, y, subjects_vec, pipeline, scale_by_subject=True)
            logger.debug("Model: %s | NO PCA | Train: %.3f | Test: %.3f | CV: %.3f",
                         m_name, train_acc, test_acc, cv_mean)
            results.append((m_name, 'None', None, train_acc, test_acc, cv_mean))

        # CSP removido: requiere epochs crudos (n_epochs, n_channels, n_times) y no features tabulares.

    res_df = pd.DataFrame(results, columns=['Model', 'Transformer', 'Components', 'Train', 'Test', 'CV'])

    # Mejor combinación
    if not res_df.empty:
        best_idx = res_df['CV'].idxmax()
        best_result = res_df.loc[best_idx]
        logger.info("Best combination: %s", best_result.to_dict())
    else:
        best_result = None
        logger.warning("No se pudo entrenar ningún modelo correctamente.")

    return res_df, best_result


def train_and_validate_csp(X, y, subject_vec, subject=None):
    from mne.decoding import CSP
    """
    Entrena modelos y evalúa PCA y CSP con distintos componentes.
    Devuelve un DataFrame con todos los resultados y la mejor combinación según CV.
    """
    models = {
        "LogisticRegression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "LinearSVM": SVC(kernel="linear", C=1.0, class_weight="balanced"),
        "RBF_SVM": SVC(kernel="rbf", C=2.0, gamma="scale", class_weight="balanced"),
        "LDA": LinearDiscriminantAnalysis(),
        "GaussianNB": GaussianNB(),
        "MLP": MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-4, max_iter=500, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42)
    }

    csp_components_list = [2, 4, 6]

    if subject is None:
        subjects = np.unique(subject_vec)
    elif isinstance(subject, int):
        subjects = [subject]
    else:
        subjects = subject

    mask = np.isin(subject_vec, subjects)
    X = X[mask]
    y = y[mask]
    subject_vec = subject_vec[mask]
    X_train, X_test, y_train, y_test, subj_train, subj_test = train_test_split(
        X, y, subject_vec, test_size=0.2, random_state=42, stratify=y
    )

    results = []

    for m_name, clf in models.items():
        for n_comp in csp_components_list:
            pipeline = Pipeline([
                ('csp', CSP(n_components=n_comp, reg=0.1, log=True)),
                ('clf', clf)
            ])
            pipeline.fit(X_train, y_train)
            train_acc = pipeline.score(X_train, y_train)
            test_acc = pipeline.score(X_test, y_test)
            cv_mean = _cv_mean_accuracy(X, y, subject_vec, pipeline, scale_by_subject=False)
            logger.debug("Model: %s | CSP components: %s | Train: %.3f | Test: %.3f | CV: %.3f",
                         m_name, n_comp, train_acc, test_acc, cv_mean)
            results.append((m_name, 'CSP', n_comp, train_acc, test_acc, cv_mean))

        # CSP removido: requiere epochs crudos (n_epochs, n_channels, n_times) y no features tabulares.

    res_df = pd.DataFrame(results, columns=['Model', 'Transformer', 'Components', 'Train', 'Test', 'CV'])

    # Mejor combinación
    if not res_df.empty:
        best_idx = res_df['CV'].idxmax()
        best_result = res_df.loc[best_idx]
        logger.info("Best combination: %s", best_result.to_dict())
    else:
        best_result = None
        logger.warning("No se pudo entrenar ningún modelo correctamente.")

    return res_df, best_result
