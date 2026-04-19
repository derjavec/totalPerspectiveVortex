"""Model definitions available for training."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def get_models():
    """Return the classifiers available for evaluation."""
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
