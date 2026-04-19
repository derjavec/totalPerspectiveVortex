import logging

import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif


logger = logging.getLogger(__name__)

METADATA_COLUMNS = ["subject", "label", "epoch_id"]


def get_feature_columns(df, metadata_columns=None):
    """Return feature columns excluding metadata columns."""
    if metadata_columns is None:
        metadata_columns = METADATA_COLUMNS

    return [col for col in df.columns if col not in metadata_columns]


def select_best_features_anova(
    df,
    k=20,
    label_col="label",
    metadata_columns=None,
    min_features=1,
):
    """Select the most relevant features using ANOVA.

    The function ranks all features based on the ANOVA F-score and keeps
    the top-k features. Metadata columns are preserved in the output.

    Returns
    -------
    pd.DataFrame
        DataFrame containing metadata columns and selected features.
    pd.DataFrame
        Ranking of all features with ANOVA scores and selection flags.
    """
    if metadata_columns is None:
        metadata_columns = METADATA_COLUMNS.copy()

    if label_col not in df.columns:
        raise ValueError(f"Column '{label_col}' not found in dataframe.")

    feature_cols = get_feature_columns(
        df,
        metadata_columns=metadata_columns,
    )

    if not feature_cols:
        raise ValueError("No feature columns available for selection.")

    X = df[feature_cols]
    y = df[label_col]

    n_features = len(feature_cols)
    k = max(min_features, min(k, n_features))

    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(X, y)

    selected_mask = selector.get_support()
    selected_features = [
        col for col, keep in zip(feature_cols, selected_mask) if keep
    ]

    ranking_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "anova_score": selector.scores_,
            "p_value": selector.pvalues_,
            "selected": selected_mask,
        }
    ).sort_values(
        by="anova_score",
        ascending=False,
        na_position="last",
    )

    keep_metadata = [col for col in metadata_columns if col in df.columns]

    selected_df = df[keep_metadata + selected_features].copy()

    logger.info(
        "ANOVA selected %d/%d features.",
        len(selected_features),
        n_features,
    )
    logger.info("Top selected features: %s", selected_features[:10])

    return selected_df, ranking_df
