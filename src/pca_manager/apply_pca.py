import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def apply_pca(df, n_components=None):
    feature_cols = [c for c in df.columns if c not in ["subject", "label", "epoch_id"]]
    X = df[feature_cols].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    print("Varianza explicada por componente:", pca.explained_variance_ratio_)

    loading_df = pd.DataFrame(
        pca.components_.T,
        index=feature_cols,
        columns=[f"PC{i+1}" for i in range(pca.n_components_)]
    )

    print(loading_df.sort_values("PC1", ascending=False).head(10))
    return loading_df, X_pca
