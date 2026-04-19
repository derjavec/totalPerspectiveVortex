import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class MyPCA(BaseEstimator, TransformerMixin):
    """Principal Component Analysis transformer.

    This transformer computes principal components from the covariance
    matrix of the centered input data and projects the data onto the
    selected components.

    """

    def __init__(self, n_components=2):
        self.n_components = n_components

    def fit(self, X, y=None):
        """Fit the PCA model on the input data.

        Returns
        -------
        MyPCA
            Fitted transformer.
        """
        if self.n_components < 1:
            raise ValueError("n_components must be at least 1")

        self.mean_ = np.mean(X, axis=0)
        x_centered = X - self.mean_

        n_samples = x_centered.shape[0]
        cov = (x_centered.T @ x_centered) / (n_samples - 1)

        eigvals, eigvecs = np.linalg.eigh(cov)

        sorted_idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[sorted_idx]
        eigvecs = eigvecs[:, sorted_idx]

        self.components_ = eigvecs[:, :self.n_components]
        self.explained_variance_ = eigvals[:self.n_components]

        return self

    def transform(self, X):
        """Project the input data onto the principal components.

        Returns
        -------
        ndarray of shape (n_samples, n_components)
            Transformed data in the reduced-dimensional space.
        """
        x_centered = X - self.mean_
        return x_centered @ self.components_
