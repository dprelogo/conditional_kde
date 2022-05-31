"""Utilities."""
import numpy as np


class DataWhitener:
    """Whitening of the data.
    Implements several algorithms, depending on the desired whitening properties.
    Args:
        algorithm (str): one of `["PCA", "ZCA", "rescale"]`.
            "PCA": data is transformed into its PCA space and divided by
                the standard deviation of each dimension
            "ZCA": equivalent to the "PCA", with additional step of rotating
                back to original space. In this case, the final data still
                outputs 'in the same direction'.
            "rescale": calculates mean and standard deviation in each dimension
                and rescales it to zero-mean, unit-variance. In the absence
                of high correlations between dimensions, this is often sufficient.
    """

    def __init__(self, algorithm="rescale"):
        self.algorithm = algorithm

    def fit(self, X, save_data=False):
        """Fitting the whitener on the data X.
        Args:
            X (array): of shape `(n_samples, n_dim)`.
            save_data (bool): if `True`, saves the data and whitened data as
                `self.data`, `self.whitened_data`.
        Returns:
            Whitened array.
        """
        self.μ = np.mean(X, axis=0, dtype=np.float128).astype(np.float32)
        Σ = np.cov(X.T)
        evals, evecs = np.linalg.eigh(Σ)

        if self.algorithm == "PCA":
            self.W = np.einsum("ij,kj->ik", np.diag(evals ** (-1 / 2)), evecs)
            self.WI = np.einsum("ij,jk->ik", evecs, np.diag(evals ** (1 / 2)))
        elif self.algorithm == "ZCA":
            self.W = np.einsum("ij,jk,lk->il", evecs, np.diag(evals ** (-1 / 2)), evecs)
            self.WI = np.einsum("ij,jk,lk->il", evecs, np.diag(evals ** (1 / 2)), evecs)
        elif self.algorithm == "rescale":
            self.W = np.identity(len(Σ)) * np.diag(Σ) ** (-1 / 2)
            self.WI = np.identity(len(Σ)) * np.diag(Σ) ** (1 / 2)
        else:
            raise ValueError("`algorithm` should be either PCA, ZCA or rescale.")

        if save_data:
            self.data = X
            self.whitened_data = self.whiten(X)

    def whiten(self, X):
        """Whiten the data by making it unit covariance.
        Args:
            X (array): of shape `(n_samples, n_dims)`.
                Data to whiten. `n_dims` has to be the same as self.data.
        Returns:
            whitened_data (array): whitened data, of shape `(n_samples, n_dims)`
        """
        if len(X.shape) == 1:
            X = np.expand_dims(X, axis=0)
            squeeze = True
        else:
            squeeze = False
        X_whitened = np.einsum("ij,kj->ki", self.W, X - self.μ)
        return np.squeeze(X_whitened) if squeeze else X_whitened
        # return (self.W @ (X - self.means).T).T

    def unwhiten(self, X_whitened):
        """Un-whiten the sample with whitening parameters from the data.
        Args:
            X_whitened (array): array_like, shape `(n_samples, n_dims)`
                Sample of the data to un-whiten.
                `n_dims` has to be the same as `self.data`.
        Returns:
            X (array): whitened data, of shape `(n_samples, n_dims)`
        """
        # return (self.WI @ X.T).T + self.means
        if len(X_whitened.shape) == 1:
            X_whitened = np.expand_dims(X_whitened, axis=0)
            squeeze = True
        else:
            squeeze = False
        X = np.einsum("ij,kj->ki", self.WI, X_whitened) + self.μ
        return np.squeeze(X) if squeeze else X
