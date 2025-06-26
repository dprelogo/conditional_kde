"""Important utilities."""

import itertools

import numpy as np
from scipy.interpolate import RegularGridInterpolator

# Removed import of private scipy function _ndim_coords_from_arrays


class DataWhitener:
    """Whitening of the data.

    Implements several algorithms, depending on the desired whitening properties.

    Args:
        algorithm (str): either `None`, `"center"`, `"rescale"`, `"PCA"` or `"ZCA"`.
            **None** - leaves the data as is. **center** - calculates mean in
            every dimension and removes it from the data. **rescale** - calculates
            mean and standard deviation in each dimension and rescales it to zero-mean,
            unit-variance. In the absence of high correlations between dimensions, this is often sufficient.
            **PCA** - data is transformed into its PCA space and divided by the
            standard deviation of each dimension. **ZCA** - equivalent to the `"PCA"`,
            with additional step of rotating back to original space, i.e. the
            orientation of space is preserved.
    """

    def __init__(self, algorithm="rescale"):
        if algorithm not in [None, "center", "rescale", "PCA", "ZCA"]:
            raise ValueError("algorithm should be None, center, rescale, PCA or ZCA.")
        self.algorithm = algorithm

        self.μ = None  # mean
        self.Σ = None  # covariance matrix
        self.W = None  # whitening matrix
        self.WI = None  # unwhitening matrix

    def fit(self, X, weights=None, save_data=False):
        """Fitting the whitener on the data X.

        Args:
            X (array): of shape `(n_samples, n_dim)`.
            weights (array): of shape `(n_samples,)`, weights of each sample in `X`.
            save_data (bool): if `True`, saves the data and whitened data as
                `self.data`, `self.whitened_data`.

        Returns:
            Whitened array.
        """
        if weights is None:
            weights = np.ones((len(X), 1), dtype=X.dtype)
        else:
            if len(weights) != len(X):
                raise ValueError("Weights and X should be of the same length.")
            weights = weights / np.sum(weights) * len(weights)
            weights = weights.reshape(-1, 1)

        if self.algorithm is None:
            self.μ = np.zeros((1, X.shape[-1]), dtype=X.dtype)
            self.Σ = np.identity(X.shape[-1], dtype=X.dtype)
        else:
            # self.μ = np.mean(X, axis=0, keepdims=True)
            self.μ = np.sum(weights * X, axis=0, keepdims=True) / np.sum(weights)
            dX = X - self.μ
            self.Σ = np.einsum("ji,jk->ik", weights * dX, dX) / (np.sum(weights) - 1)
            if self.algorithm == "rescale":
                # self.Σ = np.diag(np.var(X, axis=0))
                self.Σ = np.diag(np.diag(self.Σ))
            elif self.algorithm in ["PCA", "ZCA"]:
                # self.Σ = np.cov(X.T)
                evals, evecs = np.linalg.eigh(self.Σ)
        if self.algorithm is None:
            self.W = np.identity(X.shape[-1], dtype=X.dtype)
            self.WI = np.identity(X.shape[-1], dtype=X.dtype)
        elif self.algorithm == "rescale":
            self.W = np.diag(np.diag(self.Σ) ** (-1 / 2))
            self.WI = np.diag(np.diag(self.Σ) ** (1 / 2))
        elif self.algorithm == "PCA":
            self.W = np.einsum("ij,kj->ik", np.diag(evals ** (-1 / 2)), evecs)
            self.WI = np.einsum("ij,jk->ik", evecs, np.diag(evals ** (1 / 2)))
        elif self.algorithm == "ZCA":
            self.W = np.einsum("ij,jk,lk->il", evecs, np.diag(evals ** (-1 / 2)), evecs)
            self.WI = np.einsum("ij,jk,lk->il", evecs, np.diag(evals ** (1 / 2)), evecs)

        if save_data:
            self.data = X
            self.whitened_data = self.whiten(X)

    def whiten(self, X):
        """Whiten the data by making it unit covariance.

        Args:
            X (array): of shape `(n_samples, n_dims)`.
                Data to whiten. `n_dims` has to be the same as self.data.

        Returns:
            Whitened data, of shape `(n_samples, n_dims)`.
        """
        if self.algorithm is None:
            return X

        if len(X.shape) == 1:
            X = np.expand_dims(X, axis=0)
            squeeze = True
        else:
            squeeze = False
        if self.algorithm == "center":
            X_whitened = X - self.μ
        else:
            X_whitened = np.einsum("ij,kj->ki", self.W, X - self.μ)

        return np.squeeze(X_whitened) if squeeze else X_whitened

    def unwhiten(self, X_whitened):
        """Un-whiten the sample with whitening parameters from the data.

        Args:
            X_whitened (array): array_like, shape `(n_samples, n_dims)`
                Sample of the data to un-whiten.
                `n_dims` has to be the same as `self.data`.

        Returns:
            Unwhitened data, of shape `(n_samples, n_dims)`.
        """
        if self.algorithm is None:
            return X_whitened

        if len(X_whitened.shape) == 1:
            X_whitened = np.expand_dims(X_whitened, axis=0)
            squeeze = True
        else:
            squeeze = False
        if self.algorithm == "center":
            X = X_whitened + self.μ
        else:
            X = np.einsum("ij,kj->ki", self.WI, X_whitened) + self.μ

        return np.squeeze(X) if squeeze else X


class Interpolator(RegularGridInterpolator):
    """Regular grid interpolator.

    Inherits from `scipy.interpolate.RegularGridInterpolator`.
    The difference with respect to the original class is to make
    weights and edges explicitly visible, for the more general usage case.

    Args:
        points (list): list of lists, where every sub-list defines the grid points.
        values (array): array of function values to interpolate.
            If not given, `Interpolator` won't return any values, only weights and edges.
        method (str): either "linear" or "nearest", defining the interpolation method.
            As the name says, "linear" returns linearly interpolated value
            and "nearest" only the nearest value on the grid.
        bounds_error (bool): either to raise an error if the point for which
            interpolation is requested is out of the grid bounds.
        fill_value (float): value to be filled for out of the grid points.
    """

    def __init__(
        self, points, values=None, method="linear", bounds_error=True, fill_value=np.nan
    ):
        self.interpolate_values = False if values is None else True
        if values is None:
            values = np.zeros(tuple(len(p) for p in points), dtype=float)
        super().__init__(
            points,
            values,
            method=method,
            bounds_error=bounds_error,
            fill_value=fill_value,
        )

    def __call__(self, xi, method=None, return_aux=True):
        """Interpolation at coordinates.

        If values were not passed during initialization, it doesn't interpolate
        but returns grid coordinates and weights only.

        Args:
            xi (array): the coordinates to sample the gridded data at, of shape `(..., ndim)`.
            method (str): The method of interpolation to perform.
                Supported are "linear" and "nearest".
            return_aux (bool): If `True`, return includes grid coordinates and weights.

        Returns:
            If function values were set during initialization, returns an array of interpolated values.
            If `return_aux is True` it will further return grid coordinates for every item in `xi`.
            In the case method is "nearest", this will be only one relevant coordinate per `xi` sample,
            or multiple ones for "linear" method. For the latter, also weights of every coordinate will be returned.
        """
        if self.interpolate_values is False and return_aux is False:
            raise ValueError(
                "Please either define `values` or set `return_aux` to `True`. "
                "Otherwise there's nothing to compute."
            )
        method = self.method if method is None else method
        if method not in ["linear", "nearest"]:
            raise ValueError(f"Method {method} is not defined")

        ndim = len(self.grid)

        # Replace _ndim_coords_from_arrays with explicit handling
        if isinstance(xi, np.ndarray):
            if xi.ndim == 1:
                # 1D array: interpret as single point with ndim coordinates
                xi = xi.reshape(1, -1)
        else:
            # xi is a tuple/list of coordinate arrays
            xi = np.column_stack([np.asarray(x).ravel() for x in xi])

        if xi.shape[-1] != len(self.grid):
            raise ValueError(
                f"The requested sample points xi have dimension "
                f"{xi.shape[1]}, but this RegularGridInterpolator has "
                f"dimension {ndim}"
            )

        xi_shape = xi.shape
        xi = xi.reshape(-1, xi_shape[-1])

        if self.bounds_error:
            for i, p in enumerate(xi.T):
                if not np.logical_and(
                    np.all(self.grid[i][0] <= p), np.all(p <= self.grid[i][-1])
                ):
                    raise ValueError(
                        f"One of the requested xi is out of bounds in dimension {i}"
                    )

        # In newer scipy versions, _find_indices only returns 2 values
        find_indices_result = self._find_indices(xi.T)
        if len(find_indices_result) == 2:
            indices, norm_distances = find_indices_result
            # Calculate out_of_bounds manually
            out_of_bounds = np.zeros(xi.shape[0], dtype=bool)
            for i, (p, grid) in enumerate(zip(xi.T, self.grid)):
                out_of_bounds |= (p < grid[0]) | (p > grid[-1])
        else:
            indices, norm_distances, out_of_bounds = find_indices_result
        if method == "linear":
            result, edges, weights = self._evaluate_linear(
                indices, norm_distances, out_of_bounds
            )
        elif method == "nearest":
            result, edges = self._evaluate_nearest(
                indices, norm_distances, out_of_bounds
            )
        if not self.bounds_error and self.fill_value is not None:
            result[out_of_bounds] = self.fill_value
        result = result.reshape(xi_shape[:-1] + self.values.shape[ndim:])

        out = ()
        if self.interpolate_values:
            out = out + (result,)
        if return_aux:
            out = out + (edges,)
            if method == "linear":
                out = out + (weights,)
        return out[0] if len(out) == 1 else out

    def _evaluate_linear(self, indices, norm_distances, out_of_bounds):
        # slice for broadcasting over trailing dimensions in self.values
        vslice = (slice(None),) + (None,) * (self.values.ndim - len(indices))

        # find relevant values
        # each i and i+1 represents a edge
        edges = list(itertools.product(*[[i, i + 1] for i in indices]))
        weights = []
        values = 0.0
        for edge_indices in edges:
            weight = 1.0
            for ei, i, yi in zip(edge_indices, indices, norm_distances):
                weight *= np.where(ei == i, 1 - yi, yi)
            values += np.asarray(self.values[edge_indices]) * weight[vslice]
            weights.append(weight[vslice])
        return values, edges, weights

    def _evaluate_nearest(self, indices, norm_distances, out_of_bounds):
        idx_res = [
            np.where(yi <= 0.5, i, i + 1) for i, yi in zip(indices, norm_distances)
        ]
        edges = tuple(idx_res)
        return self.values[edges], edges
