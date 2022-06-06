"""Main module."""
from multiprocessing.sharedctypes import Value
import numpy as np
from sklearn.neighbors import KernelDensity
from .util import DataWhitener, Interpolator


class ConditionalKernelDensity(KernelDensity):
    """Conditional Kernel Density estimator.

    Probability calculations are inherited from `sklearn.neighbors.KernelDensity`.
    The only addition is a sampling function, where one can "cut" the pdf at
    the conditioned values and pull samples directly from conditional density.
    Currently implemented only for a gaussian kernel.

    Args:
        rescale (bool): either to rescale the data or not. Good to use with
            `optimal_bandwidth` flag in `self.fit`.
        For other arguments see `sklearn.neighbors.KernelDensity`.
    """

    def __init__(
        self,
        bandwidth=1.0,
        rescale=True,
        kernel="gaussian",
        algorithm="auto",
        metric="euclidean",
        atol=0,
        rtol=0,
        breadth_first=True,
        leaf_size=40,
        metric_params=None,
    ):
        if kernel != "gaussian":
            raise NotImplementedError("Not implemented for a non-gaussian kernel.")

        self.algorithm = "rescale" if rescale else None

        super().__init__(
            bandwidth=bandwidth,
            algorithm=algorithm,
            kernel=kernel,
            metric=metric,
            atol=atol,
            rtol=rtol,
            breadth_first=breadth_first,
            leaf_size=leaf_size,
            metric_params=metric_params,
        )

    @staticmethod
    def log_scott(n_samples, n_features):
        """Scott's parameter."""
        return -1 / (n_features + 4) * np.log10(n_samples)

    def fit(self, X, optimal_bandwidth=True, features=None, **kwargs):
        """Fitting the Conditional Kernel Density.

        Args:
            X (array): data of shape `(n_samples, n_features)`.
            optimal_bandwidth (bool): if `True`, uses Scott's parameter
                for the bandwith. For the best results, use with
                `rescale=True`.
            features (list): optional, list defining names for every feature.
                It's used for referencing conditional dimensions.
                Defaults to `[0, 1, ..., n_features - 1]`.
            **kwargs: see `sklearn.neighbors.KernelDensity`.
        """
        n_samples, n_features = X.shape

        if optimal_bandwidth:
            self.bandwidth = 10 ** self.log_scott(n_samples, n_features)

        if features is None:
            self.features = list(range(n_features))
        else:
            if not isinstance(features, list) or n_features != len(features):
                raise ValueError(
                    f"`features` {features} should be a `list` of the same lenght "
                    f"as the dimensionality of the data ({n_features})."
                )
            self.features = features

        self.dw = DataWhitener(self.algorithm)
        self.dw.fit(X)
        X = self.dw.whiten(X)

        return super().fit(X, **kwargs)

    def score_samples(self, X):
        """Compute the log-likelihood of each sample under the model.

        Args:
            X (array): data of shape `(n_samples, n_features)`.
                Last dimension should match dimension of training data `(n_features)`.

        Returns:
            density (array): of shape `(n_samples,)`. Log-likelihood of each sample in `X`.
                These are normalized to be probability densities,
                so values will be low for high-dimensional data.
        """
        # TODO: write score_samples for the conditional distribution.
        X = self.dw.whiten(X)

        return super().score_samples(X)

    def sample(
        self,
        n_samples=1,
        random_state=None,
        conditionals=None,
        keep_dims=False,
    ):
        """Generate random samples from the conditional model.

        Currently, it is implemented only for a gaussian kernel.

        Args:
            n_samples (int): number of samples to generate. Defaults to 1.
            random_state (int): `RandomState` instance, default=None
                Determines random number generation used to generate
                random samples. Pass `int` for reproducible results
                across multiple function calls. See `Glossary <random_state>`.
            conditionals (dict): desired variables (features) to condition upon.
                Dictionary keys should be only feature names from `features`.
                For example, if `self.features == ["a", "b", "c"]` and one would like to condition
                on "a" and "c", then `conditionas = {"a": cond_val_a, "c": cond_val_c}`.
                Defaults to `None`, i.e. normal KDE.
            keep_dims (bool): whether to return non-conditioned dimensions only
                or keep given `conditional_values` for conditioned dimensions.
                Defaults to `None`.

        Returns:
            samples (array): array of samples, of shape `(n_samples, n_features)`
                if `conditional_variables is None`, else
                `(n_samples, n_features - sum(conditional_variables))`
        """
        data = np.asarray(self.tree_.data)
        rs = np.random.RandomState(seed=random_state)

        if conditionals is None:
            idx = rs.choice(data.shape[0], n_samples)
            sample = np.atleast_2d(rs.normal(data[idx], self.bandwidth))
            return self.dw.unwhiten(sample)
        else:
            if not isinstance(conditionals, dict):
                raise ValueError(
                    "`conditional_features` should be dictionary, but is "
                    f"{type(conditionals).__name__}."
                )
            if not all(k in self.features for k in conditionals.keys()):
                raise ValueError(
                    "All conditionals should be in features. If you haven't "
                    "specified features, pick integers from `[0, 1, ..., n_features - 1]`."
                )
            if len(conditionals) == data.shape[-1]:
                raise ValueError("One cannot condition on all features.")

            # scaling conditional variables
            cond_values = np.zeros(len(self.features), dtype=np.float32)
            cond_variables = np.zeros(len(self.features), dtype=bool)
            for c_val, c_var, f in zip(cond_values, cond_variables, self.features):
                if f in conditionals.keys():
                    c_val = conditionals[f]
                    c_var = True
            cond_values = self.dw.whiten(cond_values)[cond_variables]

            weights = np.exp(
                -np.sum((cond_values - data[:, cond_variables]) ** 2, axis=1)
                / (2 * self.bandwidth**2)
            )
            weights /= np.sum(weights)
            idx = rs.choice(data.shape[0], n_samples, p=weights)

            sample = np.atleast_2d(rs.normal(data[idx], self.bandwidth))
            sample = self.dw.unwhiten(sample)

            if keep_dims is False:
                return sample[:, ~cond_variables]
            else:
                sample[:, cond_variables] = np.broadcast_to(
                    cond_values, (n_samples, len(cond_values))
                )
                return sample


class ConditionalGaussianKernelDensity:
    """Conditional Kernel Density estimator.

    Args:
        bandwidth (float): the width of the Gaussian centered arount every point.
            By default, it uses "optimal" bandwidth - Scott's parameter.
        rescale (bool): either to rescale the data or not.
    """

    def __init__(
        self,
        bandwidth=None,
        rescale=True,
    ):
        if bandwidth is not None and not isinstance(bandwidth, (int, float)):
            raise ValueError(
                f"Bandwith should be a number, but is {type(bandwidth).__name__}."
            )
        self.bandwidth = bandwidth
        self.algorithm = "rescale" if rescale else None

        self.features = None  # names of features
        self.dw = None  # data whitener

    @staticmethod
    def log_scott(n_samples, n_features):
        """Scott's parameter."""
        return -1 / (n_features + 4) * np.log10(n_samples)

    @staticmethod
    def _log_prob(X, data, sigma, add_norm=True):
        """Log probability of a gaussian KDE distribution.

        Args:
            X (array): array of samples for which probability is calculated.
                Of shape `(n, n_features)`.
            data (array): KDE data, of shape `(n_samples, n_features)`.
            sigma (float, list, array): sigma of a gaussian distribution.
                If float, it is shared for all features, otherwise it should be
                a list/array of size `n_features`.
            add_norm (bool): either to add normalization factor to the calculation or not.

        Returns:
            log_prob (array): log probabilities, of shape `(n,)`.
        """
        n_samples, n_features = data.shape
        X = np.atleast_2d(X)
        if X.shape[-1] != n_features:
            raise ValueError("`n_features` of both arrays should be the same.")
        if not isinstance(sigma, (int, float, list, np.ndarray)):
            raise ValueError(
                f"`sigma` should be a number, list or `numpy.ndarray`, "
                f"but is {type(sigma).__name__}"
            )
        if isinstance(sigma, (list, np.ndarray)) and len(sigma) != n_features:
            raise ValueError("`sigma` should be of length `n_features`.")
        if isinstance(sigma, list):
            sigma = np.array(sigma, dtype=np.float32)

        log_prob = np.empty(X.shape[0])
        for x, p in zip(X, log_prob):
            if isinstance(sigma, (int, float)):
                delta = x - data
                p = -0.5 * np.einsum("ij,ij", delta, delta) / sigma**2
            else:
                delta = x - data
                p = -0.5 * np.einsum("ij,j,ij", delta, 1 / sigma**2, delta)
        if add_norm:
            norm = np.log(n_samples) + 0.5 * n_features * np.log(2 * np.pi)
            if isinstance(sigma, (int, float)):
                norm += n_features * np.log(sigma)
            else:
                norm += np.sum(np.log(sigma))
            return log_prob - norm
        else:
            return log_prob

    @staticmethod
    def _conditional_weights(conditional_values, conditional_data, sigma):
        """Weights for the sampling from the conditional distribution.

        They amount to the conditioned part of the gaussian for every data point.

        Args:
            conditional_values (array): of length `n_conditionals`.
            cond_data (array): of shape `(n_samples, n_conditionals)`.
                Here non-conditional dimensions are already removed.
            sigma (float, array): sigma of a gaussian distribution.
                If float, it is shared for all conditioned features, otherwise it should be
                array of size `n_conditionals`.
        """
        weights = np.exp(
            -0.5
            * np.sum((conditional_values - conditional_data) ** 2 / sigma**2, axis=1)
        )

        return weights / np.sum(weights)

    def fit(self, X, bandwidth=None, features=None):
        """Fitting the Conditional Kernel Density.

        Args:
            X (array): data of shape `(n_samples, n_features)`.
            bandwidth (float): the width of the Gaussian centered arount every point.
                By default, it uses "optimal" bandwidth - Scott's parameter.
            features (list): optional, list defining names for every feature.
                It's used for referencing conditional dimensions.
                Defaults to `[0, 1, ..., n_features - 1]`.
        """
        n_samples, n_features = X.shape

        if bandwidth is not None and not isinstance(bandwidth, (int, float)):
            raise ValueError(
                f"Bandwith should be a number, but is {type(bandwidth).__name__}."
            )
        if bandwidth is not None:
            self.bandwidth = bandwidth
        else:
            if self.bandwidth is None:
                self.bandwidth = 10 ** self.log_scott(n_samples, n_features)

        if features is None:
            self.features = list(range(n_features))
        else:
            if not isinstance(features, list):
                raise ValueError("`features` should be a `list`.")
            elif n_features != len(features):
                raise ValueError(
                    f"`n_features` ({n_features}) should be equal to "
                    f"the length of `features` ({len(features)})."
                )
            self.features = features

        self.dw = DataWhitener(self.algorithm)
        self.dw.fit(X, save_data=True)

    def score_samples(self, X):
        """Compute the log-probability of each sample under the model.

        Args:
            X (array): data of shape `(n, n_features)`.
                Last dimension should match dimension of training data `(n_features)`.

        Returns:
            p (array): of shape `(n,)`. Log probability of each sample in `X`.
                These are normalized to be probability densities,
                so values will be low for high-dimensional data.
        """
        return self._log_prob(self.dw.whiten(X), self.dw.whitened_data, self.bandwidth)

    def score_samples_conditional(self, X, conditional_features=None):
        """Compute the conditional log-probability of each sample under the model.

        Args:
            X (array): data of shape `(n, n_features)`.
                Last dimension should match dimension of training data `(n_features)`.
            conditional_features (list): subset of `self.features`, which dimensions of data
                to condition upon.

        Returns:
            p (array): of shape `(n,)`. Conditional log probability of each sample in `X`.
                These are normalized to be probability densities,
                so values will be low for high-dimensional data.
        """
        if conditional_features is None:
            return self.score_samples(X)
        else:
            if not all(k in self.features for k in conditional_features):
                raise ValueError(
                    "All conditional_features should be in features. If you haven't "
                    "specified features, pick integers from `[0, 1, ..., n_features - 1]`."
                )
            if len(conditional_features) == self.features:
                raise ValueError(
                    "Doesn't make much sense to condition on all features. "
                    "Probability of that is 1."
                )

            # scaling conditional variables
            cond_variables = [
                True if f in conditional_features else False for f in self.features
            ]
            cond_variables = np.array(cond_variables, dtype=bool)

            X = np.atleast_2d(X)
            X = self.dw.whiten(X)
            p_full = self._log_prob(X, self.dw.whitened_data, self.bandwidth)
            p_marginal = self._log_prob(
                X[:, cond_variables],
                self.dw.whitened_data[:, cond_variables],
                self.bandwidth,
            )
            return p_full - p_marginal

    def sample(
        self,
        n_samples=1,
        random_state=None,
        conditionals=None,
        keep_dims=False,
    ):
        """Generate random samples from the conditional model.

        Args:
            n_samples (int): number of samples to generate. Defaults to 1.
            random_state (int): `RandomState` instance, defaults to `None`.
                Determines random number generation used to generate
                random samples. Pass `int` for reproducible results
                across multiple function calls. See `Glossary <random_state>`.
            conditionals (dict): desired variables (features) to condition upon.
                Dictionary keys should be only feature names from `features`.
                For example, if `self.features == ["a", "b", "c"]` and one would like to condition
                on "a" and "c", then `conditionas = {"a": cond_val_a, "c": cond_val_c}`.
                Defaults to `None`, i.e. normal KDE.
            keep_dims (bool): whether to return non-conditioned dimensions only
                or keep given conditional values. Defaults to `False`.

        Returns:
            samples (array): array of samples, of shape `(n_samples, n_features)`
                if `conditional_variables is None`, else
                `(n_samples, n_features - sum(conditional_variables))`
        """
        data = self.dw.whitened_data
        rs = np.random.RandomState(seed=random_state)

        if conditionals is None:
            idx = rs.choice(data.shape[0], n_samples)
            sample = rs.normal(np.atleast_2d(data[idx]), self.bandwidth)
            return self.dw.unwhiten(sample)
        else:
            if not isinstance(conditionals, dict):
                raise ValueError(
                    "`conditional_features` should be dictionary, but is "
                    f"{type(conditionals).__name__}."
                )
            if not all(k in self.features for k in conditionals.keys()):
                raise ValueError(
                    "All conditionals should be in features. If you haven't "
                    "specified features, pick integers from `[0, 1, ..., n_features - 1]`."
                )
            if len(conditionals) == len(self.features):
                raise ValueError("One cannot condition on all features.")

            # scaling conditional variables
            cond_values = np.zeros(len(self.features), dtype=np.float32)
            cond_variables = np.zeros(len(self.features), dtype=bool)
            for c_val, c_var, f in zip(cond_values, cond_variables, self.features):
                if f in conditionals.keys():
                    c_val = conditionals[f]
                    c_var = True
            cond_values = self.dw.whiten(cond_values)[cond_variables]

            weights = self._conditional_weights(
                cond_values, data[:, cond_variables], self.bandwidth
            )
            idx = rs.choice(data.shape[0], n_samples, p=weights)

            # pulling the samples
            sample = np.empty((n_samples, len(self.features)))
            sample[:, ~cond_variables] = rs.normal(
                np.atleast_2d(data[idx])[:, ~cond_variables], self.bandwidth
            )
            sample[:, cond_variables] = np.broadcast_to(
                cond_values, (n_samples, len(cond_values))
            )
            sample = self.dw.unwhiten(sample)

            if keep_dims:
                return sample
            else:
                return sample[:, ~cond_variables]


class InterpolatedConditionalKernelDensity:
    """Interpolated Conditional Kernel Density estimator.

    With respect to the `ConditionalKernelDensity`, which fits full distribution
    and cuts through it to obtain the conditional distribution, here we allow
    for some dimensions of the data to be inherently conditional.
    For such dimensions, data should be available for every point on a grid.

    To compute the final conditional density, one then interpolates
    for the inherently conditional dimensions, and slices through others as before.

    Args:
        rescale (bool): either to rescale the data or not. Good to use with
            `optimal_bandwidth` flag in `self.fit`.
        For other arguments see `sklearn.neighbors.KernelDensity`.
    """

    pass
