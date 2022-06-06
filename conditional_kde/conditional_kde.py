"""Main module."""
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
            raise ValueError(f"Bandwith should be a number, but is {type(bandwidth)}.")
        self.bandwidth = bandwidth
        self.algorithm = "rescale" if rescale else None

        self.features = None  # names of features
        self.dw = None  # data whitener

    @staticmethod
    def log_scott(n_samples, n_features):
        """Scott's parameter."""
        return -1 / (n_features + 4) * np.log10(n_samples)

    @staticmethod
    def _log_gauss(x, X, sigma):
        

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
            raise ValueError(f"Bandwith should be a number, but is {type(bandwidth)}.")
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
        """Compute the log-likelihood of each sample under the model.

        Args:
            X (array): data of shape `(n_samples, n_features)`.
                Last dimension should match dimension of training data `(n_features)`.

        Returns:
            density (array): of shape `(n_samples,)`. Log-likelihood of each sample in `X`.
                These are normalized to be probability densities,
                so values will be low for high-dimensional data.
        """
        # TODO: write score_samples for unconditional and  conditional distribution.
        X = self.dw.whiten(X)
        pass

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
        data = self.dw.whitened_data
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
