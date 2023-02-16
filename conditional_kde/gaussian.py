"""Module containing Gaussian versions of the Conditional KDE."""

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import train_test_split, GridSearchCV
from .util import DataWhitener


class ConditionalGaussianKernelDensity:
    """Conditional Kernel Density estimator.

    Args:
        whitening_algorithm (str): data whitening algorithm, either `None`, "rescale" or "ZCA".
            See `util.DataWhitener` for more details. "rescale" by default.
        bandwidth (str, float): the width of the Gaussian centered around every point.

            It can be either:\n
            (1) "scott", using Scott's parameter,\n
            (2) "optimized", which minimizes cross entropy to find the optimal bandwidth, or\n
            (3) `float`, specifying the actual value.\n
            By default, it uses Scott's parameter.
        **kwargs: additional kwargs used in the case of "optimized" bandwidth.

            steps (int): how many steps to use in optimization, 10 by default.\n
            cv_fold (int): cross validation fold, 5 by default.\n
            n_jobs (int): number of jobs to run cross validation in parallel,
            -1 by default, i.e. using all available processors.\n
            verbose (int): verbosity of the cross validation run,
            for more details see `sklearn.model_selection.GridSearchCV`.
    """

    def __init__(
        self,
        whitening_algorithm="rescale",
        bandwidth="scott",
        **kwargs,
    ):
        if whitening_algorithm not in [None, "rescale", "ZCA"]:
            raise ValueError(
                f"Whitening lgorithm should be None, rescale or ZCA, but is {whitening_algorithm}."
            )
        self.algorithm = whitening_algorithm

        if not isinstance(bandwidth, (int, float)):
            if not bandwidth in ["scott", "optimized"]:
                raise ValueError(
                    f"""Bandwidth should be a number, "scott" or "optimized", 
                    but has value {bandwidth} and type {type(bandwidth).__name__}."""
                )
        self.bandwidth = bandwidth

        if self.bandwidth == "optimized":
            self.bandwidth_kwargs = {
                "steps": kwargs.get("steps", 10),
                "cv_fold": kwargs.get("cv_fold", 5),
                "n_jobs": kwargs.get("n_jobs", -1),
                "verbose": kwargs.get("verbose", 0),
            }
        else:
            self.bandwidth_kwargs = {}

        self.features = None  # names of features
        self.dw = None  # data whitener

    @staticmethod
    def log_scott(n_samples, n_features):
        """Scott's parameter."""
        return -1 / (n_features + 4) * np.log10(n_samples)

    @staticmethod
    def _log_prob(X, data, cov, add_norm=True):
        """Log probability of a gaussian KDE distribution.

        Args:
            X (array): array of samples for which probability is calculated.
                Of shape `(n, n_features)`.
            data (array): KDE data, of shape `(n_samples, n_features)`.
            cov (float, array): covariance matrix of a gaussian distribution.
                If float, it is a variance shared for all features.
                If 1D array, it is a variance for every feature separately.
                if 2D array, it is a full covariance matrix.
            add_norm (bool): either to add normalization factor to the calculation or not.

        Returns:
            Log probabilities.
        """
        n_samples, n_features = data.shape
        X = np.atleast_2d(X)
        if X.shape[-1] != n_features:
            raise ValueError("`n_features` of both arrays should be the same.")
        if not isinstance(cov, (int, float, np.ndarray)):
            raise TypeError(
                f"`cov` should be a number or `numpy.ndarray`, "
                f"but is {type(cov).__name__}"
            )
        if isinstance(cov, (int, float)):
            Σ_inv = np.identity(n_features, dtype=np.float32) / cov
            Σ_det = cov**n_features
        elif isinstance(cov, np.ndarray):
            if len(cov.shape) == 1:
                if len(cov) != n_features:
                    raise ValueError("`cov` should be of length `n_features`.")
                Σ_inv = np.diag(1 / cov)
                Σ_det = np.prod(cov)
            elif len(cov.shape) == 2:
                if cov.shape != (n_features, n_features):
                    raise ValueError(
                        "`cov` should be of shape `(n_features, n_features)`."
                    )
                Σ_inv = np.linalg.inv(cov)
                Σ_det = np.linalg.det(cov)
            else:
                raise ValueError(
                    "Dimensionality of a covariance matrix cannot be larger than 2."
                )

        def calculate_log_prob(x):
            delta = x - data
            res = -0.5 * np.einsum("ij,jk,ik->i", delta, Σ_inv, delta)
            return logsumexp(res)

        log_prob = np.apply_along_axis(calculate_log_prob, 1, X)

        if add_norm:
            norm = (
                np.log(n_samples)
                + 0.5 * n_features * np.log(2 * np.pi)
                + 0.5 * np.log(Σ_det)
            )
            return log_prob - norm
        else:
            return log_prob

    @staticmethod
    def _covariance_decomposition(cov, cond_mask, cond_only=False):
        """Decomposing covariance matrix into the unconditional, conditional and cross terms.

        Args:
            cov (array): covariance matrix.
            cond_mask (array): boolean array defining conditional dimensions.
            cond_only (bool): to return only conditional matrix or all decompositions.

        Returns:
            If `cond_only is True`, only conditional part of the covariance,
            otherwise: conditional, unconditional and cross parts, respectively.
        """
        if len(cov) != len(cond_mask):
            raise ValueError(
                "Dimensionality of `cov` and `cond_mask` should be the same."
            )
        if len(cov.shape) != 2 or cov.shape[0] != cov.shape[-1]:
            raise ValueError("`cov` should be 2D square matrix.")
        mask_cond = np.outer(cond_mask, cond_mask)
        shape_cond = (sum(cond_mask), sum(cond_mask))
        if cond_only:
            return cov[mask_cond].reshape(shape_cond)
        else:
            mask_uncond = np.outer(~cond_mask, ~cond_mask)
            shape_uncond = (sum(~cond_mask), sum(~cond_mask))

            mask_cross = np.outer(cond_mask, ~cond_mask)
            shape_cross = (sum(cond_mask), sum(~cond_mask))
            return (
                cov[mask_cond].reshape(shape_cond),
                cov[mask_uncond].reshape(shape_uncond),
                cov[mask_cross].reshape(shape_cross),
            )

    @staticmethod
    def _conditional_weights(conditional_values, conditional_data, cov):
        """Weights for the sampling from the conditional distribution.

        They amount to the conditioned part of the gaussian for every data point.

        Args:
            conditional_values (array): of length `n_conditionals`.
            cond_data (array): of shape `(n_samples, n_conditionals)`.
                Here non-conditional dimensions are already removed.
            cov (float, array): covariance matrix.
                If float, it is a variance shared for all features.
                If 1D array, it is a variance for every feature separately.
                if 2D array, it is a full covariance matrix.

        Returns:
            Normalized weights.
        """
        if conditional_values.ndim == 1:
            delta = conditional_values - conditional_data
            if isinstance(cov, float):
                log_weights = -0.5 / cov * np.einsum("ij,ij->i", delta, delta)
            elif isinstance(cov, np.ndarray) and len(cov.shape) == 1:
                log_weights = -0.5 * np.einsum("ij,j,ij->i", delta, 1 / cov, delta)
            elif isinstance(cov, np.ndarray) and len(cov.shape) == 2:
                cov_inv = np.linalg.inv(cov)
                log_weights = -0.5 * np.einsum("ij,jk,ik->i", delta, cov_inv, delta)
            else:
                raise ValueError("`cov` cannot be more than 2D.")

            # calculate exp(weights) in a more stable way
            log_weights_sum = logsumexp(log_weights)
        else:
            if isinstance(cov, float):
                log_weights = (
                    -0.5
                    / cov
                    * (
                        np.einsum("ij,ij->i", conditional_values, conditional_values)[
                            :, np.newaxis
                        ]
                        + np.einsum("ij,ij->i", conditional_data, conditional_data)[
                            np.newaxis, :
                        ]
                        - 2
                        * np.einsum("ij,kj->ik", conditional_values, conditional_data)
                    )
                )
            elif isinstance(cov, np.ndarray) and len(cov.shape) == 1:
                log_weights = (
                    -0.5
                    * np.einsum(
                        "ij,j,ij->i", conditional_values, 1 / cov, conditional_values
                    )[:, np.newaxis]
                    - 0.5
                    * np.einsum(
                        "ij,j,ij->i", conditional_data, 1 / cov, conditional_data
                    )[np.newaxis, :]
                    + np.einsum(
                        "ij,j,kj->ik", conditional_values, 1 / cov, conditional_data
                    )
                )
            elif isinstance(cov, np.ndarray) and len(cov.shape) == 2:
                cov_inv = np.linalg.inv(cov)
                log_weights = (
                    -0.5
                    * np.einsum(
                        "ij,jk,ik->i", conditional_values, cov_inv, conditional_values
                    )[:, np.newaxis]
                    - 0.5
                    * np.einsum(
                        "ij,jk,ik->i", conditional_data, cov_inv, conditional_data
                    )[np.newaxis, :]
                    + np.einsum(
                        "ij,jk,lk->il", conditional_values, cov_inv, conditional_data
                    )
                )
            else:
                raise ValueError("`cov` cannot be more than 2D.")

            log_weights_sum = logsumexp(log_weights, axis=1, keepdims=True)

        log_weights -= log_weights_sum
        mask = log_weights < -22
        weights = np.exp(log_weights)
        weights[mask] = 0.0
        if conditional_values.ndim == 1:
            return weights / np.sum(weights)
        else:
            return weights / np.sum(weights, axis=1, keepdims=True)

    def fit(
        self,
        X,
        features=None,
    ):
        """Fitting the Conditional Kernel Density.

        Args:
            X (array): data of shape `(n_samples, n_features)`.
            features (list): optional, list defining names for every feature.
                It's used for referencing conditional dimensions.
                Defaults to `[0, 1, ..., n_features - 1]`.

        Returns:
            An instance of itself.
        """
        n_samples, n_features = X.shape

        if features is None:
            self.features = list(range(n_features))
        else:
            if not isinstance(features, list):
                raise TypeError("`features` should be a `list`.")
            elif n_features != len(features):
                raise ValueError(
                    f"`n_features` ({n_features}) should be equal to "
                    f"the length of `features` ({len(features)})."
                )
            elif len(features) != len(set(features)):
                raise ValueError("All `features` should be unique.")
            self.features = features

        self.dw = DataWhitener(self.algorithm)
        self.dw.fit(X, save_data=True)

        if self.bandwidth == "scott":
            self.bandwidth = 10 ** self.log_scott(n_samples, n_features)
        elif self.bandwidth == "optimized":
            log_scott = self.log_scott(n_samples, n_features)
            model = GridSearchCV(
                KernelDensity(),
                {
                    "bandwidth": np.logspace(
                        log_scott - 1, log_scott + 1, num=self.bandwidth_kwargs["steps"]
                    )
                },
                cv=self.bandwidth_kwargs["cv_fold"],
                n_jobs=self.bandwidth_kwargs["n_jobs"],
                verbose=self.bandwidth_kwargs["verbose"],
            )
            model.fit(self.dw.whitened_data)
            self.bandwidth = model.best_params_["bandwidth"]

        return self

    def score_samples(self, X, conditional_features=None):
        """Compute the (un)conditional log-probability of each sample under the model.

        Args:
            X (array): data of shape `(n, n_features)`.
                Last dimension should match dimension of training data `(n_features)`.
            conditional_features (list): subset of `self.features`, which dimensions of data
                to condition upon. Defaults to `None`, meaning unconditional log-probability.

        Returns:
            Conditional log probability for each sample in `X`.
        """
        if conditional_features is None:
            X = np.atleast_2d(X)
            return self._log_prob(X, self.dw.data, self.dw.Σ * self.bandwidth**2)
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

            cond_variables = [
                True if f in conditional_features else False for f in self.features
            ]
            cond_variables = np.array(cond_variables, dtype=bool)

            X = np.atleast_2d(X)
            p_full = self._log_prob(X, self.dw.data, self.dw.Σ * self.bandwidth**2)
            Σ_marginal = self._covariance_decomposition(
                self.dw.Σ * self.bandwidth**2, cond_variables, cond_only=True
            )
            p_marginal = self._log_prob(
                X[:, cond_variables],
                self.dw.data[:, cond_variables],
                Σ_marginal,
            )
            return p_full - p_marginal

    def _check_conditionals(self, conditionals, n_samples):
        if not isinstance(conditionals, dict):
            raise TypeError(
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
        if any(not isinstance(v, (float, int)) for v in conditionals.values()):
            if not all(isinstance(v, np.ndarray) for v in conditionals.values()):
                raise ValueError(
                    "For vectorized conditionals, all should be `np.ndarray`."
                )
            if not all(v.ndim == 1 for v in conditionals.values()):
                raise ValueError("For vectorized conditionals, all should be 1D.")
            lengths = [len(v) for v in conditionals.values()]
            if not all(l == lengths[0] for l in lengths):
                raise ValueError(
                    "For vectorized conditionals, all should have the same length."
                )
            vectorized_conditionals = True
            n_samples = lengths[0]
        else:
            vectorized_conditionals = False
        return vectorized_conditionals, n_samples

    def _sample(
        self,
        conditionals=None,
        n_samples=1,
        random_state=None,
        keep_dims=False,
    ):
        """Generate random samples from the conditional model.

        Here there is an assumption that all dimensions have not been distorted,
        but only rescaled. In other words, it works for `None` and "rescale"
        whitening algorithms, but not for "ZCA".
        """
        data = self.dw.whitened_data
        if isinstance(random_state, np.random.RandomState):
            rs = random_state
        elif random_state is None or isinstance(random_state, int):
            rs = np.random.RandomState(seed=random_state)
        else:
            raise TypeError("`random_state` should be `int` or `RandomState`.")

        if conditionals is None:
            idx = rs.choice(data.shape[0], n_samples)
            sample = rs.normal(np.atleast_2d(data[idx]), self.bandwidth)
            return self.dw.unwhiten(sample)
        else:
            vectorized_conditionals, n_samples = self._check_conditionals(
                conditionals, n_samples
            )

            # scaling conditional variables
            cond_variables = [
                True if f in conditionals.keys() else False for f in self.features
            ]
            cond_variables = np.array(cond_variables, dtype=bool)

            if vectorized_conditionals:
                cond_values = np.zeros(n_samples, len(self.features), dtype=np.float32)
                for i, f in enumerate(self.features):
                    if f in conditionals.keys():
                        cond_values[:, i] = conditionals[f]
            else:
                cond_values = np.zeros(len(self.features), dtype=np.float32)
                for i, f in enumerate(self.features):
                    if f in conditionals.keys():
                        cond_values[i] = conditionals[f]
            cond_values = self.dw.whiten(cond_values)[cond_variables]

            weights = self._conditional_weights(
                cond_values, data[:, cond_variables], self.bandwidth**2
            )
            if vectorized_conditionals:
                idx = np.apply_along_axis(
                    lambda x: rs.choice(data.shape[0], p=x),
                    1,
                    weights,
                )
            else:
                idx = rs.choice(data.shape[0], n_samples, p=weights)

            # pulling the samples
            sample = np.empty((n_samples, len(self.features)))
            sample[:, ~cond_variables] = rs.normal(
                np.atleast_2d(data[idx])[:, ~cond_variables], self.bandwidth
            )
            if vectorized_conditionals:
                sample[:, cond_variables] = cond_values
            else:
                sample[:, cond_variables] = np.broadcast_to(
                    cond_values, (n_samples, len(cond_values))
                )
            sample = self.dw.unwhiten(sample)

            if keep_dims:
                return sample
            else:
                return sample[:, ~cond_variables]

    def _sample_general(
        self,
        conditionals=None,
        n_samples=1,
        random_state=None,
        keep_dims=False,
    ):
        """Generate random samples from the conditional model.

        This function is the most general sampler, without any assumptions.
        It should be used for ZCA.
        """
        if isinstance(random_state, np.random.RandomState):
            rs = random_state
        elif random_state is None or isinstance(random_state, int):
            rs = np.random.RandomState(seed=random_state)
        else:
            raise TypeError("`random_state` should be `int` or `RandomState`.")

        if conditionals is None:
            idx = rs.choice(self.dw.data.shape[0], n_samples)
            return rs.multivariate_normal(
                np.zeros(len(self.dw.Σ)), self.dw.Σ * self.bandwidth**2, n_samples
            ) + np.atleast_2d(self.dw.data[idx])
        else:
            vectorized_conditionals, n_samples = self._check_conditionals(
                conditionals, n_samples
            )

            # helping variables for conditionals
            cond_variables = [
                True if f in conditionals.keys() else False for f in self.features
            ]
            cond_values = [
                conditionals[f] for f in self.features if f in conditionals.keys()
            ]
            cond_variables = np.concatenate(cond_variables, dtype=bool)
            cond_values = np.stack(cond_values, axis=-1, dtype=np.float32)

            # decomposing the covarianve
            Σ_cond, Σ_uncond, Σ_cross = self._covariance_decomposition(
                self.dw.Σ * self.bandwidth**2, cond_variables
            )
            # weights are defined from conditional part
            weights = self._conditional_weights(
                cond_values,
                self.dw.data[:, cond_variables],
                Σ_cond,
            )
            if vectorized_conditionals:
                idx = np.apply_along_axis(
                    lambda x: rs.choice(self.dw.data.shape[0], p=x),
                    1,
                    weights,
                )
            else:
                idx = rs.choice(self.dw.data.shape[0], n_samples, p=weights)
            selected_data = np.atleast_2d(self.dw.data[idx])

            # distribution is defined from corrected unconditional part
            Σ_cond_inv = np.linalg.inv(Σ_cond)
            corr_Σ = Σ_uncond - Σ_cross.T @ Σ_cond_inv @ Σ_cross
            corr_data = (
                selected_data[:, ~cond_variables]
                + (cond_values - selected_data[:, cond_variables])
                @ Σ_cond_inv
                @ Σ_cross
            )
            sample = np.empty((n_samples, len(self.features)))

            sample[:, ~cond_variables] = (
                rs.multivariate_normal(np.zeros(len(corr_Σ)), corr_Σ, n_samples)
                + corr_data
            )

            if vectorized_conditionals:
                sample[:, cond_variables] = cond_values
            else:
                sample[:, cond_variables] = np.broadcast_to(
                    cond_values, (n_samples, len(cond_values))
                )

            if keep_dims:
                return sample
            else:
                return sample[:, ~cond_variables]

    def sample(
        self,
        conditionals=None,
        n_samples=1,
        random_state=None,
        keep_dims=False,
    ):
        """Generate random samples from the conditional model. There are two modes
        of sampling:
        (1) specify conditionals as scalar values and sample `n_samples` out of distribution.
        (2) specify conditionals as an array, where the number of samples will be the length of an array.

        Args:
            conditionals (dict): desired variables (features) to condition upon.
                Dictionary keys should be only feature names from `features`.
                For example, if `self.features == ["a", "b", "c"]` and one would like to condition
                on "a" and "c", then `conditionals = {"a": cond_val_a, "c": cond_val_c}`.
                Conditioned values can be either `float` or `array`, where in the case of the
                latter, all conditioned arrays have to be of the same size.
                Defaults to `None`, i.e. normal KDE.
            n_samples (int): number of samples to generate. Ignored in the case
                conditional arrays have been passed in `conditionals`. Defaults to 1.
            random_state (np.random.RandomState, int): seed or `RandomState` instance, optional.
                Determines random number generation used to generate
                random samples. See `Glossary <random_state>`.
            keep_dims (bool): whether to return non-conditioned dimensions only
                or keep given conditional values. Defaults to `False`.

        Returns:
            Array of samples, of shape `(n_samples, n_features)` if `conditional_variables is None`,
            or `(n_samples, n_features - len(conditionals))` otherwise.
        """
        if self.algorithm == "ZCA":
            return self._sample_general(
                conditionals=conditionals,
                n_samples=n_samples,
                random_state=random_state,
                keep_dims=keep_dims,
            )
        else:
            return self._sample(
                conditionals=conditionals,
                n_samples=n_samples,
                random_state=random_state,
                keep_dims=keep_dims,
            )
