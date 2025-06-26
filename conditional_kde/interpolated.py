"""Module containing Interpolated Conditional KDE."""

import numpy as np
from scipy.special import logsumexp

from .gaussian import ConditionalGaussian, ConditionalGaussianKernelDensity
from .util import Interpolator


class InterpolatedConditionalGaussian:
    """Interpolated Conditional Gaussian estimator.

    With respect to the `ConditionalGaussian`, which fits full distribution
    and slices through it to obtain the conditional distribution, here we allow
    for some dimensions of the data to be inherently conditional.
    For such dimensions, data should be available for every point on a grid.

    To compute the final conditional density, one then interpolates
    for the inherently conditional dimensions, and slices through others as before.

    Args:
        bandwidth (float): allows for the additional smoothing/shrinking of the covariance.
            In most cases, it should be left as 1.
    """

    def __init__(
        self,
        bandwidth=1.0,
    ):
        if not isinstance(bandwidth, (int, float)):
            raise ValueError("Bandwidth should be a number.")
        self.bandwidth = bandwidth

        self.inherent_features = None  # all inherently conditional features
        self.features = None  # all other features
        self.interpolator = None  # interpolator
        self.gaussians = None  # fitted array of ConditionalGaussian

    def fit(
        self,
        data,
        inherent_features=None,
        features=None,
        interpolation_points=None,
        interpolation_method="linear",
    ):
        """Fitting the Interpolated Conditional Gaussian.

        Let's define by Y = (y1, y2, ..., yN) inherently conditional random variables of the dataset,
        and by X = (x1, x2, ..., xM) other variables, for which one has a sample of points.
        This function then fits P(X | Y) for every point on a gridded Y space.
        To make this possible, one needs to pass a set of X samples for every point on a grid.
        Later, one can use interpolation in Y and slicing in X
        to compute P(x1, x2 | x3, ..., xM, y1, ..., yN), or similar.
        Note that all Y values need to be conditioned.

        Args:
            data (list of arrays, array): data to fit.
                Of shape `(n_interp_1, n_interp_2, ..., n_samples, n_features)`.
                For every point on a grid `(n_interp_1, n_interp_2, ..., n_interp_N)`
                one needs to pass `(n_samples, n_features)` dataset, for which
                a separate `n_features`-dim Gaussian KDE is fitted.
                All points on a grid have to have the same number of features (`n_features`).
                In the case `n_samples` is not the same for every point,
                one needs to pass a nested list of arrays.
            inherent_features (list): optional, list defining name of every
                inherently conditional feature. It is used for referencing conditional dimensions.
                Defaults to `[-1, -2, ..., -N]`, where `N` is the number of inherently conditional features.
            features (list): optional, list defining name for every other feature.
                It's used for referencing conditional dimensions.
                Defaults to `[0, 1, ..., n_features - 1]`.
            interpolation_points (dict): optional, a dictionary of `feature: list_of_values` pairs.
                This defines the grid points for every inherently conditional feature.
                Every list of values should be a strictly ascending.
                By default it amounts to:
                `{-1: np.linspace(0, 1, n_interp_1), ..., -N: np.linspace(0, 1, n_interp_N)}`.
            interpolation_method (str): either "linear" or "nearest",
                making linear interpolation between distributions or picking the closest one, respectively.

        Returns:
            An instance of itself.
        """
        if isinstance(data, np.ndarray):
            if len(data.shape) < 3:
                raise ValueError(
                    "`data` should have at least 3 axes: one for 1D grid, one for samples, "
                    f"one for the rest of features, but its shape is {data.shape}"
                )
            N = len(data.shape) - 2
            number_of_samples = data.shape[:N]
            n_features = data.shape[-1]
            data = data.reshape((np.prod(number_of_samples),) + data.shape[-2:])
        elif isinstance(data, list):
            # only calculating N, leaving main checks for later
            def list_depth(lst, ns=[]):
                ns.append(len(lst))
                if isinstance(lst[0], list):
                    ns, nf = list_depth(lst[0], ns)
                elif isinstance(lst[0], np.ndarray):
                    nf = lst[0].shape[-1]
                else:
                    raise ValueError(
                        "`data` of type `list` should not contain anything else "
                        "besides other sublists or `np.ndarray`."
                    )
                return ns, nf

            number_of_samples, n_features = list_depth(data)
            N = len(number_of_samples)

            data = np.array(data, dtype=object)

            if len(data.shape) != N and len(data.shape) != N + 2:
                raise ValueError(
                    f"Total shape of the data should be equal to `N` ({N})"
                    "if `n_samples` is different for different points on the grid, "
                    "or `N + 2` if all points on the grid have the same number of samples."
                )
            if data.shape[:N] != tuple(number_of_samples):
                raise ValueError(
                    f"Something is wrong with the shape of the data ({data.shape}). "
                    f"Are you sure you defined all points on a grid?"
                )

            if len(data.shape) == N:
                data = data.flatten()
            else:
                data = data.reshape((np.prod(number_of_samples),) + data.shape[-2:])
        else:
            raise TypeError(
                f"`data` should be `np.ndarray` or `list`, but is {type(data).__name__}"
            )

        if inherent_features is None:
            self.inherent_features = list(range(-1, -N - 1, -1))
        else:
            if len(inherent_features) != N:
                raise ValueError(
                    "Number of `inherent_features` should be equal to "
                    f"{N}, but is {len(features)}."
                )
            if len(inherent_features) != len(set(inherent_features)):
                raise ValueError("All `inherent_features` should be unique.")
            self.inherent_features = inherent_features
        if features is None:
            self.features = list(range(n_features))
        else:
            if len(features) != n_features:
                raise ValueError(
                    "Number of `features` should be equal to "
                    f"{n_features}, but is {len(features)}"
                )
            if len(features) != len(set(features)):
                raise ValueError("All `features` should be unique.")
            self.features = features

        if interpolation_points is None:
            interpolation_points = {
                i: np.linspace(0, 1, n) for i, n in enumerate(number_of_samples)
            }
        else:
            if len(interpolation_points) != N:
                raise ValueError(
                    f"Number of interpolation points ({len(interpolation_points)}) "
                    f"should be equal to the number of inherently conditional dimensions ({N}), "
                    f"but is ({len(interpolation_points)})."
                )
            if not all(
                k in self.inherent_features for k in interpolation_points.keys()
            ):
                raise ValueError(
                    f"All keys of `interpolation_points` ({interpolation_points.keys()}) "
                    f"should be in `inherent_features` ({self.inherent_features})."
                )
        points = [v for k, v in interpolation_points.items()]

        self.interpolator = Interpolator(points, method=interpolation_method)

        self.gaussians = []
        for d in data:
            gaussian = ConditionalGaussian(
                bandwidth=self.bandwidth,
            )
            gaussian.fit(d.astype(float), features=self.features)
            self.gaussians.append(gaussian)
        self.gaussians = np.array(self.gaussians, dtype=object)
        self.gaussians = self.gaussians.reshape(tuple(number_of_samples))

        return self

    def score_samples(self, X, inherent_conditionals, conditional_features=None):
        """Compute the conditional log-probability of each sample under the model.

        For the simplicity of calculation, here the grid point is fixed by defining
        a point in inherently conditional dimensions.
        `X` is then an array of shape `(n, n_features)`, including all other dimensions of the data.

        Args:
            X (array): data of shape `(n, n_features)`.
                Last dimension should match dimension of training data `(n_features)`.
            inherent_conditionals (dict): values of inherent (grid) features.
                This values are used to interpolate on the grid.
                All inherently conditional dimensions must be defined.
            conditional_features (list): subset of `self.features`, which dimensions of data
                to additionally condition upon. Defaults to `None`, meaning no additionally conditioned dimensions.

        Returns:
            Conditional log probability for each sample in `X`, conditioned on
            inherently conditional dimensions by `inherent_conditionals`
            and other dimensions by `conditional_features`.
        """
        # N = len(self.inherent_features)  # Not used in this method

        if not isinstance(inherent_conditionals, dict):
            raise TypeError(
                f"`inherent_conditionals` should be dictionary, but is {type(inherent_conditionals).__name__}"
            )
        if sorted(inherent_conditionals.keys()) != sorted(self.inherent_features):
            raise ValueError(
                "`inherent_conditionals` keys should be equal to `inherent_features`."
            )

        inherently_conditional_values = np.array(
            [inherent_conditionals[k] for k in self.inherent_features], dtype=np.float64
        )

        if self.interpolator.method == "linear":
            edges, weights = self.interpolator(
                inherently_conditional_values, return_aux=True
            )
            weights = np.array([float(weight.squeeze()) for weight in weights]).reshape(
                -1, 1
            )
            gaussians = [self.gaussians[edge][0] for edge in edges]
            # Use highest precision float available on the platform
            float_dtype = np.longdouble if hasattr(np, "longdouble") else np.float64
            log_probs = np.zeros((len(gaussians), len(X)), dtype=float_dtype)
            for i, gaussian in enumerate(gaussians):
                log_probs[i, :] = gaussian.score_samples(X, conditional_features)
            return logsumexp(log_probs, axis=0, b=weights)
        else:
            edge = self.interpolator(inherently_conditional_values, return_aux=True)
            gaussian = self.gaussian[edge][0]
            return gaussian.score_samples(X, conditional_features)

    def sample(
        self,
        inherent_conditionals,
        conditionals=None,
        n_samples=1,
        random_state=None,
        keep_dims=False,
    ):
        """Generate random samples from the conditional model. For `inherent_condtitionals`,
        there's only one mode of sampling, where only scalar values are accepted.
        For `conditionals` there are two different modes:
        (1) specify conditionals as scalar values and sample `n_samples` out of distribution.
        (2) specify conditionals as an array, where the number of samples will be the length of an array.

        Args:
            inherent_conditionals (dict): values of inherent (grid) features.
                This values are used to interpolate on the grid.
                All inherently conditional dimensions must be defined.
            conditionals (dict): desired variables (features) to condition upon.
                Dictionary keys should be only feature names from `features`.
                For example, if `self.features == ["a", "b", "c"]` and one would like to condition
                on "a" and "c", then `conditionals = {"a": cond_val_a, "c": cond_val_c}`.
                Conditioned values can be either `float` or `array`, where in the case of the
                latter, all conditioned arrays have to be of the same size.
                Defaults to `None`, i.e. normal KDE.
            n_samples (int): number of samples to generate. Defaults to 1.
            random_state (np.random.RandomState, int): seed or `RandomState` instance, optional.
                Determines random number generation used to generate
                random samples. See `Glossary <random_state>`.
            keep_dims (bool): whether to return non-conditioned dimensions only
                or keep given conditional values. Defaults to `False`.

        Returns:
            Array of samples of shape `(n_samples, N + n_features)` if `conditional_variables is None`,
            or `(n_samples, n_features - len(conditionals))` otherwise.
        """
        N = len(self.inherent_features)

        if not isinstance(inherent_conditionals, dict):
            raise TypeError(
                f"`inherent_conditionals` should be dictionary, but is {type(inherent_conditionals).__name__}"
            )
        if sorted(inherent_conditionals.keys()) != sorted(self.inherent_features):
            raise ValueError(
                "`inherent_conditionals` keys should be equal to `inherent_features`."
            )

        inherently_conditional_values = np.array(
            [inherent_conditionals[k] for k in self.inherent_features], dtype=np.float64
        )

        if isinstance(random_state, np.random.RandomState):
            rs = random_state
        elif random_state is None or isinstance(random_state, int):
            rs = np.random.RandomState(seed=random_state)
        else:
            raise TypeError("`random_state` should be `int` or `RandomState`.")

        if self.interpolator.method == "linear":
            edges, weights = self.interpolator(
                inherently_conditional_values, return_aux=True
            )
            weights = [float(weight.squeeze()) for weight in weights]
            gaussians = [self.gaussians[edge][0] for edge in edges]

            all_samples = [
                gaussian.sample(
                    conditionals=conditionals,
                    n_samples=n_samples,
                    random_state=rs,
                    keep_dims=keep_dims,
                )
                for gaussian in gaussians
            ]

            # I shouldn't use old n_samples further, as it can differ for the vectorized conditionals
            n_samples = len(all_samples[0])

            all_samples = np.concatenate(all_samples, axis=0)
            all_weights = [np.ones(n_samples) * w for w in weights]
            all_weights = np.concatenate(all_weights)
            all_weights /= all_weights.sum()

            idx = rs.choice(len(all_samples), n_samples, p=all_weights)
            samples = all_samples[idx]
            if keep_dims:
                return np.hstack(
                    [
                        samples,
                        np.broadcast_to(inherently_conditional_values, (n_samples, N)),
                    ]
                )
            else:
                return samples
        else:
            edge = self.interpolator(inherently_conditional_values, return_aux=True)
            gaussian = self.gaussians[edge][0]
            samples = gaussian.sample(
                conditionals=conditionals,
                n_samples=n_samples,
                random_state=rs,
                keep_dims=keep_dims,
            )

            # I shouldn't use old n_samples further, as it can differ for the vectorized conditionals
            n_samples = len(samples)

            if keep_dims:
                return np.hstack(
                    [
                        samples,
                        np.broadcast_to(inherently_conditional_values, (n_samples, N)),
                    ]
                )
            else:
                return samples


class InterpolatedConditionalKernelDensity:
    """Interpolated Conditional Kernel Density estimator.

    With respect to the `ConditionalKernelDensity`, which fits full distribution
    and slices through it to obtain the conditional distribution, here we allow
    for some dimensions of the data to be inherently conditional.
    For such dimensions, data should be available for every point on a grid.

    To compute the final conditional density, one then interpolates
    for the inherently conditional dimensions, and slices through others as before.

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
            if bandwidth not in ["scott", "optimized"]:
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

        self.inherent_features = None  # all inherently conditional features
        self.features = None  # all other features
        self.interpolator = None  # interpolator
        self.kdes = None  # fitted array of Kernel Density Estimators

    def fit(
        self,
        data,
        inherent_features=None,
        features=None,
        interpolation_points=None,
        interpolation_method="linear",
    ):
        """Fitting the Interpolated Conditional Kernel Density.

        Let's define by Y = (y1, y2, ..., yN) inherently conditional random variables of the dataset,
        and by X = (x1, x2, ..., xM) other variables, for which one has a sample of points.
        This function then fits P(X | Y) for every point on a gridded Y space.
        To make this possible, one needs to pass a set of X samples for every point on a grid.
        Later, one can use interpolation in Y and slicing in X
        to compute P(x1, x2 | x3, ..., xM, y1, ..., yN), or similar.
        Note that all Y values need to be conditioned.

        Args:
            data (list of arrays, array): data to fit.
                Of shape `(n_interp_1, n_interp_2, ..., n_samples, n_features)`.
                For every point on a grid `(n_interp_1, n_interp_2, ..., n_interp_N)`
                one needs to pass `(n_samples, n_features)` dataset, for which
                a separate `n_features`-dim Gaussian KDE is fitted.
                All points on a grid have to have the same number of features (`n_features`).
                In the case `n_samples` is not the same for every point,
                one needs to pass a nested list of arrays.
            inherent_features (list): optional, list defining name of every
                inherently conditional feature. It is used for referencing conditional dimensions.
                Defaults to `[-1, -2, ..., -N]`, where `N` is the number of inherently conditional features.
            features (list): optional, list defining name for every other feature.
                It's used for referencing conditional dimensions.
                Defaults to `[0, 1, ..., n_features - 1]`.
            interpolation_points (dict): optional, a dictionary of `feature: list_of_values` pairs.
                This defines the grid points for every inherently conditional feature.
                Every list of values should be a strictly ascending.
                By default it amounts to:
                `{-1: np.linspace(0, 1, n_interp_1), ..., -N: np.linspace(0, 1, n_interp_N)}`.
            interpolation_method (str): either "linear" or "nearest",
                making linear interpolation between distributions or picking the closest one, respectively.

        Returns:
            An instance of itself.
        """
        if isinstance(data, np.ndarray):
            if len(data.shape) < 3:
                raise ValueError(
                    "`data` should have at least 3 axes: one for 1D grid, one for samples, "
                    f"one for the rest of features, but its shape is {data.shape}"
                )
            N = len(data.shape) - 2
            number_of_samples = data.shape[:N]
            n_features = data.shape[-1]
            data = data.reshape((np.prod(number_of_samples),) + data.shape[-2:])
        elif isinstance(data, list):
            # only calculating N, leaving main checks for later
            def list_depth(lst, ns=[]):
                ns.append(len(lst))
                if isinstance(lst[0], list):
                    ns, nf = list_depth(lst[0], ns)
                elif isinstance(lst[0], np.ndarray):
                    nf = lst[0].shape[-1]
                else:
                    raise ValueError(
                        "`data` of type `list` should not contain anything else "
                        "besides other sublists or `np.ndarray`."
                    )
                return ns, nf

            number_of_samples, n_features = list_depth(data)
            N = len(number_of_samples)

            data = np.array(data, dtype=object)

            if len(data.shape) != N and len(data.shape) != N + 2:
                raise ValueError(
                    f"Total shape of the data should be equal to `N` ({N})"
                    "if `n_samples` is different for different points on the grid, "
                    "or `N + 2` if all points on the grid have the same number of samples."
                )
            if data.shape[:N] != tuple(number_of_samples):
                raise ValueError(
                    f"Something is wrong with the shape of the data ({data.shape}). "
                    f"Are you sure you defined all points on a grid?"
                )

            if len(data.shape) == N:
                data = data.flatten()
            else:
                data = data.reshape((np.prod(number_of_samples),) + data.shape[-2:])
        else:
            raise TypeError(
                f"`data` should be `np.ndarray` or `list`, but is {type(data).__name__}"
            )

        if inherent_features is None:
            self.inherent_features = list(range(-1, -N - 1, -1))
        else:
            if len(inherent_features) != N:
                raise ValueError(
                    "Number of `inherent_features` should be equal to "
                    f"{N}, but is {len(features)}."
                )
            if len(inherent_features) != len(set(inherent_features)):
                raise ValueError("All `inherent_features` should be unique.")
            self.inherent_features = inherent_features
        if features is None:
            self.features = list(range(n_features))
        else:
            if len(features) != n_features:
                raise ValueError(
                    "Number of `features` should be equal to "
                    f"{n_features}, but is {len(features)}"
                )
            if len(features) != len(set(features)):
                raise ValueError("All `features` should be unique.")
            self.features = features

        if interpolation_points is None:
            interpolation_points = {
                i: np.linspace(0, 1, n) for i, n in enumerate(number_of_samples)
            }
        else:
            if len(interpolation_points) != N:
                raise ValueError(
                    f"Number of interpolation points ({len(interpolation_points)}) "
                    f"should be equal to the number of inherently conditional dimensions ({N}), "
                    f"but is ({len(interpolation_points)})."
                )
            if not all(
                k in self.inherent_features for k in interpolation_points.keys()
            ):
                raise ValueError(
                    f"All keys of `interpolation_points` ({interpolation_points.keys()}) "
                    f"should be in `inherent_features` ({self.inherent_features})."
                )
        points = [v for k, v in interpolation_points.items()]

        self.interpolator = Interpolator(points, method=interpolation_method)

        self.kdes = []
        for d in data:
            kde = ConditionalGaussianKernelDensity(
                whitening_algorithm=self.algorithm,
                bandwidth=self.bandwidth,
                **self.bandwidth_kwargs,
            )
            kde.fit(d.astype(float), features=self.features)
            self.kdes.append(kde)
        self.kdes = np.array(self.kdes, dtype=object)
        self.kdes = self.kdes.reshape(tuple(number_of_samples))

        return self

    def score_samples(self, X, inherent_conditionals, conditional_features=None):
        """Compute the conditional log-probability of each sample under the model.

        For the simplicity of calculation, here the grid point is fixed by defining
        a point in inherently conditional dimensions.
        `X` is then an array of shape `(n, n_features)`, including all other dimensions of the data.

        Args:
            X (array): data of shape `(n, n_features)`.
                Last dimension should match dimension of training data `(n_features)`.
            inherent_conditionals (dict): values of inherent (grid) features.
                This values are used to interpolate on the grid.
                All inherently conditional dimensions must be defined.
            conditional_features (list): subset of `self.features`, which dimensions of data
                to additionally condition upon. Defaults to `None`, meaning no additionally conditioned dimensions.

        Returns:
            Conditional log probability for each sample in `X`, conditioned on
            inherently conditional dimensions by `inherent_conditionals`
            and other dimensions by `conditional_features`.
        """
        # N = len(self.inherent_features)  # Not used in this method

        if not isinstance(inherent_conditionals, dict):
            raise TypeError(
                f"`inherent_conditionals` should be dictionary, but is {type(inherent_conditionals).__name__}"
            )
        if sorted(inherent_conditionals.keys()) != sorted(self.inherent_features):
            raise ValueError(
                "`inherent_conditionals` keys should be equal to `inherent_features`."
            )

        inherently_conditional_values = np.array(
            [inherent_conditionals[k] for k in self.inherent_features], dtype=np.float64
        )

        if self.interpolator.method == "linear":
            edges, weights = self.interpolator(
                inherently_conditional_values, return_aux=True
            )
            weights = np.array([float(weight.squeeze()) for weight in weights]).reshape(
                -1, 1
            )
            kdes = [self.kdes[edge][0] for edge in edges]
            # Use highest precision float available on the platform
            float_dtype = np.longdouble if hasattr(np, "longdouble") else np.float64
            log_probs = np.zeros((len(kdes), len(X)), dtype=float_dtype)
            for i, kde in enumerate(kdes):
                log_probs[i, :] = kde.score_samples(X, conditional_features)
            return logsumexp(log_probs, axis=0, b=weights)
        else:
            edge = self.interpolator(inherently_conditional_values, return_aux=True)
            kde = self.kdes[edge][0]
            return kde.score_samples(X, conditional_features)

    def sample(
        self,
        inherent_conditionals,
        conditionals=None,
        n_samples=1,
        random_state=None,
        keep_dims=False,
    ):
        """Generate random samples from the conditional model. For `inherent_condtitionals`,
        there's only one mode of sampling, where only scalar values are accepted.
        For `conditionals` there are two different modes:
        (1) specify conditionals as scalar values and sample `n_samples` out of distribution.
        (2) specify conditionals as an array, where the number of samples will be the length of an array.

        Args:
            inherent_conditionals (dict): values of inherent (grid) features.
                This values are used to interpolate on the grid.
                All inherently conditional dimensions must be defined.
            conditionals (dict): desired variables (features) to condition upon.
                Dictionary keys should be only feature names from `features`.
                For example, if `self.features == ["a", "b", "c"]` and one would like to condition
                on "a" and "c", then `conditionals = {"a": cond_val_a, "c": cond_val_c}`.
                Conditioned values can be either `float` or `array`, where in the case of the
                latter, all conditioned arrays have to be of the same size.
                Defaults to `None`, i.e. normal KDE.
            n_samples (int): number of samples to generate. Defaults to 1.
            random_state (np.random.RandomState, int): seed or `RandomState` instance, optional.
                Determines random number generation used to generate
                random samples. See `Glossary <random_state>`.
            keep_dims (bool): whether to return non-conditioned dimensions only
                or keep given conditional values. Defaults to `False`.

        Returns:
            Array of samples of shape `(n_samples, N + n_features)` if `conditional_variables is None`,
            or `(n_samples, n_features - len(conditionals))` otherwise.
        """
        N = len(self.inherent_features)

        if not isinstance(inherent_conditionals, dict):
            raise TypeError(
                f"`inherent_conditionals` should be dictionary, but is {type(inherent_conditionals).__name__}"
            )
        if sorted(inherent_conditionals.keys()) != sorted(self.inherent_features):
            raise ValueError(
                "`inherent_conditionals` keys should be equal to `inherent_features`."
            )

        inherently_conditional_values = np.array(
            [inherent_conditionals[k] for k in self.inherent_features], dtype=np.float64
        )

        if isinstance(random_state, np.random.RandomState):
            rs = random_state
        elif random_state is None or isinstance(random_state, int):
            rs = np.random.RandomState(seed=random_state)
        else:
            raise TypeError("`random_state` should be `int` or `RandomState`.")

        if self.interpolator.method == "linear":
            edges, weights = self.interpolator(
                inherently_conditional_values, return_aux=True
            )
            weights = [float(weight.squeeze()) for weight in weights]
            kdes = [self.kdes[edge][0] for edge in edges]

            all_samples = [
                kde.sample(
                    conditionals=conditionals,
                    n_samples=n_samples,
                    random_state=rs,
                    keep_dims=keep_dims,
                )
                for kde in kdes
            ]

            # I shouldn't use old n_samples further, as it can differ for the vectorized conditionals
            n_samples = len(all_samples[0])

            all_samples = np.concatenate(all_samples, axis=0)
            all_weights = [np.ones(n_samples) * w for w in weights]
            all_weights = np.concatenate(all_weights)
            all_weights /= all_weights.sum()

            idx = rs.choice(len(all_samples), n_samples, p=all_weights)
            samples = all_samples[idx]
            if keep_dims:
                return np.hstack(
                    [
                        samples,
                        np.broadcast_to(inherently_conditional_values, (n_samples, N)),
                    ]
                )
            else:
                return samples
        else:
            edge = self.interpolator(inherently_conditional_values, return_aux=True)
            kde = self.kdes[edge][0]
            samples = kde.sample(
                conditionals=conditionals,
                n_samples=n_samples,
                random_state=rs,
                keep_dims=keep_dims,
            )

            # I shouldn't use old n_samples further, as it can differ for the vectorized conditionals
            n_samples = len(samples)

            if keep_dims:
                return np.hstack(
                    [
                        samples,
                        np.broadcast_to(inherently_conditional_values, (n_samples, N)),
                    ]
                )
            else:
                return samples
