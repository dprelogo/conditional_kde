"""Module containing Interpolated Conditional KDE."""

import numpy as np
from .gaussian import ConditionalGaussianKernelDensity
from .util import Interpolator


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
            def list_depth(l, ns=[]):
                ns.append(len(l))
                if isinstance(l[0], list):
                    ns, nf = list_depth(l[0], ns)
                elif isinstance(l[0], np.ndarray):
                    nf = l[0].shape[-1]
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
            [inherent_conditionals[k] for k in self.inherent_features], dtype=np.float32
        )

        if self.interpolator.method == "linear":
            edges, weights = self.interpolator(
                inherently_conditional_values, return_aux=True
            )
            weights = [float(weight.squeeze()) for weight in weights]
            kdes = [self.kdes[edge][0] for edge in edges]
            probs = np.zeros(len(X), dtype=np.float128)
            for weight, kde in zip(weights, kdes):
                probs += np.exp(kde.score_samples(X, conditional_features)) * weight
            return np.log(probs).astype(np.float32)
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
        """Generate random samples from the conditional model.

        Args:
            inherent_conditionals (dict): values of inherent (grid) features.
                This values are used to interpolate on the grid.
                All inherently conditional dimensions must be defined.
            conditionals (dict): other desired variables (features) to condition upon.
                Dictionary keys should be only feature names from `features`.
                For example, if `self.features == ["a", "b", "c"]` and one would like to condition
                on "a" and "c", then `conditionas = {"a": cond_val_a, "c": cond_val_c}`.
                Defaults to `None`, i.e. no additionally conditioned variables.
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
            [inherent_conditionals[k] for k in self.inherent_features], dtype=np.float32
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
            all_samples = np.concatenate(all_samples, axis=0)
            all_weights = [np.ones(n_samples) * w for w in weights]
            all_weights = np.concatenate(all_weights)
            all_weights /= all_weights.sum()

            idx = rs.choice(len(all_samples), n_samples, p=all_weights)
            samples = all_samples[idx]
            if keep_dims:
                return np.stack(
                    [
                        np.broadcast_to(inherently_conditional_values, (n_samples, N)),
                        samples,
                    ],
                    axis=-1,
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
            if keep_dims:
                np.stack(
                    [
                        np.broadcast_to(inherently_conditional_values, (n_samples, N)),
                        samples,
                    ],
                    axis=-1,
                )
            else:
                return samples
