"""Main module."""
import numpy as np
from sklearn.neighbors import KernelDensity
from .utils import DataWhitener


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

        self.rescale = rescale

        super(ConditionalKernelDensity, self).__init__(
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
        return -1 / (n_features + 4) * np.log10(n_samples)

    def fit(self, X, optimal_bandwidth=True, **kwargs):
        """Fitting the Conditional Kernel Density.
        Args:
            X (array): data of shape `(n_samples, n_features)`.
            optimal_bandwidth (bool): if `True`, uses Scott's parameter
                for the bandwith. For the best results, use with
                `rescale = True`.
            **kwargs: see `sklearn.neighbors.KernelDensity`.
        """
        if optimal_bandwidth:
            n_samples, n_features = X.shape
            self.bandwidth = 10 ** self.log_scott(n_samples, n_features)

        if self.rescale:
            self.dw = DataWhitener("rescale")
            self.dw.fit(X)
            X = self.dw.whiten(X)

        super(ConditionalKernelDensity, self).fit(X, **kwargs)

    def sample(
        self,
        n_samples=1,
        random_state=None,
        conditional_variables=None,
        conditional_values=None,
        keep_dims=False,
    ):
        """Generate random samples from the model.
        Currently, this is implemented only for gaussian kernel.
        Args:
            n_samples (int): number of samples to generate. Defaults to 1.
            random_state (int): `RandomState` instance, default=None
                Determines random number generation used to generate
                random samples. Pass `int` for reproducible results
                across multiple function calls. See `Glossary <random_state>`.
            conditional_variables (bool array): desired variables to condition upon.
                `len(conditional_variables)` has to be equal to `n_features`.
                Defaults to `None`.
            conditional_values (array): values on which one wants to fix
                conditional variables. `len(conditional_values)` has to be equal
                to the `sum(conditional_variables)`. Defaults to `None`.
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

        if conditional_variables is None:
            i = rs.choice(data.shape[0], n_samples)
            sample = np.atleast_2d(rs.normal(data[i], self.bandwidth))
            if self.rescale:
                return self.dw.unwhiten(sample)
            else:
                return sample
        else:
            if conditional_variables.dtype != bool:
                raise ValueError(
                    f"Conditional variables` should be `np.bool` array, "
                    f"but is of type {conditional_variables.dtype}."
                )
            if len(conditional_variables) != data.shape[-1]:
                raise ValueError(
                    "`n_dim` of data should be the same as `len(conditional_variables)`, "
                    f"but is {len(conditional_variables)} != {data.shape[-1]}."
                )

            # scaling conditional variables if rescale == True
            # for this to work, it was crucial to use "rescale" in the data whitener
            if self.rescale:
                cond_values = np.zeros(len(conditional_variables))
                cond_values[conditional_variables] = np.array(conditional_values)
                cond_values = cond_values.reshape(1, -1)
                cond_values = self.dw.whiten(cond_values).flatten()[
                    conditional_variables
                ]
            else:
                cond_values = conditional_values

            weights = np.exp(
                -np.sum((cond_values - data[:, conditional_variables]) ** 2, axis=1)
                / (2 * self.bandwidth**2)
            )
            weights /= np.sum(weights)
            i = rs.choice(data.shape[0], n_samples, p=weights)

            sample = np.atleast_2d(rs.normal(data[i], self.bandwidth))

            if self.rescale:
                sample = self.dw.unwhiten(sample)

            if keep_dims is False:
                return sample[:, np.logical_not(conditional_variables)]
            else:
                sample[:, conditional_variables] = np.broadcast_to(
                    conditional_values, (n_samples, len(conditional_values))
                )
                return sample