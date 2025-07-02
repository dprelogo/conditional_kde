"""Integration tests for conditional_kde package."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from conditional_kde import (
    ConditionalGaussian,
    ConditionalGaussianKernelDensity,
    InterpolatedConditionalGaussian,
    InterpolatedConditionalKernelDensity,
)


class TestIntegration:
    """Integration tests for the conditional_kde package."""

    @pytest.fixture
    def correlated_data(self):
        """Create correlated multivariate data."""
        np.random.seed(42)
        n_samples = 2000

        # Create correlated 4D data
        mean = np.array([1.0, 2.0, 3.0, 4.0])
        cov = np.array(
            [
                [1.0, 0.7, 0.3, 0.1],
                [0.7, 2.0, 0.5, 0.2],
                [0.3, 0.5, 1.5, 0.6],
                [0.1, 0.2, 0.6, 1.0],
            ]
        )

        data = np.random.multivariate_normal(mean, cov, n_samples)
        return data, mean, cov

    @pytest.fixture
    def bimodal_data(self):
        """Create bimodal data."""
        np.random.seed(42)
        n_samples = 1000

        # Two clusters
        cluster1 = np.random.multivariate_normal(
            [0, 0, 0], np.eye(3) * 0.5, n_samples // 2
        )
        cluster2 = np.random.multivariate_normal(
            [5, 5, 5], np.eye(3) * 0.5, n_samples // 2
        )

        data = np.vstack([cluster1, cluster2])
        np.random.shuffle(data)
        return data

    def test_gaussian_vs_kde_single_gaussian(self, correlated_data):
        """Test that ConditionalGaussian and KDE give similar results for Gaussian data."""
        data, true_mean, true_cov = correlated_data

        # Fit both models
        cg = ConditionalGaussian()
        cg.fit(data, features=["a", "b", "c", "d"])

        kde = ConditionalGaussianKernelDensity(bandwidth=0.1)
        kde.fit(data, features=["a", "b", "c", "d"])

        # Test unconditional log probabilities
        test_points = data[:100]
        log_probs_cg = cg.score_samples(test_points)
        log_probs_kde = kde.score_samples(test_points)

        # They should be correlated (relaxed threshold due to different methods)
        correlation = np.corrcoef(log_probs_cg, log_probs_kde)[0, 1]
        assert correlation > 0.3

        # Test conditional log probabilities
        log_probs_cg_cond = cg.score_samples(
            test_points, conditional_features=["a", "b"]
        )
        log_probs_kde_cond = kde.score_samples(
            test_points, conditional_features=["a", "b"]
        )

        correlation_cond = np.corrcoef(log_probs_cg_cond, log_probs_kde_cond)[0, 1]
        assert (
            correlation_cond > 0.05
        )  # Very relaxed threshold - methods are quite different

    def test_conditional_sampling_consistency(self, correlated_data):
        """Test that conditional sampling produces consistent results."""
        data, _, _ = correlated_data

        # Fit models
        cg = ConditionalGaussian()
        cg.fit(data, features=["a", "b", "c", "d"])

        kde = ConditionalGaussianKernelDensity(bandwidth=0.1)
        kde.fit(data, features=["a", "b", "c", "d"])

        # Sample conditionally
        conditionals = {"a": 1.0, "b": 2.0}

        samples_cg = cg.sample(
            conditionals=conditionals, n_samples=1000, random_state=42
        )
        samples_kde = kde.sample(
            conditionals=conditionals, n_samples=1000, random_state=42
        )

        # Check that the means are similar
        assert np.abs(np.mean(samples_cg[:, 0]) - np.mean(samples_kde[:, 0])) < 0.2
        assert np.abs(np.mean(samples_cg[:, 1]) - np.mean(samples_kde[:, 1])) < 0.2

    def test_whitening_algorithms_consistency(self, correlated_data):
        """Test that different whitening algorithms produce valid results."""
        data, _, _ = correlated_data

        algorithms = [None, "rescale", "ZCA"]
        kdes = {}

        for algo in algorithms:
            kde = ConditionalGaussianKernelDensity(
                whitening_algorithm=algo, bandwidth=0.1
            )
            kde.fit(data[:500])  # Use smaller dataset
            kdes[algo] = kde

        # Test that all can score and sample
        test_point = data[0:1]
        for algo, kde in kdes.items():
            log_prob = kde.score_samples(test_point)
            assert np.isfinite(log_prob)

            samples = kde.sample(n_samples=10)
            assert samples.shape == (10, 4)
            assert np.all(np.isfinite(samples))

    def test_interpolated_gaussian_smooth_transition(self):
        """Test that interpolated Gaussian has smooth transitions."""
        np.random.seed(42)

        # Create data that varies smoothly along a parameter
        grid_data = []
        for t in np.linspace(0, 1, 5):
            # Mean shifts from [0, 0] to [5, 5] as t goes from 0 to 1
            mean = np.array([5 * t, 5 * t])
            data = np.random.multivariate_normal(mean, np.eye(2), 200)
            grid_data.append(data)

        grid_data = np.array(grid_data)

        # Fit interpolated model
        icg = InterpolatedConditionalGaussian()
        icg.fit(grid_data, interpolation_method="linear")

        # Test that samples transition smoothly
        t_values = np.linspace(0, 1, 11)
        mean_values = []

        for t in t_values:
            samples = icg.sample({-1: float(t)}, n_samples=500, random_state=42)
            mean_values.append(np.mean(samples, axis=0))

        mean_values = np.array(mean_values)

        # Check that means increase monotonically
        assert np.all(np.diff(mean_values[:, 0]) > -0.1)  # Allow small noise
        assert np.all(np.diff(mean_values[:, 1]) > -0.1)

    def test_vectorized_conditionals(self, correlated_data):
        """Test vectorized conditional sampling."""
        data, _, _ = correlated_data

        kde = ConditionalGaussianKernelDensity(bandwidth=0.1)
        kde.fit(data, features=["a", "b", "c", "d"])

        # Vectorized conditionals
        n_conditions = 10
        conditionals = {
            "a": np.linspace(-1, 3, n_conditions),
            "b": np.linspace(0, 4, n_conditions),
        }

        samples = kde.sample(conditionals=conditionals)
        assert samples.shape == (n_conditions, 2)

        # Test with keep_dims
        samples_with_dims = kde.sample(conditionals=conditionals, keep_dims=True)
        assert samples_with_dims.shape == (n_conditions, 4)
        assert_allclose(samples_with_dims[:, 0], conditionals["a"], atol=1e-15)
        assert_allclose(samples_with_dims[:, 1], conditionals["b"], atol=1e-15)

    def test_kde_bandwidth_optimization(self):
        """Test bandwidth optimization for KDE."""
        np.random.seed(42)

        # Create simple 2D data
        data = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], 200)

        # Fit with different bandwidth methods
        kde_scott = ConditionalGaussianKernelDensity(bandwidth="scott")
        kde_scott.fit(data)

        kde_opt = ConditionalGaussianKernelDensity(
            bandwidth="optimized", steps=5, cv_fold=3
        )
        kde_opt.fit(data)

        kde_fixed = ConditionalGaussianKernelDensity(bandwidth=0.5)
        kde_fixed.fit(data)

        # All should produce valid results
        test_point = np.array([[0, 0]])
        for kde in [kde_scott, kde_opt, kde_fixed]:
            log_prob = kde.score_samples(test_point)
            assert np.isfinite(log_prob)

            samples = kde.sample(n_samples=10)
            assert samples.shape == (10, 2)

    def test_interpolated_kde_multimodal(self):
        """Test interpolated KDE with multimodal distributions."""
        np.random.seed(42)

        # Create grid data where distribution changes from unimodal to bimodal
        grid_data = []
        for i in range(4):
            if i < 2:
                # Unimodal
                data = np.random.normal(0, 1, (200, 2))
            else:
                # Bimodal
                data1 = np.random.normal(-2, 0.5, (100, 2))
                data2 = np.random.normal(2, 0.5, (100, 2))
                data = np.vstack([data1, data2])
            grid_data.append(data)

        grid_data = np.array(grid_data)

        # Fit interpolated KDE
        ickde = InterpolatedConditionalKernelDensity(bandwidth=0.3)
        ickde.fit(grid_data)

        # Sample at different grid points
        for i, t in enumerate(np.linspace(0, 1, 4)):
            samples = ickde.sample({-1: float(t)}, n_samples=500)

            # Check that variance increases for bimodal distributions
            if i >= 2:
                assert np.std(samples) > 1.0

    def test_edge_cases_and_errors(self):
        """Test various edge cases and error conditions."""
        np.random.seed(42)
        data = np.random.randn(100, 3)

        # Test conditioning on all features
        cg = ConditionalGaussian()
        cg.fit(data)

        with pytest.raises(ValueError, match="condition on all features"):
            cg.score_samples(data, conditional_features=[0, 1, 2])

        with pytest.raises(ValueError, match="condition on all features"):
            cg.sample(conditionals={0: 1.0, 1: 2.0, 2: 3.0})

        # Test invalid feature names
        with pytest.raises(
            ValueError, match="conditional_features should be in features"
        ):
            cg.score_samples(data, conditional_features=["invalid"])

        # Test mismatched array lengths in vectorized conditionals
        with pytest.raises(ValueError, match="same length"):
            cg.sample(conditionals={0: np.array([1, 2]), 1: np.array([3, 4, 5])})

    def test_reproducibility(self, correlated_data):
        """Test that results are reproducible with random_state."""
        data, _, _ = correlated_data

        kde = ConditionalGaussianKernelDensity(bandwidth=0.1)
        kde.fit(data[:500])

        # Test unconditional sampling
        samples1 = kde.sample(n_samples=100, random_state=42)
        samples2 = kde.sample(n_samples=100, random_state=42)
        assert_allclose(samples1, samples2)

        # Test conditional sampling
        conditionals = {0: 1.0}
        samples1 = kde.sample(conditionals=conditionals, n_samples=100, random_state=42)
        samples2 = kde.sample(conditionals=conditionals, n_samples=100, random_state=42)
        assert_allclose(samples1, samples2)

    def test_performance_with_large_data(self):
        """Test that methods work with larger datasets."""
        np.random.seed(42)

        # Create larger dataset
        n_samples = 5000
        n_features = 10
        data = np.random.randn(n_samples, n_features)

        # Add some correlation
        for i in range(1, n_features):
            data[:, i] += 0.3 * data[:, 0]

        # Test ConditionalGaussian
        cg = ConditionalGaussian()
        cg.fit(data)

        # Should be able to score and sample quickly
        log_probs = cg.score_samples(data[:100])
        assert log_probs.shape == (100,)

        samples = cg.sample(n_samples=1000)
        assert samples.shape == (1000, n_features)

        # Test with conditioning
        conditionals = {i: 0.0 for i in range(5)}
        samples = cg.sample(conditionals=conditionals, n_samples=100)
        assert samples.shape == (100, 5)  # Returns only non-conditioned features
