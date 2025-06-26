"""Tests for ConditionalGaussian and ConditionalGaussianKernelDensity classes."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from conditional_kde.gaussian import (
    ConditionalGaussian,
    ConditionalGaussianKernelDensity,
)


class TestConditionalGaussian:
    """Test suite for ConditionalGaussian class."""

    @pytest.fixture
    def simple_data(self):
        """Create simple 2D gaussian data for testing."""
        np.random.seed(42)
        mean = np.array([1.0, 2.0])
        cov = np.array([[1.0, 0.5], [0.5, 2.0]])
        n_samples = 1000
        data = np.random.multivariate_normal(mean, cov, n_samples)
        return data

    @pytest.fixture
    def simple_3d_data(self):
        """Create simple 3D gaussian data for testing."""
        np.random.seed(42)
        mean = np.array([1.0, 2.0, 3.0])
        cov = np.array([[1.0, 0.5, 0.2], [0.5, 2.0, 0.3], [0.2, 0.3, 1.5]])
        n_samples = 1000
        data = np.random.multivariate_normal(mean, cov, n_samples)
        return data

    def test_init(self):
        """Test ConditionalGaussian initialization."""
        # Test default initialization
        cg = ConditionalGaussian()
        assert cg.bandwidth == 1.0
        assert cg.features is None
        assert cg.dw is None

        # Test with custom bandwidth
        cg = ConditionalGaussian(bandwidth=2.0)
        assert cg.bandwidth == 2.0

        # Test invalid bandwidth
        with pytest.raises(ValueError, match="Bandwidth should be a number"):
            ConditionalGaussian(bandwidth="invalid")

    def test_log_prob_scalar_cov(self):
        """Test _log_prob with scalar covariance."""
        mean = np.array([0.0, 0.0])
        cov = 1.0
        X = np.array([[0.0, 0.0], [1.0, 1.0]])

        log_probs = ConditionalGaussian._log_prob(X, mean, cov)
        assert log_probs.shape == (2,)
        # At mean, log prob should be highest
        assert log_probs[0] > log_probs[1]

    def test_log_prob_diagonal_cov(self):
        """Test _log_prob with diagonal covariance."""
        mean = np.array([0.0, 0.0])
        cov = np.array([1.0, 2.0])
        X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

        log_probs = ConditionalGaussian._log_prob(X, mean, cov)
        assert log_probs.shape == (3,)
        # At mean, log prob should be highest
        assert log_probs[0] > log_probs[1]
        assert log_probs[0] > log_probs[2]

    def test_log_prob_full_cov(self):
        """Test _log_prob with full covariance matrix."""
        mean = np.array([0.0, 0.0])
        cov = np.array([[1.0, 0.5], [0.5, 2.0]])
        X = np.array([[0.0, 0.0], [1.0, 1.0]])

        log_probs = ConditionalGaussian._log_prob(X, mean, cov)
        assert log_probs.shape == (2,)
        assert log_probs[0] > log_probs[1]

    def test_log_prob_invalid_inputs(self):
        """Test _log_prob with invalid inputs."""
        mean = np.array([0.0, 0.0])
        X = np.array([[0.0, 0.0]])

        # Test invalid covariance type
        with pytest.raises(TypeError):
            ConditionalGaussian._log_prob(X, mean, "invalid")

        # Test mismatched dimensions
        with pytest.raises(ValueError):
            ConditionalGaussian._log_prob(np.array([[0.0, 0.0, 0.0]]), mean, 1.0)

        # Test invalid covariance dimensions
        with pytest.raises(ValueError):
            ConditionalGaussian._log_prob(X, mean, np.array([1.0]))

        with pytest.raises(ValueError):
            ConditionalGaussian._log_prob(X, mean, np.array([[1.0, 0.0]]))

    def test_covariance_decomposition(self):
        """Test _covariance_decomposition method."""
        cov = np.array(
            [
                [1.0, 0.5, 0.2, 0.1],
                [0.5, 2.0, 0.3, 0.2],
                [0.2, 0.3, 1.5, 0.4],
                [0.1, 0.2, 0.4, 1.2],
            ]
        )
        cond_mask = np.array([True, False, True, False])

        # Test getting all components
        Σ_cond, Σ_uncond, Σ_cross = ConditionalGaussian._covariance_decomposition(
            cov, cond_mask, cond_only=False
        )

        assert Σ_cond.shape == (2, 2)
        assert Σ_uncond.shape == (2, 2)
        assert Σ_cross.shape == (2, 2)

        # Test getting only conditional component
        Σ_cond_only = ConditionalGaussian._covariance_decomposition(
            cov, cond_mask, cond_only=True
        )
        assert_array_equal(Σ_cond_only, Σ_cond)

        # Test invalid inputs
        with pytest.raises(ValueError):
            ConditionalGaussian._covariance_decomposition(cov, np.array([True, False]))

        with pytest.raises(ValueError):
            ConditionalGaussian._covariance_decomposition(cov.flatten(), cond_mask)

    def test_fit(self, simple_data):
        """Test fitting the model."""
        cg = ConditionalGaussian()

        # Test basic fit
        cg.fit(simple_data)
        assert cg.features == [0, 1]
        assert cg.dw is not None
        assert_allclose(cg.dw.μ.shape, (1, 2))
        assert_allclose(cg.dw.Σ.shape, (2, 2))

        # Test fit with custom features
        cg.fit(simple_data, features=["x", "y"])
        assert cg.features == ["x", "y"]

        # Test fit with weights
        weights = np.ones(len(simple_data))
        cg.fit(simple_data, weights=weights)

        # Test invalid features
        with pytest.raises(TypeError):
            cg.fit(simple_data, features="invalid")

        with pytest.raises(ValueError):
            cg.fit(simple_data, features=["x"])

        with pytest.raises(ValueError):
            cg.fit(simple_data, features=["x", "x"])

    def test_score_samples_unconditional(self, simple_data):
        """Test score_samples without conditioning."""
        cg = ConditionalGaussian()
        cg.fit(simple_data)

        # Test single sample
        log_prob = cg.score_samples(simple_data[0])
        assert isinstance(log_prob, np.ndarray)
        assert log_prob.shape == (1,)

        # Test multiple samples
        log_probs = cg.score_samples(simple_data[:10])
        assert log_probs.shape == (10,)

    def test_score_samples_conditional(self, simple_3d_data):
        """Test score_samples with conditioning."""
        cg = ConditionalGaussian()
        cg.fit(simple_3d_data, features=["a", "b", "c"])

        # Test conditioning on one feature
        log_probs = cg.score_samples(simple_3d_data[:10], conditional_features=["a"])
        assert log_probs.shape == (10,)

        # Test conditioning on multiple features
        log_probs = cg.score_samples(
            simple_3d_data[:10], conditional_features=["a", "c"]
        )
        assert log_probs.shape == (10,)

        # Test invalid conditional features
        with pytest.raises(ValueError):
            cg.score_samples(simple_3d_data[:10], conditional_features=["invalid"])

        # Test conditioning on all features
        with pytest.raises(ValueError):
            cg.score_samples(simple_3d_data[:10], conditional_features=["a", "b", "c"])

    def test_check_conditionals(self):
        """Test _check_conditionals method."""
        cg = ConditionalGaussian()
        cg.features = ["a", "b", "c"]

        # Test scalar conditionals
        conditionals = {"a": 1.0, "b": 2.0}
        vectorized, n_samples = cg._check_conditionals(conditionals, 10)
        assert not vectorized
        assert n_samples == 10

        # Test vectorized conditionals
        conditionals = {"a": np.array([1.0, 2.0]), "b": np.array([3.0, 4.0])}
        vectorized, n_samples = cg._check_conditionals(conditionals, 10)
        assert vectorized
        assert n_samples == 2

        # Test invalid conditionals
        with pytest.raises(TypeError):
            cg._check_conditionals("invalid", 10)

        with pytest.raises(ValueError):
            cg._check_conditionals({"invalid": 1.0}, 10)

        with pytest.raises(ValueError):
            cg._check_conditionals({"a": 1.0, "b": 2.0, "c": 3.0}, 10)

        # Test mixed types
        with pytest.raises(ValueError):
            cg._check_conditionals({"a": 1.0, "b": np.array([1.0, 2.0])}, 10)

        # Test mismatched lengths
        with pytest.raises(ValueError):
            cg._check_conditionals(
                {"a": np.array([1.0]), "b": np.array([1.0, 2.0])}, 10
            )

    def test_sample_unconditional(self, simple_data):
        """Test sampling without conditioning."""
        cg = ConditionalGaussian()
        cg.fit(simple_data)

        # Test single sample
        sample = cg.sample(n_samples=1)
        assert sample.shape == (1, 2)

        # Test multiple samples
        samples = cg.sample(n_samples=100)
        assert samples.shape == (100, 2)

        # Test with random state
        samples1 = cg.sample(n_samples=10, random_state=42)
        samples2 = cg.sample(n_samples=10, random_state=42)
        assert_array_equal(samples1, samples2)

    def test_sample_conditional_scalar(self, simple_3d_data):
        """Test sampling with scalar conditional values."""
        cg = ConditionalGaussian()
        cg.fit(simple_3d_data, features=["a", "b", "c"])

        # Test conditioning on one feature
        conditionals = {"a": 1.0}
        samples = cg.sample(conditionals=conditionals, n_samples=100)
        assert samples.shape == (100, 2)  # Returns only non-conditioned features

        # Test with keep_dims=True
        samples = cg.sample(conditionals=conditionals, n_samples=100, keep_dims=True)
        assert samples.shape == (100, 3)
        assert np.all(samples[:, 0] == 1.0)  # Check conditional value is preserved

        # Test conditioning on multiple features
        conditionals = {"a": 1.0, "c": 3.0}
        samples = cg.sample(conditionals=conditionals, n_samples=100)
        assert samples.shape == (100, 1)

    def test_sample_conditional_vectorized(self, simple_3d_data):
        """Test sampling with vectorized conditional values."""
        cg = ConditionalGaussian()
        cg.fit(simple_3d_data, features=["a", "b", "c"])

        # Test with vectorized conditionals
        conditionals = {"a": np.array([1.0, 2.0, 3.0])}
        samples = cg.sample(conditionals=conditionals)
        assert samples.shape == (3, 2)

        # Test with keep_dims=True
        samples = cg.sample(conditionals=conditionals, keep_dims=True)
        assert samples.shape == (3, 3)
        assert_array_equal(samples[:, 0], [1.0, 2.0, 3.0])

    def test_sample_invalid_random_state(self, simple_data):
        """Test sampling with invalid random state."""
        cg = ConditionalGaussian()
        cg.fit(simple_data)

        with pytest.raises(TypeError):
            cg.sample(n_samples=10, random_state="invalid")


class TestConditionalGaussianKernelDensity:
    """Test suite for ConditionalGaussianKernelDensity class."""

    @pytest.fixture
    def simple_data(self):
        """Create simple 2D data for testing."""
        np.random.seed(42)
        # Create two clusters
        cluster1 = np.random.normal([0, 0], 0.5, (500, 2))
        cluster2 = np.random.normal([3, 3], 0.5, (500, 2))
        data = np.vstack([cluster1, cluster2])
        return data

    @pytest.fixture
    def simple_3d_data(self):
        """Create simple 3D data for testing."""
        np.random.seed(42)
        n_samples = 1000
        data = np.random.randn(n_samples, 3)
        # Add some correlation
        data[:, 1] += 0.5 * data[:, 0]
        data[:, 2] += 0.3 * data[:, 0] + 0.4 * data[:, 1]
        return data

    def test_init(self):
        """Test ConditionalGaussianKernelDensity initialization."""
        # Test default initialization
        kde = ConditionalGaussianKernelDensity()
        assert kde.algorithm == "rescale"
        assert kde.bandwidth == "scott"
        assert kde.bandwidth_kwargs == {}

        # Test with custom parameters
        kde = ConditionalGaussianKernelDensity(whitening_algorithm="ZCA", bandwidth=0.5)
        assert kde.algorithm == "ZCA"
        assert kde.bandwidth == 0.5

        # Test optimized bandwidth
        kde = ConditionalGaussianKernelDensity(
            bandwidth="optimized", steps=20, cv_fold=3
        )
        assert kde.bandwidth == "optimized"
        assert kde.bandwidth_kwargs["steps"] == 20
        assert kde.bandwidth_kwargs["cv_fold"] == 3

        # Test invalid whitening algorithm
        with pytest.raises(ValueError):
            ConditionalGaussianKernelDensity(whitening_algorithm="invalid")

        # Test invalid bandwidth
        with pytest.raises(ValueError):
            ConditionalGaussianKernelDensity(bandwidth="invalid")

    def test_log_scott(self):
        """Test Scott's parameter calculation."""
        log_scott = ConditionalGaussianKernelDensity.log_scott(100, 2)
        assert isinstance(log_scott, float)
        assert log_scott < 0

        # Test with different parameters
        log_scott1 = ConditionalGaussianKernelDensity.log_scott(1000, 2)
        log_scott2 = ConditionalGaussianKernelDensity.log_scott(100, 2)
        assert log_scott1 < log_scott2  # More samples -> smaller bandwidth

    def test_log_prob_kde(self):
        """Test _log_prob for KDE."""
        data = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        X = np.array([[0.5, 0.5], [1.5, 1.5]])
        cov = 0.1

        log_probs = ConditionalGaussianKernelDensity._log_prob(X, data, cov)
        assert log_probs.shape == (2,)

        # Test with different covariance types
        cov_diag = np.array([0.1, 0.2])
        log_probs = ConditionalGaussianKernelDensity._log_prob(X, data, cov_diag)
        assert log_probs.shape == (2,)

        cov_full = np.array([[0.1, 0.05], [0.05, 0.2]])
        log_probs = ConditionalGaussianKernelDensity._log_prob(X, data, cov_full)
        assert log_probs.shape == (2,)

    def test_conditional_weights(self):
        """Test _conditional_weights calculation."""
        conditional_data = np.array([[0.0], [1.0], [2.0]])
        conditional_values = np.array([1.0])
        cov = 0.5

        weights = ConditionalGaussianKernelDensity._conditional_weights(
            conditional_values, conditional_data, cov
        )
        assert weights.shape == (3,)
        assert np.allclose(np.sum(weights), 1.0)
        assert weights[1] > weights[0]  # Closest point should have highest weight
        assert weights[1] > weights[2]

        # Test vectorized conditionals
        conditional_values = np.array([[0.0], [1.0], [2.0]])
        weights = ConditionalGaussianKernelDensity._conditional_weights(
            conditional_values, conditional_data, cov
        )
        assert weights.shape == (3, 3)
        assert np.allclose(np.sum(weights, axis=1), 1.0)

    def test_fit_scott_bandwidth(self, simple_data):
        """Test fitting with Scott's bandwidth."""
        kde = ConditionalGaussianKernelDensity(bandwidth="scott")
        kde.fit(simple_data)

        assert kde.features == [0, 1]
        assert kde.dw is not None
        assert isinstance(kde.bandwidth, float)
        assert kde.bandwidth > 0

    def test_fit_fixed_bandwidth(self, simple_data):
        """Test fitting with fixed bandwidth."""
        kde = ConditionalGaussianKernelDensity(bandwidth=0.5)
        kde.fit(simple_data)

        assert kde.bandwidth == 0.5

    def test_fit_optimized_bandwidth(self, simple_data):
        """Test fitting with optimized bandwidth."""
        # Use smaller dataset for faster test
        small_data = simple_data[:100]
        kde = ConditionalGaussianKernelDensity(
            bandwidth="optimized", steps=3, cv_fold=2
        )
        kde.fit(small_data)

        assert isinstance(kde.bandwidth, float)
        assert kde.bandwidth > 0

    def test_fit_with_features(self, simple_data):
        """Test fitting with custom features."""
        kde = ConditionalGaussianKernelDensity()
        kde.fit(simple_data, features=["x", "y"])

        assert kde.features == ["x", "y"]

    def test_score_samples_unconditional(self, simple_data):
        """Test score_samples without conditioning."""
        kde = ConditionalGaussianKernelDensity(bandwidth=0.5)
        kde.fit(simple_data[:100])  # Use smaller dataset

        log_probs = kde.score_samples(simple_data[:10])
        assert log_probs.shape == (10,)
        assert np.all(np.isfinite(log_probs))

    def test_score_samples_conditional(self, simple_3d_data):
        """Test score_samples with conditioning."""
        kde = ConditionalGaussianKernelDensity(bandwidth=0.5)
        kde.fit(simple_3d_data[:100], features=["a", "b", "c"])

        log_probs = kde.score_samples(simple_3d_data[:10], conditional_features=["a"])
        assert log_probs.shape == (10,)
        assert np.all(np.isfinite(log_probs))

    def test_sample_unconditional(self, simple_data):
        """Test sampling without conditioning."""
        kde = ConditionalGaussianKernelDensity(bandwidth=0.5)
        kde.fit(simple_data[:100])

        samples = kde.sample(n_samples=50)
        assert samples.shape == (50, 2)

        # Test reproducibility
        samples1 = kde.sample(n_samples=10, random_state=42)
        samples2 = kde.sample(n_samples=10, random_state=42)
        assert_array_equal(samples1, samples2)

    def test_sample_conditional_rescale(self, simple_3d_data):
        """Test conditional sampling with rescale whitening."""
        kde = ConditionalGaussianKernelDensity(
            whitening_algorithm="rescale", bandwidth=0.5
        )
        kde.fit(simple_3d_data[:100], features=["a", "b", "c"])

        conditionals = {"a": 1.0}
        samples = kde.sample(conditionals=conditionals, n_samples=50)
        assert samples.shape == (50, 2)

        # Test with keep_dims
        samples = kde.sample(conditionals=conditionals, n_samples=50, keep_dims=True)
        assert samples.shape == (50, 3)
        assert np.all(
            np.abs(samples[:, 0] - 1.0) < 0.1
        )  # Should be close to conditional value

    def test_sample_conditional_zca(self, simple_3d_data):
        """Test conditional sampling with ZCA whitening."""
        kde = ConditionalGaussianKernelDensity(whitening_algorithm="ZCA", bandwidth=0.5)
        kde.fit(simple_3d_data[:100], features=["a", "b", "c"])

        conditionals = {"a": 1.0}
        samples = kde.sample(conditionals=conditionals, n_samples=50)
        assert samples.shape == (50, 2)

    def test_sample_vectorized_conditionals(self, simple_3d_data):
        """Test sampling with vectorized conditionals."""
        kde = ConditionalGaussianKernelDensity(bandwidth=0.5)
        kde.fit(simple_3d_data[:100], features=["a", "b", "c"])

        conditionals = {"a": np.array([0.0, 1.0, 2.0])}
        samples = kde.sample(conditionals=conditionals)
        assert samples.shape == (3, 2)

    def test_sample_methods_consistency(self, simple_3d_data):
        """Test that _sample and _sample_general give similar results."""
        # For non-ZCA algorithms, both methods should work
        kde = ConditionalGaussianKernelDensity(
            whitening_algorithm="rescale", bandwidth=0.5
        )
        kde.fit(simple_3d_data[:100])

        # Set random state for reproducibility
        rs1 = np.random.RandomState(42)
        rs2 = np.random.RandomState(42)

        samples1 = kde._sample(n_samples=100, random_state=rs1)
        samples2 = kde._sample_general(n_samples=100, random_state=rs2)

        # Check that distributions are similar (not exact due to different methods)
        assert np.abs(np.mean(samples1) - np.mean(samples2)) < 0.5
        assert np.abs(np.std(samples1) - np.std(samples2)) < 0.5
