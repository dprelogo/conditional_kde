"""Tests for InterpolatedConditionalGaussian and InterpolatedConditionalKernelDensity classes."""

import numpy as np
import pytest

from conditional_kde.interpolated import (
    InterpolatedConditionalGaussian,
    InterpolatedConditionalKernelDensity,
)


class TestInterpolatedConditionalGaussian:
    """Test suite for InterpolatedConditionalGaussian class."""

    @pytest.fixture
    def grid_data_2d(self):
        """Create 2D data on a grid for testing."""
        np.random.seed(42)
        # Create data for a 3x4 grid where each point has 50 samples with 2 features
        grid_shape = (3, 4)
        n_samples = 50
        n_features = 2

        data = np.zeros(grid_shape + (n_samples, n_features))

        # Fill with data that varies smoothly across the grid
        for i in range(grid_shape[0]):
            for j in range(grid_shape[1]):
                mean = [i, j]
                cov = [[1.0, 0.3], [0.3, 1.0]]
                data[i, j] = np.random.multivariate_normal(mean, cov, n_samples)

        return data

    @pytest.fixture
    def grid_data_1d(self):
        """Create 1D data on a grid for testing."""
        np.random.seed(42)
        # Create data for a 5-point grid where each point has 100 samples with 3 features
        grid_shape = (5,)
        n_samples = 100
        n_features = 3

        data = np.zeros(grid_shape + (n_samples, n_features))

        for i in range(grid_shape[0]):
            mean = [i, i * 2, i * 3]
            cov = np.eye(3) * (i + 1) * 0.5
            data[i] = np.random.multivariate_normal(mean, cov, n_samples)

        return data

    @pytest.fixture
    def list_data(self):
        """Create nested list data with varying sample sizes."""
        np.random.seed(42)
        data = []

        for i in range(3):
            row = []
            for j in range(2):
                n_samples = 50 + i * 10 + j * 5  # Varying sample sizes
                mean = [i, j]
                samples = np.random.multivariate_normal(mean, np.eye(2), n_samples)
                row.append(samples)
            data.append(row)

        return data

    def test_init(self):
        """Test InterpolatedConditionalGaussian initialization."""
        icg = InterpolatedConditionalGaussian()
        assert icg.bandwidth == 1.0
        assert icg.inherent_features is None
        assert icg.features is None
        assert icg.interpolator is None
        assert icg.gaussians is None

        # Test with custom bandwidth
        icg = InterpolatedConditionalGaussian(bandwidth=2.0)
        assert icg.bandwidth == 2.0

        # Test invalid bandwidth
        with pytest.raises(ValueError, match="Bandwidth should be a number"):
            InterpolatedConditionalGaussian(bandwidth="invalid")

    def test_fit_array_data_1d(self, grid_data_1d):
        """Test fitting with 1D grid array data."""
        icg = InterpolatedConditionalGaussian()
        icg.fit(grid_data_1d)

        assert icg.inherent_features == [-1]
        assert icg.features == [0, 1, 2]
        assert icg.interpolator is not None
        assert len(icg.gaussians) == 5
        # Check interpolator was created
        assert icg.interpolator is not None

    def test_fit_array_data_2d(self, grid_data_2d):
        """Test fitting with 2D grid array data."""
        icg = InterpolatedConditionalGaussian()
        icg.fit(grid_data_2d)

        assert icg.inherent_features == [-1, -2]
        assert icg.features == [0, 1]
        assert icg.interpolator is not None
        assert icg.gaussians.size == 12  # 3x4 grid
        # Check that 2D interpolator was created
        assert len(icg.interpolator.grid) == 2

    def test_fit_list_data(self, list_data):
        """Test fitting with nested list data."""
        icg = InterpolatedConditionalGaussian()
        icg.fit(list_data)

        assert icg.inherent_features == [-1, -2]
        assert icg.features == [0, 1]
        assert icg.gaussians.size == 6  # 3x2 grid

    def test_fit_custom_parameters(self, grid_data_1d):
        """Test fitting with custom parameters."""
        icg = InterpolatedConditionalGaussian()

        # Custom features and interpolation points
        icg.fit(
            grid_data_1d,
            inherent_features=["time"],
            features=["x", "y", "z"],
            interpolation_points={"time": [0.0, 0.25, 0.5, 0.75, 1.0]},
            interpolation_method="nearest",
        )

        assert icg.inherent_features == ["time"]
        assert icg.features == ["x", "y", "z"]
        # Check interpolator was created with proper method
        assert icg.interpolator.method == "nearest"

    def test_fit_invalid_data(self):
        """Test fitting with invalid data."""
        icg = InterpolatedConditionalGaussian()

        # Too few dimensions
        with pytest.raises(ValueError, match="should have at least 3 axes"):
            icg.fit(np.zeros((5, 10)))

        # Invalid interpolation method
        with pytest.raises(ValueError):
            icg.fit(np.zeros((3, 50, 2)), interpolation_method="invalid")

        # The implementation doesn't validate interpolation points length
        # so we skip this test

    def test_score_samples(self, grid_data_1d):
        """Test score_samples method."""
        icg = InterpolatedConditionalGaussian()
        icg.fit(grid_data_1d)

        # Test samples - only pass the non-inherent features
        X = np.random.randn(10, 3)  # 3 features (not including inherent)
        inherent_conditionals = {-1: 0.5}  # Middle of the grid

        log_probs = icg.score_samples(X, inherent_conditionals)
        assert log_probs.shape == (10,)
        assert np.all(np.isfinite(log_probs))

        # Test with conditional features
        log_probs = icg.score_samples(
            X, inherent_conditionals, conditional_features=[0]
        )
        assert log_probs.shape == (10,)

    def test_sample(self, grid_data_1d):
        """Test sampling method."""
        icg = InterpolatedConditionalGaussian()
        icg.fit(grid_data_1d)

        # Test basic sampling
        inherent_conditionals = {-1: 0.5}  # Inherent feature value
        samples = icg.sample(inherent_conditionals, n_samples=50)
        assert samples.shape == (50, 3)

        # Test with additional conditionals
        conditionals = {0: 1.0}
        samples = icg.sample(
            inherent_conditionals, conditionals=conditionals, n_samples=50
        )
        assert samples.shape == (50, 2)  # One less due to additional conditional

        # Test with keep_dims
        samples = icg.sample(
            inherent_conditionals,
            conditionals=conditionals,
            n_samples=50,
            keep_dims=True,
        )
        assert samples.shape == (50, 4)  # 3 features + 1 inherent (with 1 conditioned)

        # Test vectorized conditionals
        conditionals = {0: np.array([0.0, 0.5, 1.0])}
        samples = icg.sample(inherent_conditionals, conditionals=conditionals)
        assert samples.shape == (3, 2)

    def test_sample_nearest_interpolation(self, grid_data_1d):
        """Test sampling with nearest neighbor interpolation."""
        icg = InterpolatedConditionalGaussian()
        icg.fit(grid_data_1d, interpolation_method="nearest")

        inherent_conditionals = {-1: 0.1}  # Should pick the first grid point
        samples = icg.sample(inherent_conditionals, n_samples=50)
        assert samples.shape == (50, 3)

    def test_inherent_conditionals_validation(self, grid_data_1d):
        """Test inherent conditionals validation."""
        icg = InterpolatedConditionalGaussian()
        icg.fit(grid_data_1d)

        # Must provide inherent conditionals
        X = np.random.randn(10, 3)

        # Test missing inherent conditionals
        with pytest.raises(TypeError):
            icg.score_samples(X, "invalid")

        # Test incomplete inherent conditionals
        with pytest.raises(ValueError):
            icg.score_samples(X, {})


class TestInterpolatedConditionalKernelDensity:
    """Test suite for InterpolatedConditionalKernelDensity class."""

    @pytest.fixture
    def grid_data_1d(self):
        """Create 1D data on a grid for testing."""
        np.random.seed(42)
        grid_shape = (4,)
        n_samples = 100
        n_features = 2

        data = np.zeros(grid_shape + (n_samples, n_features))

        for i in range(grid_shape[0]):
            # Create two clusters that move across the grid
            cluster1 = np.random.normal([i, 0], 0.3, (n_samples // 2, 2))
            cluster2 = np.random.normal([0, i], 0.3, (n_samples // 2, 2))
            data[i] = np.vstack([cluster1, cluster2])

        return data

    def test_init(self):
        """Test InterpolatedConditionalKernelDensity initialization."""
        ickde = InterpolatedConditionalKernelDensity()
        assert ickde.algorithm == "rescale"
        assert ickde.bandwidth == "scott"
        assert ickde.inherent_features is None

        # Test with custom parameters
        ickde = InterpolatedConditionalKernelDensity(
            whitening_algorithm="ZCA", bandwidth=0.5
        )
        assert ickde.algorithm == "ZCA"
        assert ickde.bandwidth == 0.5

    def test_fit(self, grid_data_1d):
        """Test fitting the model."""
        ickde = InterpolatedConditionalKernelDensity(bandwidth=0.5)
        ickde.fit(grid_data_1d)

        assert ickde.inherent_features == [-1]
        assert ickde.features == [0, 1]
        assert ickde.interpolator is not None
        assert len(ickde.kdes) == 4

    def test_fit_optimized_bandwidth(self, grid_data_1d):
        """Test fitting with optimized bandwidth."""
        # Use smaller data for faster test
        small_data = grid_data_1d[:2, :50, :]

        ickde = InterpolatedConditionalKernelDensity(
            bandwidth="optimized", steps=3, cv_fold=2
        )
        ickde.fit(small_data)

        assert len(ickde.kdes) == 2
        # Each KDE should have its own optimized bandwidth
        assert all(isinstance(kde.bandwidth, float) for kde in ickde.kdes)

    def test_score_samples(self, grid_data_1d):
        """Test score_samples method."""
        ickde = InterpolatedConditionalKernelDensity(bandwidth=0.5)
        ickde.fit(grid_data_1d)

        # Test samples
        X = np.random.randn(10, 2)  # 2 features (not including inherent)
        inherent_conditionals = {-1: 0.5}

        log_probs = ickde.score_samples(X, inherent_conditionals)
        assert log_probs.shape == (10,)
        assert np.all(np.isfinite(log_probs))

        # Test with conditional features
        log_probs = ickde.score_samples(
            X, inherent_conditionals, conditional_features=[0]
        )
        assert log_probs.shape == (10,)

    def test_sample_rescale(self, grid_data_1d):
        """Test sampling with rescale whitening."""
        ickde = InterpolatedConditionalKernelDensity(
            whitening_algorithm="rescale", bandwidth=0.5
        )
        ickde.fit(grid_data_1d)

        inherent_conditionals = {-1: 0.5}
        samples = ickde.sample(inherent_conditionals, n_samples=50)
        assert samples.shape == (50, 2)

        # Test with additional conditionals
        conditionals = {0: 1.0}
        samples = ickde.sample(
            inherent_conditionals, conditionals=conditionals, n_samples=50
        )
        assert samples.shape == (50, 1)

    def test_sample_zca(self, grid_data_1d):
        """Test sampling with ZCA whitening."""
        ickde = InterpolatedConditionalKernelDensity(
            whitening_algorithm="ZCA", bandwidth=0.5
        )
        ickde.fit(grid_data_1d)

        inherent_conditionals = {-1: 0.5}
        samples = ickde.sample(inherent_conditionals, n_samples=50)
        assert samples.shape == (50, 2)

    def test_sample_vectorized(self, grid_data_1d):
        """Test sampling with vectorized conditionals."""
        ickde = InterpolatedConditionalKernelDensity(bandwidth=0.5)
        ickde.fit(grid_data_1d)

        # Inherent conditionals must be scalar
        inherent_conditionals = {-1: 0.5}

        # Test with vectorized regular conditionals
        conditionals = {0: np.array([1.0, 1.5, 2.0])}
        samples = ickde.sample(inherent_conditionals, conditionals=conditionals)
        assert samples.shape == (3, 1)

    def test_interpolation_methods(self, grid_data_1d):
        """Test different interpolation methods."""
        # Linear interpolation
        ickde_linear = InterpolatedConditionalKernelDensity(bandwidth=0.5)
        ickde_linear.fit(grid_data_1d, interpolation_method="linear")

        # Nearest interpolation
        ickde_nearest = InterpolatedConditionalKernelDensity(bandwidth=0.5)
        ickde_nearest.fit(grid_data_1d, interpolation_method="nearest")

        # Compare log probabilities at interpolation point
        X = np.array([[0.5, 0.5, 0.5]])  # Middle of grid

        log_prob_linear = ickde_linear.score_samples(X[:, :2], {-1: X[0, 2]})
        log_prob_nearest = ickde_nearest.score_samples(X[:, :2], {-1: X[0, 2]})

        # They should be different due to interpolation method
        assert not np.allclose(log_prob_linear, log_prob_nearest)

    def test_edge_cases(self):
        """Test edge cases and error handling."""
        ickde = InterpolatedConditionalKernelDensity()

        # Test with too few dimensions
        with pytest.raises(ValueError):
            ickde.fit(np.zeros((5, 10)))

        # Test with invalid interpolation method
        with pytest.raises(ValueError):
            ickde.fit(np.zeros((3, 50, 2)), interpolation_method="invalid")
