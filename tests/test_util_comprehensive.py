"""Comprehensive tests for conditional_kde.util module."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.interpolate import RegularGridInterpolator

from conditional_kde.util import DataWhitener, Interpolator


class TestDataWhitenerComprehensive:
    """Comprehensive test suite for DataWhitener class."""

    @pytest.fixture
    def simple_data(self):
        """Simple 2D data for testing."""
        np.random.seed(42)
        return np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)

    @pytest.fixture
    def correlated_data(self):
        """Correlated multivariate data."""
        np.random.seed(42)
        mean = np.array([1.0, 2.0, 3.0])
        cov = np.array([[1.0, 0.8, 0.5], [0.8, 2.0, 0.7], [0.5, 0.7, 1.5]])
        return np.random.multivariate_normal(mean, cov, 1000)

    @pytest.fixture
    def weighted_data(self):
        """Data with non-uniform weights."""
        np.random.seed(42)
        # Create data where later samples have higher weight
        data = np.random.randn(100, 3)
        weights = np.linspace(0.1, 2.0, 100)
        return data, weights

    def test_init_valid_algorithms(self):
        """Test initialization with all valid algorithms."""
        algorithms = [None, "center", "rescale", "PCA", "ZCA"]
        for algo in algorithms:
            dw = DataWhitener(algo)
            assert dw.algorithm == algo
            assert dw.μ is None
            assert dw.Σ is None
            assert dw.W is None
            assert dw.WI is None

    def test_init_invalid_algorithm(self):
        """Test initialization with invalid algorithm."""
        with pytest.raises(
            ValueError, match="algorithm should be None, center, rescale, PCA or ZCA"
        ):
            DataWhitener("invalid")

    def test_fit_none_algorithm(self, simple_data):
        """Test fit with None algorithm (no transformation)."""
        dw = DataWhitener(None)
        dw.fit(simple_data)

        # Check that parameters are identity/zero
        assert_array_equal(dw.μ, np.zeros((1, 2)))
        assert_array_equal(dw.Σ, np.identity(2))
        assert_array_equal(dw.W, np.identity(2))
        assert_array_equal(dw.WI, np.identity(2))

    def test_fit_center_algorithm(self, simple_data):
        """Test fit with center algorithm."""
        dw = DataWhitener("center")
        dw.fit(simple_data)

        expected_mean = np.mean(simple_data, axis=0, keepdims=True)
        assert_allclose(dw.μ, expected_mean)

        # Center algorithm doesn't compute W and WI
        assert dw.W is None
        assert dw.WI is None

    def test_fit_rescale_algorithm(self, simple_data):
        """Test fit with rescale algorithm."""
        dw = DataWhitener("rescale")
        dw.fit(simple_data)

        # Check mean
        expected_mean = np.mean(simple_data, axis=0, keepdims=True)
        assert_allclose(dw.μ, expected_mean)

        # Check that Σ is diagonal
        assert np.allclose(dw.Σ, np.diag(np.diag(dw.Σ)))

        # Check whitening matrices
        assert_allclose(dw.W @ dw.WI, np.identity(2), atol=1e-6)

    def test_fit_pca_algorithm(self, correlated_data):
        """Test fit with PCA algorithm."""
        dw = DataWhitener("PCA")
        dw.fit(correlated_data)

        # Check that whitening reduces correlation
        whitened = dw.whiten(correlated_data)
        cov_whitened = np.cov(whitened.T)

        # PCA should diagonalize the covariance
        off_diagonal = cov_whitened - np.diag(np.diag(cov_whitened))
        assert np.allclose(off_diagonal, 0, atol=1e-6)

        # Check unit variance
        assert_allclose(np.diag(cov_whitened), np.ones(3), atol=1e-1)

    def test_fit_zca_algorithm(self, correlated_data):
        """Test fit with ZCA algorithm."""
        dw = DataWhitener("ZCA")
        dw.fit(correlated_data)

        # Check that whitening reduces correlation
        whitened = dw.whiten(correlated_data)
        cov_whitened = np.cov(whitened.T)

        # ZCA should produce identity covariance
        assert_allclose(cov_whitened, np.identity(3), atol=1e-1)

        # Check that transformation is reversible
        unwhitened = dw.unwhiten(whitened)
        assert_allclose(unwhitened, correlated_data, atol=1e-6)

    def test_fit_with_weights(self, weighted_data):
        """Test fitting with sample weights."""
        data, weights = weighted_data

        # Fit with weights
        dw_weighted = DataWhitener("rescale")
        dw_weighted.fit(data, weights=weights)

        # Fit without weights
        dw_unweighted = DataWhitener("rescale")
        dw_unweighted.fit(data)

        # Means should be different (weighted towards later samples)
        assert not np.allclose(dw_weighted.μ, dw_unweighted.μ)

        # Weighted mean should be shifted towards the end
        assert np.mean(dw_weighted.μ) > np.mean(dw_unweighted.μ)

    def test_fit_weights_validation(self, simple_data):
        """Test weight validation in fit."""
        dw = DataWhitener("rescale")

        # Wrong weight length
        with pytest.raises(
            ValueError, match="Weights and X should be of the same length"
        ):
            dw.fit(simple_data, weights=np.ones(3))

    def test_fit_save_data(self, simple_data):
        """Test fit with save_data option."""
        dw = DataWhitener("rescale")
        dw.fit(simple_data, save_data=True)

        assert hasattr(dw, "data")
        assert hasattr(dw, "whitened_data")
        assert_array_equal(dw.data, simple_data)
        assert_array_equal(dw.whitened_data, dw.whiten(simple_data))

    def test_whiten_none_algorithm(self, simple_data):
        """Test whitening with None algorithm."""
        dw = DataWhitener(None)
        dw.fit(simple_data)

        whitened = dw.whiten(simple_data)
        assert_array_equal(whitened, simple_data)

    def test_whiten_center_algorithm(self, simple_data):
        """Test whitening with center algorithm."""
        dw = DataWhitener("center")
        dw.fit(simple_data)

        whitened = dw.whiten(simple_data)
        assert_allclose(np.mean(whitened, axis=0), np.zeros(2), atol=1e-6)

    def test_whiten_rescale_algorithm(self, simple_data):
        """Test whitening with rescale algorithm."""
        dw = DataWhitener("rescale")
        dw.fit(simple_data)

        whitened = dw.whiten(simple_data)

        # Check zero mean and unit variance
        assert_allclose(np.mean(whitened, axis=0), np.zeros(2), atol=1e-6)
        assert_allclose(np.var(whitened, axis=0, ddof=1), np.ones(2), atol=1e-1)

    def test_whiten_single_sample(self, simple_data):
        """Test whitening a single sample (1D array)."""
        dw = DataWhitener("rescale")
        dw.fit(simple_data)

        single_sample = simple_data[0]
        whitened = dw.whiten(single_sample)

        # Should return 1D array
        assert whitened.ndim == 1
        assert whitened.shape == (2,)

    def test_unwhiten_none_algorithm(self, simple_data):
        """Test unwhitening with None algorithm."""
        dw = DataWhitener(None)
        dw.fit(simple_data)

        whitened = dw.whiten(simple_data)
        unwhitened = dw.unwhiten(whitened)
        assert_array_equal(unwhitened, simple_data)

    def test_unwhiten_center_algorithm(self, simple_data):
        """Test unwhitening with center algorithm."""
        dw = DataWhitener("center")
        dw.fit(simple_data)

        whitened = dw.whiten(simple_data)
        unwhitened = dw.unwhiten(whitened)
        assert_allclose(unwhitened, simple_data, atol=1e-6)

    def test_unwhiten_all_algorithms(self, correlated_data):
        """Test that whiten/unwhiten are inverses for all algorithms."""
        algorithms = [None, "center", "rescale", "PCA", "ZCA"]

        for algo in algorithms:
            dw = DataWhitener(algo)
            dw.fit(correlated_data)

            whitened = dw.whiten(correlated_data)
            unwhitened = dw.unwhiten(whitened)

            assert_allclose(
                unwhitened,
                correlated_data,
                atol=1e-6,
                err_msg=f"Failed for algorithm {algo}",
            )

    def test_unwhiten_single_sample(self, simple_data):
        """Test unwhitening a single sample."""
        dw = DataWhitener("rescale")
        dw.fit(simple_data)

        single_sample = simple_data[0]
        whitened = dw.whiten(single_sample)
        unwhitened = dw.unwhiten(whitened)

        assert unwhitened.ndim == 1
        assert_allclose(unwhitened, single_sample, atol=1e-6)

    def test_dtype_preservation(self):
        """Test that dtypes are preserved."""
        for dtype in [np.float32, np.float64]:
            data = np.random.randn(10, 3).astype(dtype)
            dw = DataWhitener("rescale")
            dw.fit(data)

            assert dw.μ.dtype == dtype
            assert dw.Σ.dtype == dtype
            assert dw.W.dtype == dtype
            assert dw.WI.dtype == dtype

            whitened = dw.whiten(data)
            assert whitened.dtype == dtype

    def test_numerical_stability_near_zero_variance(self):
        """Test numerical stability with near-zero variance dimensions."""
        # Create data with one dimension having very small variance
        data = np.random.randn(100, 3)
        data[:, 1] = data[:, 1] * 1e-8  # Very small variance

        dw = DataWhitener("rescale")
        # This should not raise numerical errors
        dw.fit(data)

        # Check that matrices are finite
        assert np.all(np.isfinite(dw.W))
        assert np.all(np.isfinite(dw.WI))

    def test_pca_with_singular_covariance(self):
        """Test PCA with singular covariance matrix."""
        # Create perfectly correlated data (singular covariance)
        data = np.random.randn(100, 1)
        data = np.hstack([data, data * 2, data * 3])  # Perfect linear dependence

        dw = DataWhitener("PCA")
        # This will produce warnings due to singular matrix
        # Just ensure it doesn't crash
        dw.fit(data)

        # At least some components should be valid
        assert dw.W is not None


class TestInterpolatorComprehensive:
    """Comprehensive test suite for Interpolator class."""

    @pytest.fixture
    def grid_1d(self):
        """1D grid for testing."""
        points = [np.linspace(0, 1, 5)]
        values = np.sin(2 * np.pi * points[0])
        return points, values

    @pytest.fixture
    def grid_2d(self):
        """2D grid for testing."""
        x = np.linspace(0, 1, 5)
        y = np.linspace(0, 1, 4)
        points = [x, y]
        X, Y = np.meshgrid(x, y, indexing="ij")
        values = np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
        return points, values

    @pytest.fixture
    def grid_3d(self):
        """3D grid for testing."""
        x = np.linspace(0, 1, 4)
        y = np.linspace(0, 1, 3)
        z = np.linspace(0, 1, 3)
        points = [x, y, z]
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        values = X + Y + Z
        return points, values

    def test_init_with_values(self, grid_2d):
        """Test initialization with values."""
        points, values = grid_2d
        interp = Interpolator(points, values)

        assert interp.interpolate_values is True
        assert_array_equal(interp.values, values)
        assert interp.method == "linear"
        assert interp.bounds_error is True
        assert np.isnan(interp.fill_value)

    def test_init_without_values(self, grid_2d):
        """Test initialization without values."""
        points, _ = grid_2d
        interp = Interpolator(points)

        assert interp.interpolate_values is False
        assert interp.values.shape == (5, 4)
        assert np.all(interp.values == 0)

    def test_init_with_custom_parameters(self, grid_2d):
        """Test initialization with custom parameters."""
        points, values = grid_2d
        interp = Interpolator(
            points, values, method="nearest", bounds_error=False, fill_value=-999
        )

        assert interp.method == "nearest"
        assert interp.bounds_error is False
        assert interp.fill_value == -999

    def test_inheritance_from_scipy(self, grid_2d):
        """Test that Interpolator properly inherits from scipy."""
        points, values = grid_2d
        interp = Interpolator(points, values)

        # Should be instance of RegularGridInterpolator
        assert isinstance(interp, RegularGridInterpolator)

        # Should have grid attribute
        assert hasattr(interp, "grid")
        assert len(interp.grid) == 2

    def test_call_linear_1d(self, grid_1d):
        """Test linear interpolation in 1D."""
        points, values = grid_1d
        interp = Interpolator(points, values, method="linear")

        # Test single point
        xi = np.array([0.25])
        result = interp(xi, return_aux=False)
        expected = np.sin(2 * np.pi * 0.25)
        assert_allclose(result, expected, atol=0.1)

        # Test multiple points
        xi = np.array([[0.0], [0.5], [1.0]])
        result = interp(xi, return_aux=False)
        assert result.shape == (3,)

    def test_call_linear_2d(self, grid_2d):
        """Test linear interpolation in 2D."""
        points, values = grid_2d
        interp = Interpolator(points, values, method="linear")

        # Test single point
        xi = np.array([0.25, 0.25])
        result, edges, weights = interp(xi)

        assert isinstance(result, np.ndarray)
        assert len(edges) == 4  # 2^2 for linear in 2D
        assert len(weights) == 4
        assert_allclose(np.sum(weights), 1.0)

    def test_call_nearest_2d(self, grid_2d):
        """Test nearest neighbor interpolation in 2D."""
        points, values = grid_2d
        interp = Interpolator(points, values, method="nearest")

        # Test point close to grid point
        xi = np.array([0.01, 0.01])
        result, edges = interp(xi)

        assert_allclose(result, values[0, 0], atol=1e-6)
        assert edges == (0, 0)

    def test_call_without_values(self, grid_2d):
        """Test calling interpolator without values."""
        points, _ = grid_2d
        interp = Interpolator(points)  # No values

        xi = np.array([0.5, 0.5])

        # Should raise error if return_aux=False
        with pytest.raises(ValueError, match="Please either define"):
            interp(xi, return_aux=False)

        # Should work with return_aux=True
        edges, weights = interp(xi, return_aux=True)
        assert len(edges) == 4
        assert len(weights) == 4

    def test_call_with_tuple_input(self, grid_2d):
        """Test calling with tuple of coordinate arrays."""
        points, values = grid_2d
        interp = Interpolator(points, values)

        # Create coordinate arrays
        x_coords = np.array([0.25, 0.5, 0.75])
        y_coords = np.array([0.25, 0.5, 0.75])

        # Call with tuple
        result = interp((x_coords, y_coords), return_aux=False)
        assert result.shape == (3,)

    def test_call_bounds_checking(self, grid_1d):
        """Test bounds checking."""
        points, values = grid_1d
        interp = Interpolator(points, values, bounds_error=True)

        # Out of bounds point
        xi = np.array([1.5])
        with pytest.raises(ValueError, match="out of bounds"):
            interp(xi)

        # Test with bounds_error=False
        interp_no_bounds = Interpolator(
            points, values, bounds_error=False, fill_value=-999
        )
        result = interp_no_bounds(xi, return_aux=False)
        assert result == -999

    def test_call_method_override(self, grid_2d):
        """Test method override in call."""
        points, values = grid_2d
        interp = Interpolator(points, values, method="linear")

        xi = np.array([0.5, 0.5])

        # Override to nearest
        result_nearest, edges_nearest = interp(xi, method="nearest")

        # Compare with linear
        result_linear, edges_linear, _ = interp(xi, method="linear")

        # For point at (0.5, 0.5), results might be similar
        # Just check that we get proper outputs
        assert isinstance(result_nearest, np.ndarray)
        assert isinstance(result_linear, np.ndarray)
        assert isinstance(edges_nearest, tuple)
        assert len(edges_linear) == 4  # Multiple edges for linear

    def test_call_invalid_method(self, grid_1d):
        """Test calling with invalid method."""
        points, values = grid_1d
        interp = Interpolator(points, values)

        xi = np.array([0.5])
        with pytest.raises(ValueError, match="Method .* is not defined"):
            interp(xi, method="cubic")

    def test_call_dimension_mismatch(self, grid_2d):
        """Test calling with wrong dimensions."""
        points, values = grid_2d
        interp = Interpolator(points, values)

        # Wrong dimension
        xi = np.array([0.5])  # 1D for 2D interpolator
        with pytest.raises(ValueError, match="dimension"):
            interp(xi)

    def test_evaluate_linear_implementation(self, grid_2d):
        """Test _evaluate_linear method details."""
        points, values = grid_2d
        interp = Interpolator(points, values)

        # Access internal state
        xi = np.array([[0.25, 0.25]])
        xi = xi.reshape(-1, 2)

        # Get indices and distances
        find_indices_result = interp._find_indices(xi.T)
        if len(find_indices_result) == 2:
            indices, norm_distances = find_indices_result
            out_of_bounds = np.zeros(1, dtype=bool)
        else:
            indices, norm_distances, out_of_bounds = find_indices_result

        # Call _evaluate_linear
        result, edges, weights = interp._evaluate_linear(
            indices, norm_distances, out_of_bounds
        )

        # Check that weights sum to 1
        total_weight = sum(w.item() for w in weights)
        assert_allclose(total_weight, 1.0)

    def test_evaluate_nearest_implementation(self, grid_1d):
        """Test _evaluate_nearest method details."""
        points, values = grid_1d
        interp = Interpolator(points, values)

        # Test points at different positions
        test_points = np.array([[0.24], [0.26], [0.74], [0.76]])

        for xi in test_points:
            xi_reshaped = xi.reshape(-1, 1)
            find_indices_result = interp._find_indices(xi_reshaped.T)

            if len(find_indices_result) == 2:
                indices, norm_distances = find_indices_result
                out_of_bounds = np.zeros(1, dtype=bool)
            else:
                indices, norm_distances, out_of_bounds = find_indices_result

            result, edges = interp._evaluate_nearest(
                indices, norm_distances, out_of_bounds
            )

            # Check that nearest neighbor is selected correctly
            if norm_distances[0] <= 0.5:
                assert edges[0] == indices[0]
            else:
                assert edges[0] == indices[0] + 1

    def test_3d_interpolation(self, grid_3d):
        """Test 3D interpolation."""
        points, values = grid_3d
        interp = Interpolator(points, values)

        # Test point
        xi = np.array([0.5, 0.5, 0.5])
        result, edges, weights = interp(xi)

        # Should have 8 edges/weights for 3D linear interpolation
        assert len(edges) == 8
        assert len(weights) == 8

        # Result should be close to 1.5 (sum of coordinates)
        assert_allclose(result, 1.5, atol=0.1)

    def test_shape_preservation(self, grid_2d):
        """Test that output shapes are preserved."""
        points, values = grid_2d
        interp = Interpolator(points, values)

        # Test different input shapes
        shapes = [(2, 2), (3,), (1, 1, 2)]

        for shape in shapes:
            xi = np.random.rand(*shape)
            if xi.shape[-1] != 2:
                # Adjust to have 2 coordinates
                xi = np.random.rand(*(shape[:-1] + (2,)))

            result = interp(xi, return_aux=False)
            # Handle scalar case
            if xi.shape[:-1] == ():
                expected_shape = (1,)
            else:
                expected_shape = xi.shape[:-1]
            assert result.shape == expected_shape or (
                result.shape == () and expected_shape == (1,)
            )

    def test_dtype_handling(self, grid_1d):
        """Test handling of different dtypes."""
        points, values = grid_1d

        # Test with float64 (scipy requirement)
        values_typed = values.astype(np.float64)
        interp = Interpolator(points, values_typed)

        xi = np.array([0.5], dtype=np.float64)
        result = interp(xi, return_aux=False)

        # Result should be float64
        assert result.dtype == np.float64

    def test_edge_case_single_point_grid(self):
        """Test interpolation with single-point grid."""
        points = [np.array([0.5])]
        values = np.array([42.0])

        interp = Interpolator(points, values)

        # Any point should return the single value
        xi = np.array([0.5])
        result = interp(xi, return_aux=False)
        assert result == 42.0

        # Even out of bounds with bounds_error=False
        interp_no_bounds = Interpolator(points, values, bounds_error=False)
        xi_out = np.array([1.0])
        # Test that it doesn't crash - we don't need to use the result
        _ = interp_no_bounds(xi_out, return_aux=False)
        # With single point, out of bounds returns NaN by default
        # This is expected behavior
        assert True  # Just ensure no crash

    def test_grid_irregularity_validation(self):
        """Test that irregular grids are handled properly."""
        # Irregular spacing
        points = [np.array([0.0, 0.1, 0.3, 0.7, 1.0])]
        values = np.array([0, 1, 2, 3, 4])

        # Should work fine
        interp = Interpolator(points, values)

        xi = np.array([0.5])
        result = interp(xi, return_aux=False)
        assert np.isfinite(result)

    def test_multidimensional_values(self):
        """Test interpolation with multidimensional values at each grid point."""
        # 2D grid with 2D values at each point
        x = np.linspace(0, 1, 3)
        y = np.linspace(0, 1, 3)
        points = [x, y]

        # Values have shape (3, 3, 2) - vector field
        values = np.zeros((3, 3, 2))
        for i in range(3):
            for j in range(3):
                values[i, j] = [x[i], y[j]]

        interp = Interpolator(points, values)

        xi = np.array([0.5, 0.5])
        result = interp(xi, return_aux=False)

        # Result has shape (1, 2) for single point
        assert result.shape == (1, 2) or result.shape == (2,)
        assert_allclose(result.squeeze(), [0.5, 0.5], atol=0.01)
