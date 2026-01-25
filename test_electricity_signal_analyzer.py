#!/usr/bin/env python3
"""
Pytest tests for ElectricitySignalAnalyzer class.
Tests focus on mathematical methods that can work with mock data.
"""

import pytest
import numpy as np
import pandas as pd
from electricity_signal_analyzer import ElectricitySignalAnalyzer


class TestElectricitySignalAnalyzer:
    """Test suite for ElectricitySignalAnalyzer mathematical methods."""
    
    @pytest.fixture
    def analyzer(self) -> ElectricitySignalAnalyzer:
        """Create an analyzer instance for testing."""
        return ElectricitySignalAnalyzer()
    
    @pytest.fixture
    def mock_data(self):
        """Create mock DataFrame for testing."""
        np.random.seed(42)  # For reproducible tests
        n_samples = 100
        microseconds = np.linspace(0, 1000, n_samples)
        signal = np.sin(2 * np.pi * 0.1 * microseconds) + 0.1 * np.random.randn(n_samples)
        
        return pd.DataFrame({
            'Microseconds': microseconds,
            'Report_1_IB': signal
        })
    
    def test_construct_hankel_odd_length(self, analyzer: ElectricitySignalAnalyzer):
        """Test Hankel matrix construction with odd-length input."""
        input_data = np.array([1, 2, 3, 4, 5])
        H = analyzer.construct_hankel(input_data)
        
        expected = np.array([
            [1, 2, 3],
            [2, 3, 4],
            [3, 4, 5]
        ])
        
        np.testing.assert_array_equal(H, expected)
        assert H.shape == (3, 3)
    
    def test_construct_hankel_even_length(self, analyzer: ElectricitySignalAnalyzer):
        """Test Hankel matrix construction with even-length input (should truncate)."""
        input_data = np.array([1, 2, 3, 4, 5, 6])
        H = analyzer.construct_hankel(input_data, verbose=True)
        
        # Should remove last element and work with [1, 2, 3, 4, 5]
        expected = np.array([
            [1, 2, 3],
            [2, 3, 4],
            [3, 4, 5]
        ])
        
        np.testing.assert_array_equal(H, expected)
        assert H.shape == (3, 3)
    
    def test_construct_hankel_small_input(self, analyzer: ElectricitySignalAnalyzer):
        """Test Hankel matrix with very small input."""
        input_data = np.array([1])
        H = analyzer.construct_hankel(input_data)
        
        expected = np.array([[1]])
        np.testing.assert_array_equal(H, expected)
        assert H.shape == (1, 1)
    
    def test_get_coneigen_real_matrix(self, analyzer: ElectricitySignalAnalyzer):
        """Test eigenvalue decomposition for real symmetric matrix."""
        # Create a simple symmetric positive definite matrix
        H = np.array([
            [4, 1, 2],
            [1, 3, 1],
            [2, 1, 5]
        ])
        
        sig, V = analyzer.get_coneigen(H, n_components=3, verbose=True, randomized=False)
        
        # Check that we get 3 singular values
        assert len(sig) == 3
        assert V.shape == (3, 3)
        
        # Singular values should be positive and sorted in descending order
        assert np.all(sig >= 0)
        assert np.all(sig[:-1] >= sig[1:])  # Descending order
    
    def test_get_coneigen_randomized(self, analyzer: ElectricitySignalAnalyzer):
        """Test randomized SVD option."""
        np.random.seed(42)
        H = np.random.randn(10, 10)
        H = H @ H.T  # Make it positive semidefinite
        
        sig, V = analyzer.get_coneigen(H, n_components=5, verbose=True, randomized=True)
        
        assert len(sig) == 5
        assert V.shape == (10, 5)
        assert np.all(sig >= 0)
    
    def test_get_nodes(self, analyzer: ElectricitySignalAnalyzer):
        """Test node extraction from eigenvectors."""
        # Create mock eigenvector matrix
        np.random.seed(42)
        U = np.random.randn(5, 3)
        
        nodes = analyzer.get_nodes(U, n_nodes=2)
        
        # Should return 2 complex nodes
        assert len(nodes) == 2
        assert nodes.dtype == np.complex128 or nodes.dtype == np.float64
    
    def test_construct_vandermonde_matrix(self, analyzer: ElectricitySignalAnalyzer):
        """Test Vandermonde matrix construction."""
        nodes = np.array([0.5, 0.8, 1.2])
        n_samples = 4
        
        V = analyzer.construct_vandermonde_matrix(nodes, n_samples)
        
        # Expected Vandermonde matrix: each row is nodes^k for k=0,1,2,3
        expected = np.array([
            [1.0, 1.0, 1.0],           # nodes^0
            [0.5, 0.8, 1.2],           # nodes^1
            [0.25, 0.64, 1.44],        # nodes^2
            [0.125, 0.512, 1.728]      # nodes^3
        ])
        
        np.testing.assert_array_almost_equal(V, expected, decimal=6)
        assert V.shape == (4, 3)
    
    def test_construct_vandermonde_matrix_with_first_power(self, analyzer: ElectricitySignalAnalyzer):
        """Test Vandermonde matrix with non-zero first power."""
        nodes = np.array([2.0, 3.0])
        n_samples = 3
        first_power = 1
        
        V = analyzer.construct_vandermonde_matrix(nodes, n_samples, first_power)
        
        # Should start from power 1: nodes^1, nodes^2, nodes^3
        expected = np.array([
            [2.0, 3.0],    # nodes^1
            [4.0, 9.0],    # nodes^2
            [8.0, 27.0]    # nodes^3
        ])
        
        np.testing.assert_array_almost_equal(V, expected)
        assert V.shape == (3, 2)
    
    def test_approximate_sequence_simple_signal(self, analyzer: ElectricitySignalAnalyzer):
        """Test sequence approximation with a simple synthetic signal."""
        # Create a simple exponential decay signal
        t = np.linspace(0, 1, 21)  # Odd length for Hankel
        signal = 2 * np.exp(-t) + 0.5 * np.exp(-2*t)
        
        nodes, coeffs, app, sig, error_2, error_inf = analyzer.approximate_sequence(
            signal, n_exponents=2, verbose=True
        )
        
        # Check output shapes and types
        assert len(nodes) == 2
        assert len(coeffs) == 2
        assert len(app) == len(signal)
        assert len(sig) >= 2
        assert isinstance(error_2, (float, np.floating))
        assert len(error_inf) == len(signal)
        
        # Approximation should be reasonably good for this simple case
        assert error_2 < 1.0  # L2 error should be small
        
        # Approximation should be real-valued
        assert np.all(np.isreal(app))
    
    def test_approximate_sequence_constant_signal(self, analyzer: ElectricitySignalAnalyzer):
        """Test approximation of a constant signal."""
        signal = np.ones(15)  # Constant signal
        
        nodes, coeffs, app, sig, error_2, error_inf = analyzer.approximate_sequence(
            signal, n_exponents=1, verbose=True
        )
        
        # Should approximate constant well with 1 exponent
        assert error_2 < 0.1
        np.testing.assert_array_almost_equal(app, signal, decimal=1)
    
    def test_approximate_sequence_noisy_signal(self, analyzer: ElectricitySignalAnalyzer):
        """Test approximation with noisy signal."""
        np.random.seed(42)
        t = np.linspace(0, 1, 25)
        clean_signal = np.exp(-t)
        noise = 0.01 * np.random.randn(len(t))
        noisy_signal = clean_signal + noise
        
        nodes, coeffs, app, sig, error_2, error_inf = analyzer.approximate_sequence(
            noisy_signal, n_exponents=1, verbose=False
        )
        
        # Should still provide reasonable approximation
        assert len(app) == len(noisy_signal)
        assert error_2 < 1.0  # Should be reasonable despite noise
    
    def test_run_simulation_in_window_with_mock_data(self, analyzer: ElectricitySignalAnalyzer, mock_data):
        """Test window simulation with mock DataFrame."""
        analyzer.data = mock_data
        analyzer.column_to_use = 'Report_1_IB'
        
        result = analyzer.run_simulation_in_window(0, 30, n_exponents=2, verbose=False)
        
        # Check that result contains expected keys
        expected_keys = ['nodes', 'coeffs', 'approximation', 'sig', 'error_norma_2', 
                        'error_norma_inf', 'window_signal', 'start_index', 'end_index', 'n_exponents']
        
        assert result is not None
        for key in expected_keys:
            assert key in result
        
        # Check data types and shapes
        assert len(result['nodes']) == 2
        assert len(result['coeffs']) == 2
        assert len(result['approximation']) == 30
        assert len(result['window_signal']) == 30
        assert result['start_index'] == 0
        assert result['end_index'] == 30
        assert result['n_exponents'] == 2
        assert isinstance(result['error_norma_2'], (float, np.floating))
    
    def test_run_simulation_in_window_small_window(self, analyzer: ElectricitySignalAnalyzer, mock_data):
        """Test simulation with window too small."""
        analyzer.data = mock_data
        analyzer.column_to_use = 'Report_1_IB'
        
        # Window with only 2 samples should fail
        result = analyzer.run_simulation_in_window(0, 2, n_exponents=2, verbose=False)
        
        assert result is None
    
    def test_run_simulation_in_window_no_data(self, analyzer: ElectricitySignalAnalyzer):
        """Test simulation when no data is loaded."""
        result = analyzer.run_simulation_in_window(0, 30, n_exponents=2, verbose=False)
        assert result is None
    
    def test_run_simulation_in_window_invalid_column(self, analyzer: ElectricitySignalAnalyzer, mock_data):
        """Test simulation with invalid column name."""
        analyzer.data = mock_data
        analyzer.column_to_use = 'NonExistentColumn'
        
        result = analyzer.run_simulation_in_window(0, 30, n_exponents=2, verbose=False)
        assert result is None


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    @pytest.fixture
    def analyzer(self):
        return ElectricitySignalAnalyzer()
    
    def test_hankel_empty_input(self, analyzer: ElectricitySignalAnalyzer):
        """Test Hankel construction with empty input."""
        result = analyzer.construct_hankel(np.array([]))
        # Empty input should return empty matrix
        assert result.shape == (0, 0)
    
    def test_vandermonde_empty_nodes(self, analyzer: ElectricitySignalAnalyzer):
        """Test Vandermonde matrix with empty nodes."""
        V = analyzer.construct_vandermonde_matrix(np.array([]), 3)
        assert V.shape == (3, 0)
    
    def test_approximate_sequence_too_few_samples(self, analyzer: ElectricitySignalAnalyzer):
        """Test approximation with very few samples."""
        signal = np.array([1.0])  # Single sample
        
        # Should handle gracefully or raise appropriate error
        try:
            result = analyzer.approximate_sequence(signal, n_exponents=1, verbose=False)
            # If it succeeds, check basic properties
            nodes, coeffs, app, sig, error_2, error_inf = result
            assert len(app) == 1
        except (ValueError, np.linalg.LinAlgError):
            # It's acceptable to fail with very small inputs
            pass
    
    def test_get_nodes_insufficient_components(self, analyzer: ElectricitySignalAnalyzer):
        """Test node extraction when asking for more nodes than available."""
        U = np.random.randn(3, 2)  # Only 2 components available
        
        # Asking for 3 nodes when only 2 components available
        nodes = analyzer.get_nodes(U, n_nodes=3)
        # Should handle gracefully, possibly returning fewer nodes or raising error
        assert len(nodes) <= 3


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])
