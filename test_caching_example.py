#!/usr/bin/env python3
"""
Example demonstrating the caching functionality of ElectricitySignalAnalyzer.
"""

import numpy as np
import pandas as pd
from electricity_signal_analyzer import ElectricitySignalAnalyzer

def test_caching_functionality():
    """Test the window data caching functionality."""
    print("=== Testing ElectricitySignalAnalyzer Caching Functionality ===\n")
    
    # Create analyzer instance
    analyzer = ElectricitySignalAnalyzer()
    
    # Create synthetic test data
    np.random.seed(42)
    n_samples = 100
    microseconds = np.linspace(0, 1000, n_samples)
    signal = np.sin(2 * np.pi * 0.01 * microseconds) + 0.1 * np.random.randn(n_samples)
    
    test_data = pd.DataFrame({
        'Microseconds': microseconds,
        'Report_1_IB': signal
    })
    
    analyzer.data = test_data
    analyzer.column_to_use = 'Report_1_IB'
    
    print("1. Running window analysis to populate cache...")
    
    # Run analysis on a few windows
    results = analyzer.iterate_through_windows(
        window_size=30, 
        step_size=15,
        n_exponents_list=[2, 4], 
        plot_windows=False,
        save_plots=False,
        max_windows=3
    )
    
    print(f"✓ Processed {len(results)} windows\n")
    
    print("2. Testing cache retrieval...")
    
    # Test retrieving specific window data
    cached_data = analyzer.get_cached_window_data(0, 30, 2)
    if cached_data:
        print("✓ Retrieved cached data for window [0:30] with 2 exponents:")
        print(f"  - Singular values: {len(cached_data['singular_values'])} values")
        print(f"  - Nodes: {len(cached_data['nodes'])} nodes")
        print(f"  - Coefficients: {len(cached_data['coeffs'])} coeffs")
        print(f"  - Normalized L2 error: {cached_data['error_l2']:.6f}")
        print(f"  - Approximation length: {len(cached_data['approximation'])}")
        print(f"  - Error infinity shape: {np.array(cached_data['error_infinity']).shape}")
    else:
        print("✗ Failed to retrieve cached data")
    
    print("\n3. Testing cache summary...")
    
    # Get summary of all cached windows
    summary = analyzer.get_cached_windows_summary()
    print(f"✓ Cache contains {len(summary)} window analyses:")
    for item in summary:
        print(f"  - Window {item['window']}, {item['n_exponents']} exp: "
              f"L2={item['error_l2_normalized']:.6f}, "
              f"SV={item['num_singular_values']}, "
              f"size={item['window_size']}")
    
    print("\n4. Testing specific data access...")
    
    # Test accessing specific data components
    window_data = analyzer.get_cached_window_data(15, 45, 4)
    if window_data:
        print("✓ Window [15:45] with 4 exponents data:")
        print(f"  - First 5 singular values: {window_data['singular_values'][:5]}")
        print(f"  - First node: {window_data['nodes'][0]}")
        print(f"  - First coefficient: {window_data['coeffs'][0]}")
        print(f"  - Max infinity error: {np.max(np.abs(window_data['error_infinity'])):.6f}")
    
    print("\n5. Testing cache management...")
    
    # Get all cached data
    all_cached = analyzer.get_all_cached_windows()
    print(f"✓ Total cached windows: {len(all_cached)}")
    
    # Clear cache
    analyzer.clear_cached_data()
    
    # Verify cache is empty
    summary_after_clear = analyzer.get_cached_windows_summary()
    print(f"✓ Cache after clearing: {len(summary_after_clear)} items")
    
    print("\n=== Caching Functionality Test Completed ===")
    
    # Demonstrate usage patterns
    print("\n6. Usage Examples:")
    print("# Retrieve specific window data:")
    print("data = analyzer.get_cached_window_data(start_idx, end_idx, n_exponents)")
    print("singular_values = data['singular_values']")
    print("nodes = data['nodes']")
    print("coeffs = data['coeffs']")
    print("error_l2 = data['error_l2']  # Normalized by window size")
    print("error_infinity = data['error_infinity']")
    print("approximation = data['approximation']")
    print()
    print("# Get summary of all cached windows:")
    print("summary = analyzer.get_cached_windows_summary()")
    print()
    print("# Clear all cached data:")
    print("analyzer.clear_cached_data()")

if __name__ == "__main__":
    test_caching_functionality()
