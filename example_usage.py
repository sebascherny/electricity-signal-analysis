#!/usr/bin/env python3
"""
Example usage of the ElectricitySignalAnalyzer class.
This script demonstrates how to use all the methods of the class.
"""

from electricity_signal_analyzer import ElectricitySignalAnalyzer
import logging

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # Create an instance of the analyzer
    analyzer = ElectricitySignalAnalyzer()
    
    logger.info("=== ElectricitySignalAnalyzer Demo ===\n")
    
    # 1. Load the default file
    logger.info("1. Loading default file...")
    data = analyzer.load_file()
    if data is None:
        logger.error("Failed to load data. Exiting.")
        return
    
    # 2. Choose column to use (default is Report_1_IB)
    logger.info("\n2. Choosing column to analyze...")
    analyzer.choose_column_to_use("Report_1_IB")
    
    # 3. Plot the entire signal
    logger.info("\n3. Plotting entire signal...")
    analyzer.plot_entire_signal()
    
    # 4. Test approximate_sequence on a small window
    logger.info("\n4. Testing approximate_sequence on a small sample...")
    sample_signal = data['Report_1_IB'].iloc[0:21].values  # 21 samples for odd length
    nodes, coeffs, app, sig, error_2, error_inf = analyzer.approximate_sequence(
        sample_signal, n_exponents=2, verbose=True
    )
    logger.info("Sample approximation completed with L2 error: {:.6f}".format(error_2))
    
    # 5. Run simulation on a specific window
    logger.info("\n5. Running simulation on a specific window...")
    result = analyzer.run_simulation_in_window(0, 30, n_exponents=3, verbose=True)
    
    # 6. Iterate through multiple windows with different n_exponents
    logger.info("\n6. Iterating through windows (first 3 windows only for demo)...")
    
    # Limit to first 120 samples for demo (4 windows of size 30)
    original_data = analyzer.data.copy()
    # analyzer.data = analyzer.data.iloc[:120].copy()
    
    results = analyzer.iterate_through_windows(
        window_size=240, 
        step_size=120,  # Windows will overlap: 0-240, 120-360, 240-480, etc.
        n_exponents_list=[2, 4, 6], 
        plot_windows=True,
        save_plots=True  # This will save all plots to a timestamped folder
    )
    
    # Restore original data
    analyzer.data = original_data
    
    logger.info("\n=== Demo completed successfully! ===")
    
    # 7. Show summary statistics
    logger.info("\n7. Summary of results:")
    if results:
        for i, window_result in enumerate(results):
            if window_result:
                logger.info("\nWindow {}:".format(i+1))
                for n_exp, res in window_result.items():
                    logger.info("  {} exponents: L2 error = {:.6f}".format(n_exp, res['error_norma_2']))

if __name__ == "__main__":
    main()
