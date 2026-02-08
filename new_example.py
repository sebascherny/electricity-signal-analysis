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
    analyzer = ElectricitySignalAnalyzer(plot_windows=True, plots_must_block=True)
    
    logger.info("=== ElectricitySignalAnalyzer Demo ===\n")
    
    # 1. Load the default file
    logger.info("1. Loading default file...")
    data = analyzer.load_file("DG_POI_DG_Disconnect_2021-02-22a.csv")
    if data is None:
        logger.error("Failed to load data. Exiting.")
        return
    
    # 2. Choose column to use (default is Report_1_IB)
    logger.info("\n2. Choosing column to analyze...")
    analyzer.choose_column_to_use("Report_1_IAX_A")
    
    # 3. Plot the entire signal
    logger.info("\n3. Plotting entire signal...")
    analyzer.plot_entire_signal()

    # 6. Iterate through multiple windows with different n_exponents
    logger.info("\n6. Iterating through windows (first 3 windows only for demo)...")
    
    # Limit to first 120 samples for demo (4 windows of size 30)
    original_data = analyzer.data.copy()
    # analyzer.data = analyzer.data.iloc[:120].copy()
    
    results = analyzer.iterate_through_windows(
        window_size=32, 
        step_size=32,  # Windows will overlap: 0-240, 120-360, 240-480, etc.
        n_exponents_list=[2, 4, 6, 8, 10],
        plot_windows=True,
        save_plots=True,
        max_windows=8
    )

    # Only one wider window
    analyzer.iterate_through_windows(
        window_size=2*32,
        step_size=1,
        n_exponents_list=[2, 4, 6, 8, 10],
        max_windows=1,
        start_index=15,
        end_index=15+2*32
    )
    
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
