import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import linalg as LA
from sklearn.utils.extmath import randomized_svd
import os
from datetime import datetime
import logging

logger = logging.getLogger(__name__)
# log in console and file
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', handlers=[
    logging.FileHandler('electricity_signal_analyzer.log'),
    logging.StreamHandler()
])

plt.ioff()



class ElectricitySignalAnalyzer:
    """
    A class for analyzing electricity signals using exponential approximation methods.
    """
    
    def __init__(
        self,
        folder=None,
        filename="128sample_Event.csv",
        column_to_use="Report_1_IB",
    ):
        self.data = None
        self.column_to_use = column_to_use
        self.default_file = None
        if folder and filename:
            self.default_file = os.path.join(folder, filename)
        else:
            if not (folder and os.path.exists(folder) and os.path.isdir(folder)):
                folder = os.getcwd()
            csv_files = [f for f in os.listdir(folder) if f.endswith('.csv')]
            if csv_files:
                if filename and os.path.exists(os.path.join(folder, filename)):
                    self.default_file = os.path.join(folder, filename)
                else:
                    self.default_file = os.path.join(folder, csv_files[0])
        
        # Data storage for window analysis results
        self.cached_results = {}  # Dictionary to store results by window key
        if self.default_file:
            self.load_file()
    
    def load_file(self, file_path=None):
        """
        Load a CSV file containing electricity signal data.
        
        :param file_path: Path to the CSV file. If None, uses default file.
        :return: Loaded DataFrame
        """
        if file_path is None:
            file_path = self.default_file
        
        try:
            self.data = pd.read_csv(file_path)
            logger.info("Successfully loaded file: {}".format(file_path))
            logger.info("Data shape: {}".format(self.data.shape))
            logger.info("Columns: {}".format(list(self.data.columns)))
            return self.data
        except Exception as e:
            logger.exception("Error loading file {}: {}".format(file_path, e))
            return None
    
    def choose_column_to_use(self, column_name="Report_1_IB"):
        """
        Choose which column to use for analysis.
        
        :param column_name: Name of the column to use for analysis
        """
        if self.data is None:
            logger.error("No data loaded. Please load a file first.")
            return
        
        if column_name not in self.data.columns:
            logger.error("Column '{}' not found in data. Available columns: {}".format(
                column_name, list(self.data.columns)))
            return
        
        self.column_to_use = column_name
        logger.info("Selected column: {}".format(self.column_to_use))
    
    def plot_entire_signal(self):
        """
        Plot the entire signal using the chosen column as y-axis and Microseconds as x-axis.
        """
        if self.data is None:
            logger.error("No data loaded. Please load a file first.")
            return
        
        if self.column_to_use not in self.data.columns:
            logger.error("Column '{}' not found in data.".format(self.column_to_use))
            return
        
        if 'Microseconds' not in self.data.columns:
            logger.error("Microseconds column not found in data.")
            return
        
        plt.figure(figsize=(12, 6))
        plt.plot(self.data['Microseconds'], self.data[self.column_to_use], 'b-', linewidth=1)
        plt.xlabel('Microseconds')
        plt.ylabel(self.column_to_use)
        plt.title('Entire Signal: {} vs Microseconds'.format(self.column_to_use))
        plt.grid(True, alpha=0.3)
        plt.show(block=False)
    
    def construct_hankel(self, input_data, verbose=False):
        """
        Construct Hankel matrix from input data.
        """
        if input_data.size % 2 != 1:
            if verbose:
                logger.warning('Hankel matrix requires odd length signal, last entry is removed')
            input_data = input_data[0:-1]

        n = (input_data.size + 1) // 2
        H = np.zeros((n, n))

        for i in range(n):
            H[i,:] = input_data[i:(n + i)]

        return H
    
    def get_coneigen(self, H, n_components, verbose=False, randomized=False):
        """
        Get singular values and vectors from Hankel matrix.
        """
        if randomized:
            if H.shape[0] > n_components and randomized:
                if verbose:
                    logger.info("Hankel matrix: using randomized")
                _, sig, Vt = randomized_svd(H, n_components, random_state=None)
            else:
                if verbose:
                    logger.info('Hankel matrix: using svd')
                _, sig, Vt = LA.svd(H)
            V = Vt.T
        else:
            if np.any(np.iscomplex(H)):
                if verbose:
                    logger.info('Complex Hankel matrix: using svd')
                _, sig, Vt = LA.svd(H)
                V = Vt.T
            else:
                if verbose:
                    logger.info('Real Hankel matrix: using eigh')
                eig, V = LA.eigh(H)
                sig = np.abs(eig)
                order = np.argsort(sig)[::-1]
                sig = sig[order]
                V = V[:,order]
                eig_mat = np.diag(eig[order])[0:3,0:3]
                Vd = V[:,0:3]
                diff = np.dot(H, Vd) - np.dot(Vd, eig_mat)
                if verbose and LA.norm(diff)>10**(-10):
                    logger.info('Errors in SVD for dominant vectors: ', diff)
        
        diff_norm = LA.norm((abs(np.dot(np.dot(V.T, H), V)) - np.diag(sig)))
        if verbose and diff_norm > 10.0**(-10):
           logger.warning('WARNING. Error in SVD: ', diff_norm)
        return sig, V
    
    def get_nodes(self, U, n_nodes):
        """
        Get nodes from eigenvectors.
        """
        U1 = U[:-1,:n_nodes]
        U2 = U[1:,:n_nodes]
        U3 = np.dot(LA.pinv(U1, rcond=1e-12), U2)
        nodes = LA.eigvals(U3)
        return nodes
    
    def construct_vandermonde_matrix(self, nodes, n_samples, first_power=0):
        """
        Construct Vandermonde matrix.
        """
        V = np.vstack([nodes**k for k in range(first_power, n_samples + first_power)])
        return V
    
    def approximate_sequence(self, signal, n_exponents, verbose=False):
        """
        Approximate a signal using exponential functions.
        
        :param signal: Input signal to approximate
        :param n_exponents: Number of exponential terms to use
        :param verbose: Print debug information
        :return: nodes, coeffs, app, sig, error_norma_2, error_norma_inf
        """
        n_samples = signal.size
        H = self.construct_hankel(signal, verbose=verbose)
        sig, U = self.get_coneigen(H, n_exponents, verbose=verbose)
        nodes = self.get_nodes(U, n_exponents)
        V = self.construct_vandermonde_matrix(nodes, n_samples)
        V_pinv = LA.pinv(V)
        coeffs = np.dot(V_pinv, signal)
        app = np.dot(V, coeffs)
        app = app.real
        
        difference = signal - app
        difference_without_small_values = np.array([
            D if np.abs(D) >= 1 else 1 for D in difference
        ])
        error_norma_inf = difference / difference_without_small_values
        error_norma_2 = np.linalg.norm(difference)
        
        return nodes, coeffs, app, sig, error_norma_2, error_norma_inf
    
    def run_simulation_in_window(self, start_index, end_index, n_exponents=2, verbose=True):
        """
        Run approximation simulation on a window of data.
        
        :param start_index: Starting index of the window
        :param end_index: Ending index of the window
        :param n_exponents: Number of exponential terms to use
        :param verbose: Print results
        :return: Dictionary with results
        """
        if self.data is None:
            logger.error("No data loaded. Please load a file first.")
            return None
        
        if self.column_to_use not in self.data.columns:
            logger.error("Column '{}' not found in data.".format(self.column_to_use))
            return None
        
        # Extract window data
        window_signal = self.data[self.column_to_use].iloc[start_index:end_index].values
        
        if len(window_signal) < 3:
            logger.error("Window too small: {} samples".format(len(window_signal)))
            return None
        
        try:
            nodes, coeffs, app, sig, error_norma_2, error_norma_inf = self.approximate_sequence(
                window_signal, n_exponents, verbose=False
            )
            
            # Normalize L2 error by window size as specified
            window_size = end_index - start_index
            normalized_error_l2 = error_norma_2 / window_size
            
            # Limit singular values to maximum 30 as specified
            sig_limited = sig[:30] if len(sig) > 30 else sig
            
            results = {
                'nodes': nodes,
                'coeffs': coeffs,
                'approximation': app,
                'sig': sig_limited,
                'error_norma_2': error_norma_2,
                'error_norma_inf': error_norma_inf,
                'window_signal': window_signal,
                'start_index': start_index,
                'end_index': end_index,
                'n_exponents': n_exponents,
                'normalized_error_l2': normalized_error_l2
            }
            
            # Store results in cache with window key
            window_key = (start_index, end_index, n_exponents)
            cached_data = {
                'singular_values': sig_limited,
                'nodes': nodes,
                'coeffs': coeffs,
                'error_l2': normalized_error_l2,
                'error_infinity': error_norma_inf,
                'approximation': app
            }
            self.cached_results[window_key] = cached_data
            
            if verbose:
                logger.info("\n=== Window [{}:{}] with {} exponents ===".format(
                    start_index, end_index, n_exponents))
                logger.info("Nodes: {}".format(nodes))
                logger.info("Coefficients: {}".format(coeffs))
                logger.info("Singular values: {}".format(sig))
                logger.info("L2 Error: {}".format(error_norma_2))
                logger.info("Inf Error (max): {}".format(np.max(np.abs(error_norma_inf))))
            
            return results
            
        except Exception as e:
            logger.exception("Error in simulation for window [{}:{}]: {}".format(
                start_index, end_index, e))
            return None
    
    def iterate_through_windows(self, window_size=30, step_size=10, n_exponents_list=[2, 4, 6], plot_windows=True, save_plots=False, max_windows=None, start_index=None, end_index=None):
        """
        Iterate through overlapping windows of data and run simulations.
        
        :param window_size: Size of each window
        :param step_size: Step size for window advancement (enables overlapping windows)
        :param n_exponents_list: List of n_exponents values to test
        :param plot_windows: Whether to plot each window
        :param save_plots: Whether to save plots to a timestamped folder
        :param max_windows: Maximum number of windows to process
        :param start_index: Starting index for data analysis (if None, uses 0)
        :param end_index: Ending index for data analysis (if None, uses full data length)
        """
        if self.data is None:
            logger.error("No data loaded. Please load a file first.")
            return
        
        if self.column_to_use not in self.data.columns:
            logger.error("Column '{}' not found in data.".format(self.column_to_use))
            return
        
        # Get full data arrays
        full_signal_data = self.data[self.column_to_use].values
        full_microseconds_data = self.data['Microseconds'].values
        
        # Apply index range if specified
        if start_index is None:
            start_index = 0
        if end_index is None:
            end_index = len(full_signal_data)
        
        # Validate index range
        if start_index < 0:
            start_index = 0
        if end_index > len(full_signal_data):
            end_index = len(full_signal_data)
        if start_index >= end_index:
            logger.error("Invalid index range: start_index ({}) must be less than end_index ({})".format(start_index, end_index))
            return
        
        # Slice data to specified range
        signal_data = full_signal_data[start_index:end_index]
        microseconds_data = full_microseconds_data[start_index:end_index]
        
        # Calculate windows within the specified range
        range_samples = len(signal_data)
        num_windows = (range_samples - window_size) // step_size + 1
        if max_windows:
            num_windows = min(num_windows, max_windows)
        
        # Create timestamped folder for saving plots if requested
        output_folder = None
        if save_plots:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_folder = "simulation_outputs/electricity_analysis_{}".format(timestamp)
            os.makedirs(output_folder, exist_ok=True)
            logger.info("Created output folder: {}".format(output_folder))
        
        logger.info("Processing {} windows of size {} with step size {} in range [{}:{}]".format(
            num_windows, window_size, step_size, start_index, end_index))
        logger.info("Testing n_exponents: {}".format(n_exponents_list))
        
        all_results = []
        
        for window_idx in range(num_windows):
            # Calculate window indices within the sliced data range
            relative_start_idx = window_idx * step_size
            relative_end_idx = min(relative_start_idx + window_size, range_samples)
            
            # Map back to original data indices for caching and display
            absolute_start_idx = start_index + relative_start_idx
            absolute_end_idx = start_index + relative_end_idx
            
            if relative_end_idx - relative_start_idx < 10:  # Skip very small windows
                continue
            
            logger.info("\n{}".format("="*60))
            logger.info("WINDOW {}: Samples [{}:{}] (absolute indices)".format(
                window_idx + 1, absolute_start_idx, absolute_end_idx))
            logger.info("{}".format("="*60))
            
            window_results = {}
            
            for n_exp in n_exponents_list:
                result = self.run_simulation_in_window(absolute_start_idx, absolute_end_idx, n_exp, verbose=True)
                if result is not None:
                    window_results[n_exp] = result
            
            all_results.append(window_results)
            
            # Plot the window if requested
            if plot_windows and window_results:
                # Use full microseconds data for plotting context
                self._plot_window_results(window_results, full_microseconds_data, absolute_start_idx, absolute_end_idx, 
                                        output_folder, window_idx + 1)
        
        return all_results
    
    def get_cached_window_data(self, start_index, end_index, n_exponents):
        """
        Retrieve cached analysis data for a specific window.
        
        :param start_index: Starting index of the window
        :param end_index: Ending index of the window
        :param n_exponents: Number of exponential terms used
        :return: Dictionary with cached data or None if not found
        """
        window_key = (start_index, end_index, n_exponents)
        return self.cached_results.get(window_key, None)
    
    def get_all_cached_windows(self):
        """
        Get all cached window analysis results.
        
        :return: Dictionary with all cached results, keyed by (start_index, end_index, n_exponents)
        """
        return self.cached_results.copy()
    
    def clear_cached_data(self):
        """
        Clear all cached window analysis data.
        """
        self.cached_results.clear()
        logger.info("Cleared all cached window data")
    
    def get_cached_windows_summary(self):
        """
        Get a summary of all cached windows.
        
        :return: List of tuples with window information
        """
        summary = []
        for (start_idx, end_idx, n_exp), data in self.cached_results.items():
            summary.append({
                'window': (start_idx, end_idx),
                'n_exponents': n_exp,
                'window_size': end_idx - start_idx,
                'error_l2_normalized': data['error_l2'],
                'num_singular_values': len(data['singular_values']),
                'num_nodes': len(data['nodes']),
                'num_coeffs': len(data['coeffs'])
            })
        return summary
    
    def _plot_window_results(self, window_results, microseconds_data, start_idx, end_idx, output_folder=None, window_number=None):
        """
        Plot results for a single window with different n_exponents.
        Shows all approximations in one left plot and complete signal with highlighted window on the right.
        
        :param window_results: Dictionary of results for different n_exponents
        :param microseconds_data: Time data for the complete signal
        :param start_idx: Starting index of the window
        :param end_idx: Ending index of the window
        :param output_folder: Folder to save plots (if None, plots are not saved)
        :param window_number: Window number for filename
        """
        n_plots = len(window_results)
        if n_plots == 0:
            return
        
        # Create side-by-side subplots: left for all window approximations, right for complete signal
        fig, (ax_window, ax_complete) = plt.subplots(1, 2, figsize=(20, 8))
        
        window_microseconds = microseconds_data[start_idx:end_idx]
        complete_signal = self.data[self.column_to_use].values
        complete_microseconds = microseconds_data
        
        # Define colors for different n_exponents
        colors = ['red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
        
        # Left subplot: All window approximations in one plot
        # Plot original window signal first
        first_result = list(window_results.values())[0]
        ax_window.plot(window_microseconds, first_result['window_signal'], 'b.-', 
                      label='Original Signal', linewidth=2, markersize=4)
        
        # Plot all approximations with different colors
        error_info = []
        for idx, (n_exp, result) in enumerate(window_results.items()):
            color = colors[idx % len(colors)]
            ax_window.plot(window_microseconds, result['approximation'], color=color, linestyle='-',
                          label='{} exp (L2: {:.4f})'.format(n_exp, result["error_norma_2"]), linewidth=2)
            error_info.append('{}exp: {:.6f}'.format(n_exp, result["error_norma_2"]))
        
        ax_window.set_xlabel('Microseconds')
        ax_window.set_ylabel(self.column_to_use)
        ax_window.set_title('Window [{}:{}] - All Approximations\nL2 Errors: {}'.format(
            start_idx, end_idx, ', '.join(error_info)))
        ax_window.legend()
        ax_window.grid(True, alpha=0.3)
        
        # Right subplot: Complete signal with highlighted window (unchanged)
        # Plot complete signal in blue
        ax_complete.plot(complete_microseconds, complete_signal, 'b-', 
                       label='Complete Signal', linewidth=1, alpha=0.7)
        
        # Highlight the current window in red
        ax_complete.plot(window_microseconds, first_result['window_signal'], 'r-', 
                       label='Current Window', linewidth=2)
        
        ax_complete.set_xlabel('Microseconds')
        ax_complete.set_ylabel(self.column_to_use)
        ax_complete.set_title('Complete Signal - Window [{}:{}] Highlighted'.format(
            start_idx, end_idx))
        ax_complete.legend()
        ax_complete.grid(True, alpha=0.3)
        
        # Add vertical lines to mark window boundaries
        ax_complete.axvline(x=window_microseconds[0], color='orange', linestyle='--', 
                          alpha=0.8, label='Window Start')
        ax_complete.axvline(x=window_microseconds[-1], color='orange', linestyle='--', 
                          alpha=0.8, label='Window End')
        
        plt.tight_layout()
        
        # Save plot if output folder is specified
        if output_folder and window_number:
            filename = "window_{:03d}_samples_{}_{}.png".format(
                window_number, start_idx, end_idx)
            filepath = os.path.join(output_folder, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info("Saved plot: {}".format(filepath))
        
        plt.show(block=False)
