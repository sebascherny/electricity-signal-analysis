# electricity-signal-analysis
Project to analyze a signal and try to estimate and categorize the relevant changing moments

# ElectricitySignalAnalyzer

A Python class for analyzing electricity signals using exponential approximation methods based on Hankel matrix decomposition.

## Local run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python example_usage.py
python test_caching_example.py
```

## Web Interface

The easiest way to use the analyzer is through the web interface:

```bash
source .venv/bin/activate
python3 start_server.py
```

Then open your browser to `http://localhost:8000` and:

1. **Upload your CSV file** with electricity signal data
2. **Configure parameters** like window size, step size, number of exponents
3. **Set optional ranges** with start/end indices to analyze specific portions
4. **Click "Analyze Signal"** to process your data
5. **Download results** as a ZIP file containing plots and analysis logs

The web interface provides:
- ✅ **File validation** and column detection
- ✅ **Parameter validation** with helpful hints
- ✅ **Progress indication** during analysis
- ✅ **Automatic ZIP download** with all results
- ✅ **Detailed logs** and analysis summary

## Test in python console

In bash (assuming you created a virtual environment as said above):

```bash
source .venv/bin/activate
python
```

Then in python:

```python
from electricity_signal_analyzer import ElectricitySignalAnalyzer
analyzer = ElectricitySignalAnalyzer(".", "128sample_Event.csv", "Report_1_IB")
analyzer.iterate_through_windows(window_size=200, step_size=120, n_exponents_list=[4, 5, 6, 8, 10], plot_windows=True, save_plots=False, max_windows=4, start_index=400, end_index=1200)

```

## Overview

The `ElectricitySignalAnalyzer` class provides methods to load electricity signal data from CSV files and perform signal approximation using exponential functions. It implements the algorithms from the existing notebooks in the `Organized_code_and_tests` folder.

## Features

- **Data Loading**: Load CSV files with electricity signal data
- **Column Selection**: Choose which signal column to analyze
- **Signal Visualization**: Plot entire signals with time series data
- **Exponential Approximation**: Approximate signals using linear combinations of exponentials
- **Window Analysis**: Analyze signals in sliding windows
- **Multi-parameter Testing**: Test different numbers of exponential terms

## Class Methods

### Core Methods

1. **`load_file(file_path=None)`**
   - Loads CSV data file
   - Defaults to `128sample_Event.csv`
   - Returns pandas DataFrame

2. **`choose_column_to_use(column_name="Report_1_IB")`**
   - Selects which column to analyze
   - Defaults to "Report_1_IB"

3. **`plot_entire_signal()`**
   - Plots the chosen signal column vs Microseconds
   - Uses matplotlib for visualization

4. **`approximate_sequence(signal, n_exponents, verbose=False)`**
   - Core approximation method similar to Step_2 notebook
   - Uses Hankel matrix decomposition
   - Returns nodes, coefficients, approximation, singular values, and errors

5. **`run_simulation_in_window(start_index, end_index, n_exponents=2, verbose=True)`**
   - Runs approximation on a specific data window
   - Prints detailed results including nodes, coefficients, and errors
   - Returns dictionary with all results

6. **`iterate_through_windows(window_size=30, n_exponents_list=[2, 4, 6], plot_windows=True, save_plots=False, max_windows=None, start_index=None, end_index=None)`**
   - Processes multiple windows of specified size
   - Tests different numbers of exponential terms
   - Generates plots for each window and parameter combination

### Helper Methods

- `construct_hankel()`: Builds Hankel matrix from signal data
- `get_coneigen()`: Computes singular value decomposition
- `get_nodes()`: Extracts exponential nodes from eigenvectors
- `construct_vandermonde_matrix()`: Creates Vandermonde matrix for coefficient computation

## Usage Example

```python
from electricity_signal_analyzer_fixed import ElectricitySignalAnalyzer

# Create analyzer instance
analyzer = ElectricitySignalAnalyzer()

# Load data (uses default file if no path provided)
data = analyzer.load_file()

# Choose column to analyze
analyzer.choose_column_to_use("Report_1_IB")

# Plot entire signal
analyzer.plot_entire_signal()

# Run approximation on a specific window
result = analyzer.run_simulation_in_window(0, 30, n_exponents=3)

# Analyze multiple windows with different parameters
results = analyzer.iterate_through_windows(
    window_size=30, 
    n_exponents_list=[2, 4, 6], 
    plot_windows=True
)
```

## Data Format

The class expects CSV files with the following columns:
- `Timestamp`: Time stamps
- `Microseconds`: Time in microseconds (used as x-axis)
- `Report_1_IB` or column name you want to analyze: Current measurements


## Dependencies

- numpy: Numerical computations and linear algebra
- pandas: Data loading and manipulation
- matplotlib: Plotting and visualization
- sklearn: Randomized SVD (optional)

## Algorithm Details

The approximation method uses:
1. **Hankel Matrix Construction**: Creates structured matrix from signal data
2. **Singular Value Decomposition**: Extracts dominant signal components
3. **Node Extraction**: Computes exponential decay/growth rates
4. **Vandermonde System**: Solves for linear coefficients
5. **Error Analysis**: Computes L2 and infinity norm errors

## Files

- `electricity_signal_analyzer_fixed.py`: Main class implementation
- `example_usage_fixed.py`: Demonstration script
- `simple_test.py`: Basic functionality test
- `README.md`: This documentation

## Notes

- The implementation is compatible with older Python versions (no f-strings or @ operator)
- Window analysis automatically handles edge cases and small windows
- Plotting is optional and can be disabled for batch processing
- Error metrics help evaluate approximation quality

## Based On

This implementation is derived from the algorithms in:
- `Step_1_Test_matrix_functions.ipynb`: Matrix construction and decomposition
- `Step_2_Test_approximate_sequence.ipynb`: Signal approximation methods
