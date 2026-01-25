# Testing Guide for ElectricitySignalAnalyzer

This document explains how to run the test suite for the ElectricitySignalAnalyzer class.

## Prerequisites

Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Running Tests

### Run all tests
```bash
pytest
```

### Run tests with verbose output
```bash
pytest -v
```

### Run specific test file
```bash
pytest test_electricity_signal_analyzer.py
```

### Run specific test class
```bash
pytest test_electricity_signal_analyzer.py::TestElectricitySignalAnalyzer
```

### Run specific test method
```bash
pytest test_electricity_signal_analyzer.py::TestElectricitySignalAnalyzer::test_construct_hankel_odd_length
```

### Run tests with coverage (optional)
```bash
pip install pytest-cov
pytest --cov=electricity_signal_analyzer
```

## Test Coverage

The test suite covers the following mathematical methods:

### Core Mathematical Functions
- `construct_hankel()` - Hankel matrix construction
- `get_coneigen()` - Eigenvalue/SVD computation
- `get_nodes()` - Node extraction from eigenvectors
- `construct_vandermonde_matrix()` - Vandermonde matrix construction
- `approximate_sequence()` - Signal approximation with exponentials

### Integration Tests
- `run_simulation_in_window()` - Window-based simulation with mock data

### Edge Cases
- Empty inputs
- Very small signals
- Invalid parameters
- Error handling scenarios

## Test Data

All tests use synthetic/mock data and do not require external files:
- Synthetic exponential signals
- Random noise signals
- Constant signals
- Mock pandas DataFrames

## Expected Output

When tests pass, you should see output like:
```
test_electricity_signal_analyzer.py::TestElectricitySignalAnalyzer::test_construct_hankel_odd_length PASSED
test_electricity_signal_analyzer.py::TestElectricitySignalAnalyzer::test_construct_hankel_even_length PASSED
...
========================= X passed in Y.YYs =========================
```

## Troubleshooting

If tests fail:
1. Check that all dependencies are installed
2. Ensure the `electricity_signal_analyzer.py` file is in the same directory
3. Review the specific error messages for debugging information
