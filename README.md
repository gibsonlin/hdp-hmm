# HDP-HMM MCMC Financial Regime Detection

A Python implementation of Hierarchical Dirichlet Process Hidden Markov Model (HDP-HMM) with Markov Chain Monte Carlo (MCMC) sampling for financial market regime detection and trading signal generation.

## Overview

This project implements a non-parametric Bayesian approach to identify market regimes in financial time series data. It uses a custom implementation of the HDP-HMM model with MCMC sampling to detect regimes in SPY daily data and generate trading signals based on regime changes.

### Key Features

- **Non-parametric Bayesian Model**: Automatically determines the optimal number of regimes
- **Sticky HDP-HMM**: Incorporates temporal persistence for more stable regime identification
- **Regime-Based Trading**: Generates long/short signals based on identified regimes
- **Performance Benchmarking**: Compares strategy performance against buy and hold

## Project Structure

- `src/`: Source code for the project
  - `data_pipeline.py`: Fetches SPY data and calculates log returns
  - `hdp_hmm.py`: Custom implementation of HDP-HMM with MCMC sampling
  - `model_training.py`: Trains the HDP-HMM model and analyzes state parameters
  - `signal_generation.py`: Generates trading signals based on regime changes
  - `benchmarking.py`: Evaluates trading strategy performance
  - `main.py`: Orchestrates the entire workflow
- `notebooks/`: Jupyter notebooks for interactive analysis
  - `hdp_hmm_analysis.py`: Python script that can be converted to a notebook
- `models/`: Saved model files
- `plots/`: Generated plots and visualizations
- `results/`: Benchmark results and performance metrics

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/hdp_hmm_project.git
cd hdp_hmm_project

# Install required packages
pip install numpy pandas matplotlib scipy yfinance
```

## Usage

### Running the Full Workflow

```bash
# Run the main script with default parameters
python src/main.py

# Run with custom parameters
python src/main.py --data_start_date 2010-01-01 --train_ratio 0.7 --n_iter 1000 --alpha 1.0 --gamma 1.0 --kappa 10.0 --max_states 20 --mapping_method mean_based --confidence_threshold 0.6 --initial_capital 10000.0 --transaction_cost 0.001 --output_dir results
```

### Interactive Analysis

```bash
# Convert the Python script to a Jupyter notebook
cd notebooks
jupyter nbconvert --to notebook --execute hdp_hmm_analysis.py

# Open the generated notebook
jupyter notebook hdp_hmm_analysis.ipynb
```

## Components

### Data Pipeline

The data pipeline fetches SPY daily data using yfinance and calculates log returns. It also provides functionality for splitting the data into training and testing sets.

```python
from data_pipeline import DataPipeline

# Initialize data pipeline
pipeline = DataPipeline(ticker="SPY", start_date="2010-01-01")

# Fetch data and calculate log returns
data = pipeline.fetch_data()
log_returns = pipeline.calculate_log_returns()

# Split data into training and testing sets
train_data, test_data = pipeline.split_data(train_ratio=0.7)
```

### HDP-HMM Model

The HDP-HMM model is a custom implementation of the Hierarchical Dirichlet Process Hidden Markov Model with MCMC sampling. It uses a sticky parameter to encourage temporal persistence in the state sequence.

```python
from hdp_hmm import HDPHMM

# Initialize model
model = HDPHMM(alpha=1.0, gamma=1.0, kappa=10.0, max_states=20, obs_dim=1, sticky=True)

# Train model
model.fit(train_data, n_iter=1000, verbose=True)

# Save model
model.save("models/hdp_hmm_model.pkl")

# Load model
model = HDPHMM.load("models/hdp_hmm_model.pkl")
```

### Signal Generation

The signal generator creates a mapping between states and trading signals based on the state parameters. It then generates trading signals based on regime changes.

```python
from signal_generation import RegimeSignalGenerator

# Initialize signal generator
signal_generator = RegimeSignalGenerator(model=model)

# Create state mapping
state_mapping = signal_generator.create_state_mapping(method="mean_based")

# Generate signals
signals = signal_generator.generate_regime_change_signals(test_data, dates=test_dates)

# Plot signals
signal_generator.plot_signals(price_data=test_prices)
```

### Benchmarking

The benchmarking module evaluates the performance of the trading strategy against a buy and hold strategy. It calculates various performance metrics such as returns, drawdowns, and Sharpe ratio.

```python
from benchmarking import StrategyBenchmark

# Initialize benchmark
benchmark = StrategyBenchmark(initial_capital=10000.0)

# Benchmark strategy
strategy_data, metrics = benchmark.benchmark_strategy(signals, test_prices)

# Print performance summary
benchmark.print_performance_summary()

# Plot equity curves
benchmark.plot_equity_curves()
```

## Parameters

- `alpha`: Concentration parameter for the Dirichlet Process prior on transitions
- `gamma`: Concentration parameter for the Dirichlet Process prior on states
- `kappa`: Self-transition bias parameter (sticky HDP-HMM)
- `max_states`: Maximum number of states to consider (truncation level)
- `mapping_method`: Method to use for state mapping ('mean_based', 'volatility_based', 'combined')
- `confidence_threshold`: Threshold for state probability to generate a signal
- `initial_capital`: Initial capital for the strategy
- `transaction_cost`: Transaction cost as a percentage of trade value

## Results

The HDP-HMM model successfully identifies different market regimes in the SPY daily data. The trading strategy based on regime changes shows promising results compared to the buy and hold strategy.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
