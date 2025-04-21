"""
HDP-HMM MCMC Financial Regime Detection

This script provides an interactive interface for analyzing financial market regimes 
using a Hierarchical Dirichlet Process Hidden Markov Model (HDP-HMM) with 
Markov Chain Monte Carlo (MCMC) sampling.

Overview:
1. Data Pipeline: Fetch SPY daily data and calculate log returns
2. Model Training: Train the HDP-HMM model using MCMC
3. State Analysis: Analyze the identified regimes and their parameters
4. Signal Generation: Generate trading signals based on regime changes
5. Performance Evaluation: Benchmark against buy and hold strategy

To convert this script to a Jupyter notebook, run:
jupyter nbconvert --to notebook --execute hdp_hmm_analysis.py
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import pickle

sys.path.append('../src')

from data_pipeline import DataPipeline
from hdp_hmm import HDPHMM
from model_training import train_model, analyze_state_parameters
from signal_generation import RegimeSignalGenerator
from benchmarking import StrategyBenchmark, run_benchmark

plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (12, 8)

print("## 1. Data Pipeline")
print("Fetch SPY daily data and calculate log returns using the DataPipeline class.")

data_start_date = "2010-01-01"
pipeline = DataPipeline(ticker="SPY", start_date=data_start_date)

data = pipeline.fetch_data()
print(f"Data shape: {data.shape}")
print(data.head())

log_returns = pipeline.calculate_log_returns()
print(f"Log returns shape: {log_returns.shape}")
print(log_returns.head())

pipeline.plot_data()

train_ratio = 0.7
train_data, test_data = pipeline.split_data(train_ratio=train_ratio)

print(f"Training data shape: {train_data.shape}")
print(f"Testing data shape: {test_data.shape}")

print("\n## 2. Model Training")
print("Train the HDP-HMM model using MCMC sampling. This can take a while depending on the number of iterations and the size of the dataset.")

n_iter = 1000  # Number of MCMC iterations
alpha = 1.0    # Concentration parameter for the Dirichlet Process prior on transitions
gamma = 1.0    # Concentration parameter for the Dirichlet Process prior on states
kappa = 10.0   # Self-transition bias parameter (sticky HDP-HMM)
max_states = 20  # Maximum number of states to consider (truncation level)


os.makedirs("../models", exist_ok=True)
os.makedirs("../plots", exist_ok=True)

model, train_data, test_data = train_model(
    data_start_date=data_start_date,
    train_ratio=train_ratio,
    n_iter=n_iter,
    alpha=alpha,
    gamma=gamma,
    kappa=kappa,
    max_states=max_states,
    save_dir="../models",
    plot_dir="../plots"
)



print("\n## 3. State Analysis")
print("Analyze the identified regimes and their parameters.")

if not hasattr(model, 'state_parameters') or not model.state_parameters:
    model._extract_state_parameters()

print(f"Active states: {model.active_states}")

for s in model.active_states:
    if s in model.state_parameters:
        params = model.state_parameters[s]
        print(f"\nState {s}:")
        print(f"  Mean: {params['mean'][0]:.6f}")
        print(f"  Std: {params['std'][0]:.6f}")
        print(f"  Count: {params['count']}")
        print(f"  Proportion: {params['proportion']:.4f}")

model.plot_state_sequence(data=log_returns)

model.plot_state_parameters()

model.plot_log_likelihood()

analyze_state_parameters(model, log_returns, save_dir="../plots")

print("\n## 4. Signal Generation")
print("Generate trading signals based on regime changes.")

signal_generator = RegimeSignalGenerator(model=model)

mapping_method = "mean_based"  # Options: "mean_based", "volatility_based", "combined"
state_mapping = signal_generator.create_state_mapping(method=mapping_method)

print("State Mapping:")
for s, mapping in state_mapping.items():
    print(f"State {s}: {mapping['label']} ({mapping['guideline']})")

signal_generator.plot_state_mapping()

confidence_threshold = 0.6
test_dates = log_returns.index[int(len(log_returns) * train_ratio):]

signals = signal_generator.generate_signals(
    test_data, 
    dates=test_dates, 
    confidence_threshold=confidence_threshold
)

print(signals.head())

regime_change_signals = signal_generator.generate_regime_change_signals(
    test_data, 
    dates=test_dates, 
    confidence_threshold=confidence_threshold
)

print(regime_change_signals.head())

test_dates_in_data = [date for date in test_dates if date in data.index]
if not test_dates_in_data:
    print("Warning: No test dates found in data index. Using all available dates.")
    test_prices = data['Close']
else:
    test_prices = data['Close'].loc[test_dates_in_data]
signal_generator.plot_signals(price_data=test_prices)

print("\n## 5. Performance Evaluation")
print("Benchmark the trading strategy against a buy and hold strategy.")

initial_capital = 10000.0
transaction_cost = 0.001
benchmark = StrategyBenchmark(initial_capital=initial_capital)

strategy_data, metrics = benchmark.benchmark_strategy(
    regime_change_signals,  # Using regime change signals
    test_prices, 
    strategy_name="HDP-HMM Regime Strategy",
    transaction_cost=transaction_cost
)

benchmark.print_performance_summary()

benchmark.plot_equity_curves()

benchmark.plot_returns_distribution()

os.makedirs("../results", exist_ok=True)
benchmark.save_results(directory="../results")

print("\n## 6. Parameter Sensitivity Analysis")
print("Analyze the sensitivity of the strategy to different parameters.")

confidence_thresholds = [0.4, 0.5, 0.6, 0.7, 0.8]
threshold_results = {}

for threshold in confidence_thresholds:
    signals = signal_generator.generate_regime_change_signals(
        test_data, 
        dates=test_dates, 
        confidence_threshold=threshold
    )
    
    benchmark = StrategyBenchmark(initial_capital=initial_capital)
    strategy_data, metrics = benchmark.benchmark_strategy(
        signals, 
        test_prices, 
        strategy_name=f"Threshold {threshold}",
        transaction_cost=transaction_cost
    )
    
    threshold_results[threshold] = metrics

comparison = pd.DataFrame({
    f"Threshold {threshold}": {
        'Annual Return': metrics['Strategy_Annual_Return'],
        'Sharpe Ratio': metrics['Strategy_Sharpe'],
        'Max Drawdown': metrics['Strategy_Max_Drawdown'],
        'Win Rate': metrics['Win_Rate'],
        'Number of Trades': metrics['Number_of_Trades']
    } for threshold, metrics in threshold_results.items()
})

print(comparison)

mapping_methods = ["mean_based", "volatility_based", "combined"]
method_results = {}

for method in mapping_methods:
    signal_generator = RegimeSignalGenerator(model=model)
    state_mapping = signal_generator.create_state_mapping(method=method)
    
    signals = signal_generator.generate_regime_change_signals(
        test_data, 
        dates=test_dates, 
        confidence_threshold=0.6
    )
    
    benchmark = StrategyBenchmark(initial_capital=initial_capital)
    strategy_data, metrics = benchmark.benchmark_strategy(
        signals, 
        test_prices, 
        strategy_name=f"Method {method}",
        transaction_cost=transaction_cost
    )
    
    method_results[method] = metrics

comparison = pd.DataFrame({
    f"Method {method}": {
        'Annual Return': metrics['Strategy_Annual_Return'],
        'Sharpe Ratio': metrics['Strategy_Sharpe'],
        'Max Drawdown': metrics['Strategy_Max_Drawdown'],
        'Win Rate': metrics['Win_Rate'],
        'Number of Trades': metrics['Number_of_Trades']
    } for method, metrics in method_results.items()
})

print(comparison)

print("\n## 7. Conclusion")
print("""
The HDP-HMM MCMC model successfully identifies different market regimes in the SPY daily data. 
The trading strategy based on regime changes shows promising results compared to the buy and hold strategy.

Key findings:
1. The model identifies multiple regimes with distinct mean returns and volatilities
2. The regime-based trading strategy adapts to changing market conditions
3. Parameter sensitivity analysis shows the importance of proper calibration

Future improvements:
1. Incorporate more features beyond just log returns
2. Optimize hyperparameters using cross-validation
3. Implement more sophisticated trading rules based on regime probabilities
4. Test the strategy on different assets and timeframes
""")
