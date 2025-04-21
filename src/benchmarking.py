"""
Benchmarking Module for HDP-HMM MCMC Financial Regime Detection
- Evaluates trading strategy performance
- Compares against buy and hold strategy
- Calculates performance metrics (returns, drawdowns, Sharpe ratio)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os

from data_pipeline import DataPipeline
from signal_generation import RegimeSignalGenerator

class StrategyBenchmark:
    """
    Benchmark for evaluating trading strategy performance
    """
    
    def __init__(self, initial_capital=10000.0):
        """
        Initialize the benchmark
        
        Parameters:
        -----------
        initial_capital : float
            Initial capital for the strategy
        """
        self.initial_capital = initial_capital
        self.results = {}
        
    def calculate_strategy_returns(self, signals, price_data, transaction_cost=0.001):
        """
        Calculate strategy returns based on signals
        
        Parameters:
        -----------
        signals : pandas.DataFrame
            DataFrame containing trading signals
        price_data : pandas.Series
            Price data with dates as index
        transaction_cost : float
            Transaction cost as a percentage of trade value
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing strategy performance metrics
        """
        if not isinstance(signals, pd.DataFrame) or 'Signal' not in signals.columns:
            raise ValueError("Signals must be a DataFrame with a 'Signal' column")
        
        aligned_data = pd.DataFrame({
            'Price': price_data,
            'Signal': signals['Signal']
        })
        
        aligned_data['Signal'] = aligned_data['Signal'].fillna(method='ffill').fillna(0)
        
        aligned_data['Returns'] = aligned_data['Price'].pct_change()
        
        aligned_data['Strategy_Returns'] = aligned_data['Signal'].shift(1) * aligned_data['Returns']
        
        aligned_data['Signal_Change'] = aligned_data['Signal'].diff().fillna(0)
        aligned_data['Transaction_Cost'] = (aligned_data['Signal_Change'] != 0) * transaction_cost
        
        aligned_data['Strategy_Returns'] = aligned_data['Strategy_Returns'] - aligned_data['Transaction_Cost']
        
        aligned_data['Cumulative_Returns'] = (1 + aligned_data['Returns']).cumprod() - 1
        aligned_data['Strategy_Cumulative_Returns'] = (1 + aligned_data['Strategy_Returns']).cumprod() - 1
        
        aligned_data['Buy_Hold_Equity'] = self.initial_capital * (1 + aligned_data['Cumulative_Returns'])
        aligned_data['Strategy_Equity'] = self.initial_capital * (1 + aligned_data['Strategy_Cumulative_Returns'])
        
        aligned_data['Buy_Hold_Peak'] = aligned_data['Buy_Hold_Equity'].cummax()
        aligned_data['Strategy_Peak'] = aligned_data['Strategy_Equity'].cummax()
        
        aligned_data['Buy_Hold_Drawdown'] = (aligned_data['Buy_Hold_Equity'] - aligned_data['Buy_Hold_Peak']) / aligned_data['Buy_Hold_Peak']
        aligned_data['Strategy_Drawdown'] = (aligned_data['Strategy_Equity'] - aligned_data['Strategy_Peak']) / aligned_data['Strategy_Peak']
        
        return aligned_data
    
    def calculate_performance_metrics(self, strategy_data, annualization_factor=252):
        """
        Calculate performance metrics for the strategy
        
        Parameters:
        -----------
        strategy_data : pandas.DataFrame
            DataFrame containing strategy performance data
        annualization_factor : int
            Factor to annualize returns (252 for daily data)
            
        Returns:
        --------
        dict
            Dictionary containing performance metrics
        """
        metrics = {}
        
        metrics['Buy_Hold_Total_Return'] = strategy_data['Cumulative_Returns'].iloc[-1]
        metrics['Strategy_Total_Return'] = strategy_data['Strategy_Cumulative_Returns'].iloc[-1]
        
        n_periods = len(strategy_data)
        metrics['Buy_Hold_Annual_Return'] = (1 + metrics['Buy_Hold_Total_Return']) ** (annualization_factor / n_periods) - 1
        metrics['Strategy_Annual_Return'] = (1 + metrics['Strategy_Total_Return']) ** (annualization_factor / n_periods) - 1
        
        metrics['Buy_Hold_Volatility'] = strategy_data['Returns'].std() * np.sqrt(annualization_factor)
        metrics['Strategy_Volatility'] = strategy_data['Strategy_Returns'].std() * np.sqrt(annualization_factor)
        
        risk_free_rate = 0.0  # Assuming zero risk-free rate for simplicity
        metrics['Buy_Hold_Sharpe'] = (metrics['Buy_Hold_Annual_Return'] - risk_free_rate) / metrics['Buy_Hold_Volatility'] if metrics['Buy_Hold_Volatility'] > 0 else 0
        metrics['Strategy_Sharpe'] = (metrics['Strategy_Annual_Return'] - risk_free_rate) / metrics['Strategy_Volatility'] if metrics['Strategy_Volatility'] > 0 else 0
        
        metrics['Buy_Hold_Max_Drawdown'] = strategy_data['Buy_Hold_Drawdown'].min()
        metrics['Strategy_Max_Drawdown'] = strategy_data['Strategy_Drawdown'].min()
        
        metrics['Buy_Hold_Calmar'] = metrics['Buy_Hold_Annual_Return'] / abs(metrics['Buy_Hold_Max_Drawdown']) if metrics['Buy_Hold_Max_Drawdown'] < 0 else 0
        metrics['Strategy_Calmar'] = metrics['Strategy_Annual_Return'] / abs(metrics['Strategy_Max_Drawdown']) if metrics['Strategy_Max_Drawdown'] < 0 else 0
        
        strategy_data['Winning_Trade'] = strategy_data['Strategy_Returns'] > 0
        metrics['Win_Rate'] = strategy_data['Winning_Trade'].mean()
        
        metrics['Number_of_Trades'] = (strategy_data['Signal_Change'] != 0).sum()
        
        return metrics
    
    def benchmark_strategy(self, signals, price_data, strategy_name="HDP-HMM Strategy", transaction_cost=0.001):
        """
        Benchmark a trading strategy against buy and hold
        
        Parameters:
        -----------
        signals : pandas.DataFrame
            DataFrame containing trading signals
        price_data : pandas.Series
            Price data with dates as index
        strategy_name : str
            Name of the strategy
        transaction_cost : float
            Transaction cost as a percentage of trade value
            
        Returns:
        --------
        tuple
            (strategy_data, metrics)
        """
        strategy_data = self.calculate_strategy_returns(signals, price_data, transaction_cost)
        
        metrics = self.calculate_performance_metrics(strategy_data)
        
        self.results[strategy_name] = {
            'data': strategy_data,
            'metrics': metrics
        }
        
        return strategy_data, metrics
    
    def plot_equity_curves(self, strategy_name="HDP-HMM Strategy", save_path=None):
        """
        Plot equity curves for the strategy and buy-hold
        
        Parameters:
        -----------
        strategy_name : str
            Name of the strategy to plot
        save_path : str
            Path to save the plot
        """
        if strategy_name not in self.results:
            raise ValueError(f"Strategy '{strategy_name}' not found in results")
        
        strategy_data = self.results[strategy_name]['data']
        
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 1, 1)
        plt.plot(strategy_data.index, strategy_data['Buy_Hold_Equity'], label='Buy & Hold')
        plt.plot(strategy_data.index, strategy_data['Strategy_Equity'], label=strategy_name)
        plt.title('Equity Curves')
        plt.ylabel('Portfolio Value ($)')
        plt.grid(True)
        plt.legend()
        
        plt.subplot(2, 1, 2)
        plt.plot(strategy_data.index, strategy_data['Buy_Hold_Drawdown'], label='Buy & Hold Drawdown', color='blue', alpha=0.5)
        plt.plot(strategy_data.index, strategy_data['Strategy_Drawdown'], label=f'{strategy_name} Drawdown', color='orange', alpha=0.5)
        plt.fill_between(strategy_data.index, 0, strategy_data['Buy_Hold_Drawdown'], color='blue', alpha=0.1)
        plt.fill_between(strategy_data.index, 0, strategy_data['Strategy_Drawdown'], color='orange', alpha=0.1)
        plt.title('Drawdowns')
        plt.xlabel('Date')
        plt.ylabel('Drawdown')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
    
    def plot_returns_distribution(self, strategy_name="HDP-HMM Strategy", save_path=None):
        """
        Plot returns distribution for the strategy and buy-hold
        
        Parameters:
        -----------
        strategy_name : str
            Name of the strategy to plot
        save_path : str
            Path to save the plot
        """
        if strategy_name not in self.results:
            raise ValueError(f"Strategy '{strategy_name}' not found in results")
        
        strategy_data = self.results[strategy_name]['data']
        
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 1, 1)
        plt.hist(strategy_data['Returns'], bins=50, alpha=0.5, label='Buy & Hold')
        plt.hist(strategy_data['Strategy_Returns'], bins=50, alpha=0.5, label=strategy_name)
        plt.title('Returns Distribution')
        plt.xlabel('Daily Return')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.legend()
        
        plt.subplot(2, 1, 2)
        plt.scatter(strategy_data['Returns'], strategy_data['Strategy_Returns'], alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        plt.axvline(x=0, color='r', linestyle='-', alpha=0.3)
        plt.title('Strategy Returns vs Market Returns')
        plt.xlabel('Market Returns')
        plt.ylabel('Strategy Returns')
        plt.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
    
    def print_performance_summary(self, strategy_name="HDP-HMM Strategy"):
        """
        Print performance summary for the strategy
        
        Parameters:
        -----------
        strategy_name : str
            Name of the strategy to summarize
        """
        if strategy_name not in self.results:
            raise ValueError(f"Strategy '{strategy_name}' not found in results")
        
        metrics = self.results[strategy_name]['metrics']
        
        print(f"\n{'=' * 50}")
        print(f"Performance Summary: {strategy_name} vs Buy & Hold")
        print(f"{'=' * 50}")
        
        print(f"\nTotal Return:")
        print(f"  Buy & Hold: {metrics['Buy_Hold_Total_Return']:.2%}")
        print(f"  {strategy_name}: {metrics['Strategy_Total_Return']:.2%}")
        
        print(f"\nAnnualized Return:")
        print(f"  Buy & Hold: {metrics['Buy_Hold_Annual_Return']:.2%}")
        print(f"  {strategy_name}: {metrics['Strategy_Annual_Return']:.2%}")
        
        print(f"\nAnnualized Volatility:")
        print(f"  Buy & Hold: {metrics['Buy_Hold_Volatility']:.2%}")
        print(f"  {strategy_name}: {metrics['Strategy_Volatility']:.2%}")
        
        print(f"\nSharpe Ratio:")
        print(f"  Buy & Hold: {metrics['Buy_Hold_Sharpe']:.2f}")
        print(f"  {strategy_name}: {metrics['Strategy_Sharpe']:.2f}")
        
        print(f"\nMaximum Drawdown:")
        print(f"  Buy & Hold: {metrics['Buy_Hold_Max_Drawdown']:.2%}")
        print(f"  {strategy_name}: {metrics['Strategy_Max_Drawdown']:.2%}")
        
        print(f"\nCalmar Ratio:")
        print(f"  Buy & Hold: {metrics['Buy_Hold_Calmar']:.2f}")
        print(f"  {strategy_name}: {metrics['Strategy_Calmar']:.2f}")
        
        print(f"\nWin Rate: {metrics['Win_Rate']:.2%}")
        print(f"Number of Trades: {metrics['Number_of_Trades']}")
        
        print(f"\n{'=' * 50}")
    
    def save_results(self, strategy_name="HDP-HMM Strategy", directory="results"):
        """
        Save benchmark results to CSV files
        
        Parameters:
        -----------
        strategy_name : str
            Name of the strategy to save
        directory : str
            Directory to save the results
        """
        if strategy_name not in self.results:
            raise ValueError(f"Strategy '{strategy_name}' not found in results")
        
        os.makedirs(directory, exist_ok=True)
        
        strategy_data = self.results[strategy_name]['data']
        strategy_data.to_csv(os.path.join(directory, f"{strategy_name.replace(' ', '_')}_data.csv"))
        
        metrics = self.results[strategy_name]['metrics']
        metrics_df = pd.DataFrame({k: [v] for k, v in metrics.items()})
        metrics_df.to_csv(os.path.join(directory, f"{strategy_name.replace(' ', '_')}_metrics.csv"), index=False)
        
        print(f"Results saved to {directory}")


def run_benchmark(model_path, data_start_date="2010-01-01", train_ratio=0.7, 
                 mapping_method="mean_based", confidence_threshold=0.6,
                 initial_capital=10000.0, transaction_cost=0.001,
                 save_dir="results"):
    """
    Run a complete benchmark of the HDP-HMM trading strategy
    
    Parameters:
    -----------
    model_path : str
        Path to the trained model
    data_start_date : str
        Start date for data fetching in YYYY-MM-DD format
    train_ratio : float
        Ratio of data to use for training
    mapping_method : str
        Method to use for state mapping ('mean_based', 'volatility_based', 'combined')
    confidence_threshold : float
        Threshold for state probability to generate a signal
    initial_capital : float
        Initial capital for the strategy
    transaction_cost : float
        Transaction cost as a percentage of trade value
    save_dir : str
        Directory to save results
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print("Fetching and preparing data...")
    pipeline = DataPipeline(ticker="SPY", start_date=data_start_date)
    data = pipeline.fetch_data()
    log_returns = pipeline.calculate_log_returns()
    
    train_data, test_data = pipeline.split_data(train_ratio=train_ratio)
    
    print("Creating signal generator...")
    signal_generator = RegimeSignalGenerator(model_path=model_path)
    
    print(f"Creating state mapping using {mapping_method} method...")
    state_mapping = signal_generator.create_state_mapping(method=mapping_method)
    
    signal_generator.plot_state_mapping(save_path=os.path.join(save_dir, "state_mapping.png"))
    
    print("Generating trading signals...")
    test_dates = log_returns.index[int(len(log_returns) * train_ratio):]
    signals = signal_generator.generate_regime_change_signals(
        test_data, 
        dates=test_dates, 
        confidence_threshold=confidence_threshold
    )
    
    test_prices = data['Close'][test_dates]
    signal_generator.plot_signals(price_data=test_prices, save_path=os.path.join(save_dir, "signals.png"))
    
    print("Benchmarking strategy...")
    benchmark = StrategyBenchmark(initial_capital=initial_capital)
    strategy_data, metrics = benchmark.benchmark_strategy(
        signals, 
        test_prices, 
        strategy_name="HDP-HMM Regime Strategy",
        transaction_cost=transaction_cost
    )
    
    benchmark.print_performance_summary()
    
    benchmark.plot_equity_curves(save_path=os.path.join(save_dir, "equity_curves.png"))
    
    benchmark.plot_returns_distribution(save_path=os.path.join(save_dir, "returns_distribution.png"))
    
    benchmark.save_results(directory=save_dir)
    
    return benchmark


if __name__ == "__main__":
    model_path = "models/hdp_hmm_model_latest.pkl"
    
    benchmark = run_benchmark(
        model_path=model_path,
        data_start_date="2010-01-01",
        train_ratio=0.7,
        mapping_method="mean_based",
        confidence_threshold=0.6,
        initial_capital=10000.0,
        transaction_cost=0.001,
        save_dir="results"
    )
