"""
Data Pipeline for HDP-HMM MCMC Financial Regime Detection
- Fetches SPY daily data using yfinance
- Calculates log returns
- Prepares data for model training
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
import os

class DataPipeline:
    def __init__(self, ticker="SPY", start_date="2000-01-01", end_date=None):
        """
        Initialize the data pipeline
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol to fetch data for (default: SPY)
        start_date : str
            Start date for data fetching in YYYY-MM-DD format
        end_date : str
            End date for data fetching in YYYY-MM-DD format (default: today)
        """
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date if end_date else datetime.now().strftime("%Y-%m-%d")
        self.data = None
        self.log_returns = None
        
    def fetch_data(self):
        """
        Fetch daily data for the specified ticker
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing the fetched data
        """
        print(f"Fetching data for {self.ticker} from {self.start_date} to {self.end_date}")
        self.data = yf.download(
            self.ticker, 
            start=self.start_date, 
            end=self.end_date, 
            progress=False
        )
        return self.data
    
    def calculate_log_returns(self):
        """
        Calculate log returns from the price data
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing the log returns
        """
        if self.data is None or self.data.empty:
            self.data = self.fetch_data()
            
        self.log_returns = np.log(self.data['Close'] / self.data['Close'].shift(1)).dropna()
        return self.log_returns
    
    def prepare_training_data(self):
        """
        Prepare data for model training
        
        Returns:
        --------
        numpy.ndarray
            Array of log returns for model training
        """
        if self.log_returns is None:
            self.log_returns = self.calculate_log_returns()
            
        return self.log_returns.values.reshape(-1, 1)
    
    def split_data(self, train_ratio=0.7):
        """
        Split data into training and testing sets
        
        Parameters:
        -----------
        train_ratio : float
            Ratio of data to use for training (default: 0.7)
            
        Returns:
        --------
        tuple
            (training_data, testing_data)
        """
        if self.log_returns is None:
            self.log_returns = self.calculate_log_returns()
            
        split_idx = int(len(self.log_returns) * train_ratio)
        train_data = self.log_returns.iloc[:split_idx].values.reshape(-1, 1)
        test_data = self.log_returns.iloc[split_idx:].values.reshape(-1, 1)
        
        return train_data, test_data
    
    def plot_data(self, save_path=None):
        """
        Plot the price data and log returns
        
        Parameters:
        -----------
        save_path : str
            Path to save the plot (default: None, display only)
        """
        if self.data is None or self.data.empty:
            self.data = self.fetch_data()
            
        if self.log_returns is None:
            self.log_returns = self.calculate_log_returns()
            
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        ax1.plot(self.data.index, self.data['Close'])
        ax1.set_title(f"{self.ticker} Close Price")
        ax1.set_ylabel("Price ($)")
        ax1.grid(True)
        
        ax2.plot(self.log_returns.index, self.log_returns)
        ax2.set_title(f"{self.ticker} Log Returns")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Log Return")
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
            
    def save_data(self, directory="data"):
        """
        Save the data and log returns to CSV files
        
        Parameters:
        -----------
        directory : str
            Directory to save the data (default: "data")
        """
        if self.data is None or self.data.empty:
            self.data = self.fetch_data()
            
        if self.log_returns is None:
            self.log_returns = self.calculate_log_returns()
            
        os.makedirs(directory, exist_ok=True)
        
        self.data.to_csv(os.path.join(directory, f"{self.ticker}_data.csv"))
        self.log_returns.to_csv(os.path.join(directory, f"{self.ticker}_log_returns.csv"))
        
        print(f"Data saved to {directory}")


if __name__ == "__main__":
    pipeline = DataPipeline(ticker="SPY", start_date="2010-01-01")
    data = pipeline.fetch_data()
    log_returns = pipeline.calculate_log_returns()
    
    print(f"Data shape: {data.shape}")
    print(f"Log returns shape: {log_returns.shape}")
    
    pipeline.plot_data(save_path="data/spy_plot.png")
    pipeline.save_data()
    
    train_data, test_data = pipeline.split_data(train_ratio=0.7)
    print(f"Training data shape: {train_data.shape}")
    print(f"Testing data shape: {test_data.shape}")
