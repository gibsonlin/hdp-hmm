"""
Data Pipeline for HDP-HMM MCMC Financial Regime Detection
- Fetches SPY daily data using yfinance with fallback to alternative sources
- Calculates log returns
- Prepares data for model training
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
import os
import time
import pandas_datareader as pdr
import requests

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
        
    def fetch_data(self, max_retries=3):
        """
        Fetch daily data for the specified ticker with fallback to alternative sources
        
        Parameters:
        -----------
        max_retries : int
            Maximum number of retry attempts if download fails (default: 3)
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing the fetched data
        """
        print(f"Fetching data for {self.ticker} from {self.start_date} to {self.end_date}")
        
        for attempt in range(max_retries):
            try:
                self.data = yf.download(
                    self.ticker, 
                    start=self.start_date, 
                    end=self.end_date, 
                    progress=False
                )
                
                if self.data is not None and not self.data.empty:
                    print(f"Successfully downloaded {len(self.data)} rows of data using yfinance")
                    return self.data
                else:
                    print(f"Warning: Downloaded data is empty. Attempt {attempt+1}/{max_retries}")
                    
            except Exception as e:
                print(f"Error downloading data from yfinance (Attempt {attempt+1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    print("Retrying with yfinance...")
                    time.sleep(1)  # Add delay between retries
        
        print("Trying alternative data sources...")
        
        try:
            print("Trying Yahoo Finance via pandas_datareader...")
            self.data = pdr.get_data_yahoo(
                self.ticker,
                start=self.start_date,
                end=self.end_date
            )
            if self.data is not None and not self.data.empty:
                print(f"Successfully downloaded {len(self.data)} rows of data using pandas_datareader (Yahoo)")
                return self.data
        except Exception as e:
            print(f"Error downloading data from pandas_datareader (Yahoo): {str(e)}")
        
        try:
            print("Trying Stooq data source...")
            self.data = pdr.stooq.StooqDailyReader(
                self.ticker,
                start=self.start_date,
                end=self.end_date
            ).read()
            if self.data is not None and not self.data.empty:
                print(f"Successfully downloaded {len(self.data)} rows of data using Stooq")
                if 'Close' not in self.data.columns and 'close' in self.data.columns:
                    self.data.rename(columns={
                        'open': 'Open',
                        'high': 'High',
                        'low': 'Low',
                        'close': 'Close',
                        'volume': 'Volume'
                    }, inplace=True)
                return self.data
        except Exception as e:
            print(f"Error downloading data from Stooq: {str(e)}")
        
        print("All data sources failed. Creating empty DataFrame.")
        self.data = pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'])
            
        return self.data
    
    def calculate_log_returns(self):
        """
        Calculate log returns from the price data
        
        Returns:
        --------
        pandas.Series
            Series containing the log returns
        """
        if self.data is None or self.data.empty:
            self.data = self.fetch_data()
            
        if self.data.empty:
            print("Warning: Cannot calculate log returns from empty data")
            self.log_returns = pd.Series(dtype=float)
            return self.log_returns
            
        close_prices = self.data['Close']
        # Calculate price ratio
        price_ratio = close_prices / close_prices.shift(1)
        self.log_returns = price_ratio.apply(np.log).dropna()
        
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
            
        if isinstance(self.log_returns, pd.Series) and self.log_returns.empty:
            print("Warning: Cannot prepare training data from empty log returns")
            return np.array([]).reshape(-1, 1)
            
        return np.array(self.log_returns).reshape(-1, 1)
    
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
            
        if isinstance(self.log_returns, pd.Series) and self.log_returns.empty:
            print("Warning: Cannot split empty log returns")
            empty_array = np.array([]).reshape(-1, 1)
            return empty_array, empty_array
            
        split_idx = int(len(self.log_returns) * train_ratio)
        
        if isinstance(self.log_returns, pd.Series):
            train_data = np.array(self.log_returns.iloc[:split_idx]).reshape(-1, 1)
            test_data = np.array(self.log_returns.iloc[split_idx:]).reshape(-1, 1)
        else:
            train_data = np.array(self.log_returns[:split_idx]).reshape(-1, 1)
            test_data = np.array(self.log_returns[split_idx:]).reshape(-1, 1)
        
        return train_data, test_data
    
    def plot_data(self, save_path=None):
        """
        Plot the price data and log returns
        
        Parameters:
        -----------
        save_path : str
            Path to save the plot (default: None, display only)
        
        Returns:
        --------
        bool
            True if plot was created, False if data was empty
        """
        if self.data is None or self.data.empty:
            self.data = self.fetch_data()
            
        if self.data.empty:
            print("Warning: Cannot create plot with empty data")
            return False
            
        if self.log_returns is None:
            self.log_returns = self.calculate_log_returns()
            
        if isinstance(self.log_returns, pd.Series) and self.log_returns.empty:
            print("Warning: Cannot create plot with empty log returns")
            return False
            
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot price data
        ax1.plot(self.data.index, self.data['Close'])
        ax1.set_title(f"{self.ticker} Close Price")
        ax1.set_ylabel("Price ($)")
        ax1.grid(True)
        
        if isinstance(self.log_returns, pd.Series):
            ax2.plot(self.log_returns.index, self.log_returns)
        else:
            dates = self.data.index[1:]  # Skip first date due to log return calculation
            ax2.plot(dates, self.log_returns)
            
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
            
        return True
            
    def save_data(self, directory="data"):
        """
        Save the data and log returns to CSV files
        
        Parameters:
        -----------
        directory : str
            Directory to save the data (default: "data")
            
        Returns:
        --------
        bool
            True if data was saved, False if data was empty
        """
        if self.data is None or self.data.empty:
            self.data = self.fetch_data()
            
        if self.data.empty:
            print("Warning: Cannot save empty data")
            return False
            
        if self.log_returns is None:
            self.log_returns = self.calculate_log_returns()
            
        if isinstance(self.log_returns, pd.Series) and self.log_returns.empty:
            print("Warning: Cannot save empty log returns")
            return False
            
        os.makedirs(directory, exist_ok=True)
        
        # Save price data
        self.data.to_csv(os.path.join(directory, f"{self.ticker}_data.csv"))
        print(f"Price data saved to {os.path.join(directory, f'{self.ticker}_data.csv')}")
        
        if isinstance(self.log_returns, pd.Series):
            self.log_returns.to_csv(os.path.join(directory, f"{self.ticker}_log_returns.csv"))
        else:
            if hasattr(self.log_returns, 'flatten'):
                pd.Series(
                    self.log_returns.flatten(), 
                    index=self.data.index[1:len(self.log_returns)+1]
                ).to_csv(os.path.join(directory, f"{self.ticker}_log_returns.csv"))
            else:
                self.log_returns.to_csv(os.path.join(directory, f"{self.ticker}_log_returns.csv"))
        
        print(f"Log returns saved to {os.path.join(directory, f'{self.ticker}_log_returns.csv')}")
        return True


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
