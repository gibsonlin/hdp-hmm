"""
Test script for the data pipeline to ensure it handles errors gracefully
"""

import os
import sys
from datetime import datetime, timedelta

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_pipeline import DataPipeline

def test_data_pipeline():
    """
    Test the data pipeline with various scenarios
    """
    print("=== Testing Data Pipeline ===")
    
    print("\nTest 1: Normal case with valid ticker and date range")
    pipeline = DataPipeline(ticker="SPY", start_date="2020-01-01", end_date="2023-01-01")
    data = pipeline.fetch_data()
    print(f"Data shape: {data.shape}")
    
    if not data.empty:
        log_returns = pipeline.calculate_log_returns()
        print(f"Log returns shape: {log_returns.shape}")
        
        train_data, test_data = pipeline.split_data(train_ratio=0.7)
        print(f"Training data shape: {train_data.shape}")
        print(f"Testing data shape: {test_data.shape}")
    
    print("\nTest 2: Invalid ticker")
    pipeline = DataPipeline(ticker="INVALID_TICKER", start_date="2020-01-01", end_date="2023-01-01")
    data = pipeline.fetch_data()
    print(f"Data shape: {data.shape}")
    
    log_returns = pipeline.calculate_log_returns()
    print(f"Log returns shape: {log_returns.shape}")
    
    train_data, test_data = pipeline.split_data(train_ratio=0.7)
    print(f"Training data shape: {train_data.shape}")
    print(f"Testing data shape: {test_data.shape}")
    
    print("\nTest 3: Future date range")
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    next_month = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
    
    pipeline = DataPipeline(ticker="SPY", start_date=tomorrow, end_date=next_month)
    data = pipeline.fetch_data()
    print(f"Data shape: {data.shape}")
    
    log_returns = pipeline.calculate_log_returns()
    print(f"Log returns shape: {log_returns.shape}")
    
    train_data, test_data = pipeline.split_data(train_ratio=0.7)
    print(f"Training data shape: {train_data.shape}")
    print(f"Testing data shape: {test_data.shape}")
    
    print("\n=== Data Pipeline Testing Complete ===")

if __name__ == "__main__":
    test_data_pipeline()
