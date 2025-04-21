"""
Test script to verify the data pipeline with valid data
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_pipeline import DataPipeline

def test_valid_data():
    """
    Test the data pipeline with valid data
    """
    print("=== Testing Data Pipeline with Valid Data ===")
    
    pipeline = DataPipeline(ticker="SPY", start_date="2015-01-01", end_date="2020-01-01")
    data = pipeline.fetch_data()
    print(f"Data shape: {data.shape}")
    
    if not data.empty:
        log_returns = pipeline.calculate_log_returns()
        print(f"Log returns shape: {log_returns.shape}")
        
        train_data, test_data = pipeline.split_data(train_ratio=0.7)
        print(f"Training data shape: {train_data.shape}")
        print(f"Testing data shape: {test_data.shape}")
        
        plot_result = pipeline.plot_data()
        print(f"Plot created: {plot_result}")
        
        save_result = pipeline.save_data(directory="test_data")
        print(f"Data saved: {save_result}")
    
    print("\n=== Data Pipeline Testing Complete ===")

if __name__ == "__main__":
    test_valid_data()
