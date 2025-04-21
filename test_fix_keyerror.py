"""
Test script to reproduce and fix the KeyError issue with test_prices
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_pipeline import DataPipeline
from src.hdp_hmm import HDPHMM
from src.signal_generation import RegimeSignalGenerator

def test_keyerror_fix():
    """
    Test the fix for the KeyError issue with test_prices
    """
    print("=== Testing KeyError Fix ===")
    
    pipeline = DataPipeline(ticker="SPY", start_date="2015-01-01", end_date="2020-01-01")
    data = pipeline.fetch_data()
    
    if data.empty:
        print("Error: Data is empty. Cannot proceed with test.")
        return
    
    log_returns = pipeline.calculate_log_returns()
    
    train_ratio = 0.7
    train_data, test_data = pipeline.split_data(train_ratio=train_ratio)
    
    test_dates = log_returns.index[int(len(log_returns) * train_ratio):]
    
    model = HDPHMM(alpha=1.0, gamma=1.0, kappa=10.0, max_states=5, obs_dim=1)
    model.fit(train_data, n_iter=10, verbose=True)
    
    signal_generator = RegimeSignalGenerator(model=model)
    state_mapping = signal_generator.create_state_mapping(method="mean_based")
    
    signals = signal_generator.generate_regime_change_signals(
        test_data, 
        dates=test_dates, 
        confidence_threshold=0.6
    )
    
    print("\n=== Testing the problematic code ===")
    print("Original code that causes KeyError:")
    print("test_prices = data['Close'][test_dates]")
    print("signal_generator.plot_signals(price_data=test_prices)")
    
    print("\n=== Testing the fixed code ===")
    print("Fixed code:")
    print("test_dates_in_data = [date for date in test_dates if date in data.index]")
    print("if not test_dates_in_data:")
    print("    print('Warning: No test dates found in data index. Using all available dates.')")
    print("    test_prices = data['Close']")
    print("else:")
    print("    test_prices = data['Close'].loc[test_dates_in_data]")
    print("signal_generator.plot_signals(price_data=test_prices)")
    
    test_dates_in_data = [date for date in test_dates if date in data.index]
    if not test_dates_in_data:
        print("Warning: No test dates found in data index. Using all available dates.")
        test_prices = data['Close']
    else:
        test_prices = data['Close'].loc[test_dates_in_data]
    
    print(f"\nTest dates count: {len(test_dates)}")
    print(f"Test dates in data count: {len(test_dates_in_data)}")
    print(f"Test prices shape: {test_prices.shape}")
    
    signal_generator.plot_signals(price_data=test_prices)
    
    print("\n=== KeyError Fix Testing Complete ===")

if __name__ == "__main__":
    test_keyerror_fix()
