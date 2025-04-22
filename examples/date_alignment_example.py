"""
Example script demonstrating how to properly align dates with data
to prevent KeyError when accessing price data with test dates
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from data_pipeline import DataPipeline
from main import align_dates_with_data

def demonstrate_date_alignment():
    """
    Demonstrate how to properly align dates with data to prevent KeyError
    """
    print("=== Date Alignment Example ===")
    
    pipeline = DataPipeline(ticker="SPY", start_date="2015-01-01", end_date="2020-01-01")
    data = pipeline.fetch_data()
    
    if data.empty:
        print("Error: Data is empty. Cannot proceed with example.")
        return
    
    test_dates = [
        datetime(2019, 1, 1),  # New Year's Day (market closed)
        datetime(2019, 1, 2),  # Regular trading day
        datetime(2019, 1, 5),  # Saturday (market closed)
        datetime(2019, 1, 6),  # Sunday (market closed)
        datetime(2019, 1, 7)   # Regular trading day
    ]
    
    print("\n=== Problematic Code (Will Cause KeyError) ===")
    print("test_prices = data['Close'][test_dates]")
    print("This will fail because some test_dates are not in the data index (weekends/holidays)")
    
    print("\n=== Solution 1: Filter dates manually ===")
    print("test_dates_in_data = [date for date in test_dates if date in data.index]")
    print("test_prices = data['Close'].loc[test_dates_in_data]")
    
    test_dates_in_data = [date for date in test_dates if date in data.index]
    if not test_dates_in_data:
        print("Warning: No test dates found in data index. Using all available dates.")
        test_prices_1 = data['Close']
    else:
        test_prices_1 = data['Close'].loc[test_dates_in_data]
    
    print(f"\nTest dates: {test_dates}")
    print(f"Test dates in data: {test_dates_in_data}")
    print(f"Test prices shape: {test_prices_1.shape}")
    
    print("\n=== Solution 2: Use the align_dates_with_data utility function ===")
    print("from main import align_dates_with_data")
    print("test_prices = align_dates_with_data(test_dates, data['Close'])")
    
    test_prices_2 = align_dates_with_data(test_dates, data['Close'])
    print(f"Test prices shape: {test_prices_2.shape}")
    
    print("\n=== Solution 3: Use the RegimeSignalGenerator's align_dates_with_data method ===")
    print("from signal_generation import RegimeSignalGenerator")
    print("signal_generator = RegimeSignalGenerator(model=model)")
    print("test_dates_in_data = signal_generator.align_dates_with_data(test_dates, data)")
    print("test_prices = data['Close'].loc[test_dates_in_data]")
    
    print("\n=== Date Alignment Example Complete ===")

if __name__ == "__main__":
    os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)
    demonstrate_date_alignment()
