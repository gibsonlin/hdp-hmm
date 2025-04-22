"""
Live Trading API for HDP-HMM MCMC Financial Regime Detection
- Provides API for live trading sessions
- Allows feeding forward data for testing
- Generates real-time trading signals
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import json
import pickle
import time
import logging

from hdp_hmm import HDPHMM
from signal_generation import RegimeSignalGenerator
from data_pipeline import DataPipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('live_trading_api')

class LiveTradingAPI:
    """
    API for live trading sessions using HDP-HMM model
    """
    
    def __init__(self, model_path, mapping_method='mean_based', 
                 confidence_threshold=0.6, lookback_window=30,
                 session_id=None, session_dir='live_sessions'):
        """
        Initialize the live trading API
        
        Parameters:
        -----------
        model_path : str
            Path to the saved HDP-HMM model
        mapping_method : str
            Method to use for state mapping ('mean_based', 'volatility_based', 'combined')
        confidence_threshold : float
            Threshold for state probability to generate a signal
        lookback_window : int
            Number of days to use for state probability calculation
        session_id : str
            Unique identifier for the trading session (default: timestamp)
        session_dir : str
            Directory to save session data
        """
        self.model_path = model_path
        self.mapping_method = mapping_method
        self.confidence_threshold = confidence_threshold
        self.lookback_window = lookback_window
        
        logger.info(f"Loading model from {model_path}")
        self.model = HDPHMM.load(model_path)
        
        self.signal_generator = RegimeSignalGenerator(
            model=self.model,
            mapping_method=mapping_method,
            confidence_threshold=confidence_threshold
        )
        
        self.signal_generator.create_state_mapping()
        
        self.session_id = session_id if session_id else datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(session_dir, self.session_id)
        os.makedirs(self.session_dir, exist_ok=True)
        
        self.historical_data = pd.DataFrame()
        self.live_data = pd.DataFrame()
        self.signals = pd.DataFrame()
        self.state_probs = pd.DataFrame()
        
        logger.info(f"Live trading API initialized with session ID: {self.session_id}")
        
    def load_historical_data(self, ticker="SPY", start_date=None, end_date=None, 
                            data_path=None):
        """
        Load historical data for reference
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol to fetch data for
        start_date : str
            Start date for data fetching in YYYY-MM-DD format
        end_date : str
            End date for data fetching in YYYY-MM-DD format
        data_path : str
            Path to CSV file with historical data
            
        Returns:
        --------
        pandas.DataFrame
            Historical data
        """
        if data_path:
            logger.info(f"Loading historical data from {data_path}")
            self.historical_data = pd.read_csv(data_path, index_col=0, parse_dates=True)
        else:
            if not start_date:
                start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
            if not end_date:
                end_date = datetime.now().strftime("%Y-%m-%d")
                
            logger.info(f"Fetching historical data for {ticker} from {start_date} to {end_date}")
            pipeline = DataPipeline(ticker=ticker, start_date=start_date, end_date=end_date)
            data = pipeline.fetch_data()
            
            if data.empty:
                logger.error("Failed to fetch historical data")
                return None
                
            self.historical_data = data
            
        historical_data_path = os.path.join(self.session_dir, "historical_data.csv")
        self.historical_data.to_csv(historical_data_path)
        logger.info(f"Historical data saved to {historical_data_path}")
        
        return self.historical_data
    
    def initialize_live_data(self, initial_data=None):
        """
        Initialize live data storage
        
        Parameters:
        -----------
        initial_data : pandas.DataFrame
            Initial data to start with
            
        Returns:
        --------
        pandas.DataFrame
            Live data
        """
        if initial_data is not None:
            logger.info("Initializing live data with provided data")
            self.live_data = initial_data.copy()
        elif not self.historical_data.empty:
            logger.info("Initializing live data with last points from historical data")
            self.live_data = self.historical_data.iloc[-self.lookback_window:].copy()
        else:
            logger.warning("No data available to initialize live data")
            self.live_data = pd.DataFrame()
            
        if not self.live_data.empty:
            live_data_path = os.path.join(self.session_dir, "live_data.csv")
            self.live_data.to_csv(live_data_path)
            logger.info(f"Initial live data saved to {live_data_path}")
            
        return self.live_data
    
    def update_live_data(self, new_data, append=True):
        """
        Update live data with new data points
        
        Parameters:
        -----------
        new_data : pandas.DataFrame or dict
            New data points to add
        append : bool
            Whether to append new data or replace existing data
            
        Returns:
        --------
        pandas.DataFrame
            Updated live data
        """
        if isinstance(new_data, dict):
            new_df = pd.DataFrame([new_data])
            if 'date' in new_data:
                new_df.set_index('date', inplace=True)
            elif 'timestamp' in new_data:
                new_df.set_index('timestamp', inplace=True)
            else:
                new_df = new_df.set_index(pd.DatetimeIndex([datetime.now()]))
        else:
            new_df = new_data
            
        logger.info(f"Updating live data with {len(new_df)} new data points")
        
        if append:
            self.live_data = pd.concat([self.live_data, new_df])
        else:
            self.live_data = new_df
            
        live_data_path = os.path.join(self.session_dir, "live_data.csv")
        self.live_data.to_csv(live_data_path)
        logger.info(f"Updated live data saved to {live_data_path}")
        
        return self.live_data
    
    def calculate_log_returns(self, data=None):
        """
        Calculate log returns from price data
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Price data to calculate log returns from
            
        Returns:
        --------
        pandas.Series
            Log returns
        """
        if data is None:
            data = self.live_data
            
        if data.empty:
            logger.warning("Cannot calculate log returns from empty data")
            return pd.Series(dtype=float)
            
        if 'Close' in data.columns:
            close_prices = data['Close']
        elif 'close' in data.columns:
            close_prices = data['close']
        else:
            logger.error("No close price column found in data")
            return pd.Series(dtype=float)
            
        price_ratio = close_prices / close_prices.shift(1)
        log_returns = price_ratio.apply(np.log).dropna()
        
        return log_returns
    
    def generate_signal(self, new_data=None):
        """
        Generate trading signal based on new data
        
        Parameters:
        -----------
        new_data : pandas.DataFrame or dict
            New data points to process
            
        Returns:
        --------
        dict
            Trading signal with state probabilities
        """
        if new_data is not None:
            self.update_live_data(new_data)
            
        if self.live_data.empty:
            logger.warning("Cannot generate signal from empty data")
            return None
            
        log_returns = self.calculate_log_returns()
        
        if len(log_returns) < 2:
            logger.warning("Not enough data points to generate signal")
            return None
            
        X = np.array(log_returns.values).reshape(-1, 1)
        
        state_probs = self.model.predict_state_probabilities(X)
        
        latest_probs = state_probs[-1]
        
        signal = self.signal_generator.generate_signal_from_probs(latest_probs)
        
        timestamp = self.live_data.index[-1]
        result = {
            'timestamp': timestamp,
            'signal': signal,
            'state_probs': {s: float(latest_probs[s]) for s in self.model.active_states},
            'price': float(self.live_data['Close'].iloc[-1]) if 'Close' in self.live_data.columns else None
        }
        
        signal_df = pd.DataFrame([result])
        signal_df.set_index('timestamp', inplace=True)
        self.signals = pd.concat([self.signals, signal_df])
        
        signals_path = os.path.join(self.session_dir, "signals.csv")
        self.signals.to_csv(signals_path)
        logger.info(f"Signals saved to {signals_path}")
        
        probs_df = pd.DataFrame([{
            'timestamp': timestamp,
            **{f'state_{s}': float(latest_probs[s]) for s in self.model.active_states}
        }])
        probs_df.set_index('timestamp', inplace=True)
        self.state_probs = pd.concat([self.state_probs, probs_df])
        
        state_probs_path = os.path.join(self.session_dir, "state_probs.csv")
        self.state_probs.to_csv(state_probs_path)
        
        return result
    
    def run_simulation(self, test_data, interval='1d'):
        """
        Run a simulation with test data
        
        Parameters:
        -----------
        test_data : pandas.DataFrame
            Test data to simulate with
        interval : str
            Time interval between data points
            
        Returns:
        --------
        pandas.DataFrame
            Simulation results
        """
        logger.info(f"Running simulation with {len(test_data)} data points")
        
        if len(test_data) <= self.lookback_window:
            logger.warning("Test data is smaller than lookback window")
            self.initialize_live_data(test_data)
            return self.generate_signal()
            
        self.initialize_live_data(test_data.iloc[:self.lookback_window])
        
        results = []
        for i in range(self.lookback_window, len(test_data)):
            new_point = test_data.iloc[i:i+1]
            signal = self.generate_signal(new_point)
            if signal:
                results.append(signal)
                
            if interval == '1d':
                time.sleep(0.01)  # 10ms delay for daily data
            elif interval == '1h':
                time.sleep(0.005)  # 5ms delay for hourly data
            elif interval == '1m':
                time.sleep(0.001)  # 1ms delay for minute data
                
        results_df = pd.DataFrame(results)
        if not results_df.empty:
            results_df.set_index('timestamp', inplace=True)
            results_path = os.path.join(self.session_dir, "simulation_results.csv")
            results_df.to_csv(results_path)
            logger.info(f"Simulation results saved to {results_path}")
            
        return results_df
    
    def plot_signals(self, save_path=None):
        """
        Plot trading signals with price data
        
        Parameters:
        -----------
        save_path : str
            Path to save the plot
            
        Returns:
        --------
        bool
            True if plot was created, False otherwise
        """
        if self.signals.empty or self.live_data.empty:
            logger.warning("Cannot create plot with empty data")
            return False
            
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        if 'Close' in self.live_data.columns:
            ax1.plot(self.live_data.index, self.live_data['Close'], label='Close Price')
        elif 'close' in self.live_data.columns:
            ax1.plot(self.live_data.index, self.live_data['close'], label='Close Price')
            
        buy_signals = self.signals[self.signals['signal'] == 1]
        sell_signals = self.signals[self.signals['signal'] == -1]
        neutral_signals = self.signals[self.signals['signal'] == 0]
        
        if not buy_signals.empty and 'price' in buy_signals.columns:
            ax1.scatter(buy_signals.index, buy_signals['price'], 
                       color='green', marker='^', s=100, label='Buy Signal')
                       
        if not sell_signals.empty and 'price' in sell_signals.columns:
            ax1.scatter(sell_signals.index, sell_signals['price'], 
                       color='red', marker='v', s=100, label='Sell Signal')
                       
        if not neutral_signals.empty and 'price' in neutral_signals.columns:
            ax1.scatter(neutral_signals.index, neutral_signals['price'], 
                       color='gray', marker='o', s=50, label='Neutral Signal')
                       
        ax1.set_title('Price and Trading Signals')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True)
        
        if not self.state_probs.empty:
            for col in self.state_probs.columns:
                ax2.plot(self.state_probs.index, self.state_probs[col], label=col)
                
            ax2.set_title('State Probabilities')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Probability')
            ax2.legend()
            ax2.grid(True)
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Plot saved to {save_path}")
            plt.close()
        else:
            save_path = os.path.join(self.session_dir, "signals_plot.png")
            plt.savefig(save_path)
            logger.info(f"Plot saved to {save_path}")
            plt.close()
            
        return True
    
    def save_session(self):
        """
        Save the current session state
        
        Returns:
        --------
        str
            Path to the saved session file
        """
        session_state = {
            'session_id': self.session_id,
            'model_path': self.model_path,
            'mapping_method': self.mapping_method,
            'confidence_threshold': self.confidence_threshold,
            'lookback_window': self.lookback_window,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        session_file = os.path.join(self.session_dir, "session_state.json")
        with open(session_file, 'w') as f:
            json.dump(session_state, f, indent=4)
            
        logger.info(f"Session state saved to {session_file}")
        
        return session_file
    
    @classmethod
    def load_session(cls, session_id, session_dir='live_sessions'):
        """
        Load a saved session
        
        Parameters:
        -----------
        session_id : str
            Session ID to load
        session_dir : str
            Directory containing session data
            
        Returns:
        --------
        LiveTradingAPI
            Loaded session
        """
        session_path = os.path.join(session_dir, session_id)
        session_file = os.path.join(session_path, "session_state.json")
        
        if not os.path.exists(session_file):
            logger.error(f"Session file not found: {session_file}")
            return None
            
        with open(session_file, 'r') as f:
            session_state = json.load(f)
            
        logger.info(f"Loading session from {session_file}")
        
        api = cls(
            model_path=session_state['model_path'],
            mapping_method=session_state['mapping_method'],
            confidence_threshold=session_state['confidence_threshold'],
            lookback_window=session_state['lookback_window'],
            session_id=session_id,
            session_dir=session_dir
        )
        
        live_data_path = os.path.join(session_path, "live_data.csv")
        if os.path.exists(live_data_path):
            api.live_data = pd.read_csv(live_data_path, index_col=0, parse_dates=True)
            
        signals_path = os.path.join(session_path, "signals.csv")
        if os.path.exists(signals_path):
            api.signals = pd.read_csv(signals_path, index_col=0, parse_dates=True)
            
        state_probs_path = os.path.join(session_path, "state_probs.csv")
        if os.path.exists(state_probs_path):
            api.state_probs = pd.read_csv(state_probs_path, index_col=0, parse_dates=True)
            
        logger.info(f"Session {session_id} loaded successfully")
        
        return api


if __name__ == "__main__":
    api = LiveTradingAPI(
        model_path="models/hdp_hmm_model.pkl",
        mapping_method="mean_based",
        confidence_threshold=0.6,
        lookback_window=30
    )
    
    api.load_historical_data(ticker="SPY", start_date="2020-01-01")
    
    api.initialize_live_data()
    
    signal = api.generate_signal()
    print(f"Signal: {signal}")
    
    api.plot_signals()
    
    api.save_session()
