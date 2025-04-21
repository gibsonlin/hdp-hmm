"""
Live Data Ingestion Module for HDP-HMM MCMC Financial Regime Detection
- Provides tools for ingesting live financial data
- Supports various data sources and formats
- Integrates with the live trading API
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import json
import logging
import time
import requests
import websocket
import threading
import queue

from data_pipeline import DataPipeline
from live_trading_api import LiveTradingAPI

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('live_data_ingest')

class LiveDataIngestor:
    """
    Module for ingesting live financial data from various sources
    """
    
    def __init__(self, ticker="SPY", data_dir="live_data", 
                 session_id=None, api_key=None):
        """
        Initialize the live data ingestor
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol to fetch data for
        data_dir : str
            Directory to save ingested data
        session_id : str
            Unique identifier for the ingestion session (default: timestamp)
        api_key : str
            API key for data providers that require authentication
        """
        self.ticker = ticker
        self.data_dir = data_dir
        self.session_id = session_id if session_id else datetime.now().strftime("%Y%m%d_%H%M%S")
        self.api_key = api_key
        
        self.session_dir = os.path.join(data_dir, self.session_id)
        os.makedirs(self.session_dir, exist_ok=True)
        
        self.data = pd.DataFrame()
        self.data_queue = queue.Queue()
        self.is_running = False
        self.ingest_thread = None
        
        logger.info(f"Live data ingestor initialized for {ticker} with session ID: {self.session_id}")
        
    def connect_to_trading_api(self, api):
        """
        Connect to a LiveTradingAPI instance
        
        Parameters:
        -----------
        api : LiveTradingAPI
            Instance of LiveTradingAPI to connect to
            
        Returns:
        --------
        bool
            True if connection successful, False otherwise
        """
        if not isinstance(api, LiveTradingAPI):
            logger.error("Invalid API instance provided")
            return False
            
        self.trading_api = api
        logger.info(f"Connected to trading API with session ID: {api.session_id}")
        return True
        
    def ingest_from_csv(self, file_path, date_column=None, timestamp_format=None):
        """
        Ingest data from a CSV file
        
        Parameters:
        -----------
        file_path : str
            Path to the CSV file
        date_column : str
            Name of the column containing dates/timestamps
        timestamp_format : str
            Format string for parsing timestamps
            
        Returns:
        --------
        pandas.DataFrame
            Ingested data
        """
        logger.info(f"Ingesting data from CSV file: {file_path}")
        
        try:
            if date_column and timestamp_format:
                df = pd.read_csv(file_path, parse_dates=[date_column], date_format=timestamp_format)
                df.set_index(date_column, inplace=True)
            elif date_column:
                df = pd.read_csv(file_path, parse_dates=[date_column])
                df.set_index(date_column, inplace=True)
            else:
                df = pd.read_csv(file_path)
                
            self.data = df
            
            output_path = os.path.join(self.session_dir, "ingested_data.csv")
            self.data.to_csv(output_path)
            logger.info(f"Ingested data saved to {output_path}")
            
            return self.data
            
        except Exception as e:
            logger.error(f"Error ingesting data from CSV: {str(e)}")
            return pd.DataFrame()
            
    def ingest_from_api(self, api_url, params=None, headers=None, 
                       data_key=None, timestamp_key=None, timestamp_format=None):
        """
        Ingest data from a REST API
        
        Parameters:
        -----------
        api_url : str
            URL of the API endpoint
        params : dict
            Query parameters for the API request
        headers : dict
            Headers for the API request
        data_key : str
            JSON key containing the data in the API response
        timestamp_key : str
            JSON key containing the timestamp in each data point
        timestamp_format : str
            Format string for parsing timestamps
            
        Returns:
        --------
        pandas.DataFrame
            Ingested data
        """
        logger.info(f"Ingesting data from API: {api_url}")
        
        if not headers and self.api_key:
            headers = {'Authorization': f'Bearer {self.api_key}'}
            
        try:
            response = requests.get(api_url, params=params, headers=headers)
            response.raise_for_status()
            
            data = response.json()
            
            if data_key:
                data = data[data_key]
                
            df = pd.DataFrame(data)
            
            if timestamp_key and timestamp_format:
                df[timestamp_key] = pd.to_datetime(df[timestamp_key], format=timestamp_format)
                df.set_index(timestamp_key, inplace=True)
            elif timestamp_key:
                df[timestamp_key] = pd.to_datetime(df[timestamp_key])
                df.set_index(timestamp_key, inplace=True)
                
            self.data = df
            
            output_path = os.path.join(self.session_dir, "ingested_data.csv")
            self.data.to_csv(output_path)
            logger.info(f"Ingested data saved to {output_path}")
            
            return self.data
            
        except Exception as e:
            logger.error(f"Error ingesting data from API: {str(e)}")
            return pd.DataFrame()
            
    def start_websocket_ingest(self, ws_url, on_message=None, on_error=None, 
                              on_close=None, on_open=None, headers=None):
        """
        Start ingesting data from a WebSocket connection
        
        Parameters:
        -----------
        ws_url : str
            WebSocket URL to connect to
        on_message : function
            Callback function for handling messages
        on_error : function
            Callback function for handling errors
        on_close : function
            Callback function for handling connection close
        on_open : function
            Callback function for handling connection open
        headers : dict
            Headers for the WebSocket connection
            
        Returns:
        --------
        bool
            True if WebSocket connection started, False otherwise
        """
        logger.info(f"Starting WebSocket ingestion from: {ws_url}")
        
        if not headers and self.api_key:
            headers = {'Authorization': f'Bearer {self.api_key}'}
            
        def default_on_message(ws, message):
            try:
                data = json.loads(message)
                self.data_queue.put(data)
                logger.debug(f"Received data: {data}")
            except Exception as e:
                logger.error(f"Error processing WebSocket message: {str(e)}")
                
        def default_on_error(ws, error):
            logger.error(f"WebSocket error: {str(error)}")
            
        def default_on_close(ws, close_status_code, close_msg):
            logger.info(f"WebSocket connection closed: {close_msg} (code: {close_status_code})")
            self.is_running = False
            
        def default_on_open(ws):
            logger.info("WebSocket connection opened")
            self.is_running = True
            
        on_message = on_message or default_on_message
        on_error = on_error or default_on_error
        on_close = on_close or default_on_close
        on_open = on_open or default_on_open
        
        try:
            self.ws = websocket.WebSocketApp(
                ws_url,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
                on_open=on_open,
                header=headers
            )
            
            self.ingest_thread = threading.Thread(target=self.ws.run_forever)
            self.ingest_thread.daemon = True
            self.ingest_thread.start()
            
            time.sleep(1)
            
            return self.is_running
            
        except Exception as e:
            logger.error(f"Error starting WebSocket ingestion: {str(e)}")
            return False
            
    def stop_websocket_ingest(self):
        """
        Stop ingesting data from WebSocket connection
        
        Returns:
        --------
        bool
            True if WebSocket connection stopped, False otherwise
        """
        if hasattr(self, 'ws') and self.is_running:
            logger.info("Stopping WebSocket ingestion")
            self.ws.close()
            self.is_running = False
            return True
        else:
            logger.warning("No active WebSocket connection to stop")
            return False
            
    def process_data_queue(self, transform_func=None, batch_size=10, 
                          batch_interval=5, send_to_api=True):
        """
        Process data from the queue and optionally send to trading API
        
        Parameters:
        -----------
        transform_func : function
            Function to transform data before processing
        batch_size : int
            Number of data points to process in each batch
        batch_interval : int
            Time interval (in seconds) between batch processing
        send_to_api : bool
            Whether to send processed data to trading API
            
        Returns:
        --------
        bool
            True if processing started, False otherwise
        """
        if not self.is_running:
            logger.warning("Cannot process data queue: WebSocket ingestion not running")
            return False
            
        logger.info(f"Starting data queue processing with batch size {batch_size} and interval {batch_interval}s")
        
        def process_queue():
            while self.is_running:
                batch = []
                
                start_time = time.time()
                while len(batch) < batch_size and time.time() - start_time < batch_interval:
                    try:
                        item = self.data_queue.get(timeout=0.1)
                        
                        if transform_func:
                            item = transform_func(item)
                            
                        batch.append(item)
                        self.data_queue.task_done()
                    except queue.Empty:
                        continue
                        
                if batch:
                    batch_df = pd.DataFrame(batch)
                    
                    if 'timestamp' not in batch_df.columns:
                        batch_df['timestamp'] = datetime.now()
                        
                    batch_df.set_index('timestamp', inplace=True)
                    
                    self.data = pd.concat([self.data, batch_df])
                    
                    output_path = os.path.join(self.session_dir, "ingested_data.csv")
                    self.data.to_csv(output_path)
                    
                    if send_to_api and hasattr(self, 'trading_api'):
                        self.trading_api.update_live_data(batch_df)
                        
                        signal = self.trading_api.generate_signal()
                        logger.info(f"Generated signal: {signal}")
                        
                    logger.info(f"Processed batch of {len(batch)} data points")
                    
                time.sleep(0.1)  # Small delay to prevent CPU hogging
                
        self.process_thread = threading.Thread(target=process_queue)
        self.process_thread.daemon = True
        self.process_thread.start()
        
        return True
        
    def simulate_live_data(self, data, interval='1d', delay=1.0, send_to_api=True):
        """
        Simulate live data ingestion from historical data
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Historical data to simulate from
        interval : str
            Time interval between data points
        delay : float
            Delay between data points (in seconds)
        send_to_api : bool
            Whether to send simulated data to trading API
            
        Returns:
        --------
        bool
            True if simulation completed successfully, False otherwise
        """
        logger.info(f"Starting live data simulation with {len(data)} data points")
        
        if data.empty:
            logger.error("Cannot simulate with empty data")
            return False
            
        try:
            self.data = pd.DataFrame()
            
            for i in range(len(data)):
                data_point = data.iloc[i:i+1]
                
                self.data = pd.concat([self.data, data_point])
                
                output_path = os.path.join(self.session_dir, "simulated_data.csv")
                self.data.to_csv(output_path)
                
                if send_to_api and hasattr(self, 'trading_api'):
                    self.trading_api.update_live_data(data_point)
                    
                    signal = self.trading_api.generate_signal()
                    logger.info(f"Generated signal: {signal}")
                    
                logger.info(f"Simulated data point {i+1}/{len(data)}")
                
                if i < len(data) - 1:
                    if interval == '1d':
                        time.sleep(delay)  # Default 1 second delay for daily data
                    elif interval == '1h':
                        time.sleep(delay / 24)  # Faster for hourly data
                    elif interval == '1m':
                        time.sleep(delay / (24 * 60))  # Even faster for minute data
                        
            logger.info("Live data simulation completed")
            return True
            
        except Exception as e:
            logger.error(f"Error during live data simulation: {str(e)}")
            return False
            
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
            'ticker': self.ticker,
            'data_dir': self.data_dir,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        session_file = os.path.join(self.session_dir, "ingestor_state.json")
        with open(session_file, 'w') as f:
            json.dump(session_state, f, indent=4)
            
        logger.info(f"Ingestor state saved to {session_file}")
        
        return session_file
    
    @classmethod
    def load_session(cls, session_id, data_dir='live_data'):
        """
        Load a saved session
        
        Parameters:
        -----------
        session_id : str
            Session ID to load
        data_dir : str
            Directory containing session data
            
        Returns:
        --------
        LiveDataIngestor
            Loaded session
        """
        session_path = os.path.join(data_dir, session_id)
        session_file = os.path.join(session_path, "ingestor_state.json")
        
        if not os.path.exists(session_file):
            logger.error(f"Session file not found: {session_file}")
            return None
            
        with open(session_file, 'r') as f:
            session_state = json.load(f)
            
        logger.info(f"Loading session from {session_file}")
        
        ingestor = cls(
            ticker=session_state['ticker'],
            data_dir=session_state['data_dir'],
            session_id=session_id
        )
        
        data_path = os.path.join(session_path, "ingested_data.csv")
        if os.path.exists(data_path):
            ingestor.data = pd.read_csv(data_path, index_col=0, parse_dates=True)
            
        logger.info(f"Session {session_id} loaded successfully")
        
        return ingestor


if __name__ == "__main__":
    ingestor = LiveDataIngestor(ticker="SPY")
    
    api = LiveTradingAPI(
        model_path="models/hdp_hmm_model.pkl",
        mapping_method="mean_based",
        confidence_threshold=0.6,
        lookback_window=30
    )
    
    ingestor.connect_to_trading_api(api)
    
    pipeline = DataPipeline(ticker="SPY", start_date="2022-01-01")
    data = pipeline.fetch_data()
    
    ingestor.simulate_live_data(data, interval='1d', delay=0.5)
    
    ingestor.save_session()
