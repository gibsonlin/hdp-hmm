"""
Signal Generation Module for HDP-HMM MCMC Financial Regime Detection
- Implements state-label-guideline mapping
- Generates trading signals based on regime changes
- Provides online filtering for real-time state probability calculation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import os
import pickle
from datetime import datetime

from hdp_hmm import HDPHMM

class RegimeSignalGenerator:
    """
    Signal generator based on HDP-HMM regime detection
    """
    
    def __init__(self, model=None, model_path=None):
        """
        Initialize the signal generator
        
        Parameters:
        -----------
        model : HDPHMM
            Trained HDP-HMM model
        model_path : str
            Path to a saved model file
        """
        if model is not None:
            self.model = model
        elif model_path is not None:
            self.model = HDPHMM.load(model_path)
        else:
            raise ValueError("Either model or model_path must be provided")
            
        self.state_labels = {}
        self.state_guidelines = {}
        self.signal_history = []
        self.current_state = None
        self.current_signal = 0  # 0: no position, 1: long, -1: short
        
    def create_state_mapping(self, method="mean_based", threshold=0.0):
        """
        Create state-label-guideline mapping
        
        Parameters:
        -----------
        method : str
            Method to use for mapping ('mean_based', 'volatility_based', 'combined')
        threshold : float
            Threshold for determining state labels
            
        Returns:
        --------
        dict
            State mapping dictionary
        """
        state_mapping = {}
        
        if not hasattr(self.model, 'state_parameters') or not self.model.state_parameters:
            raise ValueError("Model state parameters not available. Run model._extract_state_parameters() first.")
        
        active_states = self.model.active_states
        
        if method == "mean_based":
            for s in active_states:
                if s in self.model.state_parameters:
                    mean = self.model.state_parameters[s]['mean'][0]
                    if mean > threshold:
                        label = "Bull"
                        guideline = "Long"
                    else:
                        label = "Bear"
                        guideline = "Short"
                    
                    state_mapping[s] = {
                        'label': label,
                        'guideline': guideline,
                        'mean': mean,
                        'std': self.model.state_parameters[s]['std'][0],
                        'proportion': self.model.state_parameters[s]['proportion']
                    }
        
        elif method == "volatility_based":
            for s in active_states:
                if s in self.model.state_parameters:
                    std = self.model.state_parameters[s]['std'][0]
                    mean = self.model.state_parameters[s]['mean'][0]
                    
                    if std < np.median([self.model.state_parameters[i]['std'][0] for i in self.model.state_parameters]):
                        if mean > 0:
                            label = "Low Volatility Bull"
                            guideline = "Long"
                        else:
                            label = "Low Volatility Bear"
                            guideline = "Short"
                    else:
                        label = "High Volatility"
                        guideline = "Neutral"
                    
                    state_mapping[s] = {
                        'label': label,
                        'guideline': guideline,
                        'mean': mean,
                        'std': std,
                        'proportion': self.model.state_parameters[s]['proportion']
                    }
        
        elif method == "combined":
            for s in active_states:
                if s in self.model.state_parameters:
                    mean = self.model.state_parameters[s]['mean'][0]
                    std = self.model.state_parameters[s]['std'][0]
                    sharpe = mean / std if std > 0 else 0
                    
                    if sharpe > threshold:
                        label = "Positive Sharpe"
                        guideline = "Long"
                    elif sharpe < -threshold:
                        label = "Negative Sharpe"
                        guideline = "Short"
                    else:
                        label = "Neutral Sharpe"
                        guideline = "Neutral"
                    
                    state_mapping[s] = {
                        'label': label,
                        'guideline': guideline,
                        'mean': mean,
                        'std': std,
                        'sharpe': sharpe,
                        'proportion': self.model.state_parameters[s]['proportion']
                    }
        
        else:
            raise ValueError(f"Unknown mapping method: {method}")
        
        for s, mapping in state_mapping.items():
            self.state_labels[s] = mapping['label']
            self.state_guidelines[s] = mapping['guideline']
        
        return state_mapping
    
    def plot_state_mapping(self, save_path=None):
        """
        Plot state mapping visualization
        
        Parameters:
        -----------
        save_path : str
            Path to save the plot
        """
        if not self.state_labels:
            self.create_state_mapping()
        
        active_states = self.model.active_states
        
        data = []
        for s in active_states:
            if s in self.model.state_parameters:
                params = self.model.state_parameters[s]
                data.append({
                    'State': s,
                    'Label': self.state_labels.get(s, "Unknown"),
                    'Guideline': self.state_guidelines.get(s, "Unknown"),
                    'Mean': params['mean'][0],
                    'Std': params['std'][0],
                    'Count': params['count'],
                    'Proportion': params['proportion']
                })
        
        df = pd.DataFrame(data)
        
        if len(df) > 0:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            for label in df['Label'].unique():
                subset = df[df['Label'] == label]
                axes[0].scatter(subset['Mean'], subset['Std'], 
                               s=subset['Proportion'] * 1000, 
                               label=label, alpha=0.7)
                
                for _, row in subset.iterrows():
                    axes[0].annotate(str(int(row['State'])), 
                                    (row['Mean'], row['Std']),
                                    fontsize=9)
            
            axes[0].set_title('State Mapping: Mean vs Std')
            axes[0].set_xlabel('Mean Return')
            axes[0].set_ylabel('Standard Deviation')
            axes[0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
            axes[0].axvline(x=0, color='k', linestyle='--', alpha=0.3)
            axes[0].grid(True, alpha=0.3)
            axes[0].legend()
            
            guideline_groups = df.groupby('Guideline')['Proportion'].sum()
            axes[1].bar(guideline_groups.index, guideline_groups.values)
            axes[1].set_title('Trading Signal Distribution')
            axes[1].set_xlabel('Trading Signal')
            axes[1].set_ylabel('Proportion of Time')
            
            for i, v in enumerate(guideline_groups.values):
                axes[1].text(i, v + 0.01, f'{v:.2f}', ha='center')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
                print(f"Plot saved to {save_path}")
            else:
                plt.show()
        else:
            print("No state mapping data available for plotting")
    
    def calculate_state_probabilities(self, data):
        """
        Calculate state probabilities for new data (online filtering)
        
        Parameters:
        -----------
        data : numpy.ndarray
            New observation data of shape (n_samples, obs_dim)
            
        Returns:
        --------
        numpy.ndarray
            State probabilities of shape (n_samples, max_states)
        """
        return self.model.predict_state_probabilities(data)
    
    def predict_most_likely_states(self, data):
        """
        Predict most likely states for new data
        
        Parameters:
        -----------
        data : numpy.ndarray
            New observation data of shape (n_samples, obs_dim)
            
        Returns:
        --------
        numpy.ndarray
            Predicted state sequence of shape (n_samples,)
        """
        return self.model.predict_states(data)
    
    def generate_signals(self, data, dates=None, confidence_threshold=0.6):
        """
        Generate trading signals based on regime detection
        
        Parameters:
        -----------
        data : numpy.ndarray
            New observation data of shape (n_samples, obs_dim)
        dates : array-like
            Dates corresponding to the data points
        confidence_threshold : float
            Threshold for state probability to generate a signal
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing signals and state information
        """
        if not self.state_guidelines:
            self.create_state_mapping()
        
        state_probs = self.calculate_state_probabilities(data)
        
        states = self.predict_most_likely_states(data)
        
        signals = np.zeros(len(data))
        confidences = np.zeros(len(data))
        
        for t in range(len(data)):
            s = states[t]
            max_prob = np.max(state_probs[t])
            confidences[t] = max_prob
            
            if max_prob >= confidence_threshold:
                if s in self.state_guidelines:
                    guideline = self.state_guidelines[s]
                    if guideline == "Long":
                        signals[t] = 1
                    elif guideline == "Short":
                        signals[t] = -1
        
        if dates is None:
            dates = np.arange(len(data))
            
        result = pd.DataFrame({
            'Date': dates,
            'State': states,
            'Confidence': confidences,
            'Signal': signals
        })
        
        if isinstance(dates[0], (datetime, np.datetime64, pd.Timestamp)):
            result.set_index('Date', inplace=True)
        
        self.signal_history = result
        
        return result
    
    def generate_regime_change_signals(self, data, dates=None, confidence_threshold=0.6):
        """
        Generate trading signals only on regime changes
        
        Parameters:
        -----------
        data : numpy.ndarray
            New observation data of shape (n_samples, obs_dim)
        dates : array-like
            Dates corresponding to the data points
        confidence_threshold : float
            Threshold for state probability to generate a signal
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing signals and state information
        """
        if not self.state_guidelines:
            self.create_state_mapping()
        
        state_probs = self.calculate_state_probabilities(data)
        
        states = self.predict_most_likely_states(data)
        
        signals = np.zeros(len(data))
        confidences = np.zeros(len(data))
        
        current_signal = 0
        
        for t in range(len(data)):
            s = states[t]
            max_prob = np.max(state_probs[t])
            confidences[t] = max_prob
            
            if max_prob >= confidence_threshold:
                if s in self.state_guidelines:
                    guideline = self.state_guidelines[s]
                    
                    if guideline == "Long" and current_signal != 1:
                        signals[t] = 1
                        current_signal = 1
                    elif guideline == "Short" and current_signal != -1:
                        signals[t] = -1
                        current_signal = -1
                    elif guideline == "Neutral" and current_signal != 0:
                        signals[t] = 0
                        current_signal = 0
                    else:
                        signals[t] = current_signal
            else:
                signals[t] = current_signal
        
        if dates is None:
            dates = np.arange(len(data))
            
        result = pd.DataFrame({
            'Date': dates,
            'State': states,
            'Confidence': confidences,
            'Signal': signals
        })
        
        if isinstance(dates[0], (datetime, np.datetime64, pd.Timestamp)):
            result.set_index('Date', inplace=True)
        
        self.signal_history = result
        
        return result
    
    def plot_signals(self, price_data=None, save_path=None):
        """
        Plot trading signals with price data
        
        Parameters:
        -----------
        price_data : pandas.Series
            Price data with dates as index
        save_path : str
            Path to save the plot
        """
        if self.signal_history is None or len(self.signal_history) == 0:
            print("No signal history available. Generate signals first.")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        
        # Plot price data
        if price_data is not None:
            ax1.plot(price_data.index, price_data, label='Price')
            
            if isinstance(self.signal_history, pd.DataFrame):
                for signal, color, label in [(1, 'green', 'Long'), (-1, 'red', 'Short')]:
                    if 'Signal' in self.signal_history.columns:
                        signal_dates = []
                        for idx, row in self.signal_history.iterrows():
                            if row.get('Signal') == signal:
                                signal_dates.append(idx)
                        
                        for i in range(len(signal_dates) - 1):
                            ax1.axvspan(signal_dates[i], signal_dates[i+1], alpha=0.2, color=color)
            
            ax1.set_title('Price with Trading Signals')
            ax1.set_ylabel('Price')
            ax1.grid(True)
            ax1.legend()
        
        if isinstance(self.signal_history, pd.DataFrame) and 'Signal' in self.signal_history.columns:
            signal_values = []
            for idx, row in self.signal_history.iterrows():
                signal_values.append(row.get('Signal', 0))
            
            ax2.plot(self.signal_history.index, signal_values, label='Signal', color='blue')
            ax2.set_title('Trading Signals')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Signal')
            ax2.set_yticks([-1, 0, 1])
            ax2.set_yticklabels(['Short', 'Neutral', 'Long'])
            ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
    
    def save(self, filepath):
        """
        Save the signal generator to a file
        
        Parameters:
        -----------
        filepath : str
            Path to save the signal generator
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        
        print(f"Signal generator saved to {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """
        Load a signal generator from a file
        
        Parameters:
        -----------
        filepath : str
            Path to the saved signal generator
            
        Returns:
        --------
        RegimeSignalGenerator
            Loaded signal generator
        """
        with open(filepath, 'rb') as f:
            generator = pickle.load(f)
        
        print(f"Signal generator loaded from {filepath}")
        return generator
