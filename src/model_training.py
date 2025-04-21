"""
Model Training Script for HDP-HMM MCMC Financial Regime Detection
- Loads data from the data pipeline
- Trains the HDP-HMM model
- Saves the trained model
- Analyzes state parameters
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
from datetime import datetime

from data_pipeline import DataPipeline
from hdp_hmm import HDPHMM

def train_model(data_start_date="2010-01-01", 
                train_ratio=0.7,
                n_iter=1000, 
                alpha=1.0, 
                gamma=1.0, 
                kappa=10.0,
                max_states=20,
                save_dir="models",
                plot_dir="plots"):
    """
    Train the HDP-HMM model on SPY data
    
    Parameters:
    -----------
    data_start_date : str
        Start date for data fetching in YYYY-MM-DD format
    train_ratio : float
        Ratio of data to use for training
    n_iter : int
        Number of MCMC iterations
    alpha : float
        Concentration parameter for the Dirichlet Process prior on transitions
    gamma : float
        Concentration parameter for the Dirichlet Process prior on states
    kappa : float
        Self-transition bias parameter (sticky HDP-HMM)
    max_states : int
        Maximum number of states to consider (truncation level)
    save_dir : str
        Directory to save the trained model
    plot_dir : str
        Directory to save plots
    
    Returns:
    --------
    tuple
        (model, train_data, test_data)
    """
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    
    print("Fetching and preparing data...")
    pipeline = DataPipeline(ticker="SPY", start_date=data_start_date)
    data = pipeline.fetch_data()
    log_returns = pipeline.calculate_log_returns()
    
    pipeline.plot_data(save_path=os.path.join(plot_dir, "spy_data.png"))
    
    train_data, test_data = pipeline.split_data(train_ratio=train_ratio)
    
    print(f"Training data shape: {train_data.shape}")
    print(f"Testing data shape: {test_data.shape}")
    
    print("Initializing model...")
    model = HDPHMM(alpha=alpha, gamma=gamma, kappa=kappa, max_states=max_states, 
                  obs_dim=train_data.shape[1], sticky=True)
    
    print(f"Training model with {n_iter} iterations...")
    start_time = time.time()
    model.fit(train_data, n_iter=n_iter, verbose=True, 
             save_history=True, history_dir=save_dir)
    elapsed_time = time.time() - start_time
    print(f"Training completed in {elapsed_time:.2f} seconds")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(save_dir, f"hdp_hmm_model_{timestamp}.pkl")
    model.save(model_path)
    
    model.plot_log_likelihood(save_path=os.path.join(plot_dir, "log_likelihood.png"))
    
    model.plot_state_parameters(save_path=os.path.join(plot_dir, "state_parameters.png"))
    
    model.plot_state_sequence(data=log_returns, 
                             save_path=os.path.join(plot_dir, "state_sequence.png"))
    
    analyze_state_parameters(model, log_returns, save_dir=plot_dir)
    
    return model, train_data, test_data

def analyze_state_parameters(model, log_returns, save_dir="plots"):
    """
    Analyze state parameters and create visualizations
    
    Parameters:
    -----------
    model : HDPHMM
        Trained HDP-HMM model
    log_returns : pandas.Series
        Log returns data with dates as index
    save_dir : str
        Directory to save plots
    """
    print("Analyzing state parameters...")
    
    state_summary = pd.DataFrame()
    
    for s in model.active_states:
        if s in model.state_parameters:
            params = model.state_parameters[s]
            state_summary.loc[s, 'Mean'] = params['mean'][0]
            state_summary.loc[s, 'Std'] = params['std'][0]
            state_summary.loc[s, 'Count'] = params['count']
            state_summary.loc[s, 'Proportion'] = params['proportion']
    
    state_summary.to_csv(os.path.join(save_dir, "state_summary.csv"))
    print("State summary:")
    print(state_summary)
    
    transition_probs = model._compute_transition_probabilities()
    active_states = model.active_states
    
    if len(active_states) > 0:
        active_transition_probs = transition_probs[np.ix_(active_states, active_states)]
        
        plt.figure(figsize=(10, 8))
        plt.imshow(active_transition_probs, cmap='viridis', interpolation='none')
        plt.colorbar(label='Transition Probability')
        plt.title('State Transition Probabilities')
        plt.xlabel('To State')
        plt.ylabel('From State')
        
        plt.xticks(range(len(active_states)), active_states)
        plt.yticks(range(len(active_states)), active_states)
        
        for i in range(len(active_states)):
            for j in range(len(active_states)):
                plt.text(j, i, f'{active_transition_probs[i, j]:.2f}', 
                        ha='center', va='center', 
                        color='white' if active_transition_probs[i, j] < 0.5 else 'black')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "transition_matrix.png"))
        
        state_sequence = model.state_sequence
        dates = log_returns.index
        
        state_df = pd.DataFrame({'Date': dates[:len(state_sequence)], 'State': state_sequence})
        state_df.set_index('Date', inplace=True)
        
        plt.figure(figsize=(12, 6))
        
        state_durations = {}
        for s in active_states:
            durations = []
            current_duration = 0
            
            for state in state_sequence:
                if state == s:
                    current_duration += 1
                elif current_duration > 0:
                    durations.append(current_duration)
                    current_duration = 0
            
            if current_duration > 0:
                durations.append(current_duration)
                
            state_durations[s] = durations
            
            if durations:
                plt.hist(durations, bins=20, alpha=0.5, label=f'State {s}')
        
        plt.title('State Duration Distribution')
        plt.xlabel('Duration (days)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "state_durations.png"))
        
        plt.figure(figsize=(14, 8))
        
        ax1 = plt.subplot(2, 1, 1)
        spy_price = np.exp(log_returns.cumsum())
        ax1.plot(log_returns.index[:len(state_sequence)], spy_price[:len(state_sequence)])
        
        min_price = spy_price.min()
        max_price = spy_price.max()
        height = max_price - min_price
        
        for s in active_states:
            for i in range(1, len(state_sequence)):
                if state_sequence[i] == s:
                    ax1.axvspan(log_returns.index[i-1], log_returns.index[i], 
                               alpha=0.3, color=f'C{s % 10}')
        
        ax1.set_title('SPY Price with Regime States')
        ax1.set_ylabel('Price (normalized)')
        ax1.grid(True)
        
        ax2 = plt.subplot(2, 1, 2, sharex=ax1)
        for s in active_states:
            state_indices = np.where(state_sequence == s)[0]
            if len(state_indices) > 0:
                ax2.scatter(log_returns.index[state_indices], 
                           [s] * len(state_indices), 
                           label=f'State {s}', 
                           color=f'C{s % 10}', 
                           s=10)
        
        ax2.set_title('State Sequence')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('State')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "price_with_regimes.png"))
        
    print("Analysis completed and saved to", save_dir)

if __name__ == "__main__":
    model, train_data, test_data = train_model(
        data_start_date="2010-01-01",
        train_ratio=0.7,
        n_iter=1000,
        alpha=1.0,
        gamma=1.0,
        kappa=10.0,
        max_states=20
    )
