"""
Custom HDP-HMM MCMC Implementation for Financial Regime Detection
- Implements Hierarchical Dirichlet Process Hidden Markov Model
- Uses MCMC sampling for inference
- Designed for financial time series data
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from collections import defaultdict
import pickle
import os
import time

class HDPHMM:
    """
    Hierarchical Dirichlet Process Hidden Markov Model with MCMC sampling
    
    This implementation follows the beam sampler approach for inference in HDP-HMM
    as described in "The Infinite HMM for Unsupervised PoS Tagging" by 
    Jurgen Van Gael, Andreas Vlachos, and Zoubin Ghahramani.
    
    The model uses a non-parametric approach to determine the number of states
    automatically from the data.
    """
    
    def __init__(self, alpha=1.0, gamma=1.0, kappa=10.0, init_state_concentration=1.0, 
                 max_states=50, obs_dim=1, sticky=True):
        """
        Initialize the HDP-HMM model
        
        Parameters:
        -----------
        alpha : float
            Concentration parameter for the Dirichlet Process prior on transitions
        gamma : float
            Concentration parameter for the Dirichlet Process prior on states
        kappa : float
            Self-transition bias parameter (sticky HDP-HMM)
        init_state_concentration : float
            Concentration parameter for the initial state distribution
        max_states : int
            Maximum number of states to consider (truncation level)
        obs_dim : int
            Dimensionality of the observations
        sticky : bool
            Whether to use the sticky HDP-HMM variant
        """
        self.alpha = alpha
        self.gamma = gamma
        self.kappa = kappa if sticky else 0.0
        self.init_state_concentration = init_state_concentration
        self.max_states = max_states
        self.obs_dim = obs_dim
        self.sticky = sticky
        
        # Initialize with empty/default values
        self.state_sequence = np.array([], dtype=int)
        self.transition_counts = np.zeros((self.max_states, self.max_states))
        self.state_counts = np.zeros(self.max_states)
        self.observations = np.array([]).reshape(0, self.obs_dim)
        self.state_parameters = {}
        self.active_states = np.array([], dtype=int)
        self.log_likelihood_history = []
        
        # Initialize observation model parameters
        self.means = np.zeros((self.max_states, self.obs_dim))
        self.covariances = np.zeros((self.max_states, self.obs_dim, self.obs_dim))
        for s in range(self.max_states):
            self.covariances[s] = np.eye(self.obs_dim)
        
        self.mu_0 = np.zeros(obs_dim)
        self.kappa_0 = 0.1
        self.nu_0 = obs_dim + 2
        self.sigma_0 = np.eye(obs_dim)
        
    def _initialize_parameters(self, data):
        """
        Initialize model parameters
        
        Parameters:
        -----------
        data : numpy.ndarray
            Observation data of shape (n_samples, obs_dim)
        """
        n_samples = data.shape[0]
        
        self.state_sequence = np.random.randint(0, min(self.max_states, n_samples // 10), size=n_samples)
        
        self.transition_counts = np.zeros((self.max_states, self.max_states))
        for t in range(1, n_samples):
            self.transition_counts[self.state_sequence[t-1], self.state_sequence[t]] += 1
            
        self.state_counts = np.zeros(self.max_states)
        for s in self.state_sequence:
            self.state_counts[s] += 1
            
        self.observations = data
        
        self._initialize_observation_model()
        
        self.active_states = np.where(self.state_counts > 0)[0]
        
    def _initialize_observation_model(self):
        """
        Initialize the observation model parameters (Gaussian)
        """
        self.means = np.zeros((self.max_states, self.obs_dim))
        self.covariances = np.zeros((self.max_states, self.obs_dim, self.obs_dim))
        
        for s in range(self.max_states):
            state_obs = self.observations[self.state_sequence == s]
            if len(state_obs) > 0:
                self.means[s] = np.mean(state_obs, axis=0)
                if len(state_obs) > 1:
                    self.covariances[s] = np.cov(state_obs, rowvar=False)
                else:
                    self.covariances[s] = np.eye(self.obs_dim)
            else:
                self.means[s] = self.mu_0
                self.covariances[s] = self.sigma_0
                
    def _sample_state_sequence(self):
        """
        Sample the state sequence using beam sampling
        """
        n_samples = len(self.observations)
        
        transition_probs = self._compute_transition_probabilities()
        
        obs_likelihoods = self._compute_observation_likelihoods()
        
        forward_probs = np.zeros((n_samples, self.max_states))
        
        forward_probs[0] = obs_likelihoods[0] * (self.state_counts + self.gamma / self.max_states)
        forward_probs[0] /= np.sum(forward_probs[0])
        
        for t in range(1, n_samples):
            for j in range(self.max_states):
                forward_probs[t, j] = obs_likelihoods[t, j] * np.sum(
                    forward_probs[t-1] * transition_probs[:, j])
            
            if np.sum(forward_probs[t]) > 0:
                forward_probs[t] /= np.sum(forward_probs[t])
        
        new_state_sequence = np.zeros(n_samples, dtype=int)
        
        new_state_sequence[-1] = np.random.choice(self.max_states, p=forward_probs[-1])
        
        for t in range(n_samples-2, -1, -1):
            backward_probs = forward_probs[t] * transition_probs[:, new_state_sequence[t+1]]
            backward_probs /= np.sum(backward_probs)
            new_state_sequence[t] = np.random.choice(self.max_states, p=backward_probs)
        
        self.state_sequence = new_state_sequence
        
        self.transition_counts = np.zeros((self.max_states, self.max_states))
        for t in range(1, n_samples):
            self.transition_counts[self.state_sequence[t-1], self.state_sequence[t]] += 1
            
        self.state_counts = np.zeros(self.max_states)
        for s in self.state_sequence:
            self.state_counts[s] += 1
            
        self.active_states = np.where(self.state_counts > 0)[0]
        
    def _compute_transition_probabilities(self):
        """
        Compute transition probabilities based on counts and hyperparameters
        
        Returns:
        --------
        numpy.ndarray
            Transition probability matrix of shape (max_states, max_states)
        """
        transition_probs = np.zeros((self.max_states, self.max_states))
        
        for i in range(self.max_states):
            transition_counts_with_prior = self.transition_counts[i].copy()
            transition_counts_with_prior[i] += self.kappa
            
            transition_counts_with_prior += self.alpha * self.state_counts / np.sum(self.state_counts)
            
            transition_probs[i] = transition_counts_with_prior / np.sum(transition_counts_with_prior)
            
        return transition_probs
    
    def _compute_observation_likelihoods(self):
        """
        Compute observation likelihoods for all states and time points
        
        Returns:
        --------
        numpy.ndarray
            Observation likelihoods of shape (n_samples, max_states)
        """
        n_samples = len(self.observations)
        obs_likelihoods = np.zeros((n_samples, self.max_states))
        
        for s in range(self.max_states):
            if s in self.active_states or np.random.rand() < 0.1:  # Also compute for some inactive states
                for t in range(n_samples):
                    obs_likelihoods[t, s] = stats.multivariate_normal.pdf(
                        x=self.observations[t], mean=self.means[s], cov=self.covariances[s])
        
        return obs_likelihoods
    
    def _sample_observation_parameters(self):
        """
        Sample observation model parameters (means and covariances) using conjugate priors
        """
        for s in range(self.max_states):
            state_obs = self.observations[self.state_sequence == s]
            n_s = len(state_obs)
            
            if n_s > 0:
                y_bar = np.mean(state_obs, axis=0)
                
                kappa_n = self.kappa_0 + n_s
                mu_n = (self.kappa_0 * self.mu_0 + n_s * y_bar) / kappa_n
                nu_n = self.nu_0 + n_s
                
                if n_s > 1:
                    S = np.sum([(y - y_bar).reshape(-1, 1) @ (y - y_bar).reshape(1, -1) for y in state_obs], axis=0)
                else:
                    S = np.zeros((self.obs_dim, self.obs_dim))
                
                sigma_n = self.sigma_0 + S + (self.kappa_0 * n_s / kappa_n) * \
                          ((y_bar - self.mu_0).reshape(-1, 1) @ (y_bar - self.mu_0).reshape(1, -1))
                
                self.covariances[s] = stats.invwishart.rvs(df=nu_n, scale=sigma_n)
                
                self.means[s] = stats.multivariate_normal.rvs(
                    mean=mu_n, cov=self.covariances[s] / kappa_n)
            else:
                self.covariances[s] = stats.invwishart.rvs(df=self.nu_0, scale=self.sigma_0)
                self.means[s] = stats.multivariate_normal.rvs(
                    mean=self.mu_0, cov=self.covariances[s] / self.kappa_0)
    
    def _compute_log_likelihood(self):
        """
        Compute the log likelihood of the current model
        
        Returns:
        --------
        float
            Log likelihood of the model
        """
        log_likelihood = 0.0
        
        for t in range(len(self.observations)):
            s = self.state_sequence[t]
            log_likelihood += stats.multivariate_normal.logpdf(
                x=self.observations[t], mean=self.means[s], cov=self.covariances[s])
        
        transition_probs = self._compute_transition_probabilities()
        for t in range(1, len(self.observations)):
            s_prev = self.state_sequence[t-1]
            s_curr = self.state_sequence[t]
            log_likelihood += np.log(transition_probs[s_prev, s_curr])
        
        return log_likelihood
    
    def fit(self, data, n_iter=1000, verbose=True, save_history=False, history_dir="models"):
        """
        Fit the HDP-HMM model to the data using MCMC sampling
        
        Parameters:
        -----------
        data : numpy.ndarray
            Observation data of shape (n_samples, obs_dim)
        n_iter : int
            Number of MCMC iterations
        verbose : bool
            Whether to print progress information
        save_history : bool
            Whether to save model history during training
        history_dir : str
            Directory to save model history
        
        Returns:
        --------
        self
        """
        self._initialize_parameters(data)
        
        if save_history:
            os.makedirs(history_dir, exist_ok=True)
        
        for it in range(n_iter):
            start_time = time.time()
            
            self._sample_state_sequence()
            
            self._sample_observation_parameters()
            
            log_likelihood = self._compute_log_likelihood()
            self.log_likelihood_history.append(log_likelihood)
            
            if verbose and (it + 1) % 10 == 0:
                active_states = len(self.active_states)
                elapsed_time = time.time() - start_time
                print(f"Iteration {it+1}/{n_iter}, Log-likelihood: {log_likelihood:.2f}, "
                      f"Active states: {active_states}, Time: {elapsed_time:.2f}s")
            
            if save_history and (it + 1) % 100 == 0:
                self.save(os.path.join(history_dir, f"hdp_hmm_iter_{it+1}.pkl"))
        
        self._extract_state_parameters()
        
        return self
    
    def _extract_state_parameters(self):
        """
        Extract parameters for each state for analysis
        """
        self.state_parameters = {}
        
        for s in self.active_states:
            state_obs = self.observations[self.state_sequence == s]
            
            if len(state_obs) > 0:
                self.state_parameters[s] = {
                    'mean': self.means[s],
                    'covariance': self.covariances[s],
                    'count': len(state_obs),
                    'proportion': len(state_obs) / len(self.observations),
                    'std': np.sqrt(np.diag(self.covariances[s])),
                    'observations': state_obs
                }
    
    def predict_state_probabilities(self, data):
        """
        Predict state probabilities for new data
        
        Parameters:
        -----------
        data : numpy.ndarray
            New observation data of shape (n_samples, obs_dim)
            
        Returns:
        --------
        numpy.ndarray
            State probabilities of shape (n_samples, max_states)
        """
        n_samples = len(data)
        state_probs = np.zeros((n_samples, self.max_states))
        
        transition_probs = self._compute_transition_probabilities()
        
        obs_likelihoods = np.zeros((n_samples, self.max_states))
        for s in self.active_states:
            for t in range(n_samples):
                obs_likelihoods[t, s] = stats.multivariate_normal.pdf(
                    x=data[t], mean=self.means[s], cov=self.covariances[s])
        
        forward_probs = np.zeros((n_samples, self.max_states))
        
        forward_probs[0] = obs_likelihoods[0] * (self.state_counts + self.gamma / self.max_states)
        forward_probs[0] /= np.sum(forward_probs[0])
        
        for t in range(1, n_samples):
            for j in range(self.max_states):
                forward_probs[t, j] = obs_likelihoods[t, j] * np.sum(
                    forward_probs[t-1] * transition_probs[:, j])
            
            if np.sum(forward_probs[t]) > 0:
                forward_probs[t] /= np.sum(forward_probs[t])
        
        return forward_probs
    
    def predict_states(self, data):
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
        state_probs = self.predict_state_probabilities(data)
        return np.argmax(state_probs, axis=1)
    
    def plot_state_sequence(self, data=None, save_path=None):
        """
        Plot the state sequence and observations
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Original time series data with dates as index
        save_path : str
            Path to save the plot
        """
        if data is None:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(self.observations)
            for s in self.active_states:
                state_indices = np.where(self.state_sequence == s)[0]
                ax.scatter(state_indices, self.observations[state_indices], 
                           label=f"State {s}", alpha=0.5)
            ax.legend()
            ax.set_title("State Sequence and Observations")
            ax.set_xlabel("Time")
            ax.set_ylabel("Observation")
        else:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(data.index, data.values)
            
            for s in self.active_states:
                state_indices = np.where(self.state_sequence == s)[0]
                if len(state_indices) > 0:
                    ax.scatter(data.index[state_indices], data.values[state_indices], 
                               label=f"State {s}", alpha=0.5)
            
            ax.legend()
            ax.set_title("State Sequence and Observations")
            ax.set_xlabel("Date")
            ax.set_ylabel("Value")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
    
    def plot_log_likelihood(self, save_path=None):
        """
        Plot the log likelihood history
        
        Parameters:
        -----------
        save_path : str
            Path to save the plot
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.log_likelihood_history)
        plt.title("Log Likelihood History")
        plt.xlabel("Iteration")
        plt.ylabel("Log Likelihood")
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
    
    def plot_state_parameters(self, save_path=None):
        """
        Plot the parameters for each state
        
        Parameters:
        -----------
        save_path : str
            Path to save the plot
        """
        n_active_states = len(self.active_states)
        
        if n_active_states == 0:
            print("No active states to plot.")
            return
        
        fig, axes = plt.subplots(n_active_states, 2, figsize=(12, 4 * n_active_states))
        
        if n_active_states == 1:
            axes = np.array([axes])
        
        for i, s in enumerate(self.active_states):
            if 'observations' in self.state_parameters[s]:
                axes[i, 0].hist(self.state_parameters[s]['observations'], bins=20, alpha=0.7)
                axes[i, 0].axvline(self.state_parameters[s]['mean'][0], color='r', linestyle='--')
                axes[i, 0].set_title(f"State {s} Observations")
                axes[i, 0].set_xlabel("Value")
                axes[i, 0].set_ylabel("Frequency")
            
            param_names = ['mean', 'std', 'count', 'proportion']
            param_values = [self.state_parameters[s]['mean'][0], 
                           self.state_parameters[s]['std'][0],
                           self.state_parameters[s]['count'],
                           self.state_parameters[s]['proportion']]
            
            axes[i, 1].bar(param_names, param_values)
            axes[i, 1].set_title(f"State {s} Parameters")
            axes[i, 1].set_ylabel("Value")
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
    
    def save(self, filepath):
        """
        Save the model to a file
        
        Parameters:
        -----------
        filepath : str
            Path to save the model
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """
        Load a model from a file
        
        Parameters:
        -----------
        filepath : str
            Path to the saved model
            
        Returns:
        --------
        HDPHMM
            Loaded model
        """
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        
        print(f"Model loaded from {filepath}")
        return model
