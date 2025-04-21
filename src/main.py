"""
Main Script for HDP-HMM MCMC Financial Regime Detection
- Orchestrates the entire workflow
- Runs data pipeline, model training, signal generation, and benchmarking
"""

import os
import argparse
from datetime import datetime

from data_pipeline import DataPipeline
from hdp_hmm import HDPHMM
from model_training import train_model
from signal_generation import RegimeSignalGenerator
from benchmarking import run_benchmark

def main():
    """
    Main function to run the HDP-HMM MCMC Financial Regime Detection workflow
    """
    parser = argparse.ArgumentParser(description='HDP-HMM MCMC Financial Regime Detection')
    
    parser.add_argument('--data_start_date', type=str, default='2010-01-01',
                        help='Start date for data fetching in YYYY-MM-DD format')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='Ratio of data to use for training')
    parser.add_argument('--n_iter', type=int, default=1000,
                        help='Number of MCMC iterations')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Concentration parameter for the Dirichlet Process prior on transitions')
    parser.add_argument('--gamma', type=float, default=1.0,
                        help='Concentration parameter for the Dirichlet Process prior on states')
    parser.add_argument('--kappa', type=float, default=10.0,
                        help='Self-transition bias parameter (sticky HDP-HMM)')
    parser.add_argument('--max_states', type=int, default=20,
                        help='Maximum number of states to consider (truncation level)')
    parser.add_argument('--mapping_method', type=str, default='mean_based',
                        choices=['mean_based', 'volatility_based', 'combined'],
                        help='Method to use for state mapping')
    parser.add_argument('--confidence_threshold', type=float, default=0.6,
                        help='Threshold for state probability to generate a signal')
    parser.add_argument('--initial_capital', type=float, default=10000.0,
                        help='Initial capital for the strategy')
    parser.add_argument('--transaction_cost', type=float, default=0.001,
                        help='Transaction cost as a percentage of trade value')
    parser.add_argument('--skip_training', action='store_true',
                        help='Skip training and use an existing model')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to an existing model (used if skip_training is True)')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save results')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    models_dir = os.path.join(args.output_dir, 'models')
    plots_dir = os.path.join(args.output_dir, 'plots')
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    if not args.skip_training:
        print("\n=== Training HDP-HMM Model ===")
        model, train_data, test_data = train_model(
            data_start_date=args.data_start_date,
            train_ratio=args.train_ratio,
            n_iter=args.n_iter,
            alpha=args.alpha,
            gamma=args.gamma,
            kappa=args.kappa,
            max_states=args.max_states,
            save_dir=models_dir,
            plot_dir=plots_dir
        )
        
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
        if model_files:
            latest_model = sorted(model_files)[-1]
            model_path = os.path.join(models_dir, latest_model)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = os.path.join(models_dir, f"hdp_hmm_model_{timestamp}.pkl")
            model.save(model_path)
    else:
        if args.model_path is None:
            raise ValueError("Model path must be provided when skip_training is True")
        model_path = args.model_path
        print(f"\n=== Using existing model: {model_path} ===")
    
    print("\n=== Benchmarking Trading Strategy ===")
    benchmark = run_benchmark(
        model_path=model_path,
        data_start_date=args.data_start_date,
        train_ratio=args.train_ratio,
        mapping_method=args.mapping_method,
        confidence_threshold=args.confidence_threshold,
        initial_capital=args.initial_capital,
        transaction_cost=args.transaction_cost,
        save_dir=args.output_dir
    )
    
    print("\n=== Workflow Completed ===")
    print(f"Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
