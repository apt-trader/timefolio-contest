# tuner.py
import logging
import optuna
import yaml
from copy import deepcopy
import argparse
import os

from backtester import Backtester
from config import Config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("HyperparameterTuner")

BASE_CONFIG_PATH = 'config/config.yaml'
with open(BASE_CONFIG_PATH, 'r') as f:
    BASE_CONFIG = yaml.safe_load(f)

def objective(trial: optuna.Trial) -> float:
    """
    The objective function for Optuna. It runs a full backtest with a set of
    hyperparameters and returns the Sharpe Ratio.
    """
    trial_config = deepcopy(BASE_CONFIG)
    
    # Suggest Hyperparameters to Tune
    trial_config['optimization']['risk_aversion'] = trial.suggest_float('risk_aversion', 0.5, 3.0, log=True)
    trial_config['optimization']['l2_penalty'] = trial.suggest_float('l2_penalty', 0.05, 0.5, log=True)
    
    trial_config['factor_engine']['vol_window'] = trial.suggest_int('vol_window', 15, 40)
    trial_config['factor_engine']['rsi_window'] = trial.suggest_int('rsi_window', 10, 20)
    
    temp_config_path = f"config/temp_trial_{trial.number}.yaml"
    with open(temp_config_path, 'w') as f:
        yaml.dump(trial_config, f)
        
    logger.info(f"--- Starting Trial #{trial.number} with params: {trial.params} ---")
    
    try:
        backtester = Backtester(
            config_path=temp_config_path,
            start_date='2023-01-01',
            end_date='2024-06-01', # Use a fixed period for fair comparison
            rebalance_freq='W-MON'
        )
        backtest_summary = backtester.run()
        
        # Clean up the temporary config file
        os.remove(temp_config_path)

        sharpe_ratio = backtest_summary.get('sharpe_ratio', -2.0) # Return a very low score on failure
        logger.info(f"--- Trial #{trial.number} Finished --- Sharpe Ratio: {sharpe_ratio:.4f}")
        
        return sharpe_ratio

    except Exception as e:
        logger.error(f"Trial #{trial.number} failed with an exception: {e}")
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)
        return -2.0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Hyperparameter tuner for the TimeFolio strategy.")
    parser.add_argument('-n', '--n-trials', type=int, default=50, help="Number of tuning trials to run.")
    parser.add_argument('--study-name', default='timefolio-strategy-v1', help="Name for the Optuna study.")
    parser.add_argument('--db-url', default='sqlite:///tuning_results.db', help="DB URL for Optuna results.")
    args = parser.parse_args()

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.db_url,
        direction='maximize',
        load_if_exists=True
    )
    
    study.optimize(objective, n_trials=args.n_trials, n_jobs=-1) # Use all available CPU cores
    
    print("\n" + "="*50)
    print("           TUNING COMPLETE")
    print("="*50)
    print(f"Study: {study.study_name}")
    print(f"Number of finished trials: {len(study.trials)}")
    
    best_trial = study.best_trial
    print("\n--- Best Trial ---")
    print(f"Value (Sharpe Ratio): {best_trial.value:.4f}")
    
    print("\nBest Parameters:")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")
        
    print("\nUpdate your config.yaml with these parameters for optimal performance.")