#!/usr/bin/env python3
"""
Bayesian optimization for TimeFolio portfolio parameters.
Tunes λ (return weight), α (CVaR confidence level), and γ (dispersion penalty).
"""
import os
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Callable, Any, Optional
from pathlib import Path
import pickle
import json

# Bayesian optimization
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from skopt.plots import plot_convergence, plot_objective
import matplotlib.pyplot as plt

# Local imports
from portfolio_optimizer import Config, DataManager, PortfolioOptimizer
from portfolio_metrics import calculate_pcr_sharpe, calculate_diversification_metrics

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(ch)


class BayesianTuner:
    """Bayesian optimization for portfolio parameters with walk-forward validation."""
    
    def __init__(self, 
                 config: Config,
                 data_manager: DataManager,
                 output_dir: str = 'tuning',
                 n_calls: int = 20,
                 lookback_days: int = 252,  # 1 year
                 test_days: int = 63,       # 3 months (quarterly)
                 random_state: int = 42,
                 max_time_minutes: Optional[int] = None,
                 n_splits: int = 3,         # Number of walk-forward validation splits
                 min_n_calls: int = 8):
        """
        Initialize Bayesian tuner with walk-forward validation.
        
        Args:
            config: Configuration object
            data_manager: DataManager instance
            output_dir: Directory to save results
            n_calls: Maximum number of calls to the objective function per split
            lookback_days: Number of days for training data
            test_days: Number of days for testing period
            random_state: Random state for reproducibility
            max_time_minutes: Maximum runtime in minutes (dynamic n_calls)
            n_splits: Number of validation splits for walk-forward validation
            min_n_calls: Minimum number of optimization calls per split
        """
        self.config = config
        self.data_manager = data_manager
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.n_calls = n_calls
        self.lookback_days = lookback_days
        self.test_days = test_days
        self.random_state = random_state
        self.max_time_minutes = max_time_minutes
        self.n_splits = n_splits
        self.min_n_calls = min_n_calls
        
        # Time tracking for dynamic n_calls
        self.start_time = None
        self.call_times = []  # Track time per optimization call
        
        # Walk-forward validation history
        self.fold_results = []
        
        # Define search space
        self.space = [
            Real(0.1, 50.0, name='lambda_hhi', prior='log-uniform'),
            Real(0.1, 50.0, name='lambda_return_hhi', prior='log-uniform'),
            Real(0.80, 0.995, name='explained_variance', prior='uniform'),
        ]
        
    def _get_train_test_data(self, end_date: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get training and testing data for a given end date.
        
        Args:
            end_date: End date for testing period (YYYY-MM-DD)
            
        Returns:
            train_returns: Training returns DataFrame
            test_returns: Testing returns DataFrame
        """
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        train_start = (end_dt - timedelta(days=self.lookback_days)).strftime("%Y-%m-%d")
        test_start = (end_dt - timedelta(days=self.test_days)).strftime("%Y-%m-%d")
        
        # Get full data range
        full_returns = self.data_manager.get_returns(train_start, end_date)
        
        # Split into train/test
        test_cutoff = full_returns.index[full_returns.index >= test_start][0]
        train_returns = full_returns[full_returns.index < test_cutoff]
        test_returns = full_returns[full_returns.index >= test_cutoff]
        
        return train_returns, test_returns
        
    def _get_walk_forward_splits(self, end_date: str) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Create walk-forward validation splits.
        
        Args:
            end_date: End date for final testing period (YYYY-MM-DD)
            
        Returns:
            List of (train, test) DataFrames for each validation fold
        """
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
        # Calculate full data range needed
        # We need additional lookback for the earliest fold
        total_days_needed = self.lookback_days + (self.n_splits * self.test_days)
        full_start = (end_dt - timedelta(days=total_days_needed)).strftime("%Y-%m-%d")
        
        # Get full data range
        try:
            full_returns = self.data_manager.get_returns(full_start, end_date)
            logger.info(f"Obtained {len(full_returns)} days of data for walk-forward validation")
        except Exception as e:
            logger.warning(f"Error getting full data range: {e}. Using shorter history.")
            # Try with original lookback as fallback
            full_start = (end_dt - timedelta(days=self.lookback_days + self.test_days)).strftime("%Y-%m-%d")
            full_returns = self.data_manager.get_returns(full_start, end_date)
            # Reduce splits if needed
            original_splits = self.n_splits
            self.n_splits = min(2, self.n_splits)
            logger.warning(f"Reduced validation splits from {original_splits} to {self.n_splits} due to data availability")
            
        # Create validation splits
        splits = []
        
        for i in range(self.n_splits):
            # Calculate test period for this fold
            test_end_idx = len(full_returns) - i * self.test_days
            test_start_idx = test_end_idx - self.test_days
            
            # Handle edge case where we don't have enough data
            if test_start_idx <= 0:
                logger.warning(f"Not enough data for fold {i+1}, using {self.n_splits-i} folds")
                self.n_splits = i
                break
                
            # Calculate train period (fixed lookback from test start)
            train_start_idx = max(0, test_start_idx - self.lookback_days)
            
            # Extract train/test data
            train_data = full_returns.iloc[train_start_idx:test_start_idx]
            test_data = full_returns.iloc[test_start_idx:test_end_idx]
            
            if len(train_data) < 126:  # Minimum 6 months of training data
                logger.warning(f"Training data for fold {i+1} too short ({len(train_data)} days), skipping")
                continue
                
            logger.info(f"Fold {i+1}: train={len(train_data)} days, test={len(test_data)} days")
            splits.append((train_data, test_data))
            
        logger.info(f"Created {len(splits)} walk-forward validation splits")
        return splits
        
    def _calculate_dynamic_n_calls(self, completed_calls: int = 0) -> int:
        """
        Calculate dynamic number of calls based on remaining time budget.
        
        Args:
            completed_calls: Number of optimization calls already completed
            
        Returns:
            Adjusted number of calls for remaining validation folds
        """
        if self.max_time_minutes is None or not self.call_times:
            return self.n_calls
            
        # Calculate average time per call
        avg_time_per_call = sum(self.call_times) / len(self.call_times)
        
        # Calculate elapsed and remaining time
        elapsed_time = (datetime.now() - self.start_time).total_seconds() / 60
        remaining_time = self.max_time_minutes - elapsed_time
        
        if remaining_time <= 0:
            logger.warning("Time budget exceeded, using minimum calls for remaining folds")
            return self.min_n_calls
            
        # Calculate remaining folds
        completed_folds = len(self.fold_results)
        remaining_folds = self.n_splits - completed_folds
        
        if remaining_folds <= 0:
            return self.n_calls
            
        # Calculate calls per remaining fold
        time_per_fold = remaining_time / remaining_folds
        calls_per_fold = int(time_per_fold / (avg_time_per_call / 60))
        
        # Ensure we do at least minimum calls but no more than original n_calls
        calls = max(self.min_n_calls, min(calls_per_fold, self.n_calls))
        
        logger.info(f"Dynamic n_calls: {calls} (avg time per call: {avg_time_per_call:.2f}s, remaining time: {remaining_time:.1f}min)")
        return calls
    
    def _evaluate_portfolio(self, train_returns: pd.DataFrame, test_returns: pd.DataFrame, expected_returns: pd.Series = None):
        """
        Evaluate portfolio performance for given parameters.
        Args:
            train_returns: Training returns DataFrame
            test_returns: Testing returns DataFrame
            expected_returns: Expected returns vector
        Returns:
            Negative score (for minimization)
        """
        # Use PortfolioOptimizer
        optimizer = PortfolioOptimizer(config=self.config, risk_free_rate=None)
        if expected_returns is None:
            expected_returns = train_returns.mean() * 252
        weights = optimizer.optimize(train_returns, expected_returns, n_positions=self.config.optimization.max_positions)
        # Compute metrics
        cov = train_returns.cov() * 252
        risk_free_rate = 0.0  # or self.config.risk_free_rate if available
        pcr_sharpe = calculate_pcr_sharpe(train_returns, risk_free_rate)
        div_metrics = calculate_diversification_metrics(weights.values, cov, train_returns)
        score = pcr_sharpe @ weights - \
            self.config.optimization.lambda_hhi * max(0, div_metrics['hhi_weight'] - self.config.optimization.hhi_weight_threshold) \
            - self.config.optimization.lambda_return_hhi * max(0, div_metrics['hhi_return'] - self.config.optimization.return_hhi_threshold)
        return -float(score)
        
    def objective_function(self, params: List[float], train_returns: pd.DataFrame, test_returns: pd.DataFrame, expected_returns: pd.Series = None):
        """
        Objective function for Bayesian optimization.
        Args:
            params: [lambda_hhi, lambda_return_hhi, explained_variance]
            train_returns: Training returns DataFrame
            test_returns: Testing returns DataFrame
            expected_returns: Expected returns vector
        Returns:
            Negative of the objective value (for minimization)
        """
        self.config.optimization.lambda_hhi = params[0]
        self.config.optimization.lambda_return_hhi = params[1]
        self.config.optimization.explained_variance = params[2]
        try:
            return self._evaluate_portfolio(train_returns, test_returns, expected_returns)
        except Exception as e:
            logger.error(f"Error in objective function: {e}")
            return 1000  # Large penalty for failures
    
    def optimize(self, end_date: str, expected_returns: Optional[pd.Series] = None) -> Dict[str, float]:
        """
        Tune hyperparameters using Bayesian optimization with walk-forward validation.
        
        Args:
            end_date: End date for optimization period (YYYY-MM-DD)
            expected_returns: Optional expected returns override
            
        Returns:
            Dictionary of optimized parameters
        """
        # Initialize time tracking
        self.start_time = datetime.now()
        self.call_times = []
        self.fold_results = []
        
        # Get walk-forward validation splits
        splits = self._get_walk_forward_splits(end_date)
        
        if not splits:
            logger.warning("No valid validation splits, falling back to single train/test split")
            train_returns, test_returns = self._get_train_test_data(end_date)
            splits = [(train_returns, test_returns)]
            
        # Track performance across all folds
        all_params = []
        all_scores = []
        
        # Run optimization for each fold
        for fold_idx, (train_returns, test_returns) in enumerate(splits):
            logger.info(f"\n--- Starting optimization for fold {fold_idx+1}/{len(splits)} ---")
            logger.info(f"Training data: {train_returns.shape}, test data: {test_returns.shape}")
            
            # Calculate dynamic n_calls based on time budget
            if fold_idx > 0 and self.max_time_minutes is not None:
                n_calls = self._calculate_dynamic_n_calls()
            else:
                n_calls = self.n_calls
                
            # Calculate beta vector
            beta_vec = self._calculate_beta_vector(train_returns)
            
            # Initialize optimizer params for this fold
            instance_params = {
                'train_returns': train_returns,
                'test_returns': test_returns,
                'expected_returns': expected_returns,
                'beta_vec': beta_vec,
                'fold_idx': fold_idx
            }
            
            # Define objective wrapper
            @use_named_args(self.space)
            def objective(**params):
                # Record call start time
                call_start = datetime.now()
                
                # Convert from list to dict
                params_dict = {
                    'return_weight': params['return_weight'],
                    'cvar_alpha': params['cvar_alpha'],
                    'dispersion_penalty': params['dispersion_penalty']
                }
                
                # Evaluate params
                score = self._evaluate_objective(params_dict, **instance_params)
                
                # Record time taken for this call
                call_time = (datetime.now() - call_start).total_seconds()
                self.call_times.append(call_time)
                
                return score
            
            logger.info(f"Starting fold {fold_idx+1} optimization with {n_calls} calls")
            try:
                result = gp_minimize(objective, self.space, n_calls=n_calls, 
                                  random_state=self.random_state, verbose=True)
                
                # Extract best parameters
                best_params = {
                    'return_weight': result.x[0],
                    'cvar_alpha': result.x[1],
                    'dispersion_penalty': result.x[2]
                }
                
                # Get performance metrics
                best_performance = self._evaluate_portfolio(best_params, 
                    train_returns, test_returns, expected_returns, beta_vec)
                
                # Track params and performance
                all_params.append(best_params)
                all_scores.append(best_performance['score'])
                
                # Save fold results
                fold_result = {
                    'fold': fold_idx,
                    'best_params': best_params,
                    'performance': best_performance,
                    'n_calls': len(result.x_iters),
                    'train_dates': (train_returns.index[0], train_returns.index[-1]),
                    'test_dates': (test_returns.index[0], test_returns.index[-1])
                }
                self.fold_results.append(fold_result)
                
                # Save interim results
                self._save_fold_results(fold_result, end_date, fold_idx)
                
            except Exception as e:
                logger.error(f"Error in fold {fold_idx+1}: {e}")
                continue
        
        # Aggregate results across folds to get final parameters
        if not self.fold_results:
            logger.error("No successful optimization folds. Returning default parameters.")
            return {
                'return_weight': 2.0,
                'cvar_alpha': 0.05,
                'dispersion_penalty': 0.1
            }
            
        # Average parameters weighted by performance
        if len(all_params) > 1:
            # Convert scores to weights (higher is better)
            weights = np.array(all_scores)
            # Handle negative scores by shifting to positive
            if np.any(weights < 0):
                weights = weights - np.min(weights) + 1e-6
            # Normalize to sum to 1
            weights = weights / weights.sum()
            
            # Weighted average of parameters
            weighted_params = {}
            for param in ['return_weight', 'cvar_alpha', 'dispersion_penalty']:
                values = np.array([p[param] for p in all_params])
                weighted_params[param] = float(np.sum(values * weights))
                
            best_params = weighted_params
            logger.info(f"Final parameters (weighted average of {len(all_params)} folds): {best_params}")
        else:
            # Just use the single fold result
            best_params = all_params[0]
            logger.info(f"Final parameters (single fold): {best_params}")
        
        # Save aggregated results
        self._save_walk_forward_results(self.fold_results, best_params, end_date)
        
        return best_params
    
    def _plot_weights(self, weights, output_path, title='Portfolio Weights'):
        """
        Plot portfolio weights.
        
        Args:
            weights: Portfolio weights Series
            output_path: Path to save the plot
            title: Plot title
        """
        if weights is None or len(weights) == 0:
            return
            
        plt.figure(figsize=(12, 8))
        weights.sort_values(ascending=False).plot(kind='bar')
        plt.title(title)
        plt.xlabel('Ticker')
        plt.ylabel('Weight')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
    def _save_fold_results(self, fold_result, end_date, fold_idx):
        """
        Save results for a single validation fold.
        
        Args:
            fold_result: Results for this fold
            end_date: End date for optimization
            fold_idx: Fold index
        """
        timestamp = datetime.now().strftime("%Y%m%d")
        results_dir = self.output_dir / f"wf_tuning_{timestamp}"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        fold_dir = results_dir / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        
        # Save best parameters
        with open(fold_dir / "params.json", "w") as f:
            json.dump(fold_result['best_params'], f, indent=4)
            
        # Save performance metrics
        with open(fold_dir / "performance.json", "w") as f:
            # Convert numpy values to Python types for JSON serialization
            serializable_perf = {}
            for k, v in fold_result['performance'].items():
                if isinstance(v, np.ndarray):
                    serializable_perf[k] = v.tolist()
                elif isinstance(v, np.floating):
                    serializable_perf[k] = float(v)
                elif isinstance(v, pd.Timestamp):
                    serializable_perf[k] = v.strftime("%Y-%m-%d")
                elif isinstance(v, pd.Series):
                    serializable_perf[k] = "Series data (not serialized)"
                else:
                    serializable_perf[k] = v
            json.dump(serializable_perf, f, indent=4)
        
        # Generate and save portfolio weights plot if available
        if 'weights' in fold_result['performance']:
            self._plot_weights(fold_result['performance']['weights'], 
                             fold_dir / "weights.png", 
                             title=f"Fold {fold_idx} Portfolio Weights")
            
        logger.info(f"Fold {fold_idx} results saved to {fold_dir}")
        
    def _save_walk_forward_results(self, fold_results, final_params, end_date):
        """
        Save aggregated walk-forward validation results.
        
        Args:
            fold_results: Results from all folds
            final_params: Final aggregated parameters
            end_date: End date for optimization
        """
        timestamp = datetime.now().strftime("%Y%m%d")
        results_dir = self.output_dir / f"wf_tuning_{timestamp}"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save final parameters
        with open(results_dir / "final_params.json", "w") as f:
            json.dump(final_params, f, indent=4)
            
        # Generate summary of all folds
        summary = {
            'end_date': end_date,
            'n_folds': len(fold_results),
            'final_params': final_params,
            'fold_performance': []
        }
        
        for fold in fold_results:
            perf = fold['performance']
            fold_summary = {
                'fold': fold['fold'],
                'params': fold['best_params']
            }
            
            # Add performance metrics, handling different types
            for k, v in perf.items():
                if k == 'weights' or isinstance(v, pd.Series) or isinstance(v, np.ndarray):
                    continue
                try:
                    if isinstance(v, (float, np.floating)):
                        fold_summary[k] = float(v)
                    elif isinstance(v, (int, np.integer)):
                        fold_summary[k] = int(v)
                    elif hasattr(v, 'strftime'):
                        fold_summary[k] = v.strftime("%Y-%m-%d")
                    else:
                        fold_summary[k] = str(v)
                except:
                    fold_summary[k] = str(v)
            
            # Handle dates specially since they might be various types
            for date_field in ['train_start', 'train_end', 'test_start', 'test_end']:
                if date_field.startswith('train'):
                    idx = 0 if date_field.endswith('start') else 1
                    date_val = fold['train_dates'][idx] if len(fold['train_dates']) > idx else None
                else:  # test dates
                    idx = 0 if date_field.endswith('start') else 1
                    date_val = fold['test_dates'][idx] if len(fold['test_dates']) > idx else None
                    
                if date_val is not None:
                    if hasattr(date_val, 'strftime'):
                        fold_summary[date_field] = date_val.strftime("%Y-%m-%d")
                    else:
                        fold_summary[date_field] = str(date_val)
            
            summary['fold_performance'].append(fold_summary)
            
        with open(results_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=4)
            
        # Generate comparative visualization
        self._plot_walk_forward_comparison(fold_results, final_params, results_dir / "fold_comparison.png")
        
        logger.info(f"Walk-forward validation results saved to {results_dir}")
        
    def _plot_walk_forward_comparison(self, fold_results, final_params, output_path):
        """
        Plot comparison of parameter values and performance across folds.
        
        Args:
            fold_results: Results from all folds
            final_params: Final aggregated parameters
            output_path: Path to save the visualization
        """
        if not fold_results:
            return
            
        # Extract data for plotting
        folds = [r['fold'] for r in fold_results]
        
        # Handle metrics gracefully (they might not exist in all fold results)
        def safe_extract(key):
            values = []
            for r in fold_results:
                val = r['performance'].get(key)
                if val is not None:
                    try:
                        values.append(float(val))
                    except (TypeError, ValueError):
                        values.append(np.nan)
                else:
                    values.append(np.nan)
            return values
            
        sharpes = safe_extract('sharpe')
        returns = safe_extract('return')
        cvars = safe_extract('cvar')
        
        # Parameter values by fold
        return_weights = [r['best_params']['return_weight'] for r in fold_results]
        alphas = [r['best_params']['cvar_alpha'] for r in fold_results]
        dispersions = [r['best_params']['dispersion_penalty'] for r in fold_results]
        
        # Create figure with subplots
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        
        # Performance metrics
        axs[0, 0].bar(folds, sharpes, color='skyblue')
        if not all(np.isnan(sharpes)):
            mean_sharpe = np.nanmean(sharpes)
            axs[0, 0].axhline(y=mean_sharpe, color='r', linestyle='--', label=f'Mean: {mean_sharpe:.4f}')
        axs[0, 0].set_title('Sharpe Ratio by Fold')
        axs[0, 0].set_xlabel('Fold')
        axs[0, 0].set_ylabel('Sharpe Ratio')
        axs[0, 0].legend()
        
        axs[0, 1].bar(folds, returns, color='lightgreen')
        if not all(np.isnan(returns)):
            mean_return = np.nanmean(returns)
            axs[0, 1].axhline(y=mean_return, color='r', linestyle='--', label=f'Mean: {mean_return:.4f}')
        axs[0, 1].set_title('Return by Fold')
        axs[0, 1].set_xlabel('Fold')
        axs[0, 1].set_ylabel('Return')
        axs[0, 1].legend()
        
        axs[0, 2].bar(folds, cvars, color='salmon')
        if not all(np.isnan(cvars)):
            mean_cvar = np.nanmean(cvars)
            axs[0, 2].axhline(y=mean_cvar, color='r', linestyle='--', label=f'Mean: {mean_cvar:.4f}')
        axs[0, 2].set_title('CVaR by Fold')
        axs[0, 2].set_xlabel('Fold')
        axs[0, 2].set_ylabel('CVaR')
        axs[0, 2].legend()
        
        # Parameter values
        axs[1, 0].plot(folds, return_weights, 'o-', color='purple')
        axs[1, 0].axhline(y=final_params['return_weight'], color='r', linestyle='--', 
                        label=f'Final: {final_params["return_weight"]:.4f}')
        axs[1, 0].set_title('Return Weight (λ) by Fold')
        axs[1, 0].set_xlabel('Fold')
        axs[1, 0].set_ylabel('Return Weight')
        axs[1, 0].legend()
        
        axs[1, 1].plot(folds, alphas, 'o-', color='brown')
        axs[1, 1].axhline(y=final_params['cvar_alpha'], color='r', linestyle='--', 
                        label=f'Final: {final_params["cvar_alpha"]:.4f}')
        axs[1, 1].set_title('CVaR Alpha (α) by Fold')
        axs[1, 1].set_xlabel('Fold')
        axs[1, 1].set_ylabel('Alpha')
        axs[1, 1].legend()
        
        axs[1, 2].plot(folds, dispersions, 'o-', color='orange')
        axs[1, 2].axhline(y=final_params['dispersion_penalty'], color='r', linestyle='--', 
                        label=f'Final: {final_params["dispersion_penalty"]:.4f}')
        axs[1, 2].set_title('Dispersion Penalty (γ) by Fold')
        axs[1, 2].set_xlabel('Fold')
        axs[1, 2].set_ylabel('Dispersion Penalty')
        axs[1, 2].legend()
        
        # Add overall title
        fig.suptitle('Walk-Forward Validation Results', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(output_path)
        plt.close()
        
    def tune_hyperparams(self,
                        end_date: Optional[str] = None,
                        expected_returns: Optional[pd.Series] = None,
                        max_time_minutes: Optional[int] = None,
                        n_calls: Optional[int] = None) -> Dict[str, float]:
        """
        Tune hyperparameters using Bayesian optimization.
        Args:
            end_date: End date for optimization period (defaults to today)
            expected_returns: Optional expected returns override
        Returns:
            Dictionary of optimized parameters
        """
        # Set default end date to today if not provided
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        # Get train/test data
        train_returns, test_returns = self._get_train_test_data(end_date)
        # Set default expected returns if not provided
        if expected_returns is None:
            expected_returns = train_returns.mean() * 252
        # Create the objective function with fixed data
        @use_named_args(self.space)
        def objective(lambda_hhi, lambda_return_hhi, explained_variance):
            self.config.optimization.lambda_hhi = lambda_hhi
            self.config.optimization.lambda_return_hhi = lambda_return_hhi
            self.config.optimization.explained_variance = explained_variance
            return self._evaluate_portfolio(train_returns, test_returns, expected_returns)
        # Run Bayesian optimization
        logger.info(f"Starting Bayesian optimization with {self.n_calls} calls")
        result = gp_minimize(
            objective,
            self.space,
            n_calls=self.n_calls,
            random_state=self.random_state,
            verbose=True
        )
        # Extract best parameters
        best_params = {
            'lambda_hhi': result.x[0],
            'lambda_return_hhi': result.x[1],
            'explained_variance': result.x[2]
        }
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d")
        result_path = self.output_dir / f"tuning_result_{timestamp}.pkl"
        with open(result_path, 'wb') as f:
            pickle.dump(result, f)
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        plot_convergence(result, ax=ax1)
        plot_objective(result, dimensions=['lambda_hhi', 'lambda_return_hhi'], ax=ax2)
        plt.tight_layout()
        plt.savefig(self.output_dir / f"tuning_plots_{timestamp}.png")
        # Log results
        logger.info(f"Optimization complete. Best parameters: {best_params}")
        logger.info(f"Best score: {-result.fun:.4f}")
        logger.info(f"Results saved to {result_path}")
        return best_params
    
    def get_latest_params(self) -> Dict[str, float]:
        """
        Get the most recent tuned parameters.
        Falls back to defaults if no tuned parameters are found.
        
        Returns:
            Dictionary of parameters
        """
        try:
            files = sorted(self.output_dir.glob("tuning_result_*.pkl"))
            if not files:
                logger.warning("No tuned parameters found, using defaults")
                return {
                    'return_weight': self.config.return_weight,
                    'cvar_alpha': self.config.cvar_alpha,
                    'dispersion_penalty': 0.0
                }
                
            latest = files[-1]
            with open(latest, 'rb') as f:
                result = pickle.load(f)
                
            return {
                'return_weight': result.x[0],
                'cvar_alpha': result.x[1],
                'dispersion_penalty': result.x[2]
            }
            
        except Exception as e:
            logger.error(f"Error loading latest parameters: {e}")
            return {
                'return_weight': self.config.return_weight,
                'cvar_alpha': self.config.cvar_alpha,
                'dispersion_penalty': 0.0
            }


def tune_hyperparams(config: Config, 
                    data_manager: DataManager,
                    expected_returns: Optional[pd.Series] = None,
                    beta_vec: Optional[pd.Series] = None,
                    end_date: Optional[str] = None,
                    n_calls: int = 20) -> Dict[str, float]:
    """
    Convenience function to tune hyperparameters.
    
    Args:
        config: Config object
        data_manager: DataManager instance
        expected_returns: Optional expected returns override
        beta_vec: Optional beta vector override
        end_date: End date for optimization period
        n_calls: Number of optimization calls
        
    Returns:
        Dictionary of optimized parameters
    """
    tuner = BayesianTuner(
        config=config,
        data_manager=data_manager,
        n_calls=n_calls
    )
    
    return tuner.tune_hyperparams(
        end_date=end_date,
        expected_returns=expected_returns,
        beta_vec=beta_vec
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Tune portfolio parameters")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--n-calls", type=int, default=20, help="Number of optimization calls")
    parser.add_argument("--output-dir", default="tuning", help="Output directory")
    args = parser.parse_args()
    
    # Load config and data manager
    config = Config(args.config)
    data_manager = DataManager(config)
    
    # Create tuner
    tuner = BayesianTuner(
        config=config,
        data_manager=data_manager,
        output_dir=args.output_dir,
        n_calls=args.n_calls
    )
    
    # Run optimization
    best_params = tuner.tune_hyperparams()
    
    print("Best parameters:")
    for k, v in best_params.items():
        print(f"  {k}: {v:.6f}")
