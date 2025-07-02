# tuner.py
import logging
import optuna
import argparse
import sys
import time
from copy import deepcopy
import pandas as pd
import traceback
import json
from pathlib import Path
from datetime import datetime
from typing import Any, Dict

# Add project root to path for clean imports
sys.path.insert(0, str(Path(__file__).parent))

from backtester import Backtester
from data_manager import DataManager
from config import Config

# Configure logging with detailed format and progress tracking
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(name)-20s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
    ]
)

# Set higher log level for noisy libraries
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('numexpr').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)
logger = logging.getLogger("HyperparameterTuner")

# Constants
BASE_CONFIG_PATH = 'config/config.yaml'
DEFAULT_N_TRIALS = 50
DEFAULT_STUDY_NAME = 'final-model-tuning-v1'
DEFAULT_DB_URL = 'sqlite:///db/tuning_results.db'

# Parameter search spaces
PARAM_SPACES = {
    'n_pca_components': {'type': 'int', 'low': 3, 'high': 14},
    'risk_aversion': {'type': 'float', 'low': 0.1, 'high': 10.0, 'log': True},
    'l2_penalty': {'type': 'float', 'low': 0.001, 'high': 1.0, 'log': True},
    'cov_l2_alpha': {'type': 'float', 'low': 0.01, 'high': 0.3, 'log': True},
    'momentum_window': {'type': 'int', 'low': 5, 'high': 63}
}

class TuningError(Exception):
    """Custom exception for tuning-specific errors."""
    pass

class ParameterValidator:
    """Validates parameter values against defined constraints."""
    
    @staticmethod
    def validate_parameter(name: str, value: Any) -> bool:
        """Validate a single parameter against its defined constraints."""
        if name not in PARAM_SPACES:
            raise ValueError(f"Unknown parameter: {name}")
            
        param = PARAM_SPACES[name]
        
        if param['type'] == 'int':
            if not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if not param['low'] <= value <= param['high']:
                raise ValueError(f"{name} must be between {param['low']} and {param['high']}")
        elif param['type'] == 'float':
            if not (isinstance(value, float) or isinstance(value, int)):
                raise TypeError(f"{name} must be a float")
            if not param['low'] <= value <= param['high']:
                raise ValueError(f"{name} must be between {param['low']} and {param['high']}")
        
        return True
    
    @classmethod
    def validate_all(cls, params: Dict[str, Any]) -> bool:
        """Validate all parameters in a dictionary."""
        for name, value in params.items():
            cls.validate_parameter(name, value)
        return True

def log_trial_start(trial: optuna.Trial) -> Dict[str, Any]:
    """Log trial start and return trial metadata."""
    trial_start = datetime.now()
    trial_id = f"{trial.number:04d}_{trial_start.strftime('%Y%m%d_%H%M%S')}"
    
    logger.info("\n" + "=" * 80)
    logger.info(f"TRIAL {trial_id} - STARTING")
    logger.info("-" * 80)
    logger.info(f"Trial #{trial.number} started at {trial_start}")
    
    return {
        'trial_id': trial_id,
        'start_time': trial_start,
        'status': 'running'
    }

def log_trial_complete(trial_meta: Dict[str, Any], metrics: Dict[str, Any]) -> None:
    """Log trial completion with metrics."""
    duration = (datetime.now() - trial_meta['start_time']).total_seconds()
    trial_meta.update({
        'status': 'completed',
        'duration_seconds': duration,
        'end_time': datetime.now().isoformat(),
        **metrics
    })
    
    logger.info("-" * 80)
    logger.info(f"TRIAL {trial_meta['trial_id']} - COMPLETED")
    logger.info(f"Duration: {duration:.2f} seconds")
    for k, v in metrics.items():
        if isinstance(v, (int, float)) and 'time' not in k.lower():
            logger.info(f"{k.replace('_', ' ').title()}: {v:.4f}")
        else:
            logger.info(f"{k.replace('_', ' ').title()}: {v}")
    logger.info("=" * 80 + "\n")

def suggest_parameters(trial: optuna.Trial) -> Dict[str, Any]:
    """Suggest parameters for the current trial."""
    params = {}
    for name, space in PARAM_SPACES.items():
        if space['type'] == 'int':
            params[name] = trial.suggest_int(
                name, 
                low=space['low'], 
                high=space['high']
            )
        elif space['type'] == 'float':
            params[name] = trial.suggest_float(
                name,
                low=space['low'],
                high=space['high'],
                log=space.get('log', False)
            )
    return params

def update_config(cfg: Config, params: Dict[str, Any]) -> None:
    """Update configuration with trial parameters."""
    # Update factor settings
    if 'n_pca_components' in params:
        cfg.factor_settings['n_pca_components'] = params['n_pca_components']
    
    # Update optimization settings
    if 'risk_aversion' in params:
        cfg.optimization_settings['risk_aversion'] = params['risk_aversion']
    if 'l2_penalty' in params:
        cfg.optimization_settings['l2_penalty'] = params['l2_penalty']
    
    # Update any other parameters
    if 'momentum_window' in params:
        cfg.factor_settings['mom_windows'] = [params['momentum_window']]

def objective(trial: optuna.Trial, start_date: str, end_date: str) -> float:
    """Optimization objective for Optuna hyperparameter tuning with enhanced logging and error handling."""
    trial_start_time = time.time()
    trial_id = f"{trial.number:04d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    trial_logger = logging.getLogger(f"Trial_{trial_id}")
    
    # Initialize metrics with default values
    metrics = {
        'status': 'running',
        'trial_id': trial_id,
        'start_time': datetime.now().isoformat(),
        'end_time': None,
        'duration_seconds': 0.0,
        'error': None,
        'sharpe_ratio': -2.0,  # Default to worst possible score
        'return': 0.0,
        'volatility': 0.0,
        'max_drawdown': 0.0,
        'parameters': {}
    }
    
    # Log trial start
    trial_logger.info("=" * 80)
    trial_logger.info(f"TRIAL {trial_id} - STARTING")
    trial_logger.info("-" * 80)
    
    try:
        # 1. Load base configuration
        try:
            trial_logger.info("Loading configuration...")
            cfg = Config(BASE_CONFIG_PATH)
            trial_logger.debug(f"Configuration loaded from {BASE_CONFIG_PATH}")
        except Exception as e:
            error_msg = f"Failed to load config: {str(e)}"
            trial_logger.error(error_msg, exc_info=True)
            raise TuningError(error_msg)
        
        # 2. Suggest and validate parameters
        try:
            trial_logger.info("Generating parameters...")
            params = suggest_parameters(trial)
            ParameterValidator.validate_all(params)
            
            # Store parameters for reference
            for k, v in params.items():
                trial.set_user_attr(k, v)
            metrics['parameters'] = params
            
            trial_logger.info("Parameters validated successfully")
            trial_logger.debug(f"Trial parameters: {json.dumps(params, indent=2, default=str)}")
            
            # Update config with suggested parameters
            update_config(cfg, params)
        except Exception as e:
            error_msg = f"Invalid parameters: {str(e)}"
            trial_logger.error(error_msg, exc_info=True)
            raise TuningError(error_msg)
        
        # 3. Initialize DataManager and Backtester
        try:
            trial_logger.info("Initializing DataManager for backtest...")
            
            # Create a dedicated config for the DataManager to avoid side effects
            dm_config = deepcopy(cfg)
            dm_config.data_settings['start_date'] = (pd.to_datetime(start_date) - pd.DateOffset(years=2)).strftime('%Y-%m-%d')
            dm_config.data_settings['end_date'] = end_date
            
            dm = DataManager(dm_config, backtest_mode=True)
            if not dm.load_data():
                raise TuningError("DataManager failed to load data.")
            
            trial_logger.info(f"DataManager loaded. Sector map contains {len(dm.sector_map)} entries.")
            if not dm.sector_map:
                trial_logger.warning("Sector map is empty after loading. Check data sources.")

            trial_logger.info("Initializing backtester with pre-loaded data...")
            backtester = Backtester(
                cfg=cfg,
                start_date=start_date,
                end_date=end_date,
                dm=dm  # Inject the pre-loaded DataManager
            )
            
            # Run backtest and collect metrics
            trial_logger.info("Starting backtest...")
            backtest_start = time.time()
            summary = backtester.run()
            backtest_duration = time.time() - backtest_start
            
            # Extract and validate metrics
            sharpe_ratio = float(summary.get('sharpe_ratio', -2.0))
            metrics.update({
                'sharpe_ratio': sharpe_ratio,
                'return': float(summary.get('total_return', 0.0)),
                'volatility': float(summary.get('volatility', 0.0)),
                'max_drawdown': float(summary.get('max_drawdown', 0.0)),
                'backtest_duration_seconds': backtest_duration,
                'status': 'completed'
            })
            
            # Report metrics to Optuna
            trial.report(sharpe_ratio, step=trial.number)
            
            # Check for pruning
            if trial.should_prune():
                metrics['status'] = 'pruned'
                trial_logger.warning("Trial pruned due to poor performance")
                raise optuna.TrialPruned("Trial pruned due to poor performance")
                
            trial_logger.info("Trial completed successfully")
            return sharpe_ratio
            
        except Exception as e:
            error_msg = f"Backtest failed: {str(e)}"
            trial_logger.error(error_msg, exc_info=True)
            raise TuningError(error_msg)
            
    except optuna.TrialPruned as e:
        metrics['status'] = 'pruned'
        metrics['error'] = str(e)
        trial_logger.warning(f"Trial pruned: {e}")
        raise
        
    except TuningError as e:
        metrics['status'] = 'failed'
        metrics['error'] = str(e)
        trial_logger.error(f"Tuning error: {e}", exc_info=True)
        raise
        
    except Exception as e:
        metrics['status'] = 'error'
        metrics['error'] = f"Unexpected error: {str(e)}"
        trial_logger.critical("Unexpected error in trial", exc_info=True)
        raise
    
    finally:
        # Finalize metrics and log completion
        try:
            end_time = datetime.now()
            duration = time.time() - trial_start_time
            metrics.update({
                'end_time': end_time.isoformat(),
                'duration_seconds': duration
            })
            
            trial_logger.info("-" * 80)
            trial_logger.info(
                f"TRIAL {trial_id} - {metrics['status'].upper()}"
                f"\nDuration: {duration:.2f}s"
                f"\nSharpe: {metrics['sharpe_ratio']:.4f}"
                f"\nReturn: {metrics['return']:.4f}"
                f"\nVolatility: {metrics['volatility']:.4f}"
                f"\nMax Drawdown: {metrics['max_drawdown']:.4f}"
            )
            if 'error' in metrics and metrics['error']:
                trial_logger.error(f"Error: {metrics['error']}")
            trial_logger.info("=" * 80 + "\n")
            
            # Clean up any resources if needed
            if 'backtester' in locals():
                try:
                    del backtester
                except Exception as e:
                    trial_logger.warning(f"Error during cleanup: {str(e)}")
                    
        except Exception as e:
            trial_logger.critical(f"Error in trial finalization: {str(e)}", exc_info=True)

def parse_arguments():
    """Parse and validate command line arguments."""
    parser = argparse.ArgumentParser(description="Hyperparameter tuner for the TimeFolio strategy.")
    
    # Required arguments
    parser.add_argument('--start-date', 
                      default='2023-01-01',
                      help='Start date for backtesting - focuses on 2023-2024 period (YYYY-MM-DD).')
    parser.add_argument('--end-date',
                      default='2024-12-27',
                      help='End date for backtesting - Friday aligned with weekend data fetching (YYYY-MM-DD).')
    
    # Tuning parameters
    parser.add_argument('-n', '--n-trials', 
                      type=int, 
                      default=DEFAULT_N_TRIALS,
                      help=f"Number of tuning trials (default: {DEFAULT_N_TRIALS}).")
    parser.add_argument('--study-name', 
                      default=DEFAULT_STUDY_NAME,
                      help=f"Name for the Optuna study (default: {DEFAULT_STUDY_NAME}).")
    parser.add_argument('--db-url', 
                      default=DEFAULT_DB_URL,
                      help=f"Database URL for Optuna results (default: {DEFAULT_DB_URL}).")
    
    # Execution control
    parser.add_argument('--n-jobs', 
                      type=int, 
                      default=1,
                      help='Number of parallel jobs to run (-1 for all cores).')
    parser.add_argument('--timeout', 
                      type=float, 
                      default=None,
                      help='Timeout in seconds for the entire optimization.')
    
    # Logging and output
    parser.add_argument('--log-level',
                      default='INFO',
                      choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                      help='Logging level (default: INFO).')
    parser.add_argument('--log-file',
                      default=None,
                      help='File to write logs to (default: tuner_<timestamp>.log).')
    
    return parser.parse_args()

def setup_logging(args):
    """Configure logging based on command line arguments."""
    log_handlers = [logging.StreamHandler()]
    
    if args.log_file:
        log_handlers.append(logging.FileHandler(args.log_file))
    
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=log_handlers
    )

def create_or_load_study(study_name: str, db_url: str) -> optuna.Study:
    """Create a new study or load an existing one."""
    try:
        storage = optuna.storages.RDBStorage(
            url=db_url,
            engine_kwargs={
                'pool_size': 20,
                'max_overflow': 10,
                'pool_timeout': 30,
                'pool_recycle': 3600
            }
        )
        
        # Check if study exists
        try:
            study = optuna.load_study(study_name=study_name, storage=storage)
            logger.info(f"Loaded existing study: {study_name}")
        except KeyError:
            study = optuna.create_study(
                study_name=study_name,
                storage=storage,
                direction='maximize',
                load_if_exists=True
            )
            logger.info(f"Created new study: {study_name}")
            
        return study
        
    except Exception as e:
        logger.critical(f"Failed to initialize study: {e}")
        raise

def print_study_summary(study: optuna.Study) -> None:
    """Print a summary of the study results."""
    if not study.trials:
        logger.warning("No trials completed in this study.")
        return
    
    best_trial = study.best_trial
    
    logger.info("\n" + "=" * 80)
    logger.info("TUNING COMPLETE")
    logger.info("=" * 80)
    
    # Basic info
    logger.info(f"Study name: {study.study_name}")
    logger.info(f"Number of completed trials: {len(study.trials)}")
    logger.info(f"Best trial number: {best_trial.number}")
    logger.info(f"Best Sharpe ratio: {best_trial.value:.4f}")
    
    # Best parameters
    logger.info("\nBest parameters:")
    for k, v in best_trial.params.items():
        logger.info(f"  {k}: {v}")
    
    # Additional metrics from the best trial
    if best_trial.user_attrs:
        logger.info("\nAdditional metrics from best trial:")
        for k, v in best_trial.user_attrs.items():
            if k not in best_trial.params:
                logger.info(f"  {k}: {v}")
    
    logger.info("=" * 80 + "\n")

def main():
    """Main entry point for the hyperparameter tuner."""
    try:
        # Parse arguments and set up logging
        args = parse_arguments()
        setup_logging(args)
        
        logger.info("=" * 80)
        logger.info("STARTING TIMEFOLIO HYPERPARAMETER TUNER")
        logger.info("=" * 80)
        logger.info(f"Start date: {args.start_date}")
        logger.info(f"End date: {args.end_date}")
        logger.info(f"Number of trials: {args.n_trials}")
        logger.info(f"Study name: {args.study_name}")
        logger.info(f"Database URL: {args.db_url}")
        logger.info("=" * 80 + "\n")
        
        # Set up the study
        study = create_or_load_study(args.study_name, args.db_url)
        
        # Run optimization
        # Create a wrapper for the objective function to pass static arguments
        objective_with_args = lambda trial: objective(
            trial, 
            start_date=args.start_date, 
            end_date=args.end_date
        )

        study.optimize(
            objective_with_args,
            n_trials=args.n_trials,
            n_jobs=args.n_jobs,
        )
        
        # Print results
        print_study_summary(study)
        
        return 0
        
    except KeyboardInterrupt:
        logger.warning("\nTuning interrupted by user.")
        return 1
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
        return 1
    finally:
        logger.info("Tuner shutdown complete.")

if __name__ == '__main__':
    sys.exit(main())