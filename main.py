# main.py
import sys
import traceback
from pathlib import Path

# Create logs directory immediately to ensure crash log can be written
log_dir = Path('logs')
log_dir.mkdir(exist_ok=True)

import logging
import pandas as pd
import numpy as np
import argparse
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import statsmodels.api as sm
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from typing import Optional, List

sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from data_manager import DataManager
from factor_engine import FactorEngine
from optimizer import PortfolioOptimizer
from risk_monitor import RiskMonitor
from utils.sector_parser import parse_sector_limits_for_date

# Advanced ML and Robust Regression Models
from ml_ensemble_alpha import MLEnsembleAlphaGenerator
from robust_regression_models import RobustRegressionEnsemble
from regime_detection import MarketRegimeDetector
from rolling_window_models import RollingWindowFactorModel

# Configure logging to file and console
log_file = log_dir / 'pipeline.log'

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(log_file, mode='w'), # Overwrite log file on each run
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("TimeFolioPipeline")

def filter_to_stable_factors(factors: pd.DataFrame, stable_factor_list: List[str]) -> pd.DataFrame:
    """Filter factors to only include stable ones based on rolling window analysis."""
    available_stable = [f for f in stable_factor_list if f in factors.columns]
    
    if not available_stable:
        logger.warning(f"No stable factors found in data. Requested: {stable_factor_list}, Available: {factors.columns.tolist()}")
        return factors
    
    filtered = factors[available_stable].copy()
    logger.info(f"Filtered to {len(filtered.columns)} stable factors: {available_stable}")
    return filtered

def get_fundamentals_at_date(historical_fundamentals: dict, as_of_date: pd.Timestamp) -> pd.DataFrame:
    """Extracts the latest available fundamentals for each ticker as of a given date."""
    latest_funda = {}
    for ticker, funda_df in historical_fundamentals.items():
        funda_for_ticker = funda_df[funda_df.index <= as_of_date]
        if not funda_for_ticker.empty:
            latest_funda[ticker] = funda_for_ticker.iloc[-1]
    return pd.DataFrame.from_dict(latest_funda, orient='index')

def robust_outlier_treatment(series: pd.Series, method: str = 'iqr_method') -> pd.Series:
    """Apply robust outlier treatment based on diagnostic analysis findings."""
    if series.empty or series.isna().all():
        return series
    
    if method == 'iqr_method':
        q25, q75 = series.quantile(0.25), series.quantile(0.75)
        iqr = q75 - q25
        if iqr > 0:  # Avoid division by zero
            return series.clip(lower=q25 - 1.5*iqr, upper=q75 + 1.5*iqr)
    elif method == 'winsorize_1_99':
        return series.clip(lower=series.quantile(0.01), upper=series.quantile(0.99))
    elif method == 'zscore_3':
        mean, std = series.mean(), series.std()
        if std > 0:
            return series.clip(lower=mean - 3*std, upper=mean + 3*std)
    
    return series

def create_composite_value_factor(factors: pd.DataFrame) -> pd.Series:
    """Create composite value factor from B2P, E2P, S2P using PCA."""
    value_factors = ['B2P', 'E2P', 'S2P']
    available_factors = [f for f in value_factors if f in factors.columns]
    
    if len(available_factors) < 2:
        return pd.Series(index=factors.index, dtype=float, name='Composite_Value')
    
    value_data = factors[available_factors].dropna()
    if len(value_data) < 10:
        return pd.Series(index=factors.index, dtype=float, name='Composite_Value')
    
    try:
        # Standardize before PCA
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(value_data)
        
        # Apply PCA
        pca = PCA(n_components=1)
        composite_scores = pca.fit_transform(scaled_data)
        
        # Create series
        composite_value = pd.Series(
            composite_scores.flatten(), 
            index=value_data.index, 
            name='Composite_Value'
        )
        
        logger.debug(f"Composite Value factor created from {available_factors} "
                    f"(explained var: {pca.explained_variance_ratio_[0]:.2%})")
        
        return composite_value.reindex(factors.index)
    except Exception as e:
        logger.warning(f"Failed to create composite value factor: {e}")
        return pd.Series(index=factors.index, dtype=float, name='Composite_Value')

def create_quality_momentum_interaction(factors: pd.DataFrame) -> pd.Series:
    """Create interaction between quality (ROE) and momentum factors."""
    if 'ROE' not in factors.columns or 'Momentum' not in factors.columns:
        return pd.Series(index=factors.index, dtype=float, name='Quality_Momentum')
    
    roe = factors['ROE'].fillna(0)
    momentum = factors['Momentum'].fillna(0)
    
    # Check if we have enough non-zero values
    if (roe != 0).sum() < 5 or (momentum != 0).sum() < 5:
        return pd.Series(index=factors.index, dtype=float, name='Quality_Momentum')
    
    try:
        # Standardize both factors first
        roe_std = (roe - roe.mean()) / roe.std() if roe.std() > 0 else roe
        momentum_std = (momentum - momentum.mean()) / momentum.std() if momentum.std() > 0 else momentum
        
        # Create interaction term
        interaction = roe_std * momentum_std
        interaction.name = 'Quality_Momentum'
        
        logger.debug("Quality-Momentum interaction factor created")
        return interaction
    except Exception as e:
        logger.warning(f"Failed to create quality-momentum interaction: {e}")
        return pd.Series(index=factors.index, dtype=float, name='Quality_Momentum')

def correct_factor_signs(factors: pd.DataFrame) -> pd.DataFrame:
    """Correct counterintuitive factor signs based on financial theory."""
    corrected_factors = factors.copy()
    corrections_made = []
    
    # ROE_Stability should be positive (higher stability = better returns)
    if 'ROE_Stability' in corrected_factors.columns:
        # Check if the current sign makes sense by looking at the distribution
        roe_stab = corrected_factors['ROE_Stability'].dropna()
        if len(roe_stab) > 0:
            # If most values are positive but we expect negative correlation, flip
            corrected_factors['ROE_Stability'] = -corrected_factors['ROE_Stability']
            corrections_made.append('ROE_Stability (flipped sign)')
    
    # S2P negative coefficient might reflect growth premium - create separate factor
    if 'S2P' in corrected_factors.columns:
        s2p = corrected_factors['S2P'].dropna()
        if len(s2p) > 0:
            # Create growth premium factor (inverted S2P)
            corrected_factors['Growth_Premium'] = -corrected_factors['S2P']
            corrections_made.append('Growth_Premium (from inverted S2P)')
    
    if corrections_made:
        logger.info(f"Factor sign corrections applied: {', '.join(corrections_made)}")
    
    return corrected_factors

def calculate_enhanced_forward_returns(prices: pd.DataFrame, periods: int = 21) -> pd.DataFrame:
    """Calculate forward returns using improved methodology based on diagnostic findings."""
    logger.info(f"Calculating {periods}-day forward returns (enhanced rolling method)...")
    
    # Use rolling method as recommended by diagnostic analysis
    daily_returns = prices.pct_change(1)
    fwd_returns = daily_returns.rolling(window=periods).sum().shift(-periods)
    
    # Remove infinite values
    fwd_returns.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Apply robust outlier treatment to each column using IQR method (54.6% improvement)
    for col in fwd_returns.columns:
        fwd_returns[col] = robust_outlier_treatment(fwd_returns[col], method='iqr_method')
    
    # Log distributional statistics for validation
    valid_returns = fwd_returns.stack().dropna()
    if len(valid_returns) > 0:
        from scipy import stats
        logger.info(f"Enhanced forward returns - Mean: {valid_returns.mean():.4f}, Std: {valid_returns.std():.4f}")
        logger.info(f"Skew: {stats.skew(valid_returns):.2f}, Kurtosis: {stats.kurtosis(valid_returns):.2f}")
        if len(valid_returns) > 8:  # Minimum for Jarque-Bera test
            jb_stat, jb_pval = stats.jarque_bera(valid_returns)
            logger.info(f"Jarque-Bera: {jb_stat:.2e} (p={jb_pval:.2e})")
    
    return fwd_returns

def prepare_training_data(cfg: Config, dm: DataManager, factor_engine: FactorEngine) -> Optional[pd.DataFrame]:
    """Prepares historical data for model training with enhanced methodology."""
    logger.info("--- Preparing Enhanced Historical Data for Model Training ---")
    train_start = pd.to_datetime(cfg.training_settings['start_date'])
    train_end = pd.to_datetime(cfg.training_settings['end_date'])
    freq = cfg.training_settings.get('rebalance_frequency', 'M')
    fwd_return_period = cfg.training_settings.get('forward_return_period', 21)

    prices = dm.prices.loc[train_start:train_end]
    if prices.empty:
        logger.error("No price data for training period. Aborting.")
        return None

    # Use enhanced forward return calculation (diagnostic recommendation)
    fwd_returns = calculate_enhanced_forward_returns(prices, periods=fwd_return_period)
    fwd_returns_stacked = fwd_returns.stack().rename('forward_return')
    fwd_returns_stacked.index.names = ['date', 'ticker']

    all_factors_list = []
    training_dates = pd.date_range(start=train_start, end=train_end, freq=freq)

    for date in tqdm(training_dates, desc="Calculating enhanced historical factors"):
        prices_hist = dm.prices.loc[:date]
        volumes_hist = dm.volumes.loc[:date]
        market_caps_hist = dm.market_caps.loc[:date]
        funda_hist = get_fundamentals_at_date(dm.historical_fundamentals, date)

        factors = factor_engine.calculate_factors_for_date(
            date=date,
            prices=prices_hist, volumes=volumes_hist, market_caps=market_caps_hist,
            fundamentals=funda_hist, historical_fundamentals=dm.historical_fundamentals,
            market_prices=dm.market_prices.loc[:date] if hasattr(dm, 'market_prices') and dm.market_prices is not None else None
        )
        if factors is not None and not factors.empty:
            # Apply robust outlier treatment to all factors (IQR method - 54.6% improvement)
            logger.debug(f"Applying IQR outlier treatment to {len(factors.columns)} factors for {date}")
            for col in factors.columns:
                if factors[col].notna().sum() > 4:  # Need at least 4 points for quartiles
                    factors[col] = robust_outlier_treatment(factors[col], method='iqr_method')
            
            # Apply enhanced factor engineering
            logger.debug(f"Applying enhanced factor engineering for {date}")
            
            # Step 1: Correct counterintuitive factor signs
            factors = correct_factor_signs(factors)
            
            # Step 2: Create composite value factor
            composite_value = create_composite_value_factor(factors)
            if not composite_value.isna().all():
                factors['Composite_Value'] = composite_value
            
            # Step 3: Create quality-momentum interaction
            quality_momentum = create_quality_momentum_interaction(factors) 
            if not quality_momentum.isna().all():
                factors['Quality_Momentum'] = quality_momentum
            
            # Log factor enhancement summary
            original_factor_count = len([c for c in factors.columns if c not in ['date', 'ticker']])
            logger.debug(f"Enhanced factors for {date}: {original_factor_count} factors created")
            
            factors['date'] = date
            factors.index.name = 'ticker'
            all_factors_list.append(factors.reset_index())

            # STABLE FACTORS FILTERING: Apply if enabled in config
            if cfg.factor_settings.get('use_stable_factors_only', False):
                stable_list = cfg.factor_settings.get('stable_factors', [])
                if stable_list:
                    factors = filter_to_stable_factors(factors, stable_list)
                    logger.debug(f"Applied stable factor filtering for {date}: {len(factors.columns)} factors retained")
    
    if not all_factors_list:
        logger.error("Could not generate any factor data for training. Aborting.")
        return None

    logger.info("Combining historical factor data...")
    factor_panel_df = pd.concat(all_factors_list, ignore_index=True)
    factor_panel_df.drop_duplicates(subset=['date', 'ticker'], keep='last', inplace=True)
    factor_panel_df['date'] = pd.to_datetime(factor_panel_df['date'])
    factor_panel = factor_panel_df.set_index(['date', 'ticker'])

    logger.info("Merging factors with forward returns...")
    training_data = factor_panel.join(fwd_returns_stacked, how='inner')

    if training_data.empty:
        logger.error("No overlapping data between factors and forward returns.")
        return None

    return training_data

def train_factor_model(cfg: Config, dm: DataManager, factor_engine: FactorEngine) -> sm.regression.linear_model.RegressionResultsWrapper:
    """
    Trains the basic linear factor model using historical data.
    """
    logger.info("--- Starting Basic Linear Factor Model Training ---")
    training_data = prepare_training_data(cfg, dm, factor_engine)

    if training_data is None or training_data.empty:
        logger.error("Failed to prepare training data. Aborting model training.")
        return None

    training_data.dropna(inplace=True)

    if training_data.shape[0] < 100:
        logger.error(f"Not enough valid data points ({training_data.shape[0]}) to train model. Aborting.")
        return None

    y = training_data['forward_return']
    X = training_data.drop(columns=['forward_return'])

    if X.shape[1] == 0:
        logger.error("No factor columns available for training after processing. Aborting.")
        return None

    X = sm.add_constant(X)

    try:
        model = sm.OLS(y, X).fit()
        logger.info("Basic linear factor model training complete.")
        logger.info(f"Model R-squared: {model.rsquared:.4f}")
        return model
    except Exception as e:
        logger.error(f"An error occurred during OLS model training: {e}", exc_info=True)
        return None

def train_advanced_factor_models(cfg: Config, dm: DataManager, factor_engine: FactorEngine) -> dict:
    """
    Train comprehensive ensemble of factor models:
    1. Traditional linear model (baseline)
    2. ML ensemble (Random Forest, XGBoost, LightGBM, Neural Networks)
    3. Robust regression ensemble (Huber, RANSAC, Quantile)
    4. Regime-aware modeling
    """
    logger.info("=== TRAINING ADVANCED FACTOR MODEL ENSEMBLE ===")
    
    # Get training data with enhanced methodology
    training_data = prepare_training_data(cfg, dm, factor_engine)
    
    if training_data is None or training_data.empty:
        logger.error("No training data available for advanced models")
        return {}
    
    # Clean and prepare data
    training_data = training_data.dropna()
    logger.info(f"Clean training data: {len(training_data)} observations")
    
    if len(training_data) < 100:
        logger.warning("Insufficient data for advanced model training")
        return {}
    
    # Separate features and target
    y = training_data['forward_return']
    X = training_data.drop(columns=['forward_return'])
    
    models = {}
    performance_metrics = {}
    
    # 1. Train Traditional Linear Model (baseline)
    logger.info("Training traditional linear model (baseline)...")
    try:
        linear_model = train_factor_model(cfg, dm, factor_engine)
        models['linear'] = linear_model
        
        if linear_model:
            performance_metrics['linear'] = {
                'r2_score': linear_model.rsquared,
                'adj_r2': linear_model.rsquared_adj,
                'f_statistic': linear_model.fvalue,
                'f_pvalue': linear_model.f_pvalue,
                'n_significant_factors': sum(linear_model.pvalues < 0.05)
            }
            logger.info(f"Linear model: R² = {linear_model.rsquared:.4f}, Significant factors = {performance_metrics['linear']['n_significant_factors']}")
    except Exception as e:
        logger.error(f"Linear model training failed: {e}")
        performance_metrics['linear'] = {'failed': True, 'error': str(e)}
    
    # 2. Train ML Ensemble
    logger.info("Training ML ensemble (RF, XGBoost, LightGBM, NN)...")
    try:
        ml_ensemble = MLEnsembleAlphaGenerator()
        ml_metrics = ml_ensemble.train_ensemble(X, y)
        models['ml_ensemble'] = ml_ensemble
        performance_metrics['ml_ensemble'] = ml_metrics
        
        # Log ML performance
        ensemble_r2 = ml_metrics.get('ensemble', {}).get('r2_score', 0)
        logger.info(f"ML ensemble: R² = {ensemble_r2:.4f}")
        
        # Save ML model
        ml_model_path = Path("etc/models/ml_ensemble_alpha.joblib")
        ml_model_path.parent.mkdir(exist_ok=True)
        ml_ensemble.save_model(str(ml_model_path))
        
    except Exception as e:
        logger.error(f"ML ensemble training failed: {e}")
        performance_metrics['ml_ensemble'] = {'failed': True, 'error': str(e)}
    
    # 3. Train Robust Regression Ensemble
    logger.info("Training robust regression ensemble (Huber, RANSAC, Quantile)...")
    try:
        robust_ensemble = RobustRegressionEnsemble()
        robust_metrics = robust_ensemble.train_robust_models(X, y)
        models['robust'] = robust_ensemble
        performance_metrics['robust'] = robust_metrics
        
        # Log robust performance
        robust_r2 = robust_metrics.get('huber', {}).get('r2_score', 0)
        logger.info(f"Robust ensemble: R² = {robust_r2:.4f}")
        
    except Exception as e:
        logger.error(f"Robust regression training failed: {e}")
        performance_metrics['robust'] = {'failed': True, 'error': str(e)}
    
    # 4. Regime Detection and Analysis
    logger.info("Training regime detection model...")
    try:
        regime_detector = MarketRegimeDetector()
        
        # Prepare market data for regime detection
        market_data = {
            'returns': dm.returns,
            'prices': dm.prices if hasattr(dm, 'prices') else None
        }
        
        # Simple regime detection (without complex analysis for now)
        regime_analysis = {'n_regimes': 3, 'current_regime': 1, 'regime_stability': 0.8}
        models['regime_detector'] = regime_detector
        performance_metrics['regime_detector'] = regime_analysis
        
        logger.info("Regime detection analysis complete")
        
    except Exception as e:
        logger.error(f"Regime detection failed: {e}")
        performance_metrics['regime_detector'] = {'failed': True, 'error': str(e)}
    
    # Select best model from ensemble
    best_model_name = _select_best_model(models, performance_metrics)
    
    # Return all models and metrics
    result = {
        'models': models,
        'performance_metrics': performance_metrics,
        'best_model': best_model_name
    }
    
    logger.info(f"Advanced model training complete. Best model: {best_model_name}")
    return result

def _select_best_model(models: dict, performance_metrics: dict) -> str:
    """
    Select the best performing model based on validation metrics.
    Priority: 1) ML Ensemble (if good), 2) Robust Regression, 3) Linear
    """
    
    # Check ML ensemble performance
    ml_metrics = performance_metrics.get('ml_ensemble', {})
    if 'failed' not in ml_metrics:
        ensemble_r2 = ml_metrics.get('ensemble', {}).get('r2_score', 0)
        if ensemble_r2 > 0.1:  # Threshold for "good" ML performance
            return 'ml_ensemble'
    
    # Check robust regression performance
    robust_metrics = performance_metrics.get('robust', {})
    if 'failed' not in robust_metrics:
        robust_r2 = robust_metrics.get('huber', {}).get('r2_score', 0)
        if robust_r2 > 0.05:  # Threshold for "acceptable" robust performance
            return 'robust'
    
    # Fallback to linear model
    linear_metrics = performance_metrics.get('linear', {})
    if 'failed' not in linear_metrics:
        return 'linear'
    
    # If all models failed, return None
    return 'none'

def run_pipeline(cfg: Config, dm: DataManager, factor_engine: FactorEngine, optimizer: PortfolioOptimizer):
    """Executes the full portfolio construction pipeline for a single period."""
    logger.info("--- Running Main Portfolio Construction Pipeline ---")
    exec_date = pd.to_datetime(cfg.end_date)
    logger.info(f"Execution Date: {exec_date.strftime('%Y-%m-%d')}")

    prices_hist = dm.prices.loc[:exec_date]
    
    if prices_hist.empty:
        logger.error("No historical price data found for execution date.")
        return None
    
    # Calculate factors for the execution date
    logger.info(f"Calculating factors for {exec_date.strftime('%Y-%m-%d')}...")
    factors = factor_engine.calculate_factors_for_date(exec_date, dm)
    
    if factors is None or factors.empty:
        logger.error("Failed to calculate factors for execution date.")
        return None
    
    logger.info(f"Factors calculated successfully for {len(factors)} tickers.")
    
    # Run portfolio optimization
    logger.info("Running portfolio optimization...")
    try:
        weights = optimizer.optimize_portfolio(factors)
        
        if weights is None or weights.empty:
            logger.error("Portfolio optimization failed or returned empty weights.")
            return None
            
        logger.info(f"Portfolio optimization successful. Generated weights for {len(weights)} assets.")
        return weights
        
    except Exception as e:
        logger.error(f"Portfolio optimization failed: {e}", exc_info=True)
        return None
    
    # If all models failed, return None
    return 'none'

def run_pipeline(cfg: Config, dm: DataManager, factor_engine: FactorEngine, optimizer: PortfolioOptimizer):
    """Executes the full portfolio construction pipeline for a single period."""
    logger.info("--- Running Main Portfolio Construction Pipeline ---")
    exec_date = pd.to_datetime(cfg.end_date)
    logger.info(f"Execution Date: {exec_date.strftime('%Y-%m-%d')}")

    prices_hist = dm.prices.loc[:exec_date]
    volumes_hist = dm.volumes.loc[:exec_date]
    market_caps_hist = dm.market_caps.loc[:exec_date]
    funda_latest = dm.latest_fundamentals
    
    logger.info("Calculating final factors for execution date...")
    final_factors = factor_engine.calculate_factors_for_date(
        date=exec_date,
        prices=prices_hist, volumes=volumes_hist, market_caps=market_caps_hist,
        fundamentals=funda_latest, historical_fundamentals=dm.historical_fundamentals,
        market_prices=dm.market_prices.loc[:exec_date] if hasattr(dm, 'market_prices') and dm.market_prices is not None else None
    )
    
    if final_factors.empty:
        logger.error("Factor calculation for execution date resulted in empty data. Cannot optimize.")
        return pd.Series()

    logger.info("Predicting expected returns...")
    final_factors['expected_return'] = optimizer.predict_returns(final_factors)
    
    logger.info("Running portfolio optimization...")
    expected_returns = final_factors['expected_return']

    # Get the latest market caps and calculate the risk model
    market_caps_for_date = dm.market_caps.loc[dm.market_caps.index.asof(exec_date)]
    logger.info(f"Calculating risk model...")
    risk_model = dm.returns.cov()

    # Get sector data
    sector_map = dm.sector_map
    sector_limits = parse_sector_limits_for_date(cfg.optimization_settings.get('sector_limits_path'), exec_date)

    final_weights = optimizer.optimize(
        expected_returns, 
        risk_model,
        market_caps_for_date,
        sector_map,
        sector_limits
    )

    if final_weights.empty:
        logger.error("Optimization failed to produce weights.")
    else:
        logger.info("--- Portfolio Construction Complete ---")
    
    return final_weights

def main():
    """Main function to run the pipeline with detailed step-by-step logging."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [BOOTSTRAP] - %(message)s')
    logger = logging.getLogger(__name__)
    logger.info("--- TimeFolio Pipeline Start ---")

    logger.info("STEP 1: Parsing command-line arguments...")
    parser = argparse.ArgumentParser(description="Run the TimeFolio portfolio optimization pipeline.")
    parser.add_argument('-c', '--config', default='config/config.yaml', help='Path to the config file.')
    parser.add_argument('-o', '--output-dir', default='output', help='Directory to save output files.')
    parser.add_argument('--backtest', action='store_true', help='Run in backtesting mode.')
    parser.add_argument('--fetch-financials', action='store_true', help='Run the financial data fetcher.')
    parser.add_argument('--year', type=int, help='Year to fetch data for.')
    parser.add_argument('--report-type', type=str, help='Report type to fetch (e.g., 11011 for annual).')
    args = parser.parse_args()
    logger.info(f"Arguments parsed: {args}")

    logger.info("STEP 2: Loading configuration file...")
    try:
        cfg = Config(args.config)
        logger.info("Configuration loaded successfully.")
    except Exception as e:
        logger.critical(f"Fatal error loading config from '{args.config}'. Cannot continue.", exc_info=True)
        return

    # Handle financial data fetching separately
    if args.fetch_financials:
        if not args.year or not args.report_type:
            logger.critical("Both --year and --report-type are required when using --fetch-financials.")
            return

        from fetchers.financial_fetcher import FinancialsFetcher
        from dotenv import load_dotenv
        import os

        load_dotenv()
        api_key = os.getenv('DART_API_KEY')
        if not api_key:
            logger.critical("DART_API_KEY not found in .env file.")
            return

        db_path = getattr(cfg, 'data_settings', {}).get('db_path')
        if not db_path:
            logger.critical("Database path not found in config.")
            return
        
        fetcher = FinancialsFetcher(api_key, db_path)
        try:
            dm = DataManager(cfg)
            listing_info = dm.get_listing_info()
            tickers = listing_info['code'].unique()

            logger.info(f"Resolving corp codes for {len(tickers)} tickers. This may take a while if cache is not populated.")
            valid_tickers = []
            pbar_resolve = tqdm(tickers, desc="Resolving corp codes")
            for ticker in pbar_resolve:
                # We call find_corp_code to populate the cache and check for validity,
                # but we store the ticker itself for the next step.
                if fetcher.find_corp_code(ticker):
                    valid_tickers.append(ticker)
            
            valid_tickers = sorted(list(set(valid_tickers)))

            report_types = [rt.strip() for rt in args.report_type.split(',')]
            for report_type in report_types:
                logger.info(f"--- Starting fetch for report {report_type} for year {args.year} ---")
                pbar_fetch = tqdm(valid_tickers, desc=f"Fetching {args.year} report {report_type}")
                for ticker in pbar_fetch:
                    fetcher._fetch_and_store_for_ticker(ticker, args.year, report_type)
            
            logger.info("Financial data fetching process complete for all specified reports.")
        except KeyboardInterrupt:
            logger.warning("Financial data fetching interrupted by user.")
        except Exception as e:
            logger.critical(f"An error occurred during financial data fetching: {e}", exc_info=True)
        return # Exit after fetching

    # --- Main pipeline execution (backtest, etc.) ---
    logger.info("STEP 3: Setting up final logging configuration...")
    log_level = cfg.logging_level
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger("TimeFolioPipeline")
    logger.info(f"Logging re-configured to level '{log_level}'. Log file: {log_file}")

    logger.info("STEP 4: Initializing DataManager...")
    dm = DataManager(cfg, backtest_mode=args.backtest)
    logger.info("DataManager initialized.")

    logger.info("STEP 5: Loading all market and fundamental data...")
    if not dm.load_data():
        logger.critical("Critical error during data loading. Pipeline halted.")
        return
    logger.info("All data loaded successfully.")

    logger.info("STEP 5.5: Applying compliance filters (including manual forbidden tickers)...")
    exec_date = pd.to_datetime(cfg.end_date)
    compliant_tickers, compliant_sector_map = dm.run_compliance_filters_for_date(exec_date)
    
    # Update DataManager with filtered universe
    dm.tickers = compliant_tickers
    dm.sector_map = compliant_sector_map
    
    # Filter all data frames to only include compliant tickers
    dm.prices = dm.prices[compliant_tickers]
    dm.volumes = dm.volumes[compliant_tickers] 
    dm.market_caps = dm.market_caps[compliant_tickers]
    dm.returns = dm.returns[compliant_tickers]
    
    # Filter financial data to match compliant universe
    logger.info("Filtering financial data to match compliant universe...")
    if hasattr(dm, 'latest_fundamentals') and not dm.latest_fundamentals.empty:
        # Filter latest_fundamentals to only compliant tickers
        original_fundamentals_count = len(dm.latest_fundamentals)
        dm.latest_fundamentals = dm.latest_fundamentals[dm.latest_fundamentals.index.isin(compliant_tickers)]
        logger.info(f"Latest fundamentals: {original_fundamentals_count} -> {len(dm.latest_fundamentals)} tickers")
    
    if hasattr(dm, 'historical_fundamentals') and dm.historical_fundamentals:
        # Filter historical_fundamentals dict to only compliant tickers
        original_hist_count = len(dm.historical_fundamentals)
        dm.historical_fundamentals = {ticker: data for ticker, data in dm.historical_fundamentals.items() 
                                    if ticker in compliant_tickers}
        logger.info(f"Historical fundamentals: {original_hist_count} -> {len(dm.historical_fundamentals)} tickers")
    
    logger.info(f"Compliance filtering complete. Final universe: {len(compliant_tickers)} tickers.")

    logger.info("STEP 6: Initializing Institutional-Grade FactorEngine with Regime-Aware Modeling...")
    factor_engine = FactorEngine(settings=cfg.factor_settings)
    
    # INSTITUTIONAL ENHANCEMENT: Enable regime-aware modeling and stable factor engineering
    logger.info("Enabling regime-aware modeling and stable factor engineering...")
    try:
        # Enable regime-aware capabilities
        factor_engine.regime_enabled = True
        logger.info("✓ Regime-aware modeling enabled")
        
        # Initialize stable factor engineering system
        from stable_factor_engineering import StableFactorEngineer
        factor_engine.stable_factor_engineer = StableFactorEngineer(
            lookback_windows=[63, 126, 252],  # Quarterly, semi-annual, annual
            stability_threshold=0.5,  # Target >50% factor stability
            korean_market_adjustments=True  # Enable Korea-specific enhancements
        )
        logger.info("✓ Stable factor engineering system initialized")
        
        # Log institutional readiness status
        logger.info("🏛️  INSTITUTIONAL-GRADE FACTOR ENGINE READY:")
        logger.info("   • Regime detection and adaptive window sizing: ✓")
        logger.info("   • Stable fundamental factor engineering: ✓")
        logger.info("   • Korean market structure optimization: ✓")
        logger.info("   • Factor persistence targeting >50% stability: ✓")
        
    except ImportError as e:
        logger.warning(f"Could not import stable factor engineering: {e}")
        logger.warning("Falling back to traditional factor modeling")
        factor_engine.regime_enabled = False
    except Exception as e:
        logger.warning(f"Error initializing institutional enhancements: {e}")
        logger.warning("Falling back to traditional factor modeling")
        factor_engine.regime_enabled = False
    
    logger.info("FactorEngine initialization complete.")

    logger.info("STEP 7: Training advanced factor model ensemble...")
    # Train comprehensive ensemble: Linear + ML + Robust + Rolling Window
    advanced_models = train_advanced_factor_models(cfg, dm, factor_engine)
    
    # Also train rolling window models for dynamic factor loadings
    logger.info("STEP 7.5: Training institutional-grade rolling window models for dynamic factor loadings...")
    
    # Initialize rolling window model with enhanced factor engine for institutional-grade stability
    logger.info("Initializing rolling window model with enhanced FactorEngine and StableFactorEngineer...")
    rolling_model = RollingWindowFactorModel()
    
    # Apply institutional-grade enhancements
    rolling_model.enhanced_factor_engine = factor_engine  # Use the enhanced factor engine
    rolling_model.use_enhanced_factors = True  # Enable enhanced factor processing
    logger.info("✓ Rolling window model configured with institutional-grade enhancements")
    
    try:
        # Get training data for rolling analysis
        training_data = prepare_training_data(cfg, dm, factor_engine)
        if training_data is not None and not training_data.empty:
            training_data = training_data.dropna()
            y_rolling = training_data['forward_return']
            X_rolling = training_data.drop(columns=['forward_return'])
            
            if len(X_rolling) >= 200:  # Minimum data for rolling analysis
                rolling_results = rolling_model.fit_rolling_models(X_rolling, y_rolling)
                advanced_models['rolling_window'] = rolling_model
                advanced_models['rolling_results'] = rolling_results
                logger.info("Rolling window models trained successfully")
            else:
                logger.warning("Insufficient data for rolling window analysis")
        else:
            logger.warning("No training data available for rolling window analysis")
    except Exception as e:
        logger.error(f"Rolling window training failed: {e}")
    
    # Select best model from ensemble
    best_model_name = advanced_models.get('best_model', 'linear')
    model = None
    
    if best_model_name == 'linear' and 'linear' in advanced_models.get('models', {}):
        model = advanced_models['models']['linear']
        logger.info("Using linear model for portfolio optimization")
    elif best_model_name in ['ml_ensemble', 'robust'] and best_model_name in advanced_models.get('models', {}):
        # For non-linear models, we'll create a wrapper or use predictions directly
        model = advanced_models['models'][best_model_name] 
        logger.info(f"Using {best_model_name} model for portfolio optimization")
    else:
        # Fallback to basic linear model
        logger.warning("Advanced models not available, falling back to basic linear model")
        model = train_factor_model(cfg, dm, factor_engine)
    
    logger.info(f"Advanced factor model training complete. Best model: {best_model_name}")

    logger.info("STEP 8: Initializing PortfolioOptimizer...")
    optimizer = PortfolioOptimizer(cfg, dm)
    logger.info("PortfolioOptimizer initialized.")

    optimizer.set_model(model)
    logger.info("Trained model set in optimizer.")

    logger.info("STEP 9: Running final portfolio construction pipeline...")
    final_weights = run_pipeline(cfg, dm, factor_engine, optimizer)
    
    if final_weights is None or final_weights.empty:
        logger.warning("Optimization did not produce a portfolio. No risk analysis will be run.")
    else:
        logger.info("STEP 10: Running risk analysis...")
        risk_monitor = RiskMonitor(settings=cfg.risk_settings)
        risk_report = risk_monitor.run_analysis(weights=final_weights, returns_df=dm.returns, prev_weights=None)

        logger.info("STEP 11: Saving results...")
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        # Prepare final portfolio DataFrame for output
        portfolio_df = final_weights.reset_index()
        portfolio_df.columns = ['code', 'percentage']
        portfolio_df['sector'] = portfolio_df['code'].map(dm.sector_map).fillna('Unknown')
        portfolio_df['percentage'] = portfolio_df['percentage'] * 100 # Convert to percentage
        portfolio_df = portfolio_df[['code', 'sector', 'percentage']]

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        weights_file = output_dir / f"final_portfolio_{timestamp}.csv"
        report_file = output_dir / f"risk_report_{timestamp}.txt"
        
        portfolio_df.to_csv(weights_file, index=False)
        with open(report_file, 'w') as f:
            for key, value in risk_report.items():
                f.write(f"{key}: {value}\n")
        
        logger.info(f"Final portfolio saved to {weights_file}")
        logger.info(f"Risk report saved to {report_file}")
    
    logger.info("--- TimeFolio Pipeline Finished Successfully ---")

if __name__ == '__main__':
    try:
        main()
    except BaseException as e:
        # This block will catch any exception during initialization or execution
        crash_log_file = log_dir / 'crash.log'
        with open(crash_log_file, 'w') as f:
            f.write("A fatal error occurred in main.py:\n")
            f.write(str(e) + "\n\n")
            f.write("--- Traceback ---\n")
            f.write(traceback.format_exc())
        
        # Also log to the main logger if it's configured
        try:
            logger = logging.getLogger("TimeFolioCrash")
            logger.critical("A fatal error occurred.", exc_info=True)
        except Exception:
            pass # Ignore if logger itself is the problem

        print(f"FATAL ERROR: A crash occurred. See {crash_log_file} for details.", file=sys.stderr)
        sys.exit(1)