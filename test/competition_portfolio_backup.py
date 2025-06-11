#!/usr/bin/env python3
import os
import sys
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import yaml
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import FinanceDataReader as fdr
from pandas_datareader import data as pdr
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import cvxpy as cp

from portfolio_optimizer import Config, DataManager, CVaROptimizer
from dynamic_weighting import get_dynamic_weights, save_beta_state, visualize_beta_history
from factor_engine import FactorEngine

# Import non-ML modules that are always needed
from risk_monitor import generate_risk_report, RiskMonitor
from compliance_filters import filter_universe, load_forbidden_tickers

# ML and tuning modules will be imported conditionally

# Utility functions
def load_market_sectors(path="market_sectors.csv") -> dict:
    limits = {}
    in_section = False
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('[') and line.endswith(']'):
                in_section = not in_section
                continue
            if in_section:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        pct = float(parts[1]) / 100.0
                        limits[parts[0]] = min(2*pct, 0.10)
                    except ValueError:
                        continue
    logging.info(f"Loaded sector limits for {len(limits)} sectors")
    return limits


def optimize_with_sectors(cfg, returns: pd.DataFrame, sector_limits: dict) -> pd.Series:
    tickers = returns.columns.tolist()
    n = len(tickers)
    w = cp.Variable(n)
    var = cp.Variable()
    aux = cp.Variable(returns.shape[0])
    R   = returns.values
    mu  = returns.mean() * 252
    rf  = cfg.risk_free_rate

    # CVaR term
    cvar = var + (1/(returns.shape[0] * cfg.cvar_alpha)) * cp.sum(aux)
    # DCP‐safe objective: linear return minus CVaR penalty
    portfolio_return = mu.values @ w
    obj = cp.Maximize(portfolio_return - cfg.return_weight * cvar)

    cons = [
        cp.sum(w) == 1,
        w >= 0,
        w <= cfg.individual_limit,
        aux >= 0,
        aux >= -R @ w - var
    ]
    sec_map = returns.attrs.get('sector_map', {})
    for sec, cap in sector_limits.items():
        idxs = [i for i, t in enumerate(tickers) if sec_map.get(t) == sec]
        if idxs:
            cons.append(cp.sum(w[idxs]) <= cap)

    prob = cp.Problem(obj, cons)
    prob.solve(solver=cp.SCS)
    wv = np.maximum(w.value, 0)
    out = pd.Series(wv, index=tickers)
    return out / out.sum()


def enforce_turnover(new_w: pd.Series, old_w: pd.Series, min_to: float) -> pd.Series:
    if old_w is None:
        return new_w
    # Remove any duplicate tickers in previous weights
    old_unique = old_w[~old_w.index.duplicated(keep='first')]
    # Align previous weights to the new portfolio's index
    old_aligned = old_unique.reindex(new_w.index, fill_value=0)
    diff = new_w - old_aligned
    net_turn = diff.abs().sum() / 2
    # If turnover requirement already met, just use new weights
    if net_turn >= min_to:
        return new_w
    # No change desired
    if net_turn == 0:
        return old_aligned
    # Otherwise, scale changes to meet minimum turnover
    factor = min_to / net_turn
    blended = old_aligned + diff * factor
    return blended / blended.sum()

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def fetch_10y_treasury_yield() -> float:
    try:
        end = datetime.today()
        start = end - timedelta(days=7)
        df = pdr.DataReader('DGS10', 'fred', start, end)
        return float(df['DGS10'].dropna().iloc[-1]) / 100.0
    except Exception as e:
        logger.warning(f"fetch_10y_treasury_yield failed: {e}")
        return None


def monte_carlo_stress_test(returns: pd.DataFrame, weights: pd.Series,
                           horizon_days: int = 30, simulations: int = 500,
                           alpha: float = 0.05) -> dict:
    start = returns.index[0].strftime("%Y%m%d")
    end = returns.index[-1].strftime("%Y%m%d")
    dm = DataManager(cfg)

    # Build a dict of price series for each numeric ticker code
    price_dict = {}
    for t in weights.index:
        code = str(t)
        if not code.isdigit():
            continue
        ser = dm.fetch_stock_prices(code, start, end)
        if ser is not None:
            price_dict[code] = ser
    price_df = pd.DataFrame(price_dict).ffill().dropna()
    # forward‐fill, then drop only rows where *all* columns are NaN
    price_df = price_df.ffill().dropna(how='all', axis=0)

    # if we still have no data, bail out gracefully
    if price_df.empty:
        logger.warning("monte_carlo_stress_test: no price series available → skipping stress test")
        return {'var': np.nan, 'cvar': np.nan, 'sims': np.zeros((horizon_days, 0))}

    port = (price_df * weights).sum(axis=1)
    lr = np.log(port / port.shift(1)).dropna()
    drift = lr.mean() - 0.5 * lr.var()
    vol = lr.std()
    last = port.iloc[-1]

    sims = np.zeros((horizon_days, simulations))
    for i in range(simulations):
        price = last
        for j in range(horizon_days):
            shock = drift + vol * np.random.randn()
            price *= np.exp(shock)
            sims[j, i] = price
    final = sims[-1] / last - 1
    var = np.percentile(final, alpha * 100)
    cvar = final[final <= var].mean()
    return {'var': var, 'cvar': cvar, 'sims': sims}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Portfolio Optimization')
    parser.add_argument('-c', '--config', default='config.yaml')
    parser.add_argument('-p', '--positions', type=int, default=12)
    parser.add_argument('--use-ml-forecast', action='store_true', 
                        help='Use ML for return forecasting')
    parser.add_argument('--ml-model', choices=['lstm', 'xgboost'], default='xgboost',
                        help='ML model to use for forecasting')
    parser.add_argument('--tune-params', action='store_true',
                        help='Enable Bayesian parameter tuning')
    parser.add_argument('--n-tuning-calls', type=int, default=20,
                        help='Number of tuning iterations')
    parser.add_argument('-o','--output-dir', dest='out_dir',
                        default='out', help='Override data_settings.output_dir')
    args = parser.parse_args()

    cfg = Config(args.config)
    default_rf = cfg.risk_free_rate
    current_y = fetch_10y_treasury_yield()
    if current_y is not None:
        cfg.risk_free_rate = current_y
        logger.info(f"Risk-free rate updated to {cfg.risk_free_rate:.4f}")
    else:
        cfg.risk_free_rate = default_rf

    full_cfg = yaml.safe_load(open(args.config))
    out_dir = args.out_dir or full_cfg['data_settings']['output_dir']
    start = full_cfg['data_settings']['start_date']
    end = full_cfg['data_settings']['end_date']
    min_to = full_cfg.get('optimization', {}).get('min_turnover', 0.05)

    # Setup output directories
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    (out_dir_path / 'plots').mkdir(parents=True, exist_ok=True)
    (out_dir_path / 'models').mkdir(parents=True, exist_ok=True)
    (out_dir_path / 'risk_reports').mkdir(parents=True, exist_ok=True)
    
    # Apply compliance filters
    logger.info("Applying compliance filters...")
    forbidden_path = 'forbidden.csv'
    dm = DataManager(cfg)
    all_tickers = dm.tickers.copy()
    
    # Filter universe based on volume and IPO age
    clean_tickers = filter_universe(
        all_tickers,
        min_avg_daily_volume=300_000_000,  # 3억원 minimum volume
        min_ipo_days=90,                   # 3 months minimum IPO age
        output_file=forbidden_path,
        cache_dir='cache',
        use_cached=True,
        cache_expiry_days=7
    )
    logger.info(f"Filtered universe contains {len(clean_tickers)} stocks")
    
    # Load forbidden tickers (includes halted and other filtered stocks)
    forbidden = load_forbidden_tickers()
    clean_tickers = [t for t in clean_tickers if t not in forbidden]
    logger.info(f"Filtered out {len(forbidden)} forbidden/halted tickers")
    
    # Load sector limits
    sector_limits = load_market_sectors()
    
    # Initialize data storage
    prices_dict = {}
    volumes_dict = {}
    failed_tickers = []
    
    logger.info(f"Fetching price and volume data for {len(clean_tickers)} tickers...")
    
    # Try to fetch data for a sample ticker first to check data availability
    sample_ticker = clean_tickers[0] if clean_tickers else None
    if sample_ticker:
        try:
            sample_data = dm.fetch_stock_prices(sample_ticker, start_date, end_date)
            if sample_data is not None and not sample_data.empty:
                logger.info(f"Sample data for {sample_ticker} has {len(sample_data)} rows from {sample_data.index[0].date()} to {sample_data.index[-1].date()}")
            else:
                logger.warning(f"No data returned for sample ticker {sample_ticker}")
        except Exception as e:
            logger.error(f"Error fetching sample data for {sample_ticker}: {str(e)}")
    
    # Fetch data for all tickers
    for ticker in clean_tickers:
        try:
            # Get price data using the correct method
            price_data = dm.fetch_stock_prices(ticker, start_date, end_date)
            if price_data is not None and not price_data.empty:
                prices_dict[ticker] = price_data
                # Using price data as volume data for now since we don't have volume data
                volumes_dict[ticker] = price_data
                
                # Log the first successful fetch as an example
                if len(prices_dict) == 1:  # Only log for the first successful fetch
                    logger.info(f"Successfully fetched {len(price_data)} days of data for {ticker} from {price_data.index[0].date()} to {price_data.index[-1].date()}")
            else:
                failed_tickers.append(ticker)
                logger.warning(f"No data returned for {ticker}")
        except Exception as e:
            failed_tickers.append(ticker)
            logger.warning(f"Failed to fetch data for {ticker}: {str(e)}")
    
    if failed_tickers:
        logger.warning(f"Failed to fetch data for {len(failed_tickers)}/{len(clean_tickers)} tickers")
    
    if not prices_dict:
        logger.error("No price data could be fetched. Exiting.")
        sys.exit(1)
    
    # Create DataFrames from the collected data
    prices = pd.DataFrame(prices_dict)
    volumes = pd.DataFrame(volumes_dict)
    
    # Calculate returns from prices
    rets = prices.pct_change().dropna()
    
    if rets.empty:
        logger.error("No valid returns data after processing. Exiting.")
        sys.exit(1)
    
    # Ensure we only keep tickers that are in our clean list
    valid_cols = [t for t in rets.columns if t in clean_tickers]
    rets = rets[valid_cols]
    prices = prices[valid_cols]
    volumes = volumes[valid_cols]

    # Attach sector map so CVaROptimizer will enforce sector caps
    univ = pd.read_csv(cfg.stock_universe_file, dtype=str)
    univ.columns = univ.columns.str.strip()
    univ['ticker'] = (
        univ['종목코드']
           .str.strip()
           .str.lstrip('A')
           .str.zfill(6)
    )
    sector_map = dict(zip(univ['ticker'], univ['섹터코드']))
    rets.attrs['sector_map'] = sector_map

    # sanity check 
    logger.info(f"Fetched {rets.shape[1]} tickers × {rets.shape[0]} days of returns")
    if rets.shape[0] < 30:
        logger.critical(
            f"Insufficient historical returns ({rets.shape[0]} days < 30). "
            "Data fetch likely failed → exiting."
        )
        sys.exit(1)

    # Clustering
    # Remove NaN values and calculate correlation matrix
    corr = rets.corr().fillna(0)
    dist_mat = np.sqrt(2 * (1 - corr))
    # Deal with inf/nan values
    dist_mat = pd.DataFrame(
         np.nan_to_num(dist_mat.values, posinf=0, neginf=0),
         index=dist_mat.index,
         columns=dist_mat.columns
    )
    condensed = squareform(dist_mat.values, checks=False)
    # Clustering via Ward on condensed distance 
    Z = linkage(condensed, method='ward')
    # determine core (diverse) picks
    k = args.positions
    labels = fcluster(Z, t=k, criterion='maxclust')
    ratio = (rets.mean() * 252) / (rets.std() * np.sqrt(252))
    diverse = [
        max([t for t, lb in zip(rets.columns, labels) if lb == i],
            key=lambda t: ratio[t])
        for i in np.unique(labels)
    ]
    logger.info(f"Cluster picks: {diverse}")

    # Run Monte Carlo stress test
    logger.info("Running Monte Carlo stress test")
    stress_results = monte_carlo_stress_test(rets, pd.Series(1/len(rets.columns), index=rets.columns), horizon_days=21, simulations=1000)
    
    # Calculate summary statistics
    port_rets = rets.mean(axis=1)
    sharpe = port_rets.mean() / port_rets.std() * np.sqrt(252)
    cumulative = (1 + port_rets).cumprod()
    drawdowns = 1 - cumulative / cumulative.cummax()
    max_dd = drawdowns.max()
    
    logger.info(f"Portfolio Sharpe: {sharpe:.4f}")
    logger.info(f"Max Drawdown: {max_dd:.4f}")
    logger.info(f"Monte Carlo 5% CVaR (21d): {stress_results['cvar']:.4f}")
    
    # Run risk monitoring
    logger.info("Generating risk report...")
    try:
        # Load previous weights if available
        prev_weights_path = out_dir_path / 'previous_weights.csv'
        prev_weights = None
        if prev_weights_path.exists():
            prev_weights = pd.read_csv(prev_weights_path, index_col=0).iloc[:, 0]
        
        # Additional info for risk report
        additional_info = {
            'sharpe': f"{sharpe:.4f}",
            'monte_carlo_cvar': f"{stress_results['cvar']:.4f}",
            'positions': str(len(rets.columns)),
            'use_elite_alpha': str(args.use_elite_alpha),
            'use_ml_forecast': str(args.use_ml_forecast),
            'ml_model': args.ml_model if args.use_ml_forecast else 'N/A',
            'tactical_allocation': f"{args.tactical_allocation:.4f}" if args.tactical_allocation else 'N/A'
        }
        
        # Generate risk report
        risk_report_path = generate_risk_report(
            returns=port_rets,
            weights=pd.Series(1/len(rets.columns), index=rets.columns),
            previous_weights=prev_weights,
            risk_report_dir=str(out_dir_path / 'risk_reports'),
            additional_info=additional_info
        )
        logger.info(f"Risk report generated: {risk_report_path}")
        
        # Save current weights for next run
        pd.Series(1/len(rets.columns), index=rets.columns).to_frame('weight').to_csv(prev_weights_path)
    except Exception as e:
        logger.error(f"Risk monitoring failed: {e}")

    # Static weights
    dyn = get_dynamic_weights(rets)
    cfg.sharpe_treynor_weight = dyn['sharpe_treynor_weight']
    logger.info(f"Dynamic Sharpe–Treynor weight set to {cfg.sharpe_treynor_weight:.2f}")

    # Beta calculation
    market_returns = rets.mean(axis=1)
    var_mkt = market_returns.var()
    beta_vec = rets.apply(lambda x: x.cov(market_returns) / var_mkt)
    logger.info(f"Computed beta for {len(beta_vec)} tickers")

    # Core portfolio optimization setup
    
    # ML-based return forecasts
    if args.use_ml_forecast:
        logger.info(f"Generating ML forecasts using {args.ml_model} model...")
        try:
            # Conditionally import ML forecasting module
            from ml_forecast import forecast_returns
            expected_returns = forecast_returns(
                rets, 
                model_type=args.ml_model,
                use_cached=True,
                train_if_missing=True
            )
            logger.info(f"ML forecast generated for {len(expected_returns)} stocks")
        except Exception as e:
            logger.error(f"ML forecast failed: {e}")
            logger.info("Falling back to historical means")
            expected_returns = rets.mean() * 252
    else:
        # Use historical mean returns
        expected_returns = rets.mean() * 252
        
    # Ensure we have a numeric Series with the correct index
    if not isinstance(expected_returns, pd.Series):
        expected_returns = pd.Series(expected_returns, index=rets.columns)
    expected_returns = expected_returns.fillna(0)  # Fill any NA values with 0

    # Dynamic beta
    logger.info("Calculating dynamic beta-adjusted weights")
    beta_weights = get_dynamic_weights(rets, cfg.sharpe_treynor_weight)
    
    # Tuning optimizer parameters with Bayesian optimization
    if args.tune_params:
        logger.info("Tuning optimization parameters with Bayesian optimization...")
        try:
            # Conditionally import Bayesian tuning module
            from bayes_tuner import tune_hyperparams
            tuned_params = tune_hyperparams(
                cfg, 
                dm, 
                expected_returns=expected_returns,
                beta_vec=beta_weights,
                n_calls=args.n_tuning_calls
            )
            # Update config with tuned parameters
            logger.info(f"Tuned parameters: {tuned_params}")
            cfg.return_weight = tuned_params['return_weight']
            cfg.cvar_alpha = tuned_params['cvar_alpha']
            dispersion_gamma = tuned_params['dispersion_penalty']
        except Exception as e:
            logger.error(f"Parameter tuning failed: {e}")
            logger.info("Using default parameters")
            dispersion_gamma = 0.0
    else:
        # Use default or latest tuned parameters
        try:
            # Conditionally import BayesianTuner class
            from bayes_tuner import BayesianTuner
            tuner = BayesianTuner(cfg, dm)
            latest_params = tuner.get_latest_params()
            if latest_params:
                logger.info(f"Using previously tuned parameters: {latest_params}")
                cfg.return_weight = latest_params['return_weight']
                cfg.cvar_alpha = latest_params['cvar_alpha']
                dispersion_gamma = latest_params['dispersion_penalty']
            else:
                dispersion_gamma = 0.0
        except Exception as e:
            logger.error(f"Error loading tuned parameters: {e}")
            dispersion_gamma = 0.0
    
    # Construct CVaR Optimizer
    cvar_opt = CVaROptimizer(cfg, dm, beta_weights)
    
    # Initialize returns for optimization
    new_rets = rets.copy()

def process_portfolio_weights(weights, sector_map, sharpe_series, vol_series, out_dir='output', n_positions=12):
    """Process and save the optimized portfolio weights.
    
    Args:
        weights: Portfolio weights as a Series
        sector_map: Dictionary mapping tickers to sectors
        sharpe_series: Series of Sharpe ratios for each asset
        vol_series: Series of annualized volatilities for each asset
        out_dir: Output directory for saving results
        n_positions: Number of top positions to include in the output
        
    Returns:
        DataFrame with processed portfolio weights (top n_positions by weight)
    """
    # Ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)
    
    # Convert weights to Series if not already
    if not isinstance(weights, pd.Series):
        weights = pd.Series(weights, index=sharpe_series.index)
    
    # Create output DataFrame
    df_out = pd.DataFrame({
        'ticker': weights.index,
        'sector': weights.index.map(sector_map),
        'weight': weights.values,
        'sharpe_ratio': sharpe_series[weights.index],
        'volatility': vol_series[weights.index]
    })
    
    # Filter out zero-weight positions
    df_out = df_out[df_out['weight'] > 1e-6].copy()
    
    # Ensure we have at least one position
    if len(df_out) == 0:
        logger.error("No valid positions in the optimized portfolio")
        return None
    
    # Sort by weight in descending order and take top n_positions positions
    df_out = df_out.nlargest(n_positions, 'weight')
    
    # Re-normalize weights to sum to 1
    df_out['weight'] = df_out['weight'] / df_out['weight'].sum()
    
    # Calculate weight percentages
    df_out['weight_pct'] = df_out['weight'] * 100
    
    # Reorder columns to place weight_pct right after sector
    cols = ['ticker', 'sector', 'weight_pct', 'weight', 'sharpe_ratio', 'volatility']
    df_out = df_out[cols]
    
    # Save the processed portfolio
    output_path = os.path.join(out_dir, 'current_portfolio.csv')
    df_out.to_csv(output_path, index=False, float_format='%.6f')
    logger.info(f"Portfolio weights saved to {output_path}")
    
    return df_out

def calculate_expected_returns(prices: pd.DataFrame, volumes: pd.DataFrame) -> pd.Series:
    """Calculate expected returns using the Factor Engine.
    
    Args:
        prices: DataFrame of historical prices (tickers as columns, datetime index)
        volumes: DataFrame of historical volumes (same shape as prices)
        
    Returns:
        Series of expected returns (aligned with input columns)
    """
    logger.info("Calculating expected returns using Factor Engine...")
    
    # Initialize and run factor engine
    engine = FactorEngine(
        mom_windows=[20, 60, 120],
        vol_window=20,
        rsi_window=14,
        min_volume=3e9,  # 3B KRW
        volume_window=5
    )
    
    # Calculate all factors
    factors = engine.calculate_factors(prices, volumes)
    
    # Get filtered universe
    valid_stocks = engine.filter_universe(factors, prices)
    
    # Use composite scores as expected returns
    expected_returns = factors['composite'].iloc[-1]  # Latest composite scores
    
    # Set returns to 0 for invalid stocks
    expected_returns[~expected_returns.index.isin(valid_stocks[valid_stocks].index)] = 0
    
    # Convert to annualized returns (assuming 252 trading days)
    # Scale factor scores to reasonable return estimates (e.g., -30% to +30% annualized)
    expected_returns = (expected_returns - 0.5) * 0.6  # Scale to [-0.3, 0.3]
    
    return expected_returns

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Portfolio Optimization')
    parser.add_argument('-c', '--config', default='config.yaml')
    parser.add_argument('-p', '--positions', type=int, default=12)
    parser.add_argument('--use-ml-forecast', action='store_true', 
                        help='Use ML-based return forecasting')
    parser.add_argument('--ml-model', choices=['lstm', 'xgboost'], default='xgboost',
                       help='ML model to use for forecasting')
    parser.add_argument('--tune-params', action='store_true',
                       help='Enable hyperparameter tuning')
    parser.add_argument('--n-tuning-calls', type=int, default=20,
                       help='Number of tuning iterations')
    parser.add_argument('-o','--output-dir', dest='out_dir',
                       default='output', help='Output directory')
    
    args = parser.parse_args()
    
    # Initialize directories
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Initialize configuration and data manager
    cfg = Config(args.config)
    dm = DataManager(cfg)
    
    # Get date range for data fetching
    end_date = pd.Timestamp(cfg.end_date)
    start_date = pd.Timestamp(cfg.start_date)
    
    # Format dates as strings for the API
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    
    logger.info(f"Fetching data from {start_date.date()} to {end_date.date()} ({(end_date - start_date).days} days)")
    
    # Validate date range
    if (end_date - start_date).days < 30:
        logger.error(f"Date range too short: {(end_date - start_date).days} days. Need at least 30 days of data.")
        logger.info("Please check your config.yaml for start_date and end_date settings")
        sys.exit(1)
    
    # Load compliance filters and forbidden tickers
    forbidden = load_forbidden_tickers()
    
    # Filter out forbidden tickers from the data manager's tickers
    clean_tickers = [t for t in dm.tickers if t not in forbidden]
    
    # Initialize data storage
    prices_dict = {}
    volumes_dict = {}
    failed_tickers = []
    
    logger.info(f"Fetching price and volume data for {len(clean_tickers)} tickers...")
    
    # Try to fetch data for a sample ticker first to check data availability
    sample_ticker = clean_tickers[0] if clean_tickers else None
    if sample_ticker:
        try:
            sample_data = dm.fetch_stock_prices(sample_ticker, start_date, end_date)
            if sample_data is not None and not sample_data.empty:
                logger.info(f"Sample data for {sample_ticker} has {len(sample_data)} rows from {sample_data.index[0].date()} to {sample_data.index[-1].date()}")
            else:
                logger.warning(f"No data returned for sample ticker {sample_ticker}")
        except Exception as e:
            logger.error(f"Error fetching sample data for {sample_ticker}: {str(e)}")
    
    # Fetch data for all tickers
    for ticker in clean_tickers:
        try:
            # Get price data using the correct method
            price_data = dm.fetch_stock_prices(ticker, start_date, end_date)
            if price_data is not None and not price_data.empty:
                prices_dict[ticker] = price_data
                # Using price data as volume data for now since we don't have volume data
                volumes_dict[ticker] = price_data
                
                # Log the first successful fetch as an example
                if len(prices_dict) == 1:  # Only log for the first successful fetch
                    logger.info(f"Successfully fetched {len(price_data)} days of data for {ticker} from {price_data.index[0].date()} to {price_data.index[-1].date()}")
            else:
                failed_tickers.append(ticker)
                logger.warning(f"No data returned for {ticker}")
        except Exception as e:
            failed_tickers.append(ticker)
            logger.warning(f"Failed to fetch data for {ticker}: {str(e)}")
    
    if failed_tickers:
        logger.warning(f"Failed to fetch data for {len(failed_tickers)}/{len(clean_tickers)} tickers")
    
    if not prices_dict:
        logger.error("No price data could be fetched. Exiting.")
        sys.exit(1)
    
    # Create DataFrames from the collected data
    prices = pd.DataFrame(prices_dict)
    volumes = pd.DataFrame(volumes_dict)
    
    # Calculate returns from prices
    rets = prices.pct_change().dropna()
    
    if rets.empty:
        logger.error("No valid returns data after processing. Exiting.")
        sys.exit(1)
    
    # Calculate new_rets using the DataManager's get_returns method
    try:
        new_rets = dm.get_returns(start_date.strftime('%Y-%m-%d'), 
                                end_date.strftime('%Y-%m-%d'))
        # Align with our filtered tickers
        new_rets = new_rets[clean_tickers].dropna(axis=1, how='all')
        logger.info(f"Successfully fetched returns for {len(new_rets.columns)} tickers")
    except Exception as e:
        logger.warning(f"Failed to fetch returns using DataManager: {str(e)}")
        logger.info("Falling back to simple returns calculation")
        new_rets = rets.copy()
    
    # Calculate expected returns using Factor Engine
    expected_returns = calculate_expected_returns(prices, volumes)
    
    # Align expected returns with our returns dataframe
    expected_returns = expected_returns[expected_returns.index.isin(rets.columns)]
    
    # Calculate contribution targets (equal risk contribution)
    cov = new_rets.cov() * 252
    vol = np.sqrt(np.diag(cov))
    contrib_target = vol / vol.sum()
    
    # Ensure expected_returns is a numeric array with correct dimensions
    if isinstance(expected_returns, (pd.Series, pd.DataFrame)):
        expected_values = expected_returns.copy()
    else:
        expected_values = pd.Series(expected_returns)
    
    # Ensure we have numeric values
    if not pd.api.types.is_numeric_dtype(expected_values):
        expected_values = pd.to_numeric(expected_values, errors='coerce')
    
    # Create a Series with the same index as new_rets columns
    aligned_expected = pd.Series(0, index=new_rets.columns, dtype=float)
    
    # Fill in the values we have, leaving others as 0
    common_idx = expected_values.index.intersection(aligned_expected.index)
    if len(common_idx) > 0:
        aligned_expected[common_idx] = expected_values[common_idx]
    
    # Fill any remaining NaN values with 0
    expected_values = aligned_expected.fillna(0)
    
    # Convert to numpy array and ensure it's 1D
    expected_values = np.asarray(expected_values).flatten()
    
    # Run optimizer with dispersion penalty and position count
    weights = cvar_opt.optimise(
        new_rets, 
        expected_values,
        dispersion_gamma=dispersion_gamma,
        contrib_target=contrib_target,
        n_positions=args.positions  # Pass the number of positions from command line
    )

    # 1) Load and sort columns & sector codes in universe file 
    univ = pd.read_csv(cfg.stock_universe_file, dtype=str)
    univ.columns = univ.columns.str.strip()  # ['섹터코드','섹터명','종목코드','종목명']

    # 2) normalize tickers to six digits
    univ['ticker'] = (
        univ['종목코드']
        .str.strip()
        .str.lstrip('A')
        .str.zfill(6)
    )
    # 3) Create sector map
    sector_map = dict(zip(univ['ticker'], univ['섹터코드']))
    
    # 4) Calculate indicators
    sharpe_series = (rets.mean() / rets.std()) * np.sqrt(252)
    vol_series = rets.std() * np.sqrt(252)
    
    # Call the portfolio allocation function
    df_out = allocate_portfolio_weights(weights, sector_map, sharpe_series, vol_series)
    
    # Process and save the portfolio weights
    df_out = process_portfolio_weights(weights, sector_map, sharpe_series, vol_series, n_positions=args.positions)
    
    return df_out

def allocate_portfolio_weights(weights, sector_map, sharpe_series, vol_series, out_dir='output'):
    """
    Allocate portfolio weights based on optimization results.
    
    Args:
        weights: Series of optimized weights
        sector_map: Dictionary mapping tickers to sectors
        sharpe_series: Series of Sharpe ratios for each asset
        vol_series: Series of volatilities for each asset
        out_dir: Output directory for saving results
        
    Returns:
        DataFrame with allocated weights and portfolio metrics
    """
    # Ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)
    
    # Convert weights to DataFrame if not already
    if not isinstance(weights, pd.Series):
        weights = pd.Series(weights, index=sharpe_series.index)
    
    # Create output DataFrame
    df_out = pd.DataFrame({
        'ticker': weights.index,
        'weight': weights.values,
        'sector': weights.index.map(sector_map),
        'sharpe': sharpe_series,
        'volatility': vol_series
    })
    
    # Drop any rows with missing data
    df_out = df_out.dropna()
    
    # Sort by weight descending
    df_out = df_out.sort_values('weight', ascending=False)
    
    return df_out

if __name__ == "__main__":
    main()