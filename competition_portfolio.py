#!/usr/bin/env python3
import os
import sys
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cvxpy as cp

from portfolio_optimizer import Config, DataManager, optimize_portfolio

from factor_engine import FactorEngine

# Import non-ML modules that are always needed
from risk_monitor import generate_risk_report, RiskMonitor
from compliance_filters import load_forbidden_tickers

# Portfolio metrics import
from portfolio_metrics import compute_covariance
from portfolio_metrics import calculate_diversification_metrics

# --- Fetcher imports for factor integration ---
import sqlite3
from krx_fetcher import KRXFetcher
from financial_fetcher import FinancialsFetcher
from macro_fetcher import MacroFetcher

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# [Previous utility functions remain the same...]

def fetch_market_data(dm: DataManager, tickers: List[str], 
                     start_date: datetime, end_date: datetime) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch market data for a list of tickers with proper error handling.
    
    Args:
        dm: Initialized DataManager instance
        tickers: List of ticker symbols to fetch
        start_date: Start date for data
        end_date: End date for data
        
    Returns:
        Tuple of (prices_df, volumes_df) DataFrames
    """
    prices_dict = {}
    volumes_dict = {}
    failed_tickers = []
    
    # Format dates as strings for the API
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    
    logger.info(f"Fetching data for {len(tickers)} tickers from {start_str} to {end_str}")
    
    # Try to fetch data for a sample ticker first
    if tickers:
        sample_ticker = tickers[0]
        try:
            logger.info(f"Testing data fetch for sample ticker: {sample_ticker}")
            sample_data = dm.fetch_stock_prices(sample_ticker, start_str, end_str)
            if sample_data is not None and not sample_data.empty:
                logger.info(f"Sample data for {sample_ticker}: {len(sample_data)} rows from {sample_data.index[0].date()} to {sample_data.index[-1].date()}")
            else:
                logger.warning(f"No data returned for sample ticker {sample_ticker}")
        except Exception as e:
            logger.error(f"Error in sample fetch for {sample_ticker}: {str(e)}")
    
    # Fetch data for all tickers
    for i, ticker in enumerate(tickers, 1):
        try:
            price_data = dm.fetch_stock_prices(ticker, start_str, end_str)
            if price_data is not None and not price_data.empty:
                prices_dict[ticker] = price_data
                volumes_dict[ticker] = price_data  # Using price as volume for now
                
                # Log progress
                if len(prices_dict) % 10 == 0 or len(prices_dict) == 1:
                    logger.info(f"Fetched {len(prices_dict)}/{len(tickers)}: {ticker} ({len(price_data)} data points)")
            else:
                failed_tickers.append(ticker)
                logger.debug(f"No data for {ticker}")
        except Exception as e:
            failed_tickers.append(ticker)
            logger.debug(f"Failed to fetch {ticker}: {str(e)}")
        
        # Periodic progress update
        if i % 50 == 0:
            logger.info(f"Progress: {i}/{len(tickers)} tickers processed")
    
    # Log summary
    if failed_tickers:
        logger.warning(f"Failed to fetch {len(failed_tickers)}/{len(tickers)} tickers")
    
    if not prices_dict:
        logger.error("No price data could be fetched. Exiting.")
        sys.exit(1)
        
    logger.info(f"Successfully fetched data for {len(prices_dict)}/{len(tickers)} tickers")
    return pd.DataFrame(prices_dict), pd.DataFrame(volumes_dict)

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
    # Inject sector limits into config for use in optimizer constraints
    cfg.sector_limits = dm.sector_limits

    # Initialize data fetchers
    krx = KRXFetcher(db_path=cfg.db_path if hasattr(cfg, 'db_path') else 'krx_data.db')
    fin = FinancialsFetcher(api_key=os.getenv('DART_API_KEY'), db_path=cfg.db_path if hasattr(cfg, 'db_path') else 'krx_data.db')
    macro = MacroFetcher(fred_api_key=os.getenv('FRED_API_KEY'))
    DB_PATH = cfg.db_path if hasattr(cfg, 'db_path') else 'krx_data.db'
    
    # Get date range for data fetching
    end_date = pd.Timestamp(cfg.end_date)
    start_date = pd.Timestamp(cfg.start_date)
    
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
    
    # Fetch the data using our new function
    prices, volumes = fetch_market_data(dm, clean_tickers, start_date, end_date)
    
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

    # --- Asset-level technical factors via KRXFetcher ---
    krx_factors = krx.fetch_and_compute(clean_tickers, end_date.strftime('%Y-%m-%d'))
    # --- Fundamental factors via FinancialsFetcher ---
    # Ensure financials for codes are fetched (annual reports)
    fin.fetch_financials(clean_tickers, years=[int(cfg.end_date[:4]), int(cfg.end_date[:4]) - 1], report_type='annual')
    prices_df = pd.DataFrame({'close': prices.iloc[-1]}).T  # code-indexed latest close
    fin_factors = fin.compute_financial_factors(prices_df, end_date.strftime('%Y-%m-%d'))
    # --- Macro & regime factors via MacroFetcher ---
    macro.fetch_and_store(end_date.strftime('%Y-%m-%d'))
    macro_df = pd.read_sql(
        "SELECT * FROM macro_data WHERE date = ?", 
        sqlite3.connect(DB_PATH), 
        params=[end_date.strftime('%Y-%m-%d')],
        parse_dates=['date']
    ).set_index('date').iloc[0]

    # 1. Calculate dynamic beta
    logger.info("Calculating dynamic beta...")
    market_returns = rets.mean(axis=1)
    var_mkt = market_returns.var()
    beta_vec = rets.apply(lambda x: x.cov(market_returns) / var_mkt)

    # 2. Initialize Factor Engine and calculate factors
    logger.info("Initializing Factor Engine...")
    factor_engine = FactorEngine(
        mom_windows=[20, 60, 120],
        vol_window=20,
        rsi_window=14,
        min_volume=3e9,
        volume_window=5
    )

    logger.info("Calculating factors...")
    try:
        # Merge all factors into a single DataFrame
        factors = prices.to_frame('close').join(krx_factors, how='left')
        factors = factors.join(fin_factors, how='left')
        for col, val in macro_df.items():
            factors[col] = val
        # Now pass `factors` to the engine
        factors_input = factors
        # Calculate all factors using the merged DataFrame
        factors = factor_engine.calculate_factors(factors_input, volumes)
        
        # Calculate factor scores with the new factor structure
        factor_scores = pd.DataFrame(index=factors['composite'].index)
        
        # Map the new factor names to expected names
        factor_mapping = {
            'momentum': 'momentum',
            'mean_reversion': 'rsi',
            'liquidity': 'volume_ratio',
            'composite': 'composite'
        }
        
        for old_name, new_name in factor_mapping.items():
            if old_name in factors:
                factor_scores[new_name] = factors[old_name]
        
        # Calculate volatility separately
        factor_scores['volatility'] = rets.rolling(20).std().mean() * np.sqrt(252)  # Annualized vol
        
        # 3. Bayesian optimization for factor weights (if enabled)
        if args.tune_params:
            logger.info("Tuning factor weights with Bayesian optimization...")
            try:
                from bayes_tuner import tune_factor_weights
                best_weights = tune_factor_weights(
                    factor_scores,
                    rets,
                    n_calls=args.n_tuning_calls
                )
                factor_weights = best_weights
                logger.info(f"Optimized factor weights: {factor_weights}")
            except Exception as e:
                logger.error(f"Factor weight tuning failed: {e}")
                # Fallback to default weights
                factor_weights = {
                    'momentum': 0.4,
                    'volatility': -0.3,
                    'rsi': 0.2,
                    'volume_ratio': 0.1
                }
        else:
            # Use default weights
            factor_weights = {
                'momentum': 0.4,
                'volatility': -0.3,
                'rsi': 0.2,
                'volume_ratio': 0.1
            }
        
        # Calculate composite score
        factor_scores['composite'] = 0
        for factor, weight in factor_weights.items():
            if factor in factor_scores:
                z_scores = factor_scores[factor].sub(factor_scores[factor].mean()).div(
                    factor_scores[factor].std()
                )
                factor_scores['composite'] += z_scores * weight
        
        latest_scores = factor_scores['composite'].iloc[-1]
        
        # 4. PCR-based return prediction
        logger.info("Generating PCR-based return predictions...")
        # Predict expected returns using PCA-based PCR
        expected_returns = factor_engine.pca_predict(
            factors,
            new_rets.mean(axis=0),  # average historical returns per ticker
            cfg.pca_components
        )
        
    except Exception as e:
        logger.error(f"Error in factor calculation: {str(e)}")
        logger.warning("Falling back to simple mean returns")
        expected_returns = rets.mean()
    
    
    # 5. Calculate risk metrics
    logger.info("Calculating risk metrics...")
    try:
        from risk_monitor import calculate_risk_metrics
        risk_metrics = calculate_risk_metrics(rets, expected_returns)
        logger.info(f"Portfolio VaR (95%): {risk_metrics['var_95']:.2%}")
        logger.info(f"Portfolio CVaR (95%): {risk_metrics['cvar_95']:.2%}")
    except Exception as e:
        logger.warning(f"Could not calculate risk metrics: {e}")

    # Run optimization with QP-based mean-variance + L2 penalty
    logger.info("Running portfolio optimization (QP)...")
    mu = expected_returns.values
    Sigma = rets.cov().values * 252  # Annualized covariance
    lambda_ = cfg.risk_aversion
    eta = cfg.l2_penalty
    w_opt = optimize_portfolio(mu, Sigma, lambda_, eta)
    weights = pd.Series(w_opt, index=rets.columns).nlargest(args.positions)
    # Calculate and log diversification metrics (effective N, diversification ratio)
    cov_matrix = rets[weights.index].cov().values * 252  # Annualized covariance
    div_metrics = calculate_diversification_metrics(weights.values, cov_matrix, rets[weights.index])
    logger.info(f"Effective N: {div_metrics['effective_n']:.2f}, Diversification Ratio: {div_metrics['diversification_ratio']:.2f}")
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare portfolio DataFrame with additional metrics
    portfolio = pd.DataFrame({
        'ticker': weights.index,
        'weight': weights.values,
        'return_1m': rets.mean() * 21,  # Monthly returns
        'volatility': rets.std() * np.sqrt(252),  # Annualized vol
        'sharpe_ratio': (rets.mean() / (rets.std() + 1e-6)) * np.sqrt(252)  # Annualized Sharpe ratio
    })
    
    # Add sector information if available
    try:
        univ = pd.read_csv(cfg.stock_universe_file, dtype=str)
        logger.info(f"Available columns in universe file: {univ.columns.tolist()}")
        
        # Map ticker and sector columns based on actual file content
        ticker_col = next((col for col in univ.columns if '종목' in col or 'ticker' in col.lower() or 'code' in col.lower()), None)
        sector_col = next((col for col in univ.columns if '섹터' in col or 'sector' in col.lower()), None)
        
        if ticker_col and sector_col:
            logger.info(f"Using ticker column: '{ticker_col}', sector column: '{sector_col}'")
            univ['ticker'] = univ[ticker_col].astype(str).str.strip().str.lstrip('A').str.zfill(6)
            # Take first character of sector code if it's a code
            if univ[sector_col].str.match(r'^[A-Za-z]\d*$').all():
                univ['sector'] = univ[sector_col].str[0]
            else:
                univ['sector'] = univ[sector_col]
            
            sector_map = dict(zip(univ['ticker'], univ['sector']))
            portfolio['sector'] = portfolio['ticker'].map(sector_map)
            logger.info(f"Successfully loaded sector information for {portfolio['sector'].count()}/{len(portfolio)} assets")
        else:
            logger.warning(f"Could not find required columns in universe file. Found: {univ.columns.tolist()}")
            portfolio['sector'] = 'UNK'  # Shorter code for unknown sectors
    except Exception as e:
        logger.warning(f"Could not load sector information: {e}")
        portfolio['sector'] = 'UNK'  # Shorter code for unknown sectors
    
    # Remove zero weights and renormalize
    portfolio = portfolio[portfolio['weight'] > 1e-6]
    portfolio['weight'] = portfolio['weight'] / portfolio['weight'].sum()
    
    # Sort by weight descending and prepare final columns
    portfolio = portfolio.sort_values('weight', ascending=False)
    
    # Prepare final columns in requested order and format
    portfolio['weight_pct'] = portfolio['weight'] * 100
    final_columns = ['ticker', 'sector', 'weight_pct', 'sharpe_ratio', 'volatility', 'return_1m']
    portfolio = portfolio[final_columns]
    
    # Save to file
    output_file = output_dir / 'current_portfolio.csv'
    portfolio.to_csv(output_file, index=False, float_format='%.6f')
    logger.info(f"Saved portfolio to {output_file}")
    
    # Generate portfolio statistics (convert weight_pct back to decimal for calculations)
    portfolio_return = (portfolio['weight_pct']/100 * portfolio['return_1m']).sum()
    portfolio_vol = (portfolio['weight_pct']/100 * portfolio['volatility']).sum()
    sharpe_ratio = portfolio_return / portfolio_vol if portfolio_vol > 0 else 0
    
    # Print portfolio summary
    print("\n=== OPTIMIZED PORTFOLIO ===")
    print(f"Number of positions: {len(portfolio)}")
    print(f"Expected monthly return: {portfolio_return:.2%}")
    print(f"Annualized volatility: {portfolio_vol:.2%}")
    print(f"Sharpe ratio: {sharpe_ratio:.2f}")
    print(f"Min weight: {portfolio['weight_pct'].min():.2f}%")
    print(f"Max weight: {portfolio['weight_pct'].max():.2f}%")
    
    # Print top positions by weight
    print("\nTop 10 positions:")
    top_positions = portfolio.nlargest(10, 'weight_pct').copy()
    
    # Format the output
    display_cols = ['ticker', 'sector', 'weight_pct', 'sharpe_ratio', 'volatility', 'return_1m']
    
    # Create formatters
    formatters = {
        'weight_pct': '{:.2f}%'.format,
        'return_1m': '{:.2%}'.format,
        'volatility': '{:.2f}'.format,
        'sharpe_ratio': '{:.2f}'.format
    }
    
    # Apply formatting
    formatted = top_positions[display_cols].copy()
    for col, formatter in formatters.items():
        if col in formatted.columns:
            formatted[col] = formatted[col].apply(formatter)
    
    print(formatted.to_string(index=False, float_format='{:.2f}'.format))
    
    # Generate and save risk report
    try:
        from risk_monitor import generate_risk_report
        risk_report = generate_risk_report(
            rets[portfolio['ticker']], 
            portfolio.set_index('ticker')['weight']
        )
        risk_file = output_dir / 'risk_report.txt'
        with open(risk_file, 'w') as f:
            f.write(risk_report)
        logger.info(f"Saved risk report to {risk_file}")
    except Exception as e:
        logger.warning(f"Could not generate risk report: {e}")
    
    # Generate and save network visualization
    try:
        import riskfolio as rp
        import matplotlib.pyplot as plt
        
        # Prepare returns for network plot
        rets_net = rets[portfolio['ticker']].copy()
        w_net = portfolio.set_index('ticker')['weight']
        
        # Create network plot
        plt.figure(figsize=(12, 10))
        ax = rp.plot_network(
            returns=rets_net,
            codependence='pearson',
            kmeans=True,
            d=0.7,
            leaf_order=True
        )
        
        # Save the plot
        network_file = output_dir / 'portfolio_network.png'
        plt.savefig(network_file, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved network visualization to {network_file}")
        
    except Exception as e:
        logger.warning(f"Could not generate network visualization: {e}")
    
    logger.info("Portfolio optimization completed successfully")

if __name__ == "__main__":
    main()
