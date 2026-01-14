"""
SIGNAL-BASED MAIN ENTRY POINT
=============================
New main entry point using signal-based MVO architecture.

This replaces the stock-level MVO approach with:
1. Signal construction (5 orthogonal signals)
2. Signal-level MVO optimization
3. Signal-to-stock mapping with cardinality constraint

Usage:
    python main_signal.py --mode pipeline
    python main_signal.py --mode backtest --start 2023-01-01 --end 2024-12-31
    python main_signal.py --mode live

Author: TimeFolio System Refactor
Date: 2025-01-11
"""

import argparse
import logging
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from data_manager import DataManager
from signals.signal_pipeline import SignalPipeline, create_pipeline_from_config
from compliance_filters import load_forbidden_tickers

# Setup logging
def setup_logging(log_level: str = 'INFO') -> logging.Logger:
    """Configure logging for the application."""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f'logs/signal_pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )
    return logging.getLogger(__name__)


def prepare_fundamentals_for_date(
    dm: DataManager,
    date: pd.Timestamp
) -> pd.DataFrame:
    """
    Prepare fundamentals DataFrame for a specific date.
    Uses point-in-time data to avoid look-ahead bias.
    
    Args:
        dm: DataManager instance
        date: Date for which to prepare fundamentals
        
    Returns:
        DataFrame indexed by ticker with fundamental metrics
    """
    fundamentals_list = []
    
    for ticker, hist_df in dm.historical_fundamentals.items():
        if hist_df.empty:
            continue
        
        # Get most recent data before or on date (point-in-time)
        available = hist_df[hist_df.index <= date]
        
        if available.empty:
            continue
        
        latest = available.iloc[-1].copy()
        latest['ticker'] = ticker
        fundamentals_list.append(latest)
    
    if not fundamentals_list:
        return pd.DataFrame()
    
    fundamentals = pd.DataFrame(fundamentals_list).set_index('ticker')
    
    # Calculate derived metrics if not present
    if 'roe' not in fundamentals.columns and 'net_income' in fundamentals.columns and 'total_equity' in fundamentals.columns:
        fundamentals['roe'] = fundamentals['net_income'] / fundamentals['total_equity'].replace(0, np.nan)
    
    if 'debt_to_equity' not in fundamentals.columns and 'total_liabilities' in fundamentals.columns and 'total_equity' in fundamentals.columns:
        fundamentals['debt_to_equity'] = fundamentals['total_liabilities'] / fundamentals['total_equity'].replace(0, np.nan)
    
    if 'gross_profit' not in fundamentals.columns and 'revenue' in fundamentals.columns and 'cost_of_sales' in fundamentals.columns:
        fundamentals['gross_profit'] = fundamentals['revenue'] - fundamentals['cost_of_sales']
    
    return fundamentals


def generate_rebalance_dates(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    frequency: str = 'W-FRI'
) -> List[pd.Timestamp]:
    """
    Generate rebalance dates.
    
    Args:
        start_date: Start of period
        end_date: End of period
        frequency: Pandas frequency string (W-FRI for weekly Friday)
        
    Returns:
        List of rebalance dates
    """
    dates = pd.date_range(start=start_date, end=end_date, freq=frequency)
    return dates.tolist()


def run_backtest(
    cfg: Config,
    dm: DataManager,
    pipeline: SignalPipeline,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    warmup_periods: int = 52
) -> Dict[str, Any]:
    """
    Run backtest of the signal-based strategy.
    
    Args:
        cfg: Configuration
        dm: DataManager with loaded data
        pipeline: SignalPipeline instance
        start_date: Backtest start date
        end_date: Backtest end date
        warmup_periods: Number of periods for warmup (signal history)
        
    Returns:
        Backtest results dictionary
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Starting backtest: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # Generate all rebalance dates
    all_dates = generate_rebalance_dates(
        start_date=dm.start_date,
        end_date=end_date,
        frequency='W-FRI'
    )
    
    # Split into warmup and backtest periods
    backtest_start_idx = None
    for i, date in enumerate(all_dates):
        if date >= start_date:
            backtest_start_idx = i
            break
    
    if backtest_start_idx is None:
        logger.error("No valid backtest dates found")
        return {}
    
    warmup_dates = all_dates[max(0, backtest_start_idx - warmup_periods):backtest_start_idx]
    backtest_dates = all_dates[backtest_start_idx:]
    backtest_dates = [d for d in backtest_dates if d <= end_date]
    
    logger.info(f"Warmup periods: {len(warmup_dates)}")
    logger.info(f"Backtest periods: {len(backtest_dates)}")
    
    # Warmup phase
    logger.info("=" * 60)
    logger.info("WARMUP PHASE")
    logger.info("=" * 60)
    
    for date in warmup_dates:
        fundamentals = prepare_fundamentals_for_date(dm, date)
        
        if fundamentals.empty:
            logger.warning(f"No fundamentals for warmup date {date}")
            continue
        
        pipeline.run(
            prices=dm.prices.loc[:date],
            returns=dm.returns.loc[:date],
            market_caps=dm.market_caps.loc[:date],
            fundamentals=fundamentals,
            sector_map=dm.sector_map,
            rebalance_date=date,
            warmup_mode=True
        )
    
    # Backtest phase
    logger.info("=" * 60)
    logger.info("BACKTEST PHASE")
    logger.info("=" * 60)
    
    results = {
        'dates': [],
        'weights': [],
        'signal_weights': [],
        'returns': [],
        'cumulative_returns': [],
        'metrics': []
    }
    
    portfolio_value = 1.0
    previous_weights = None
    
    # Load forbidden tickers once before backtest loop
    forbidden_tickers = load_forbidden_tickers('forbidden.csv')
    if forbidden_tickers:
        logger.info(f"Loaded {len(forbidden_tickers)} forbidden tickers for backtest")
    
    for i, date in enumerate(backtest_dates):
        logger.info(f"\n{'='*60}")
        logger.info(f"Rebalance {i+1}/{len(backtest_dates)}: {date.strftime('%Y-%m-%d')}")
        logger.info(f"{'='*60}")
        
        # Prepare fundamentals
        fundamentals = prepare_fundamentals_for_date(dm, date)
        
        if fundamentals.empty:
            logger.warning(f"No fundamentals for {date}, skipping")
            continue
        
        # Run pipeline with forbidden tickers excluded BEFORE optimization
        result = pipeline.run(
            prices=dm.prices.loc[:date],
            returns=dm.returns.loc[:date],
            market_caps=dm.market_caps.loc[:date],
            fundamentals=fundamentals,
            sector_map=dm.sector_map,
            rebalance_date=date,
            warmup_mode=False,
            forbidden_tickers=forbidden_tickers,
            trading_values=dm.trading_values.loc[:date]
        )
        
        if result['weights'].empty:
            logger.warning(f"No weights generated for {date}")
            continue
        
        # Calculate return since last rebalance
        if previous_weights is not None and i > 0:
            prev_date = backtest_dates[i - 1]
            
            # Get returns between dates
            period_returns = dm.returns.loc[
                (dm.returns.index > prev_date) & (dm.returns.index <= date)
            ]
            
            if not period_returns.empty:
                # Calculate portfolio return
                aligned_weights = previous_weights.reindex(period_returns.columns).fillna(0)
                
                # Compound daily returns
                daily_port_returns = (period_returns * aligned_weights).sum(axis=1)
                period_return = (1 + daily_port_returns).prod() - 1
                
                portfolio_value *= (1 + period_return)
                
                results['returns'].append(period_return)
                results['cumulative_returns'].append(portfolio_value - 1)
                
                logger.info(f"Period return: {period_return:.2%}")
                logger.info(f"Cumulative return: {portfolio_value - 1:.2%}")
        
        # Store results
        results['dates'].append(date)
        results['weights'].append(result['weights'])
        results['signal_weights'].append(result['signal_weights'])
        results['metrics'].append(result['metrics'])
        
        previous_weights = result['weights'].copy()
    
    # Calculate performance metrics
    if results['returns']:
        returns_series = pd.Series(results['returns'])
        
        results['total_return'] = portfolio_value - 1
        results['annualized_return'] = (portfolio_value ** (52 / len(results['returns']))) - 1
        results['volatility'] = returns_series.std() * np.sqrt(52)
        results['sharpe_ratio'] = results['annualized_return'] / results['volatility'] if results['volatility'] > 0 else 0
        results['max_drawdown'] = calculate_max_drawdown(results['cumulative_returns'])
        results['win_rate'] = (returns_series > 0).mean()
        
        logger.info("\n" + "=" * 60)
        logger.info("BACKTEST SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total Return: {results['total_return']:.2%}")
        logger.info(f"Annualized Return: {results['annualized_return']:.2%}")
        logger.info(f"Volatility: {results['volatility']:.2%}")
        logger.info(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        logger.info(f"Max Drawdown: {results['max_drawdown']:.2%}")
        logger.info(f"Win Rate: {results['win_rate']:.2%}")
        
        # IC Summary from backtest
        ic_summary = pipeline.ic_monitor.get_ic_summary()
        if not ic_summary.empty:
            logger.info("\n" + "=" * 60)
            logger.info("SIGNAL IC SUMMARY (Backtest)")
            logger.info("=" * 60)
            for signal_name, row in ic_summary.iterrows():
                status = "OK" if row['mean_ic'] >= 0.02 else "⚠️ LOW"
                logger.info(f"  {signal_name}: Mean IC={row['mean_ic']:+.4f}, "
                           f"Std={row['std_ic']:.4f}, "
                           f"N={int(row['n_observations'])} [{status}]")
            
            # Store IC summary in results
            results['ic_summary'] = ic_summary.to_dict()
    
    return results


def calculate_max_drawdown(cumulative_returns: List[float]) -> float:
    """Calculate maximum drawdown from cumulative returns."""
    if not cumulative_returns:
        return 0.0
    
    values = [1 + r for r in cumulative_returns]
    peak = values[0]
    max_dd = 0.0
    
    for value in values:
        if value > peak:
            peak = value
        dd = (peak - value) / peak
        max_dd = max(max_dd, dd)
    
    return max_dd


def run_live(
    cfg: Config,
    dm: DataManager,
    pipeline: SignalPipeline
) -> Dict[str, Any]:
    """
    Run live portfolio generation.
    
    Args:
        cfg: Configuration
        dm: DataManager with loaded data
        pipeline: SignalPipeline instance
        
    Returns:
        Live portfolio weights and metrics
    """
    logger = logging.getLogger(__name__)
    
    # Use today as rebalance date
    today = pd.Timestamp.now().normalize()
    
    # Find most recent trading day
    available_dates = dm.prices.index
    valid_dates = available_dates[available_dates <= today]
    
    if valid_dates.empty:
        logger.error("No valid dates for live run")
        return {}
    
    rebalance_date = valid_dates[-1]
    logger.info(f"Live rebalance date: {rebalance_date.strftime('%Y-%m-%d')}")
    
    # Warmup with historical data
    warmup_dates = generate_rebalance_dates(
        start_date=rebalance_date - timedelta(days=400),
        end_date=rebalance_date - timedelta(days=7),
        frequency='W-FRI'
    )
    
    logger.info(f"Warming up with {len(warmup_dates)} historical periods")
    
    for date in warmup_dates:
        fundamentals = prepare_fundamentals_for_date(dm, date)
        
        if fundamentals.empty:
            continue
        
        pipeline.run(
            prices=dm.prices.loc[:date],
            returns=dm.returns.loc[:date],
            market_caps=dm.market_caps.loc[:date],
            fundamentals=fundamentals,
            sector_map=dm.sector_map,
            rebalance_date=date,
            warmup_mode=True
        )
    
    # Load forbidden tickers BEFORE running pipeline
    forbidden_tickers = load_forbidden_tickers('forbidden.csv')
    if forbidden_tickers:
        logger.info(f"Loaded {len(forbidden_tickers)} forbidden tickers")
    
    # Run live
    fundamentals = prepare_fundamentals_for_date(dm, rebalance_date)
    
    result = pipeline.run(
        prices=dm.prices,
        returns=dm.returns,
        market_caps=dm.market_caps,
        fundamentals=fundamentals,
        sector_map=dm.sector_map,
        rebalance_date=rebalance_date,
        warmup_mode=False,
        forbidden_tickers=forbidden_tickers,
        trading_values=dm.trading_values
    )
    
    # Output portfolio
    if not result['weights'].empty:
        logger.info("\n" + "=" * 60)
        logger.info("LIVE PORTFOLIO")
        logger.info("=" * 60)
        
        for ticker, weight in result['weights'].sort_values(ascending=False).items():
            sector = dm.sector_map.get(ticker, 'Unknown')
            logger.info(f"  {ticker}: {weight:.2%} (sector: {sector})")
        
        # Log IC summary if available
        if result.get('ic_values'):
            logger.info("\n" + "=" * 60)
            logger.info("SIGNAL INFORMATION COEFFICIENTS")
            logger.info("=" * 60)
            for signal_name, ic in sorted(result['ic_values'].items(), key=lambda x: -x[1] if not np.isnan(x[1]) else -999):
                status = "OK" if ic >= 0.02 else "⚠️ LOW"
                logger.info(f"  {signal_name}: IC={ic:+.4f} [{status}]")
            
            if result.get('ic_alerts'):
                logger.info("\n⚠️ IC ALERTS:")
                for signal_name, alert_msg in result['ic_alerts'].items():
                    logger.info(f"  {signal_name}: {alert_msg}")
        
        # Save to file
        output_path = Path('output')
        output_path.mkdir(exist_ok=True)
        
        portfolio_df = pd.DataFrame({
            'ticker': result['weights'].index,
            'weight': result['weights'].values,
            'sector': [dm.sector_map.get(t, 'Unknown') for t in result['weights'].index]
        })
        
        output_file = output_path / f"portfolio_{rebalance_date.strftime('%Y%m%d')}.csv"
        portfolio_df.to_csv(output_file, index=False)
        logger.info(f"Portfolio saved to {output_file}")
    
    return result


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Signal-Based Portfolio System')
    parser.add_argument('--mode', type=str, default='backtest',
                       choices=['backtest', 'live', 'diagnostics'],
                       help='Run mode')
    parser.add_argument('--start', type=str, default='2023-01-01',
                       help='Backtest start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2024-12-31',
                       help='Backtest end date (YYYY-MM-DD)')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to config file')
    parser.add_argument('--log-level', type=str, default='INFO',
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    Path('logs').mkdir(exist_ok=True)
    logger = setup_logging(args.log_level)
    
    logger.info("=" * 60)
    logger.info("SIGNAL-BASED PORTFOLIO SYSTEM")
    logger.info("=" * 60)
    logger.info(f"Mode: {args.mode}")
    
    try:
        # Load configuration
        logger.info("Loading configuration...")
        cfg = Config(args.config)
        
        # Initialize data manager
        logger.info("Initializing data manager...")
        dm = DataManager(cfg, backtest_mode=(args.mode == 'backtest'))
        
        # Load data
        logger.info("Loading data...")
        if not dm.load_data():
            logger.error("Failed to load data")
            sys.exit(1)
        
        logger.info(f"Data loaded: {len(dm.tickers)} tickers, "
                   f"{len(dm.prices)} days, "
                   f"{len(dm.historical_fundamentals)} with fundamentals")
        
        # Create pipeline
        logger.info("Creating signal pipeline...")
        pipeline = create_pipeline_from_config(cfg)
        
        # Run based on mode
        if args.mode == 'backtest':
            start_date = pd.Timestamp(args.start)
            end_date = pd.Timestamp(args.end)
            
            results = run_backtest(
                cfg=cfg,
                dm=dm,
                pipeline=pipeline,
                start_date=start_date,
                end_date=end_date
            )
            
        elif args.mode == 'live':
            results = run_live(
                cfg=cfg,
                dm=dm,
                pipeline=pipeline
            )
            
        elif args.mode == 'diagnostics':
            # Run diagnostics on signal performance
            logger.info("Running signal diagnostics...")
            
            # Warmup to build signal history
            warmup_dates = generate_rebalance_dates(
                start_date=pd.Timestamp(args.start),
                end_date=pd.Timestamp(args.end),
                frequency='W-FRI'
            )
            
            for date in warmup_dates:
                fundamentals = prepare_fundamentals_for_date(dm, date)
                if fundamentals.empty:
                    continue
                
                pipeline.run(
                    prices=dm.prices.loc[:date],
                    returns=dm.returns.loc[:date],
                    market_caps=dm.market_caps.loc[:date],
                    fundamentals=fundamentals,
                    sector_map=dm.sector_map,
                    rebalance_date=date,
                    warmup_mode=True
                )
            
            diagnostics = pipeline.get_signal_diagnostics()
            
            logger.info("\nSignal Diagnostics:")
            for key, value in diagnostics.items():
                logger.info(f"  {key}: {value:.4f}")
        
        logger.info("\nPipeline execution complete.")
        
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
