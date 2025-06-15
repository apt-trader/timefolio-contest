# main.py
import logging
import pandas as pd
from pathlib import Path
import argparse
from datetime import datetime

from config import Config
from data_manager import DataManager
from factor_engine import FactorEngine
from optimizer import PortfolioOptimizer
from risk_monitor import RiskMonitor
from utils.sector_parser import parse_sector_limits

# Main logger for the pipeline
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("TimeFolioPipeline")

def run_pipeline(config_path: str,
                 output_dir: str,
                 start_date: str = None,
                 end_date: str = None) -> pd.Series:
    """
    Executes the full portfolio construction pipeline. Can be run for a single period
    or called by a backtester with dynamic dates.

    Args:
        config_path (str): Path to the configuration YAML file.
        output_dir (str): Directory to save outputs for this run.
        start_date (str, optional): Override for the data lookback start date.
        end_date (str, optional): Override for the data lookback end date (the 'as of' date).

    Returns:
        pd.Series: A Series of the final portfolio weights, indexed by ticker.
                   Returns an empty Series on failure.
    """
    try:
        cfg = Config(config_path)
        # --- Override dates if provided by backtester ---
        if start_date: cfg.start_date = start_date
        if end_date: cfg.end_date = end_date
        
        logger.info(f"--- Running Pipeline for period: {cfg.start_date} to {cfg.end_date} ---")

        dm = DataManager(cfg)
        factor_engine = FactorEngine(cfg.factor_settings)
        portfolio_optimizer = PortfolioOptimizer(cfg.optimization_settings)
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        prices, volumes, rets, market_index, market_caps = dm.get_market_data()
        sector_map = dm.get_sector_mappings()
        sector_limits = parse_sector_limits(cfg.market_sectors_file, as_of_date=datetime.strptime(cfg.end_date, '%Y-%m-%d'))
        
        fundamental_data = dm.get_fundamental_data()
        macro_data = dm.get_macro_data()

        if rets.empty:
            logger.error("No returns data available. Cannot proceed.")
            return pd.Series(dtype=float)

        factors = factor_engine.calculate_all_factors(prices, volumes, market_index, fundamental_data, market_caps, macro_data)
        expected_returns = factor_engine.predict_returns(factors)

        if expected_returns.empty:
            logger.error("Return prediction failed. Halting optimization.")
            return pd.Series(dtype=float)

        weights = portfolio_optimizer.optimize(expected_returns, rets, market_caps, sector_map, sector_limits)

        if weights.empty or weights.sum() < 0.99:
            logger.error("Optimization failed to produce a valid portfolio.")
            return pd.Series(dtype=float)

        final_weights = weights[weights > 1e-6]
        return final_weights

    except Exception:
        logger.exception("A critical error occurred in the pipeline run.")
        return pd.Series(dtype=float)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the TimeFolio portfolio optimization pipeline for a single period.")
    parser.add_argument('-c', '--config', default='config/config.yaml', help='Path to the config file.')
    parser.add_argument('-o', '--output-dir', default='output', help='Directory to save output files.')
    args = parser.parse_args()
    
    # When run directly, it executes for the period defined in the config file
    final_weights = run_pipeline(config_path=args.config, output_dir=args.output_dir)
    
    if not final_weights.empty:
        # Create and save the final report for the single run
        # This part could be expanded with RiskMonitor as before
        final_portfolio_df = final_weights.to_frame(name='weight')
        final_portfolio_df.index.name = 'ticker'
        
        output_file = Path(args.output_dir) / 'final_portfolio.csv'
        final_portfolio_df.to_csv(output_file)
        logger.info(f"Successfully saved final portfolio to {output_file}")
        
        print("\n" + "="*50)
        print("      FINAL OPTIMIZED PORTFOLIO")
        print("="*50)
        print(final_portfolio_df.to_string(float_format='{:.4%}'.format))
        print("="*50 + "\n")
    else:
        logger.error("Pipeline run failed to produce a portfolio.")