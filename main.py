# main.py
import logging, pandas as pd, argparse, os
from pathlib import Path
from datetime import datetime
from config import Config
from data_manager import DataManager
from factor_engine import FactorEngine
from optimizer import PortfolioOptimizer
from utils.sector_parser import parse_sector_limits
import sys
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("TimeFolioPipeline")

def run_pipeline(config_path: str, output_dir: str, start_date: str = None, end_date: str = None) -> pd.Series:
    try:
        cfg = Config(config_path)
        if start_date: cfg.start_date = start_date
        if end_date: cfg.end_date = end_date
        logger.info(f"--- Running Pipeline for period: {cfg.start_date} to {cfg.end_date} ---")

        dm = DataManager(cfg); factor_engine = FactorEngine(cfg.factor_settings); portfolio_optimizer = PortfolioOptimizer(cfg.optimization_settings)
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        prices, volumes, rets, market_caps_df = dm.get_market_data()
        latest_fundamental_data = dm.get_fundamental_data()
        historical_fundamental_data = dm.get_historical_fundamental_data()

        if rets.empty: logger.error("No market data available."); return pd.Series(dtype=float)

        factors = factor_engine.calculate_all_factors(prices, volumes, market_caps_df, latest_fundamental_data, historical_fundamental_data)
        expected_returns = factor_engine.predict_returns(factors, rets)

        if expected_returns.empty: logger.error("Return prediction failed."); return pd.Series(dtype=float)

        sector_map = dm.get_sector_mappings()
        sector_limits = parse_sector_limits(cfg.market_sectors_file, as_of_date=datetime.strptime(cfg.end_date, '%Y-%m-%d'))
        
        weights = portfolio_optimizer.optimize(expected_returns, rets, market_caps_df.iloc[-1], sector_map, sector_limits)

        if weights is None or weights.empty: logger.error("Optimization failed."); return pd.Series(dtype=float)
        return weights[weights > 1e-6]
    except Exception:
        logger.exception("A critical error occurred in the pipeline run.")
        return pd.Series(dtype=float)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the TimeFolio portfolio optimization pipeline.")
    parser.add_argument('-c', '--config', default='config/config.yaml', help='Path to the config file.')
    parser.add_argument('-o', '--output-dir', default='output', help='Directory to save output files.')
    args = parser.parse_args()
    
    if not (final_weights := run_pipeline(config_path=args.config, output_dir=args.output_dir)).empty:
        final_portfolio_df = final_weights.to_frame(name='weight')
        output_file = Path(args.output_dir) / 'final_portfolio.csv'
        final_portfolio_df.to_csv(output_file, float_format='%.6f')
        print(f"\nSuccessfully saved final portfolio to {output_file}\n")
    else:
        logger.error("Pipeline run failed to produce a portfolio.")