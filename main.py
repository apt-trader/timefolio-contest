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
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from data_manager import DataManager
from factor_engine import FactorEngine
from optimizer import PortfolioOptimizer
from risk_monitor import RiskMonitor
from utils.sector_parser import parse_sector_limits_for_date

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

def get_fundamentals_at_date(historical_fundamentals: dict, as_of_date: pd.Timestamp) -> pd.DataFrame:
    """Extracts the latest available fundamentals for each ticker as of a given date."""
    latest_funda = {}
    for ticker, funda_df in historical_fundamentals.items():
        funda_for_ticker = funda_df[funda_df.index <= as_of_date]
        if not funda_for_ticker.empty:
            latest_funda[ticker] = funda_for_ticker.iloc[-1]
    return pd.DataFrame.from_dict(latest_funda, orient='index')

def prepare_training_data(cfg: Config, dm: DataManager, factor_engine: FactorEngine) -> Optional[pd.DataFrame]:
    """Prepares historical data for model training."""
    logger.info("--- Preparing Historical Data for Model Training ---")
    train_start = pd.to_datetime(cfg.training_settings['start_date'])
    train_end = pd.to_datetime(cfg.training_settings['end_date'])
    freq = cfg.training_settings.get('rebalance_frequency', 'M')
    fwd_return_period = cfg.training_settings.get('forward_return_period', 21)

    prices = dm.prices.loc[train_start:train_end]
    if prices.empty:
        logger.error("No price data for training period. Aborting.")
        return None

    fwd_returns = prices.pct_change(fwd_return_period).shift(-fwd_return_period)
    fwd_returns.replace([np.inf, -np.inf], np.nan, inplace=True)
    fwd_returns_stacked = fwd_returns.stack().rename('forward_return')
    fwd_returns_stacked.index.names = ['date', 'ticker']

    all_factors_list = []
    training_dates = pd.date_range(start=train_start, end=train_end, freq=freq)

    for date in tqdm(training_dates, desc="Calculating historical factors"):
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
            factors['date'] = date
            factors.index.name = 'ticker'
            all_factors_list.append(factors.reset_index())

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
    Trains the factor model using historical data.
    """
    logger.info("--- Starting Factor Model Training ---")
    training_data = prepare_training_data(cfg, dm, factor_engine)

    if training_data is None or training_data.empty:
        logger.error("Failed to prepare training data. Aborting model training.")
        return

    training_data.dropna(inplace=True)

    if training_data.shape[0] < 100:
        logger.error(f"Not enough valid data points ({training_data.shape[0]}) to train model. Aborting.")
        return

    y = training_data['forward_return']
    X = training_data.drop(columns=['forward_return'])

    if X.shape[1] == 0:
        logger.error("No factor columns available for training after processing. Aborting.")
        return

    X = sm.add_constant(X)

    try:
        model = sm.OLS(y, X).fit()
        logger.info("Factor model training complete.")
        logger.info(model.summary())
        return model
    except Exception as e:
        logger.error(f"An error occurred during OLS model training: {e}", exc_info=True)
        return None

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

    logger.info("STEP 6: Initializing FactorEngine...")
    factor_engine = FactorEngine(settings=cfg.factor_settings)
    logger.info("FactorEngine initialized.")

    logger.info("STEP 7: Training factor model...")
    model = train_factor_model(cfg, dm, factor_engine)
    logger.info("Factor model training complete.")

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
        portfolio_df = pd.DataFrame(final_weights, columns=['percentage']).reset_index()
        portfolio_df.columns = ['code', 'percentage']
        portfolio_df['sector'] = portfolio_df['code'].map(dm.sector_map).fillna('Unknown')
        portfolio_df['percentage'] = portfolio_df['percentage'] * 100 # Convert to percentage
        portfolio_df = portfolio_df[['code', 'sector', 'percentage']]

        # Save results
        weights_file = output_dir / "final_portfolio.csv"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
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