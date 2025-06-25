# main.py
import sys
import traceback
from pathlib import Path

# Create logs directory immediately to ensure crash log can be written
log_dir = Path('logs')
log_dir.mkdir(exist_ok=True)

try:
    import logging
    import pandas as pd
    import numpy as np
    import argparse
    from datetime import datetime

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

    def train_factor_model(cfg: Config, dm: DataManager, factor_engine: FactorEngine):
        """Prepares historical data and trains the factor model."""
        logger.info("--- Starting Factor Model Training ---")
        train_start = pd.to_datetime(cfg.training_settings['start_date'])
        train_end = pd.to_datetime(cfg.training_settings['end_date'])
        freq = cfg.training_settings.get('rebalance_frequency', 'M')
        fwd_return_period = cfg.training_settings.get('forward_return_period', 21) # days

        prices = dm.prices.loc[train_start:train_end]
        if prices.empty:
            logger.error("No price data for training period. Aborting training.")
            return

        fwd_returns = prices.pct_change(fwd_return_period).shift(-fwd_return_period)
        fwd_returns.replace([np.inf, -np.inf], np.nan, inplace=True)
        fwd_returns_stacked = fwd_returns.stack().rename('fwd_ret')
        fwd_returns_stacked.index.names = ['date', 'ticker']

        all_factors_list = []
        training_dates = pd.date_range(start=train_start, end=train_end, freq=freq)

        for date in training_dates:
            logger.info(f"Calculating factors for training date {date.strftime('%Y-%m-%d')}")
            prices_hist = dm.prices.loc[:date]
            volumes_hist = dm.volumes.loc[:date]
            market_caps_hist = dm.market_caps.loc[:date]
            funda_hist = get_fundamentals_at_date(dm.historical_fundamentals, date)

            factors = factor_engine.calculate_factors_for_date(
                prices=prices_hist, volumes=volumes_hist, market_caps=market_caps_hist,
                fundamentals=funda_hist, historical_fundamentals=dm.historical_fundamentals,
                market_prices=dm.market_prices.loc[:date] if hasattr(dm, 'market_prices') and dm.market_prices is not None else None
            )
            factors['date'] = date
            all_factors_list.append(factors.reset_index().rename(columns={'index': 'ticker'}))

        if not all_factors_list:
            logger.error("Could not generate any factor data for training. Aborting.")
            return

        logger.info("Preparing historical factor data for model training...")
        factor_panel_df = pd.concat(all_factors_list, ignore_index=True)
        factor_panel_df.drop_duplicates(subset=['date', 'ticker'], keep='last', inplace=True)
        factor_panel_df['date'] = pd.to_datetime(factor_panel_df['date'])
        factor_panel = factor_panel_df.set_index(['date', 'ticker'])

        if not factor_panel.index.is_unique:
            logger.warning("Duplicate (date, ticker) entries found in factor data. Dropping duplicates.")
            factor_panel = factor_panel[~factor_panel.index.duplicated(keep='last')]

        logger.info("Merging factors with forward returns...")
        training_data = factor_panel.join(fwd_returns_stacked, how='inner')
        training_data.dropna(inplace=True)

        if training_data.empty:
            logger.error("No overlapping data between factors and forward returns. Aborting training.")
            return

        logger.info(f"Training data prepared. Shape: {training_data.shape}")
        factor_engine.train_model(training_data)
        logger.info("--- Factor Model Training Complete ---")

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
            prices=prices_hist, volumes=volumes_hist, market_caps=market_caps_hist,
            fundamentals=funda_latest, historical_fundamentals=dm.historical_fundamentals,
            market_prices=dm.market_prices.loc[:exec_date] if hasattr(dm, 'market_prices') and dm.market_prices is not None else None
        )
        
        if final_factors.empty:
            logger.error("Factor calculation for execution date resulted in empty data. Cannot optimize.")
            return pd.Series()

        logger.info("Predicting expected returns...")
        final_factors['expected_return'] = factor_engine.predict_returns(final_factors)
        
        logger.info("Running portfolio optimization...")
        sector_limits = parse_sector_limits_for_date(cfg.optimization_settings.get('sector_limits_path'), exec_date)
        final_weights = optimizer.optimize(final_factors, dm.sector_map, sector_limits)

        if final_weights.empty:
            logger.error("Optimization failed to produce weights.")
        else:
            logger.info("--- Portfolio Construction Complete ---")
        
        return final_weights

    def main():
        """Main function to run the pipeline with detailed step-by-step logging."""
        # This initial logger will catch issues even before the config file is loaded.
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [BOOTSTRAP] - %(message)s')
        logger = logging.getLogger(__name__)
        logger.info("--- TimeFolio Pipeline Start ---")

        logger.info("STEP 1: Parsing command-line arguments...")
        parser = argparse.ArgumentParser(description="Run the TimeFolio portfolio optimization pipeline.")
        parser.add_argument('-c', '--config', default='config/config.yaml', help='Path to the config file.')
        parser.add_argument('-o', '--output-dir', default='output', help='Directory to save output files.')
        parser.add_argument('--backtest', action='store_true', help='Run in backtesting mode.')
        args = parser.parse_args()
        logger.info(f"Arguments parsed: {args}")

        logger.info("STEP 2: Loading configuration file...")
        try:
            cfg = Config(args.config)
            logger.info("Configuration loaded successfully.")
        except Exception as e:
            logger.critical(f"Fatal error loading config from '{args.config}'. Cannot continue.", exc_info=True)
            return

        logger.info("STEP 3: Setting up final logging configuration...")
        # Reconfigure logging with the level from the config file
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
        logger = logging.getLogger("TimeFolioPipeline") # Get logger with the new config
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
        train_factor_model(cfg, dm, factor_engine)
        logger.info("Factor model training complete.")

        logger.info("STEP 8: Initializing PortfolioOptimizer...")
        optimizer = PortfolioOptimizer(factor_engine, settings=cfg.optimization_settings)
        logger.info("PortfolioOptimizer initialized.")

        logger.info("STEP 9: Running final portfolio construction pipeline...")
        final_weights = run_pipeline(cfg, dm, factor_engine, optimizer)

        if final_weights is None or final_weights.empty:
            logger.warning("Optimization did not produce a portfolio. No risk analysis will be run.")
        else:
            logger.info("STEP 10: Running risk analysis...")
            # In the original code, RiskMonitor was initialized with returns and sector map.
            # This seems incorrect as it should probably use the full returns_df from DataManager.
            # Correcting this to pass the necessary data.
            risk_monitor = RiskMonitor(settings=cfg.risk_settings)
            risk_report = risk_monitor.run_analysis(weights=final_weights, returns_df=dm.returns, prev_weights=None)

            logger.info("STEP 11: Saving results...")
            output_dir = Path(args.output_dir)
            output_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            weights_file = output_dir / f"final_weights_{timestamp}.csv"
            report_file = output_dir / f"risk_report_{timestamp}.txt"
            
            final_weights.to_csv(weights_file)
            # The report is a dict, so we'll save it in a more readable format.
            with open(report_file, 'w') as f:
                for key, value in risk_report.items():
                    f.write(f"{key}: {value}\n")
            
            logger.info(f"Final weights saved to {weights_file}")
            logger.info(f"Risk report saved to {report_file}")
        
        logger.info("--- TimeFolio Pipeline Finished Successfully ---")

    if __name__ == '__main__':
        main()

except BaseException as e:
    # This block will catch any exception during initialization or execution
    crash_log_file = log_dir / 'crash.log'
    with open(crash_log_file, 'w') as f:
        f.write("A fatal error occurred in main.py:\n")
        f.write(str(e) + "\n\n")
        f.write("--- Traceback ---\n")
        f.write(traceback.format_exc())
    
    print(f"FATAL ERROR: A crash occurred. See {crash_log_file} for details.", file=sys.stderr)
    sys.exit(1)