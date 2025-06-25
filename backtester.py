# backtester.py
import logging
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path
import sys
from copy import deepcopy
from typing import Optional
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV

sys.path.insert(0, str(Path(__file__).parent))
from config import Config
from data_manager import DataManager
from factor_engine import FactorEngine
from optimizer import PortfolioOptimizer
from utils.portfolio_metrics import calculate_portfolio_performance
from utils.sector_parser import parse_sector_limits_for_date

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("Backtester")

class Backtester:
    def __init__(self, cfg: Config, start_date: str, end_date: str, rebalance_freq: str = 'W-MON'):
        self.cfg = cfg
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.rebalance_freq = rebalance_freq
        self.pipeline = None  # The ML model will live here
        # Initialize DataManager in backtest mode
        self.dm = self._load_data_for_backtest()
        if not self.dm:
            raise RuntimeError("Failed to initialize DataManager for backtest.")

    def run(self) -> dict:
        logger.info(f"Starting backtest from {self.start_date.date()} to {self.end_date.date()}...")

        # Initialize engines
        factor_engine = FactorEngine(self.cfg.factor_settings)
        optimizer = PortfolioOptimizer(factor_engine, self.cfg.optimization_settings)

        # Generate the full panel of historical factors and train the prediction model
        historical_factors = self._generate_historical_factor_panel(factor_engine, self.dm)
        if historical_factors.empty:
            logger.error("Factor panel is empty. Cannot proceed with backtest.")
            return {}
        self._train_model(historical_factors, self.dm.returns)

        # Main backtesting loop
        rebalance_dates = pd.date_range(self.start_date, self.end_date, freq=self.rebalance_freq)
        all_daily_returns = []

        for rebal_date in tqdm(rebalance_dates[:-1], desc="Backtesting Periods"):
            next_rebal_date = rebalance_dates[rebalance_dates.get_loc(rebal_date) + 1]
            holding_period = pd.date_range(rebal_date, next_rebal_date - pd.Timedelta(days=1))

            try:
                # 1. Get the latest available factor data for the current rebalance date.
                available_dates = historical_factors.index.get_level_values('date')
                valid_dates = available_dates[available_dates <= rebal_date]
                if valid_dates.empty:
                    logger.warning(f"No historical factors found on or before {rebal_date.date()}. Skipping.")
                    all_daily_returns.append(pd.Series(0, index=holding_period))
                    continue
                
                latest_factors = historical_factors.loc[valid_dates.max()]

                # 2. Determine the point-in-time tradable universe.
                compliant_tickers, sector_map = self.dm.run_compliance_filters_for_date(rebal_date)
                if not compliant_tickers:
                    logger.warning(f"No compliant tickers found for {rebal_date.date()}. Skipping.")
                    all_daily_returns.append(pd.Series(0, index=holding_period))
                    continue

                # 3. Predict returns for the compliant universe.
                latest_factors = latest_factors.loc[latest_factors.index.isin(compliant_tickers)]
                if latest_factors.empty:
                    logger.warning(f"No factor data for compliant tickers on {rebal_date.date()}. Skipping.")
                    all_daily_returns.append(pd.Series(0, index=holding_period))
                    continue
                expected_returns = self.predict(latest_factors)

                # 4. Get sector risk limits for the period.
                sector_limits = parse_sector_limits_for_date(
                    self.cfg.data_settings['sector_file_path'],
                    target_date=rebal_date
                )

                # 5. Optimize the portfolio to get target weights.
                weights = optimizer.optimize(
                    expected_returns,
                    self.dm.returns.loc[:rebal_date],
                    self.dm.market_caps.loc[rebal_date],
                    sector_map,
                    sector_limits
                )

                # 6. Calculate portfolio returns for the holding period.
                holding_period_rets = self.dm.returns.loc[holding_period.min():holding_period.max()]
                if weights is None or weights.empty:
                    daily_pnl = pd.Series(0, index=holding_period_rets.index)
                else:
                    valid_tickers = holding_period_rets.columns.intersection(weights.index)
                    daily_pnl = holding_period_rets[valid_tickers].mul(weights[valid_tickers], axis=1).sum(axis=1)
                
                all_daily_returns.append(daily_pnl)

            except Exception as e:
                logger.error(f"Error processing period for {rebal_date.date()}: {e}", exc_info=True)
                all_daily_returns.append(pd.Series(0, index=holding_period))

        # Consolidate results and generate summary
        if not all_daily_returns:
            logger.warning("No returns were generated during the backtest.")
            return {}
            
        portfolio_returns = pd.concat(all_daily_returns).sort_index()
        return self.generate_summary(portfolio_returns)

    def _load_data_for_backtest(self) -> Optional[DataManager]:
        """Loads data for the entire backtest period plus a lookback."""
        dm_config = deepcopy(self.cfg)
        dm_config.start_date = (self.start_date - pd.DateOffset(years=2)).strftime('%Y-%m-%d')
        dm_config.end_date = self.end_date.strftime('%Y-%m-%d')
        # Initialize DataManager in backtest mode
        dm = DataManager(dm_config, backtest_mode=True)
        if not dm.load_data():
            logger.critical("Backtest failed: Initial data load failed.")
            return None
        return dm
        
    def _generate_historical_factor_panel(self, factor_engine: FactorEngine, dm: DataManager) -> pd.DataFrame:
        """Uses the factor engine to compute factors for the entire historical period."""
        logger.info("Generating historical point-in-time factors for model training...")
        factor_list = []
        
        # Ensure we have valid data
        if dm.prices.empty or dm.volumes.empty or dm.market_caps.empty or not dm.historical_fundamentals:
            logger.error("Missing required data for factor calculation")
            return pd.DataFrame()
            
        for i in tqdm(range(252, len(dm.prices)), desc="Calculating Historical Factors"):
            try:
                date = dm.prices.index[i]
                
                # Get point-in-time fundamentals
                pit_fundamentals = []
                for ticker, hist_df in dm.historical_fundamentals.items():
                    if ticker not in dm.prices.columns:
                        continue
                    mask = hist_df.index.to_series().dt.date <= date.date()
                    if not mask.any():
                        continue
                    try:
                        latest = hist_df.loc[mask].iloc[-1]
                        latest.name = ticker  # Use ticker as the index
                        pit_fundamentals.append(latest)
                    except Exception as e:
                        logger.debug(f"Error processing fundamentals for {ticker} on {date}: {e}")
                
                if not pit_fundamentals:
                    continue
                    
                # Create DataFrame with tickers as index
                pit_fundamentals_df = pd.DataFrame(pit_fundamentals)
                
                # Calculate factors for this date
                factors = factor_engine.calculate_factors_for_date(
                    prices=dm.prices.iloc[:i+1],
                    volumes=dm.volumes.iloc[:i+1],
                    market_caps=dm.market_caps.loc[date],
                    fundamentals=pit_fundamentals_df,
                    historical_fundamentals=dm.historical_fundamentals
                )
                
                if not factors.empty:
                    # Ensure we have a proper MultiIndex with date and ticker
                    factors = factors.copy()
                    factors['date'] = date
                    factors = factors.reset_index()
                    if 'index' in factors.columns:
                        factors = factors.rename(columns={'index': 'ticker'})
                    factor_list.append(factors)
                    
            except Exception as e:
                logger.error(f"Error processing date {date}: {e}", exc_info=True)
                continue
        
        if not factor_list:
            logger.error("Failed to generate any historical factors.")
            return pd.DataFrame()
            
        try:
            # Concatenate all factor DataFrames
            result = pd.concat(factor_list, ignore_index=False)
            
            # Ensure we have the required columns for the MultiIndex
            if 'ticker' not in result.columns or 'date' not in result.columns:
                logger.error("Missing required columns 'ticker' or 'date' in factor data")
                return pd.DataFrame()
                
            # Set MultiIndex and sort
            result = result.set_index(['date', 'ticker']).sort_index()
            return result
            
        except Exception as e:
            logger.error(f"Error concatenating factor data: {e}", exc_info=True)
            return pd.DataFrame()

    def _train_model(self, historical_factors: pd.DataFrame, returns: pd.DataFrame):
        """Trains the prediction model on the full historical factor panel."""
        logger.info("Training predictive model...")
        target_days = self.cfg.factor_settings.get('prediction_target_days', 21)
        y = returns.shift(-target_days).rolling(target_days).mean().stack().rename("forward_return")
        
        # Name the index levels of the returns series to match the factors
        y.index.names = ['date', 'ticker']

        # Debugging: Log index names and check for uniqueness
        logger.info(f"Aligning factors with index names: {historical_factors.index.names}")
        logger.info(f"Aligning returns with index names: {y.index.names}")
        logger.info(f"Factors index is unique: {historical_factors.index.is_unique}")
        logger.info(f"Returns index is unique: {y.index.is_unique}")

        if not historical_factors.index.is_unique:
            duplicates = historical_factors.index[historical_factors.index.duplicated()].unique()
            logger.warning(f"Found {len(duplicates)} duplicate indices in historical_factors. Example: {duplicates[:5].tolist()}")

        # Align factors and returns
        X, y = historical_factors.align(y, join='inner', axis=0)
        if X.empty: raise ValueError("No aligned data for training.")
        
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=self.cfg.factor_settings.get('n_pca_components', 5))),
            ('model', RidgeCV(alphas=np.logspace(-4, 4, 10)))
        ])
        self.pipeline.fit(X, y)
        logger.info("Model training complete.")

    def predict(self, latest_factors: pd.DataFrame) -> pd.Series:
        """Generates predictions using the pre-trained model."""
        if self.pipeline is None: raise RuntimeError("Model is not trained.")
        predictions = self.pipeline.predict(latest_factors)
        return pd.Series(predictions, index=latest_factors.index).rank(pct=True).sub(0.5).mul(2)

    def generate_summary(self, portfolio_returns: pd.Series) -> dict:
        """
        Calculates and logs a summary of the backtest performance.
        """
        if portfolio_returns.empty:
            logger.warning("Portfolio returns are empty. Cannot generate summary.")
            return {}

        # Clean returns data to prevent calculation errors with NaN or Inf
        portfolio_returns = portfolio_returns.replace([np.inf, -np.inf], np.nan).fillna(0)

        # --- Performance Metrics Calculation ---
        risk_free_rate = self.cfg.optimization_settings.get('risk_free_rate', 0.02)
        annualized_return = portfolio_returns.mean() * 252
        annualized_volatility = portfolio_returns.std() * np.sqrt(252)

        if annualized_volatility is None or annualized_volatility < 1e-8:
            sharpe_ratio = 0.0
        else:
            sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility

        # --- Drawdown Calculation ---
        equity_curve = (1 + portfolio_returns).cumprod()
        peak = equity_curve.cummax()
        drawdown = (equity_curve - peak) / peak
        max_drawdown = drawdown.min()

        summary = {
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }

        logger.info(f"""\
--- Backtest Summary --- 

Sharpe Ratio: {summary['sharpe_ratio']:.2f}
Max Drawdown: {summary['max_drawdown']:.2%}
Annualized Return: {summary['annualized_return']:.2%}
Annualized Volatility: {summary['annualized_volatility']:.2%}
""")
        return summary

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run a backtest of the TimeFolio strategy.")
    parser.add_argument('-c', '--config', default='config/config.yaml', help='Path to the config file.')
    parser.add_argument('-s', '--start', required=True, help='Backtest start date (YYYY-MM-DD).')
    parser.add_argument('-e', '--end', required=True, help='Backtest end date (YYYY-MM-DD).')
    args = parser.parse_args()
    Backtester(cfg=Config(args.config), start_date=args.start, end_date=args.end).run()