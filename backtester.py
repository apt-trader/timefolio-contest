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
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV, LinearRegression

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
    def __init__(self, cfg: Config, start_date: str, end_date: str, rebalance_freq: str = 'W-MON', dm: Optional[DataManager] = None):
        self.cfg = cfg
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.rebalance_freq = rebalance_freq
        self.pipeline = None

        if dm:
            self.dm = dm
            logger.info("Backtester initialized with a pre-loaded DataManager.")
        else:
            logger.info("No pre-loaded DataManager provided. Initializing a new one.")
            self.dm = self._load_data_for_backtest()
        
        if not self.dm:
            raise RuntimeError("Failed to initialize or receive a valid DataManager.")

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
                    self.cfg.data_settings['market_sectors_file'],
                    target_date=rebal_date
                )

                # 5. Optimize the portfolio to get target weights.
                weights = optimizer.optimize(
                    expected_returns,
                    self.dm.returns.loc[:rebal_date],
                    self.dm.market_caps.loc[[rebal_date]].iloc[0],
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
        # Set the date range for the DataManager to load all necessary historical data
        dm_config.data_settings['start_date'] = (self.start_date - pd.DateOffset(years=2)).strftime('%Y-%m-%d')
        dm_config.data_settings['end_date'] = self.end_date.strftime('%Y-%m-%d')
        
        # Initialize DataManager in backtest mode with the adjusted config
        dm = DataManager(dm_config, backtest_mode=True)
        
        # Load all the data required for the backtest period
        logger.info("Backtester is attempting to call DataManager.load_data()...")
        if not dm.load_data():
            logger.critical("Backtest failed: Initial data load failed.")
            return None
            
        return dm
        
    def _generate_historical_factor_panel(self, factor_engine: FactorEngine, dm: DataManager) -> pd.DataFrame:
        """Uses the factor engine to compute factors for the entire historical period."""
        logger.info("Generating historical point-in-time factors for model training...")
        factor_list = []
        
        if dm.prices.empty or dm.volumes.empty or dm.market_caps.empty or not dm.historical_fundamentals:
            logger.error("Missing required data for factor calculation")
            return pd.DataFrame()

        start_date = pd.to_datetime(self.cfg.start_date)
        end_date = pd.to_datetime(self.cfg.end_date)
        relevant_dates = dm.prices.loc[start_date:end_date].index
            
        for date in tqdm(relevant_dates, desc="Calculating Historical Factors"):
            try:
                # 1. Get market caps for the current date, skipping if not available.
                try:
                    market_caps_for_date = dm.market_caps.loc[date]
                    if not isinstance(market_caps_for_date, pd.Series):
                        continue
                except KeyError:
                    continue
                
                if market_caps_for_date.empty:
                    continue

                # 2. Get point-in-time fundamentals for all tickers.
                pit_fundamentals_dict = {}
                for ticker, hist_df in dm.historical_fundamentals.items():
                    if ticker not in dm.prices.columns:
                        continue
                    
                    if not isinstance(hist_df.index, pd.DatetimeIndex):
                        try:
                            hist_df.index = pd.to_datetime(hist_df.index)
                        except Exception:
                            logger.warning(f"Could not convert index to DatetimeIndex for {ticker}. Skipping.")
                            continue

                    mask = hist_df.index.date <= date.date()
                    if not mask.any():
                        continue
                    
                    try:
                        latest = hist_df.loc[mask].iloc[-1]
                        pit_fundamentals_dict[ticker] = latest
                    except Exception as e:
                        logger.debug(f"Error processing fundamentals for {ticker} on {date}: {e}")


                
                pit_fundamentals_df = pd.DataFrame.from_dict(pit_fundamentals_dict, orient='index')
                
                # 3. Get historical price and volume data up to the current date.
                prices_up_to_date = dm.prices.loc[:date]
                volumes_up_to_date = dm.volumes.loc[:date]

                # 4. Calculate factors for the current date.
                factors = factor_engine.calculate_factors_for_date(
                    date=date,
                    prices=prices_up_to_date,
                    volumes=volumes_up_to_date,
                    market_caps=market_caps_for_date,
                    fundamentals=pit_fundamentals_df,
                    historical_fundamentals=dm.historical_fundamentals
                )
                
                # 5. Append a copy of the factors to the list.
                if not factors.empty:
                    factors['date'] = date
                    factors['ticker'] = factors.index
                    factor_list.append(factors.copy())
                    
            except Exception as e:
                logger.error(f"Error processing date {date}: {e}", exc_info=True)
                continue
        
        if not factor_list:
            logger.error("Failed to generate any historical factors.")
            return pd.DataFrame()

        try:
            logger.info(f"Factor list contains {len(factor_list)} DataFrames before concatenation.")
            if not factor_list:
                logger.error("Factor list is empty. Cannot create panel.")
                return pd.DataFrame()

            # Diagnostic logging
            first_df_date = factor_list[0]['date'].iloc[0]
            last_df_date = factor_list[-1]['date'].iloc[0]
            logger.info(f"Date in first DataFrame: {first_df_date}. Date in last DataFrame: {last_df_date}.")

            factor_panel = pd.concat(factor_list, ignore_index=True)
            factor_panel.set_index(['date', 'ticker'], inplace=True)
            factor_panel.sort_index(inplace=True)
            
            min_date, max_date = factor_panel.index.get_level_values('date').min(), factor_panel.index.get_level_values('date').max()
            logger.info(f"Successfully generated historical factor panel with shape: {factor_panel.shape}. Date range: {min_date} to {max_date}")
            return factor_panel
            
        except Exception as e:
            logger.error(f"Error creating final factor panel: {e}", exc_info=True)
            return pd.DataFrame()

    def _train_model(self, historical_factors: pd.DataFrame, returns: pd.DataFrame):
        """Trains the prediction model on the full historical factor panel."""
        logger.info("Training predictive model...")
        target_days = self.cfg.factor_settings.get('prediction_target_days', 21)
        y = returns.shift(-target_days).rolling(target_days).mean().stack().rename("forward_return")
        
        # Name the index levels of the returns series to match the factors
        y.index.names = ['date', 'ticker']

        # Ensure the 'ticker' level of the index is string type for alignment
        if len(y.index.levels) > 1:
            y.index = y.index.set_levels(y.index.levels[1].astype(str), level='ticker')

        # Align factors and returns
        aligned_X, aligned_y = historical_factors.align(y, join='inner', axis=0)

        if aligned_X.empty or aligned_y.empty:
            err_msg = "Factor and return data alignment resulted in empty DataFrames. Cannot train model."
            logger.error(err_msg)
            raise ValueError(err_msg)

        n_features = aligned_X.shape[1]
        # Get n_components from config, default to n_features if not present
        n_pca_components = self.cfg.factor_settings.get('n_pca_components', n_features)

        # PCA n_components cannot be greater than the number of features
        effective_n_components = min(n_pca_components, n_features)

        if effective_n_components < n_pca_components:
            logger.warning(
                f"Requested PCA components ({n_pca_components}) exceeds number of features ({n_features}). "
                f"Using {effective_n_components} components instead."
            )

        self.pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=effective_n_components)),
            ('regressor', LinearRegression())
        ])
        self.pipeline.fit(aligned_X, aligned_y)
        logger.info("Model training complete.")

    def predict(self, latest_factors: pd.DataFrame) -> pd.Series:
        """Generates predictions using the pre-trained model."""
        if self.pipeline is None:
            raise RuntimeError("Model is not trained.")
        
        predictions = self.pipeline.predict(latest_factors)
        
        # Rank-normalize predictions to get alpha scores from -1 to 1
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