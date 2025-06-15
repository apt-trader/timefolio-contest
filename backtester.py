# backtester.py
import logging
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path

# Import the refactored, callable pipeline
from main import run_pipeline
from data_manager import DataManager
from config import Config
from utils.portfolio_metrics import calculate_portfolio_performance

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("Backtester")

class Backtester:
    def __init__(self, config_path: str, start_date: str, end_date: str, rebalance_freq: str = 'W-MON'):
        self.config_path = config_path
        self.cfg = Config(config_path)
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.rebalance_freq = rebalance_freq
        self.lookback_period = pd.DateOffset(years=1)
        
        self.strategy_returns = []
        self.dates = []

    def run(self):
        logger.info(f"Starting backtest from {self.start_date.date()} to {self.end_date.date()} "
                    f"with {self.rebalance_freq} rebalancing.")

        # 1. Fetch all market data for the entire backtest period once
        logger.info("Pre-fetching all market data for the backtest period...")
        dm = DataManager(self.cfg)
        # We need a longer history to accommodate the first lookback period
        full_history_start = (self.start_date - self.lookback_period).strftime('%Y-%m-%d')
        full_history_end = self.end_date.strftime('%Y-%m-%d')
        _, _, all_returns, _, _ = dm.get_market_data(
            start_date_override=full_history_start,
            end_date_override=full_history_end
        )

        # 2. Define rebalancing dates
        rebalance_dates = pd.date_range(self.start_date, self.end_date, freq=self.rebalance_freq)

        # 3. Main backtesting loop
        for i in tqdm(range(len(rebalance_dates) - 1), desc="Backtesting Periods"):
            rebal_date = rebalance_dates[i]
            next_rebal_date = rebalance_dates[i+1]
            
            # Define the lookback window for this iteration
            lookback_start = (rebal_date - self.lookback_period).strftime('%Y-%m-%d')
            lookback_end = rebal_date.strftime('%Y-%m-%d')
            
            logger.debug(f"\nProcessing period starting {rebal_date.date()}")
            
            # 4. Run the full pipeline to get target weights for this period
            output_dir = f"backtest_artefacts/{rebal_date.strftime('%Y%m%d')}"
            target_weights = run_pipeline(self.config_path, output_dir, lookback_start, lookback_end)
            
            if target_weights.empty:
                logger.warning(f"No portfolio generated for {rebal_date.date()}. Assuming 0% return for the period.")
                period_return = 0.0
            else:
                # 5. Calculate performance for the holding period (from rebal_date to next_rebal_date)
                holding_period_returns = all_returns.loc[rebal_date:next_rebal_date]
                
                # Align weights with available return data
                valid_tickers = holding_period_returns.columns.intersection(target_weights.index)
                aligned_weights = target_weights.loc[valid_tickers]
                
                # Calculate daily portfolio returns for the period
                daily_pnl = holding_period_returns[valid_tickers].mul(aligned_weights, axis=1).sum(axis=1)
                
                # Calculate the compounded return over the holding period
                period_return = (1 + daily_pnl).prod() - 1

            self.strategy_returns.append(period_return)
            self.dates.append(next_rebal_date)

        logger.info("Backtest loop completed.")
        self.generate_summary()

    def generate_summary(self):
        if not self.strategy_returns:
            logger.error("No returns were generated during the backtest.")
            return

        returns_series = pd.Series(self.strategy_returns, index=self.dates, name="Strategy")
        
        # --- Performance Metrics Calculation ---
        equity_curve = (1 + returns_series).cumprod()
        
        total_return = equity_curve.iloc[-1] - 1
        num_years = (equity_curve.index[-1] - equity_curve.index[0]).days / 365.25
        annualized_return = (1 + total_return) ** (1 / num_years) - 1
        
        # Calculate annualized volatility from periodic returns
        annualized_volatility = returns_series.std() * np.sqrt(52) # Assuming weekly rebalancing

        risk_free_rate = self.cfg.optimization_settings.get('risk_free_rate', 0.02)
        sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility if annualized_volatility > 0 else 0
        
        # Calculate Maximum Drawdown
        rolling_max = equity_curve.cummax()
        drawdown = (equity_curve / rolling_max) - 1
        max_drawdown = drawdown.min()

        # --- Print Summary ---
        print("\n" + "="*50)
        print("          BACKTEST PERFORMANCE SUMMARY")
        print("="*50)
        print(f"Period: {self.start_date.date()} to {self.end_date.date()}")
        print(f"Total Return: {total_return: .2%}")
        print(f"Annualized Return: {annualized_return: .2%}")
        print(f"Annualized Volatility: {annualized_volatility: .2%}")
        print(f"Sharpe Ratio: {sharpe_ratio: .2f}")
        print(f"Maximum Drawdown: {max_drawdown: .2%}")
        print("="*50)

        # Optional: Plotting
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(12, 6))
            equity_curve.plot(title='Strategy Equity Curve', grid=True)
            plt.ylabel("Cumulative Growth")
            plt.xlabel("Date")
            plot_path = Path('output') / 'backtest_equity_curve.png'
            plt.savefig(plot_path)
            plt.show()
            logger.info(f"Equity curve plot saved to {plot_path}")
        except ImportError:
            logger.warning("matplotlib not found. Skipping plot generation.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run a backtest of the TimeFolio strategy.")
    parser.add_argument('-c', '--config', default='config/config.yaml', help='Path to the config file.')
    parser.add_argument('-s', '--start', required=True, help='Backtest start date (YYYY-MM-DD).')
    parser.add_argument('-e', '--end', required=True, help='Backtest end date (YYYY-MM-DD).')
    parser.add_argument('-f', '--freq', default='W-MON', help='Rebalancing frequency (e.g., "W-MON", "M").')
    args = parser.parse_args()

    backtester = Backtester(
        config_path=args.config,
        start_date=args.start,
        end_date=args.end,
        rebalance_freq=args.freq
    )
    backtester.run()