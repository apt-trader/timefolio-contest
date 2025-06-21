# backtester.py
import logging
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from main import run_pipeline
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
        self.lookback_period = pd.DateOffset(years=1) # Standard lookback for covariance
        self.portfolio_returns = pd.Series(dtype=float)

    def run(self) -> dict:
        logger.info(f"Starting backtest from {self.start_date.date()} to {self.end_date.date()} with {self.rebalance_freq} rebalancing.")
        rebalance_dates = pd.date_range(self.start_date, self.end_date, freq=self.rebalance_freq)
        
        all_daily_returns = []
        
        for i in tqdm(range(len(rebalance_dates) - 1), desc="Backtesting Periods"):
            rebal_date = rebalance_dates[i]
            next_rebal_date = rebalance_dates[i+1]
            
            lookback_start = (rebal_date - self.lookback_period).strftime('%Y-%m-%d')
            lookback_end = rebal_date.strftime('%Y-%m-%d')
            
            target_weights = run_pipeline(self.config_path, f"backtest_artefacts/{lookback_end}", lookback_start, lookback_end)
            
            # Fetch returns for the holding period
            from data_manager import DataManager # Local import to avoid circular dependency issues
            dm = DataManager(self.cfg)
            _, _, period_rets, _ = dm.get_market_data(start_date_override=rebal_date.strftime('%Y-%m-%d'), end_date_override=next_rebal_date.strftime('%Y-%m-%d'))
            
            if target_weights.empty or period_rets.empty:
                # If no portfolio, create a Series of zeros for this period
                zero_returns = pd.Series(0, index=pd.date_range(rebal_date, next_rebal_date, closed='left'))
                all_daily_returns.append(zero_returns)
                continue

            valid_tickers = period_rets.columns.intersection(target_weights.index)
            daily_pnl = period_rets[valid_tickers].mul(target_weights.loc[valid_tickers], axis=1).sum(axis=1)
            all_daily_returns.append(daily_pnl)

        self.portfolio_returns = pd.concat(all_daily_returns)
        return self.generate_summary()

    def generate_summary(self) -> dict:
        if self.portfolio_returns.empty:
            logger.error("No returns were generated.")
            return {}

        summary = calculate_portfolio_performance(
            self.portfolio_returns,
            risk_free_rate=self.cfg.optimization_settings.get('risk_free_rate', 0.02)
        )
        
        equity_curve = (1 + self.portfolio_returns).cumprod()
        rolling_max = equity_curve.cummax()
        drawdown = (equity_curve / rolling_max) - 1
        summary['max_drawdown'] = drawdown.min()

        print("\n" + "="*50 + "\n          BACKTEST PERFORMANCE SUMMARY\n" + "="*50)
        print(f"Period: {self.start_date.date()} to {self.end_date.date()}")
        print(f"Annualized Return: {summary['annualized_return']:.2%}")
        print(f"Annualized Volatility: {summary['annualized_volatility']:.2%}")
        print(f"Sharpe Ratio: {summary['sharpe_ratio']:.2f}")
        print(f"Maximum Drawdown: {summary['max_drawdown']:.2%}")
        print("="*50)
        
        # Plotting
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(12, 6)); equity_curve.plot(title='Strategy Equity Curve', grid=True)
            plt.ylabel("Cumulative Growth"); plt.xlabel("Date")
            plot_path = Path('output') / 'backtest_equity_curve.png'; plt.savefig(plot_path)
            plt.show(); logger.info(f"Equity curve plot saved to {plot_path}")
        except ImportError:
            logger.warning("matplotlib not found. Skipping plot generation.")
            
        return summary

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run a backtest of the TimeFolio strategy.")
    parser.add_argument('-c', '--config', default='config/config.yaml', help='Path to the config file.')
    parser.add_argument('-s', '--start', required=True, help='Backtest start date (YYYY-MM-DD).')
    parser.add_argument('-e', '--end', required=True, help='Backtest end date (YYYY-MM-DD).')
    args = parser.parse_args()
    Backtester(config_path=args.config, start_date=args.start, end_date=args.end).run()