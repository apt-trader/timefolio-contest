#!/usr/bin/env python3
"""
Backtesting script to compare portfolio performance with and without Elite Alpha signals.
"""
import os
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from competition_portfolio import main as run_optimization
from elite_alpha_stack import EliteAlphaSignalStack
from portfolio_optimizer import Config, DataManager, CVaROptimizer
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class Backtester:
    def __init__(self, config_path='config.yaml', start_date='2020-01-01', end_date=None):
        """Initialize the backtester."""
        self.config = Config(config_path)
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date) if end_date else datetime.now()
        self.results = []
        
    def run_backtest(self, use_elite_alpha=False):
        """Run backtest for a single configuration."""
        logger.info(f"Running backtest with {'Elite Alpha' if use_elite_alpha else 'Baseline'}")
        
        # Set up data manager
        dm = DataManager(self.config)
        
        # Get historical price data
        prices = dm.get_returns(self.start_date.strftime('%Y-%m-%d'), 
                              self.end_date.strftime('%Y-%m-%d'))
        
        # Convert returns to price levels for signal calculation
        price_levels = (1 + prices).cumprod()
        price_levels = price_levels.fillna(method='ffill')  # Forward fill any NaN values
        
        # Initialize portfolio value and weights
        initial_capital = 1000000  # Start with 1M KRW
        portfolio_value = initial_capital
        current_weights = None
        
        # Rebalance monthly
        dates = pd.date_range(start=self.start_date, end=self.end_date, freq='M')
        
        for i, date in enumerate(dates[:-1]):  # Exclude last date as we need next period returns
            try:
                # Get data up to current date
                train_data = prices[prices.index <= date]
                
                if len(train_data) < 60:  # Need at least 3 months of data
                    continue
                
                # Run optimization with current settings
                if use_elite_alpha:
                    try:
                        # Get price data for the training period
                        train_prices = price_levels.loc[train_data.index]
                        
                        # Ensure we have enough data
                        if len(train_prices) < 20:  # Minimum 20 days of data needed
                            logger.warning(f"Not enough price data for {date}, using baseline")
                            adjusted_rets = train_data
                        else:
                            # Initialize Elite Alpha Signal Stack with price data
                            alpha_signal = EliteAlphaSignalStack(
                                price_df=train_prices,
                                market_returns=train_data.mean(axis=1),  # Simple market proxy
                                previous_weights=current_weights  # For retention bias
                            )
                            
                            # Get alpha scores and adjust returns
                            alpha_scores = alpha_signal.compute_signals()
                            # Ensure we have valid numeric scores
                            if isinstance(alpha_scores, pd.Series):
                                # Normalize scores to be between -0.1 and 0.1 (10% adjustment max)
                                alpha_scores = (alpha_scores - alpha_scores.mean()) / (alpha_scores.max() - alpha_scores.min()) * 0.2 - 0.1
                                adjusted_rets = train_data * (1 + alpha_scores)
                            else:
                                logger.warning("Invalid alpha scores format, using baseline returns")
                                adjusted_rets = train_data
                    except Exception as e:
                        logger.error(f"Error in Elite Alpha signal generation: {str(e)}")
                        adjusted_rets = train_data
                else:
                    adjusted_rets = train_data
                # Use the adjusted returns from the Elite Alpha processing
                # or baseline returns if Elite Alpha is not used
                
                # Run optimization (simplified - in practice, you'd use your full optimization)
                # This is a placeholder - you'd need to adapt this to your actual optimization
                weights = self.optimize_portfolio(adjusted_rets)
                
                # Calculate period returns
                next_period = dates[i+1]
                period_rets = prices[(prices.index > date) & (prices.index <= next_period)]
                
                if len(period_rets) == 0:
                    continue
                    
                # Calculate portfolio return for the period
                if current_weights is not None:
                    # Calculate turnover cost (assume 10bps per turnover)
                    turnover = np.abs(weights - current_weights).sum() / 2
                    turnover_cost = turnover * 0.001  # 10bps cost
                else:
                    turnover_cost = 0
                
                # Update portfolio value
                period_portfolio_return = (weights * period_rets).sum(axis=1).mean()
                portfolio_value *= (1 + period_portfolio_return - turnover_cost)
                current_weights = weights
                
                # Log results
                self.results.append({
                    'date': next_period,
                    'strategy': 'Elite Alpha' if use_elite_alpha else 'Baseline',
                    'portfolio_value': portfolio_value,
                    'return': period_portfolio_return,
                    'turnover': turnover if 'turnover' in locals() else 0,
                    'turnover_cost': turnover_cost
                })
                
                logger.info(f"{next_period.strftime('%Y-%m')}: "
                           f"Return: {period_portfolio_return:.2%}, "
                           f"Value: {portfolio_value:,.0f} KRW")
                
            except Exception as e:
                logger.error(f"Error processing {date}: {str(e)}")
                continue
    
    def optimize_portfolio(self, returns):
        """Simplified portfolio optimization."""
        # This is a placeholder - replace with your actual optimization logic
        # For demonstration, we'll use simple inverse volatility weighting
        vol = returns.std()
        weights = (1 / vol) / (1 / vol).sum()
        return weights
    
    def analyze_results(self):
        """Analyze and plot backtest results."""
        if not self.results:
            logger.warning("No results to analyze")
            return
        
        df = pd.DataFrame(self.results)
        
        # Calculate performance metrics
        metrics = {}
        for strategy in df['strategy'].unique():
            strat_df = df[df['strategy'] == strategy]
            returns = strat_df['return']
            metrics[strategy] = {
                'Total Return': strat_df['portfolio_value'].iloc[-1] / 1000000 - 1,
                'Annualized Return': (1 + returns).prod() ** (12/len(returns)) - 1,
                'Annualized Vol': returns.std() * np.sqrt(12),
                'Max Drawdown': self.calculate_max_drawdown(returns),
                'Sharpe Ratio': returns.mean() / returns.std() * np.sqrt(12),
                'Avg Turnover': strat_df['turnover'].mean(),
                'Turnover Cost': strat_df['turnover_cost'].sum()
            }
        
        # Print metrics
        metrics_df = pd.DataFrame(metrics).T
        print("\nPerformance Metrics:")
        print(metrics_df)
        
        # Plot portfolio values
        plt.figure(figsize=(12, 6))
        for strategy in df['strategy'].unique():
            strat_df = df[df['strategy'] == strategy]
            plt.plot(strat_df['date'], strat_df['portfolio_value'], label=strategy)
        
        plt.title('Portfolio Value Over Time')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value (KRW)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # Save plot
        os.makedirs('output/backtest', exist_ok=True)
        plt.savefig('output/backtest/portfolio_comparison.png')
        plt.close()
        
        # Save results to CSV
        metrics_df.to_csv('output/backtest/performance_metrics.csv')
        df.to_csv('output/backtest/detailed_results.csv', index=False)
        
        return metrics_df
    
    @staticmethod
    def calculate_max_drawdown(returns):
        """Calculate maximum drawdown."""
        cum_returns = (1 + returns).cumprod()
        rolling_max = cum_returns.cummax()
        drawdowns = (cum_returns - rolling_max) / rolling_max
        return drawdowns.min()

def main():
    parser = argparse.ArgumentParser(description='Backtest Elite Alpha vs Baseline')
    parser.add_argument('--start-date', default='2020-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='End date (YYYY-MM-DD)')
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    args = parser.parse_args()
    
    # Run backtests
    backtester = Backtester(
        config_path=args.config,
        start_date=args.start_date,
        end_date=args.end_date
    )
    
    # Run both strategies
    backtester.run_backtest(use_elite_alpha=False)  # Baseline
    backtester.run_backtest(use_elite_alpha=True)   # With Elite Alpha
    
    # Analyze and plot results
    metrics = backtester.analyze_results()
    
    # Print final recommendation
    print("\nRecommendation:")
    if metrics.loc['Elite Alpha', 'Sharpe Ratio'] > metrics.loc['Baseline', 'Sharpe Ratio']:
        print("Elite Alpha shows better risk-adjusted returns. Consider integrating it.")
    else:
        print("Baseline performs better. Elite Alpha may not provide additional value.")

if __name__ == "__main__":
    main()
