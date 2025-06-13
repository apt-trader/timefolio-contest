#!/usr/bin/env python3
import pandas as pd
import numpy as np
import logging
import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('optimization_test.log')
    ]
)
logger = logging.getLogger(__name__)

# Import after setting up path
from portfolio_optimizer import Config, DataManager, CVaROptimizer

def run_minimal_optimization():
    try:
        # Initialize config and data manager
        config_path = 'config.yaml'
        cfg = Config(config_path)
        dm = DataManager(cfg)
        
        # Use a small subset of tickers for testing
        test_tickers = ['005930', '000660', '035420', '006400']
        
        # Get returns for test tickers
        logger.info("Fetching returns data...")
        rets = dm.get_returns('20230101', '20231231')
        
        # Filter for our test tickers
        common_tickers = list(set(test_tickers) & set(rets.columns))
        if not common_tickers:
            logger.error("No common tickers found in returns data")
            return
            
        rets = rets[common_tickers]
        logger.info(f"Using tickers: {', '.join(common_tickers)}")
        
        # Calculate expected returns (simple mean for testing)
        expected_returns = rets.mean()
        
        # Create beta vector (mock data for testing)
        beta_vec = pd.Series(1.0, index=common_tickers)
        
        # Initialize optimizer
        optimizer = CVaROptimizer(cfg, dm, beta_vec)
        
        # Run optimization
        logger.info("\n" + "="*80)
        logger.info("RUNNING OPTIMIZATION")
        logger.info("="*80)
        
        # Run optimization with actual weights
        weights = optimizer.optimise(
            rets=rets,
            expected=expected_returns
        )
        
        logger.info("Optimization completed successfully!")
        
        # Print results
        logger.info("\n" + "="*80)
        logger.info("OPTIMIZATION RESULTS")
        logger.info("="*80)
        
        # Calculate portfolio statistics
        cov_matrix = rets.cov() * 252  # Annualized covariance
        portfolio_return = (weights * expected_returns).sum()  # Already in expected returns space
        portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
        sharpe = portfolio_return / portfolio_vol if portfolio_vol > 0 else 0
        
        logger.info(f"\nPortfolio Statistics:")
        logger.info(f"- Expected Return (annualized): {portfolio_return:.2%}")
        logger.info(f"- Volatility (annualized): {portfolio_vol:.2%}")
        logger.info(f"- Sharpe Ratio: {sharpe:.2f}")
        
        # Log the weights
        logger.info("\nOptimal Weights:")
        for ticker, weight in weights.items():
            logger.info(f"{ticker}: {weight:.2%}")
        
    except Exception as e:
        logger.error(f"Error in optimization: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    run_minimal_optimization()
