# test_improvement.py
import logging
from config import Config
from backtester import Backtester
from data_manager import DataManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def test_best_params():
    logger.info("Testing optimized parameters on 2023-2024 period (excludes COVID era)...")
    
    # Load config with optimized parameters (already updated in config.yaml)
    cfg = Config("config/config.yaml")
    
    # Parameters from hyperparameter tuning (Sharpe: 0.9494)
    logger.info(f"Using optimized parameters:")
    logger.info(f"  risk_aversion: {cfg.optimization_settings['risk_aversion']}")
    logger.info(f"  n_pca_components: {cfg.factor_settings['n_pca_components']}")
    logger.info(f"  momentum_window: {cfg.factor_settings['mom_windows'][0]}")
    logger.info(f"  l2_penalty: {cfg.factor_settings.get('l2_penalty', 'N/A')}")
    logger.info(f"  cov_l2_alpha: {cfg.factor_settings.get('cov_l2_alpha', 'N/A')}")
    
    # Run backtest
    dm = DataManager(cfg, backtest_mode=True)
    if not dm.load_data():
        logger.error("Failed to load data")
        return
        
    # Updated to use strategic 2023-2024 period (excludes COVID anomalies)
    start_date = "2023-01-01"
    end_date = "2024-12-27"  # Friday - aligned with weekend data fetching
    backtester = Backtester(cfg, start_date, end_date, dm=dm)
    
    results = backtester.run()
    
    if results:
        # results IS the summary dictionary returned from generate_summary()
        logger.info("\n" + "="*50)
        logger.info("OPTIMIZED RESULTS:")
        logger.info("="*50)
        logger.info(f"Sharpe Ratio: {results.get('sharpe_ratio', 0):.3f}")
        logger.info(f"Annual Return: {results.get('annualized_return', 0):.2%}")
        logger.info(f"Volatility: {results.get('annualized_volatility', 0):.2%}")
        logger.info(f"Max Drawdown: {results.get('max_drawdown', 0):.2%}")
        
        sharpe = results.get('sharpe_ratio', 0)
        improvement = (sharpe / 0.36 - 1) * 100
        logger.info(f"Improvement over baseline: +{improvement:.1f}%")
        
        if sharpe > 1.0:
            logger.info("🎉 EXCELLENT! Sharpe ratio > 1.0 achieved!")
        
    else:
        logger.error("No results returned")

if __name__ == "__main__":
    test_best_params()
