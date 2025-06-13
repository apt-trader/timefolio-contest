"""
Integration module for EliteAlphaSignalStack with the main CVaR portfolio optimization.
"""

import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime

# Set up logging
log_dir = "../logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "enhanced_workflow.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Try to import the EliteAlphaSignalStack
try:
    from .elite_alpha_stack import EliteAlphaSignalStack
    elite_alpha_available = True
    logger.info("EliteAlphaSignalStack is available for integration")
except ImportError:
    elite_alpha_available = False
    logger.warning("EliteAlphaSignalStack is not available. Install the module to enable enhanced alpha signals.")

def enhance_portfolio_signals(optimizer, cluster_df):
    """
    Enhance portfolio signals by integrating EliteAlphaSignalStack
    with existing combined_ratio
    
    Args:
        optimizer: The CVaRPortfolioOptimizer instance
        cluster_df: DataFrame with clustering results including combined_ratio
        
    Returns:
        DataFrame: Enhanced cluster_df with updated combined_ratio values
    """
    logger.info("Enhancing portfolio signals with EliteAlphaSignalStack...")
    
    if not elite_alpha_available:
        logger.warning("EliteAlphaSignalStack not available. Returning original signals.")
        return cluster_df
    
    try:
        # 1. Prepare data for alpha engine
        price_data = optimizer.returns_df.add(1).cumprod()  # Convert returns to price-like data
        
        # Get market returns (try multiple methods)
        market_returns = None
        
        # Define a list of possible sources for market data (Korean indices only)
        market_data_sources = [
            # KRX main index
            {"source": "KS11", "name": "KOSPI"},
            # Alternative Korean indices
            {"source": "KQ11", "name": "KOSDAQ"},
            {"source": "KS200", "name": "KOSPI 200"}
        ]
        
        logger.info("Fetching recent market data...")
        
        # Method 1: Try using FinanceDataReader with multiple sources
        for market_source in market_data_sources:
            if market_returns is not None and len(market_returns) >= 10:
                break
                
            try:
                import FinanceDataReader as fdr
                market_start = price_data.index[0].strftime('%Y-%m-%d')
                market_end = price_data.index[-1].strftime('%Y-%m-%d')
                
                logger.info(f"Trying to fetch {market_source['name']} data from {market_start} to {market_end}...")
                market_data = fdr.DataReader(market_source['source'], market_start, market_end)
                
                if not market_data.empty and "Close" in market_data.columns:
                    market_returns = market_data["Close"].pct_change().dropna()
                    # Align with our returns data
                    market_returns = market_returns.reindex(price_data.index).dropna()
                    if len(market_returns) >= 10:
                        logger.info(f"Successfully loaded {market_source['name']} returns from FinanceDataReader")
                        
                        # Save this data for future use
                        try:
                            data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                                "enhanced_workflow_integration/data")
                            os.makedirs(data_dir, exist_ok=True)
                            market_returns_path = os.path.join(data_dir, "market_returns.csv")
                            market_returns.to_frame(market_source['source']).to_csv(market_returns_path)
                            logger.info(f"Saved {market_source['name']} returns to cache for future use")
                        except Exception as save_e:
                            logger.warning(f"Could not save market returns: {save_e}")
                            
                        break
            except Exception as e:
                logger.warning(f"Could not fetch {market_source['name']} returns: {str(e)[:200]}...")
        
        # Method 2: Try using cached data if available (with most recent first)
        if market_returns is None or len(market_returns) < 10:
            cache_locations = [
                os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                             "enhanced_workflow_integration/data/market_returns.csv"),
                os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                             "data/market_returns.csv"),
                os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                             "market_returns.csv")
            ]
            
            for cached_returns_path in cache_locations:
                if market_returns is not None and len(market_returns) >= 10:
                    break
                    
                try:
                    if os.path.exists(cached_returns_path):
                        market_cached = pd.read_csv(cached_returns_path, index_col=0, parse_dates=True)
                        # Use first column, whatever it's called
                        market_symbol = market_cached.columns[0]
                        market_returns = market_cached.reindex(price_data.index).dropna()[market_symbol]
                        if len(market_returns) >= 10:
                            logger.info(f"Using cached {market_symbol} returns data from {cached_returns_path}")
                            break
                except Exception as e:
                    logger.warning(f"Could not load cached market returns from {cached_returns_path}: {e}")
        
        # Method 3: Generate from our portfolio returns as a last resort
        if market_returns is None or len(market_returns) < 10:
            logger.warning("No Korean market data available. Using portfolio average returns as market proxy (preferred over international indices).")
            market_returns = price_data.mean(axis=1).pct_change().dropna()
        
        # Get previous weights from current portfolio
        previous_weights = None
        if optimizer.previous_portfolio is not None:
            try:
                previous_weights = pd.Series(
                    index=optimizer.previous_portfolio['ticker'],
                    data=optimizer.previous_portfolio['weight']
                )
            except Exception as e:
                logger.warning(f"Error creating previous weights: {e}")
        
        # Create theme exposure dict for election themes with varying intensities
        theme_exposure = {}
        try:
            theme_files = ["politic_theme_stocks.csv"]
            for theme_file in theme_files:
                theme_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), theme_file)
                if os.path.exists(theme_path):
                    theme_df = pd.read_csv(theme_path)
                    
                    # Extract sections/themes from comments in file
                    current_theme = "general"
                    theme_weights = {"general": 0.3}  # Default
                    
                    # Process the theme file by groups from comments
                    with open(theme_path, 'r') as f:
                        for line in f:
                            if line.startswith('#'):
                                current_theme = line[1:].strip()
                                # Dynamically weight themes based on popularity/momentum
                                # Higher weights for trending themes
                                if "Lee Jae Myung" in current_theme:
                                    theme_weights[current_theme] = 0.7  # Higher weight
                                elif "Hong Junpyo" in current_theme:
                                    theme_weights[current_theme] = 0.5
                                else:
                                    theme_weights[current_theme] = 0.3
                    
                    # Extract unique rows with tickers
                    for i, row in theme_df.iterrows():
                        if isinstance(row.get('ticker'), str):
                            ticker = row['ticker']
                            if ticker.startswith('A'):
                                ticker = ticker[1:]
                            ticker = ticker.zfill(6)
                            
                            # Assign theme score based on current theme section
                            score = theme_weights.get(current_theme, 0.3)
                            theme_exposure[ticker] = score
        except Exception as e:
            logger.warning(f"Error processing theme exposure: {e}")
        
        # 2. Generate alpha signals
        alpha_engine = EliteAlphaSignalStack(
            price_df=price_data,  # Already in Date x Ticker format
            market_returns=market_returns,
            theme_exposure=theme_exposure,
            previous_weights=previous_weights
        )
        alpha_scores = alpha_engine.compute_signals()
        
        # 3. Detect current market regime for adaptive blending
        is_volatile = False
        try:
            market_vol = market_returns[-20:].std() * np.sqrt(252)
            hist_vol = market_returns.std() * np.sqrt(252)
            is_volatile = market_vol > 1.2 * hist_vol
            if is_volatile:
                logger.info(f"Volatile market detected: Current vol {market_vol:.2%} vs historical {hist_vol:.2%}")
            else:
                logger.info(f"Normal volatility: Current vol {market_vol:.2%} vs historical {hist_vol:.2%}")
        except Exception as e:
            logger.warning(f"Error detecting volatility: {e}")
        
        # 4. Adaptive blending based on market regime
        alpha_weight = 0.25 if is_volatile else 0.4  # Lower alpha weight in volatile markets
        original_weight = 1 - alpha_weight
        
        logger.info(f"Using adaptive blend: {original_weight:.2f} original, {alpha_weight:.2f} alpha signals")
        
        # 5. Apply the blend to cluster_df's combined_ratio
        common_tickers = set(cluster_df['ticker']).intersection(alpha_scores.index)
        blended_count = 0
        
        # First normalize alpha scores to match the scale of combined_ratio
        mean_ratio = cluster_df['combined_ratio'].mean()
        std_ratio = cluster_df['combined_ratio'].std()
        
        for ticker in common_tickers:
            idx = cluster_df[cluster_df['ticker'] == ticker].index
            if len(idx) > 0:
                # Normalize alpha score to similar range as combined_ratio
                norm_alpha = (alpha_scores[ticker] - alpha_scores.mean()) / alpha_scores.std()
                norm_alpha = norm_alpha * std_ratio + mean_ratio
                
                # Update with blended score
                original_value = cluster_df.loc[idx, 'combined_ratio'].values[0]
                new_value = (original_weight * original_value + alpha_weight * norm_alpha)
                cluster_df.loc[idx, 'combined_ratio'] = new_value
                blended_count += 1
        
        # 6. Add EliteAlpha column for transparency
        cluster_df['elite_alpha'] = 0.0
        for ticker in common_tickers:
            idx = cluster_df[cluster_df['ticker'] == ticker].index
            if len(idx) > 0:
                cluster_df.loc[idx, 'elite_alpha'] = alpha_scores[ticker]
        
        logger.info(f"Enhanced {blended_count} tickers with EliteAlphaSignalStack")
        return cluster_df
        
    except Exception as e:
        logger.error(f"Error enhancing portfolio signals: {e}")
        # Return original data if enhancement fails
        return cluster_df