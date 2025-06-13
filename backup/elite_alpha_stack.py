"""
Elite Alpha Signal Stack implementation for enhanced alpha generation.
Implements short-term alpha signals for tactical portfolio allocation.
"""

import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime
import statsmodels.api as sm

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

class EliteAlphaSignalStack:
    """
    Implements multi-factor alpha signals for short-term tactical allocation.
    
    Combines multiple signals:
    1. Multi-timeframe momentum (5/10/20 days)
    2. Volatility-adjusted returns
    3. Market-neutral (residual) alpha
    4. Theme exposure integration
    5. Retention bias to reduce turnover
    """
    
    def __init__(self, price_df, market_returns=None, theme_exposure=None, previous_weights=None):
        """
        Initialize the EliteAlphaSignalStack with required data.
        
        Args:
            price_df (DataFrame): Price data in DataFrame format with tickers as columns
            market_returns (Series, optional): Market returns for residual alpha calculation
            theme_exposure (dict, optional): Dictionary mapping tickers to theme exposure scores
            previous_weights (Series, optional): Previous portfolio weights for retention bias
        """
        self.price_df = price_df
        self.market_returns = market_returns
        self.theme_exposure = theme_exposure or {}
        self.previous_weights = previous_weights
        
        # Convert prices to returns if needed
        if self._is_price_data(price_df):
            logger.info("Converting price data to returns")
            self.returns_df = price_df.pct_change().dropna()
        else:
            self.returns_df = price_df
            
        # Fill any remaining NaN values with 0
        self.returns_df = self.returns_df.fillna(0)
        
        logger.info(f"Initialized EliteAlphaSignalStack with {len(self.returns_df.columns)} tickers")
    
    def _is_price_data(self, df):
        """Check if the data looks like price data rather than returns"""
        # Price data typically has values > 1, doesn't change by >20% daily, and is always positive
        sample = df.iloc[-20:].mean()
        return (sample > 1).mean() > 0.9  # >90% of columns have mean > 1
    
    def compute_signals(self):
        """
        Compute combined alpha signals from multiple factors.
        
        Returns:
            Series: Alpha scores for each ticker
        """
        logger.info("Computing EliteAlphaSignalStack signals")
        
        # 1. Multi-timeframe momentum signals
        mom_5d = self._compute_momentum(5)
        mom_10d = self._compute_momentum(10)
        mom_20d = self._compute_momentum(20)
        
        # 2. Volatility adjustment (penalize high volatility)
        vol_adj = self._compute_volatility_adjustment()
        
        # 3. Residual alpha (market-neutral component)
        residual_alpha = self._compute_residual_alpha()
        
        # 4. Theme exposure integration
        theme_scores = self._compute_theme_scores()
        
        # 5. Retention bias (reduce turnover by favoring existing positions)
        retention_scores = self._compute_retention_bias()
        
        # Combine all signals with different weights
        alpha_scores = pd.DataFrame({
            'mom_5d': mom_5d,
            'mom_10d': mom_10d, 
            'mom_20d': mom_20d,
            'vol_adj': vol_adj,
            'residual_alpha': residual_alpha,
            'theme': theme_scores,
            'retention': retention_scores
        })
        
        # Normalize each signal to have mean 0 and std 1
        for col in alpha_scores.columns:
            if alpha_scores[col].std() > 0:
                alpha_scores[col] = (alpha_scores[col] - alpha_scores[col].mean()) / alpha_scores[col].std()
            else:
                alpha_scores[col] = 0  # If std=0, set all values to 0
        
        # Apply signal weights to get combined score
        weights = {
            'mom_5d': 0.25,          # Short-term momentum
            'mom_10d': 0.20,         # Medium-term momentum
            'mom_20d': 0.15,         # Longer-term momentum
            'vol_adj': 0.10,         # Volatility penalty
            'residual_alpha': 0.15,  # Market-neutral component
            'theme': 0.10,           # Theme exposure
            'retention': 0.05        # Retention bias
        }
        
        combined_score = pd.Series(0, index=alpha_scores.index)
        for col, weight in weights.items():
            combined_score += alpha_scores[col] * weight
        
        # Final normalization to get z-scores
        if combined_score.std() > 0:
            combined_score = (combined_score - combined_score.mean()) / combined_score.std()
        
        # Log signal statistics 
        logger.info(f"Generated alpha signals for {len(combined_score)} tickers")
        logger.info(f"Alpha signal stats: min={combined_score.min():.2f}, max={combined_score.max():.2f}, "
                   f"mean={combined_score.mean():.2f}, std={combined_score.std():.2f}")
        
        return combined_score
    
    def _compute_momentum(self, window):
        """Compute momentum over the specified window"""
        try:
            # If we have enough data, use cumulative return
            if len(self.returns_df) >= window:
                # Compute cumulative return for each column over the specified window
                cumulative_returns = ((1 + self.returns_df.iloc[-window:]).prod() - 1)
                return cumulative_returns
            # Otherwise, use available data
            else:
                logger.warning(f"Insufficient data for {window}-day momentum, using available data")
                cumulative_returns = ((1 + self.returns_df).prod() - 1)
                return cumulative_returns
        except Exception as e:
            logger.error(f"Error in momentum calculation: {e}")
            # Return zeros as fallback
            return pd.Series(0, index=self.returns_df.columns)
    
    def _compute_volatility_adjustment(self):
        """Compute volatility adjustment (lower is better)"""
        # Use negative volatility so higher values are better, like our other signals
        annualization_factor = np.sqrt(252)  # Define this here to avoid scoping issues
        
        if len(self.returns_df) >= 20:
            # Calculate annualized standard deviation directly
            volatility = self.returns_df.iloc[-20:].std() * annualization_factor
            return -1 * volatility  # Negative so higher values are better
        else:
            volatility = self.returns_df.std() * annualization_factor
            return -1 * volatility
    
    def _compute_residual_alpha(self):
        """Compute residual alpha (market-neutral returns)"""
        if self.market_returns is None or len(self.market_returns) < 20:
            logger.warning("Insufficient market data for residual alpha calculation")
            return pd.Series(0, index=self.returns_df.columns)
        
        try:
            # Align market returns with return data
            aligned_market = self.market_returns.reindex(self.returns_df.index).dropna()
            aligned_returns = self.returns_df.reindex(aligned_market.index)
            
            if len(aligned_market) < 15:
                logger.warning(f"Insufficient aligned data for residual calculation: {len(aligned_market)} points")
                return pd.Series(0, index=self.returns_df.columns)
            
            # For each stock, regress against market
            residual_returns = {}
            
            for ticker in aligned_returns.columns:
                # Add constant to market returns for regression
                X = sm.add_constant(aligned_market)
                y = aligned_returns[ticker]
                
                try:
                    # Run regression
                    model = sm.OLS(y, X).fit()
                    
                    # Get residuals (alpha)
                    resid = model.resid
                    
                    # Use recent residuals as signal
                    recent_window = min(10, len(resid))
                    residual_returns[ticker] = resid[-recent_window:].mean()
                except Exception as e:
                    logger.warning(f"Error in market regression for {ticker}: {e}")
                    residual_returns[ticker] = 0
            
            return pd.Series(residual_returns)
        except Exception as e:
            logger.error(f"Error computing residual alpha: {e}")
            return pd.Series(0, index=self.returns_df.columns)
    
    def _compute_theme_scores(self):
        """Compute theme exposure scores"""
        if not self.theme_exposure:
            return pd.Series(0, index=self.returns_df.columns)
        
        scores = {}
        for ticker in self.returns_df.columns:
            # Get correct ticker format (6 digits with leading zeros)
            norm_ticker = ticker
            if norm_ticker.isdigit() and len(norm_ticker) < 6:
                norm_ticker = norm_ticker.zfill(6)
            
            # Apply theme score if available, otherwise 0
            scores[ticker] = self.theme_exposure.get(norm_ticker, 0)
        
        return pd.Series(scores)
    
    def _compute_retention_bias(self):
        """Compute retention bias to reduce turnover"""
        if self.previous_weights is None:
            return pd.Series(0, index=self.returns_df.columns)
        
        # Create series of previous weights aligned with current tickers
        retention = pd.Series(0, index=self.returns_df.columns)
        
        # Fill with previous weights where available
        for ticker in retention.index:
            if ticker in self.previous_weights.index:
                retention[ticker] = self.previous_weights[ticker]
        
        return retention