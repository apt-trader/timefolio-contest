"""Enhanced factor engine for Korean equity portfolio optimization."""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy.stats import zscore
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FactorEngine:
    def __init__(self, 
                 mom_windows: List[int] = [20, 60, 120],
                 vol_window: int = 20,
                 rsi_window: int = 14,
                 min_volume: float = 3e9,  # 3B KRW minimum daily volume
                 volume_window: int = 5):
        """
        Initialize the factor engine.
        
        Args:
            mom_windows: Lookback periods for momentum factors (in days)
            vol_window: Window for volatility calculation (in days)
            rsi_window: Window for RSI calculation (in days)
            min_volume: Minimum average daily volume (KRW)
            volume_window: Window for volume averaging (in days)
        """
        self.mom_windows = mom_windows
        self.vol_window = vol_window
        self.rsi_window = rsi_window
        self.min_volume = min_volume
        self.volume_window = volume_window

    def calculate_factors(self, 
                        prices: pd.DataFrame, 
                        volumes: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Calculate all factors.
        
        Args:
            prices: DataFrame with price data (tickers as columns, datetime index)
            volumes: DataFrame with volume data (same structure as prices)
            
        Returns:
            Dictionary of factor DataFrames
        """
        logger.info("Calculating factors...")
        returns = prices.pct_change()
        
        # 1. Calculate individual factors
        vol_adj_momentum = self._calculate_vol_adj_momentum(prices, returns)
        mean_reversion = self._calculate_mean_reversion(prices, returns)
        liquidity = self._calculate_liquidity(volumes)
        
        # 2. Combine factors
        factors = {
            'momentum': vol_adj_momentum,
            'mean_reversion': mean_reversion,
            'liquidity': liquidity
        }
        
        # 3. Create composite score (equal-weighted for now)
        composite = sum(factor for factor in factors.values()) / len(factors)
        factors['composite'] = composite
        
        return factors
    
    def _calculate_vol_adj_momentum(self, 
                                   prices: pd.DataFrame, 
                                   returns: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility-adjusted momentum factors."""
        logger.debug("Calculating volatility-adjusted momentum...")
        vol = returns.rolling(self.vol_window).std() * np.sqrt(252)  # Annualized vol
        
        momentum_factors = []
        for window in self.mom_windows:
            # Simple momentum
            mom = prices.pct_change(window)
            
            # Volatility adjustment (avoid division by zero)
            vol_adj = vol.mask(vol < 1e-6, np.nan)
            vol_adj_mom = mom / (vol_adj + 1e-6)
            
            # Z-score normalization
            z_mom = vol_adj_mom.apply(zscore)
            momentum_factors.append(z_mom)
        
        # Average across different lookback periods
        if momentum_factors:
            avg_momentum = sum(momentum_factors) / len(momentum_factors)
            return avg_momentum
        return pd.DataFrame(0, index=prices.index, columns=prices.columns)
    
    def _calculate_mean_reversion(self, 
                                prices: pd.DataFrame, 
                                returns: pd.DataFrame) -> pd.DataFrame:
        """Calculate mean reversion factors."""
        logger.debug("Calculating mean reversion factors...")
        
        # 1. RSI (Relative Strength Index)
        delta = returns.diff()
        gain = delta.where(delta > 0, 0).rolling(self.rsi_window).mean()
        loss = -delta.where(delta < 0, 0).rolling(self.rsi_window).mean()
        rs = gain / (loss + 1e-6)
        rsi = 100 - (100 / (1 + rs))
        
        # 2. Bollinger Bands %B
        rolling_mean = prices.rolling(20).mean()
        rolling_std = prices.rolling(20).std()
        bollinger_pct = (prices - rolling_mean) / (2 * rolling_std + 1e-6)
        
        # Combine signals (equal weight)
        mean_rev = (rsi.rank(axis=1, pct=True) + bollinger_pct.rank(axis=1, pct=True)) / 2
        return mean_rev
    
    def _calculate_liquidity(self, volumes: pd.DataFrame) -> pd.DataFrame:
        """Calculate liquidity factors."""
        logger.debug("Calculating liquidity factors...")
        
        # 1. Volume-based liquidity
        avg_volume = volumes.rolling(self.volume_window).mean()
        volume_liquidity = avg_volume.apply(lambda x: x / self.min_volume).clip(upper=1.0)
        
        # 2. Volume trend (increasing volume is better)
        volume_trend = volumes.pct_change(self.volume_window).clip(lower=-1.0, upper=1.0)
        
        # Combine liquidity factors
        liquidity = (volume_liquidity * 0.7 + (volume_trend + 1) * 0.15)  # 70% level, 15% trend
        
        return liquidity
    
    def filter_universe(self, 
                       factors: Dict[str, pd.DataFrame], 
                       prices: pd.DataFrame,
                       min_liquidity: float = 0.5) -> pd.DataFrame:
        """
        Filter universe based on liquidity and other criteria.
        
        Args:
            factors: Dictionary of factor DataFrames
            prices: Price data
            min_liquidity: Minimum liquidity score (0-1)
            
        Returns:
            Boolean mask of valid securities
        """
        liquidity_mask = factors['liquidity'] >= min_liquidity
        price_mask = prices > 1000  # Minimum price filter (1000 KRW)
        
        # Combine masks
        valid_mask = liquidity_mask & price_mask
        return valid_mask

# Example usage
if __name__ == "__main__":
    # Example data (replace with actual data loading)
    dates = pd.date_range(start='2024-01-01', periods=100)
    tickers = ['005930', '000660', '035420']  # Example tickers
    
    # Generate random price and volume data for demonstration
    np.random.seed(42)
    prices = pd.DataFrame(
        np.cumprod(1 + np.random.normal(0.001, 0.02, (100, 3)), axis=0) * 50000,
        index=dates,
        columns=tickers
    )
    
    volumes = pd.DataFrame(
        np.random.lognormal(15, 0.5, (100, 3)),
        index=dates,
        columns=tickers
    ) * 1e6  # Convert to KRW
    
    # Initialize and run factor engine
    engine = FactorEngine()
    factors = engine.calculate_factors(prices, volumes)
    
    # Show latest factor values
    print("\nLatest Composite Scores:")
    print(factors['composite'].iloc[-1].sort_values(ascending=False))
