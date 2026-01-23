"""
TORANIKO SIGNALS
================
Enhanced signal construction using Toraniko's mathematical utilities.

Provides improved implementations of:
1. Momentum signal with exponential decay weighting
2. Value signal with proper cross-sectional standardization
3. Size signal (SMB - Small Minus Big)

These can be used alongside or as replacements for the original
SignalConstructor methods.

Author: TimeFolio-Toraniko Integration
Date: 2026-01-23
"""

import pandas as pd
import numpy as np
import polars as pl
from typing import Dict, List, Optional, Tuple
import logging

from toraniko.math import (
    winsorize,
    center_xsection,
    winsorize_xsection,
    exp_weights,
    norm_xsection
)
from toraniko.styles import factor_mom, factor_val, factor_sze
from toraniko.utils import fill_features, smooth_features, top_n_by_group

logger = logging.getLogger(__name__)


class ToranikoSignalEnhancer:
    """
    Enhanced signal construction using Toraniko's mathematical utilities.
    
    Provides methods that can be used to enhance or replace the original
    SignalConstructor implementations with more rigorous statistical methods.
    """
    
    def __init__(
        self,
        momentum_lookback: int = 252,
        momentum_halflife: int = 126,
        momentum_lag: int = 21,
        winsor_factor: float = 0.01,
        size_lower_decile: float = 0.2,
        size_upper_decile: float = 0.8
    ):
        """
        Initialize signal enhancer.
        
        Args:
            momentum_lookback: Days for momentum calculation (default 252 = 12 months)
            momentum_halflife: Half-life for exponential decay (default 126 = 6 months)
            momentum_lag: Days to skip for momentum (default 21 = 1 month)
            winsor_factor: Winsorization percentile (default 0.01 = 1%)
            size_lower_decile: Lower percentile for size factor
            size_upper_decile: Upper percentile for size factor
        """
        self.momentum_lookback = momentum_lookback
        self.momentum_halflife = momentum_halflife
        self.momentum_lag = momentum_lag
        self.winsor_factor = winsor_factor
        self.size_lower_decile = size_lower_decile
        self.size_upper_decile = size_upper_decile
        
        logger.info(f"ToranikoSignalEnhancer initialized:")
        logger.info(f"  - Momentum: {momentum_lookback}d lookback, {momentum_halflife}d halflife, {momentum_lag}d lag")
        logger.info(f"  - Winsorization: {winsor_factor}")
    
    # ==================== Momentum Signal ====================
    
    def construct_momentum_signal(
        self,
        returns_df: pd.DataFrame,
        date: pd.Timestamp
    ) -> pd.Series:
        """
        Construct momentum signal using Toraniko's exponential weighting.
        
        Unlike simple 12-1 month momentum, this uses:
        - Exponential decay weighting (more recent returns weighted higher)
        - Proper cross-sectional standardization
        - Robust winsorization
        
        Args:
            returns_df: DataFrame of returns (date index, ticker columns)
            date: Date for signal construction
            
        Returns:
            Series of momentum scores indexed by ticker
        """
        # Convert to Polars long format
        returns_pl = self._returns_to_polars(returns_df)
        
        if returns_pl.is_empty():
            logger.warning("Empty returns data for momentum signal")
            return pd.Series(dtype=float)
        
        try:
            # Use Toraniko's factor_mom
            mom_df = factor_mom(
                returns_pl,
                trailing_days=self.momentum_lookback,
                half_life=self.momentum_halflife,
                lag=self.momentum_lag,
                winsor_factor=self.winsor_factor
            ).collect()
            
            # Filter to target date (or latest available)
            if date is not None:
                target_date = pd.Timestamp(date).date()
                mom_df = mom_df.filter(pl.col('date') == target_date)
            
            if mom_df.is_empty():
                # Use latest available date
                mom_df = factor_mom(
                    returns_pl,
                    trailing_days=self.momentum_lookback,
                    half_life=self.momentum_halflife,
                    lag=self.momentum_lag,
                    winsor_factor=self.winsor_factor
                ).collect()
                latest_date = mom_df['date'].max()
                mom_df = mom_df.filter(pl.col('date') == latest_date)
            
            # Convert to Pandas Series
            result = mom_df.select(['symbol', 'mom_score']).to_pandas()
            result = result.set_index('symbol')['mom_score']
            
            logger.info(f"Momentum signal: {len(result)} stocks, mean={result.mean():.4f}, std={result.std():.4f}")
            return result
            
        except Exception as e:
            logger.error(f"Momentum signal construction failed: {e}")
            return pd.Series(dtype=float)
    
    # ==================== Value Signal ====================
    
    def construct_value_signal(
        self,
        fundamentals: pd.DataFrame,
        market_caps: pd.Series,
        date: pd.Timestamp
    ) -> pd.Series:
        """
        Construct value signal using Toraniko's methodology.
        
        Combines:
        - Book-to-Price (B/P)
        - Sales-to-Price (S/P)
        - Cash Flow-to-Price (CF/P)
        
        With proper log transformation and cross-sectional standardization.
        
        Args:
            fundamentals: DataFrame with columns [total_equity, revenue, operating_cash_flow]
            market_caps: Series of market caps indexed by ticker
            date: Date for signal construction
            
        Returns:
            Series of value scores indexed by ticker
        """
        # Calculate price ratios
        common_tickers = list(set(fundamentals.index) & set(market_caps.index))
        
        if len(common_tickers) == 0:
            logger.warning("No common tickers for value signal")
            return pd.Series(dtype=float)
        
        # Extract data
        fund = fundamentals.loc[common_tickers]
        caps = market_caps.loc[common_tickers]
        
        # Calculate ratios (handle division by zero)
        ratios = pd.DataFrame(index=common_tickers)
        
        if 'total_equity' in fund.columns:
            ratios['book_price'] = fund['total_equity'] / caps.replace(0, np.nan)
        
        if 'revenue' in fund.columns:
            ratios['sales_price'] = fund['revenue'] / caps.replace(0, np.nan)
        
        if 'operating_cash_flow' in fund.columns:
            ratios['cf_price'] = fund['operating_cash_flow'] / caps.replace(0, np.nan)
        
        if ratios.empty:
            logger.warning("No value metrics available")
            return pd.Series(dtype=float)
        
        # Convert to Polars for Toraniko processing
        ratios['symbol'] = ratios.index
        ratios['date'] = date
        ratios_pl = pl.from_pandas(ratios)
        
        try:
            # Use Toraniko's factor_val if we have all required columns
            required_cols = ['book_price', 'sales_price', 'cf_price']
            available_cols = [c for c in required_cols if c in ratios.columns]
            
            if len(available_cols) == 3:
                val_df = factor_val(
                    ratios_pl.select(['date', 'symbol'] + available_cols),
                    winsorize_features=self.winsor_factor
                ).collect()
                
                result = val_df.select(['symbol', 'val_score']).to_pandas()
                result = result.set_index('symbol')['val_score']
            else:
                # Manual calculation for partial data
                result = self._manual_value_signal(ratios, available_cols)
            
            logger.info(f"Value signal: {len(result)} stocks, mean={result.mean():.4f}, std={result.std():.4f}")
            return result
            
        except Exception as e:
            logger.error(f"Value signal construction failed: {e}")
            # Fallback to manual calculation
            return self._manual_value_signal(ratios, available_cols)
    
    def _manual_value_signal(
        self,
        ratios: pd.DataFrame,
        value_cols: List[str]
    ) -> pd.Series:
        """Manual value signal calculation as fallback."""
        scores = []
        
        for col in value_cols:
            if col in ratios.columns:
                # Log transform (for book_price and sales_price)
                if col in ['book_price', 'sales_price']:
                    vals = np.log(ratios[col].clip(lower=1e-10))
                else:
                    vals = ratios[col]
                
                # Winsorize
                vals_np = vals.values
                vals_win = winsorize(vals_np, percentile=self.winsor_factor)
                
                # Standardize
                z_score = (vals_win - np.nanmean(vals_win)) / np.nanstd(vals_win)
                scores.append(pd.Series(z_score, index=ratios.index))
        
        if not scores:
            return pd.Series(dtype=float)
        
        # Average z-scores
        composite = pd.concat(scores, axis=1).mean(axis=1)
        
        # Final standardization
        composite = (composite - composite.mean()) / composite.std()
        
        return composite
    
    # ==================== Size Signal ====================
    
    def construct_size_signal(
        self,
        market_caps: pd.Series,
        date: pd.Timestamp
    ) -> pd.Series:
        """
        Construct size signal (SMB - Small Minus Big) using Toraniko.
        
        The size premium is on smaller firms, so we use negative log market cap.
        
        Args:
            market_caps: Series of market caps indexed by ticker
            date: Date for signal construction
            
        Returns:
            Series of size scores indexed by ticker (higher = smaller cap)
        """
        if market_caps.empty:
            logger.warning("Empty market caps for size signal")
            return pd.Series(dtype=float)
        
        # Convert to Polars
        mkt_cap_df = pd.DataFrame({
            'date': date,
            'symbol': market_caps.index,
            'market_cap': market_caps.values
        })
        mkt_cap_pl = pl.from_pandas(mkt_cap_df)
        
        try:
            # Use Toraniko's factor_sze
            sze_df = factor_sze(
                mkt_cap_pl,
                lower_decile=self.size_lower_decile,
                upper_decile=self.size_upper_decile
            ).collect()
            
            result = sze_df.select(['symbol', 'sze_score']).to_pandas()
            result = result.set_index('symbol')['sze_score']
            
            logger.info(f"Size signal: {len(result)} stocks, mean={result.mean():.4f}, std={result.std():.4f}")
            return result
            
        except Exception as e:
            logger.error(f"Size signal construction failed: {e}")
            # Fallback to manual calculation
            return self._manual_size_signal(market_caps)
    
    def _manual_size_signal(self, market_caps: pd.Series) -> pd.Series:
        """Manual size signal calculation as fallback."""
        # Negative log market cap (smaller = higher score)
        log_cap = -np.log(market_caps.clip(lower=1e-10))
        
        # Winsorize
        log_cap_win = winsorize(log_cap.values, percentile=self.winsor_factor)
        
        # Standardize
        z_score = (log_cap_win - np.nanmean(log_cap_win)) / np.nanstd(log_cap_win)
        
        return pd.Series(z_score, index=market_caps.index)
    
    # ==================== Utility Methods ====================
    
    def _returns_to_polars(self, returns_df: pd.DataFrame) -> pl.DataFrame:
        """Convert returns DataFrame to Polars long format."""
        if returns_df.empty:
            return pl.DataFrame()
        
        # Reset index if date is index
        if isinstance(returns_df.index, pd.DatetimeIndex):
            returns_df = returns_df.reset_index()
            returns_df.columns = ['date'] + list(returns_df.columns[1:])
        
        # Melt to long format
        melted = returns_df.melt(
            id_vars=['date'],
            var_name='symbol',
            value_name='asset_returns'
        )
        
        # Convert to Polars
        pl_df = pl.from_pandas(melted)
        
        # Ensure proper types
        pl_df = pl_df.with_columns([
            pl.col('date').cast(pl.Date),
            pl.col('symbol').cast(pl.Utf8),
            pl.col('asset_returns').cast(pl.Float64)
        ])
        
        # Drop nulls
        pl_df = pl_df.drop_nulls(subset=['asset_returns'])
        
        return pl_df
    
    def winsorize_series(self, series: pd.Series, percentile: Optional[float] = None) -> pd.Series:
        """Winsorize a Pandas Series using Toraniko's winsorize function."""
        if percentile is None:
            percentile = self.winsor_factor
        
        values = series.values
        winsorized = winsorize(values, percentile=percentile)
        return pd.Series(winsorized, index=series.index)
    
    def standardize_cross_sectional(
        self,
        series: pd.Series,
        winsorize_first: bool = True
    ) -> pd.Series:
        """
        Cross-sectionally standardize a series (z-score).
        
        Args:
            series: Input series
            winsorize_first: Whether to winsorize before standardizing
            
        Returns:
            Standardized series
        """
        if winsorize_first:
            series = self.winsorize_series(series)
        
        mean = series.mean()
        std = series.std()
        
        if std == 0 or np.isnan(std):
            return pd.Series(0, index=series.index)
        
        return (series - mean) / std
    
    def construct_all_enhanced_signals(
        self,
        returns: pd.DataFrame,
        market_caps: pd.DataFrame,
        fundamentals: pd.DataFrame,
        date: pd.Timestamp
    ) -> pd.DataFrame:
        """
        Construct all enhanced signals for a given date.
        
        Args:
            returns: DataFrame of returns (date x ticker)
            market_caps: DataFrame of market caps (date x ticker)
            fundamentals: DataFrame of fundamentals (ticker x metrics)
            date: Date for signal construction
            
        Returns:
            DataFrame with columns ['Momentum', 'Value', 'Size'] indexed by ticker
        """
        signals = {}
        
        # Get market caps for the date
        if isinstance(market_caps.index, pd.DatetimeIndex):
            if date in market_caps.index:
                caps_series = market_caps.loc[date]
            else:
                # Use latest available
                caps_series = market_caps.iloc[-1]
        else:
            caps_series = market_caps.iloc[-1] if len(market_caps) > 0 else pd.Series()
        
        # Momentum
        mom_signal = self.construct_momentum_signal(returns, date)
        if not mom_signal.empty:
            signals['Momentum'] = mom_signal
        
        # Value
        val_signal = self.construct_value_signal(fundamentals, caps_series, date)
        if not val_signal.empty:
            signals['Value'] = val_signal
        
        # Size
        sze_signal = self.construct_size_signal(caps_series, date)
        if not sze_signal.empty:
            signals['Size'] = sze_signal
        
        if not signals:
            return pd.DataFrame()
        
        # Combine into DataFrame
        result = pd.DataFrame(signals)
        
        # Align to common tickers
        result = result.dropna(how='all')
        
        logger.info(f"Enhanced signals constructed: {len(result)} stocks, {len(signals)} signals")
        return result


def create_signal_enhancer(**kwargs) -> ToranikoSignalEnhancer:
    """Factory function to create ToranikoSignalEnhancer."""
    return ToranikoSignalEnhancer(**kwargs)
