"""
SIGNAL CONSTRUCTOR
==================
Builds 5 orthogonal trading signals from raw factor data.
Each signal represents a tradeable long-only portfolio strategy.

Signals:
1. Value: Composite of B/P, E/P, S/P (stability-weighted)
2. Quality: ROE stability + low leverage + earnings quality
3. Momentum: 12-1 month momentum (skip recent month)
4. Low Volatility: Inverse realized volatility (defensive)
5. Growth: Revenue/earnings growth stability

Architecture follows "Introduction To Forecast Signals.txt":
- Start with established features
- Regularize everything
- Preprocess carefully (winsorization, standardization)
- Meaningful dimensionality reduction
- Evaluate on trading performance

Author: TimeFolio System Refactor
Date: 2025-01-11
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats
from sklearn.preprocessing import RobustScaler
import logging

logger = logging.getLogger(__name__)


class SignalConstructor:
    """
    Constructs orthogonal trading signals from raw market and fundamental data.
    
    Each signal is a cross-sectional score (higher = more attractive).
    Signals are designed to be:
    1. Orthogonal (low correlation with each other)
    2. Stable (persistent over time)
    3. Tradeable (can be converted to long-only portfolios)
    """
    
    def __init__(
        self,
        winsorize_limits: Tuple[float, float] = (0.02, 0.98),
        momentum_lookback: int = 252,
        momentum_skip: int = 21,
        volatility_lookback: int = 63,
        min_history_days: int = 126
    ):
        """
        Initialize signal constructor.
        
        Args:
            winsorize_limits: Quantile limits for winsorization
            momentum_lookback: Days for momentum calculation (default 252 = 12 months)
            momentum_skip: Days to skip for momentum (default 21 = 1 month)
            volatility_lookback: Days for volatility calculation
            min_history_days: Minimum price history required
        """
        self.winsorize_limits = winsorize_limits
        self.momentum_lookback = momentum_lookback
        self.momentum_skip = momentum_skip
        self.volatility_lookback = volatility_lookback
        self.min_history_days = min_history_days
        
        self.scaler = RobustScaler(quantile_range=(10.0, 90.0))
        
        logger.info(f"SignalConstructor initialized:")
        logger.info(f"  - Winsorize limits: {winsorize_limits}")
        logger.info(f"  - Momentum: {momentum_lookback}d lookback, {momentum_skip}d skip")
        logger.info(f"  - Volatility lookback: {volatility_lookback}d")
    
    def construct_all_signals(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        market_caps: pd.DataFrame,
        fundamentals: pd.DataFrame,
        date: pd.Timestamp
    ) -> pd.DataFrame:
        """
        Construct all 5 signals for a given date.
        
        Args:
            prices: DataFrame of prices (date x ticker)
            returns: DataFrame of returns (date x ticker)
            market_caps: DataFrame of market caps (date x ticker)
            fundamentals: DataFrame of fundamentals (ticker x metrics)
            date: Date for signal construction
            
        Returns:
            DataFrame with columns ['Value', 'Quality', 'Momentum', 'LowVol', 'Growth']
            indexed by ticker
        """
        logger.info(f"Constructing signals for {date.strftime('%Y-%m-%d')}")
        
        # Get data up to date
        prices_to_date = prices.loc[:date]
        returns_to_date = returns.loc[:date]
        market_caps_at_date = market_caps.loc[date] if date in market_caps.index else market_caps.iloc[-1]
        
        # Filter tickers with sufficient history
        valid_tickers = self._get_valid_tickers(prices_to_date, fundamentals)
        logger.info(f"Valid tickers with sufficient data: {len(valid_tickers)}")
        
        if len(valid_tickers) < 20:
            logger.warning(f"Too few valid tickers ({len(valid_tickers)}), returning empty signals")
            return pd.DataFrame()
        
        # Construct individual signals
        signals = {}
        
        # 1. Value Signal
        value_signal = self._construct_value_signal(
            fundamentals.loc[fundamentals.index.isin(valid_tickers)],
            market_caps_at_date[valid_tickers]
        )
        signals['Value'] = value_signal
        
        # 2. Quality Signal
        quality_signal = self._construct_quality_signal(
            fundamentals.loc[fundamentals.index.isin(valid_tickers)]
        )
        signals['Quality'] = quality_signal
        
        # 3. Momentum Signal
        momentum_signal = self._construct_momentum_signal(
            prices_to_date[valid_tickers]
        )
        signals['Momentum'] = momentum_signal
        
        # 4. Low Volatility Signal
        lowvol_signal = self._construct_lowvol_signal(
            returns_to_date[valid_tickers]
        )
        signals['LowVol'] = lowvol_signal
        
        # 5. Growth Signal
        growth_signal = self._construct_growth_signal(
            fundamentals.loc[fundamentals.index.isin(valid_tickers)]
        )
        signals['Growth'] = growth_signal
        
        # Combine into DataFrame
        signal_df = pd.DataFrame(signals)
        
        # Standardize all signals to z-scores
        signal_df = self._standardize_signals(signal_df)
        
        # Log signal statistics
        self._log_signal_stats(signal_df)
        
        return signal_df
    
    def _get_valid_tickers(
        self,
        prices: pd.DataFrame,
        fundamentals: pd.DataFrame
    ) -> List[str]:
        """Get tickers with sufficient price history and fundamental data."""
        valid_tickers = []
        
        for ticker in prices.columns:
            # Check price history
            ticker_prices = prices[ticker].dropna()
            if len(ticker_prices) < self.min_history_days:
                continue
            
            # Check if in fundamentals
            if ticker not in fundamentals.index:
                continue
            
            # Check for essential fundamental fields
            ticker_fundamentals = fundamentals.loc[ticker]
            essential_fields = ['total_equity', 'net_income', 'revenue']
            has_essentials = all(
                field in ticker_fundamentals.index and pd.notna(ticker_fundamentals.get(field))
                for field in essential_fields
            )
            
            if has_essentials:
                valid_tickers.append(ticker)
        
        return valid_tickers
    
    def _construct_value_signal(
        self,
        fundamentals: pd.DataFrame,
        market_caps: pd.Series
    ) -> pd.Series:
        """
        Construct Value signal: composite of B/P, E/P, S/P.
        Higher value = cheaper stock.
        """
        value_components = {}
        
        # Book-to-Price (most stable)
        if 'total_equity' in fundamentals.columns:
            b2p = fundamentals['total_equity'] / market_caps
            b2p = self._winsorize(b2p)
            value_components['B2P'] = b2p
        
        # Earnings-to-Price
        if 'net_income' in fundamentals.columns:
            e2p = fundamentals['net_income'] / market_caps
            e2p = self._winsorize(e2p)
            value_components['E2P'] = e2p
        
        # Sales-to-Price
        if 'revenue' in fundamentals.columns:
            s2p = fundamentals['revenue'] / market_caps
            s2p = self._winsorize(s2p)
            value_components['S2P'] = s2p
        
        # Cash-to-Price (if available)
        if 'operating_cash_flow' in fundamentals.columns:
            c2p = fundamentals['operating_cash_flow'] / market_caps
            c2p = self._winsorize(c2p)
            value_components['C2P'] = c2p
        
        if not value_components:
            logger.warning("No value components available")
            return pd.Series(index=fundamentals.index, data=np.nan)
        
        # Equal-weight composite (simple and robust)
        composite = pd.DataFrame(value_components).rank(pct=True).mean(axis=1)
        
        logger.info(f"Value signal: {len(value_components)} components, {composite.notna().sum()} valid")
        return composite
    
    def _construct_quality_signal(
        self,
        fundamentals: pd.DataFrame
    ) -> pd.Series:
        """
        Construct Quality signal: ROE, low leverage, earnings quality.
        Higher quality = better fundamentals.
        """
        quality_components = {}
        
        # ROE (profitability)
        if 'roe' in fundamentals.columns:
            roe = fundamentals['roe']
            roe = self._winsorize(roe)
            quality_components['ROE'] = roe
        elif 'net_income' in fundamentals.columns and 'total_equity' in fundamentals.columns:
            roe = fundamentals['net_income'] / fundamentals['total_equity'].replace(0, np.nan)
            roe = self._winsorize(roe)
            quality_components['ROE'] = roe
        
        # Low Leverage (inverse debt-to-equity)
        if 'debt_to_equity' in fundamentals.columns:
            leverage = fundamentals['debt_to_equity']
            low_leverage = 1 / (1 + leverage.clip(lower=0))  # Higher = lower leverage
            low_leverage = self._winsorize(low_leverage)
            quality_components['LowLeverage'] = low_leverage
        elif 'total_liabilities' in fundamentals.columns and 'total_equity' in fundamentals.columns:
            leverage = fundamentals['total_liabilities'] / fundamentals['total_equity'].replace(0, np.nan)
            low_leverage = 1 / (1 + leverage.clip(lower=0))
            low_leverage = self._winsorize(low_leverage)
            quality_components['LowLeverage'] = low_leverage
        
        # Earnings Quality (OCF / Net Income)
        if 'operating_cash_flow' in fundamentals.columns and 'net_income' in fundamentals.columns:
            earnings_quality = fundamentals['operating_cash_flow'] / fundamentals['net_income'].abs().replace(0, np.nan)
            earnings_quality = earnings_quality.clip(-3, 3)  # Bound extremes
            earnings_quality = self._winsorize(earnings_quality)
            quality_components['EarningsQuality'] = earnings_quality
        
        # Gross Profit / Assets (profitability efficiency)
        if 'gross_profit' in fundamentals.columns and 'total_assets' in fundamentals.columns:
            gpa = fundamentals['gross_profit'] / fundamentals['total_assets'].replace(0, np.nan)
            gpa = self._winsorize(gpa)
            quality_components['GPA'] = gpa
        
        if not quality_components:
            logger.warning("No quality components available")
            return pd.Series(index=fundamentals.index, data=np.nan)
        
        # Equal-weight composite
        composite = pd.DataFrame(quality_components).rank(pct=True).mean(axis=1)
        
        logger.info(f"Quality signal: {len(quality_components)} components, {composite.notna().sum()} valid")
        return composite
    
    def _construct_momentum_signal(
        self,
        prices: pd.DataFrame
    ) -> pd.Series:
        """
        Construct Momentum signal: 12-1 month return.
        Skip recent month to avoid short-term reversal.
        """
        if len(prices) < self.momentum_lookback:
            logger.warning(f"Insufficient price history for momentum ({len(prices)} < {self.momentum_lookback})")
            return pd.Series(index=prices.columns, data=np.nan)
        
        # 12-month return (skip last month)
        end_idx = -self.momentum_skip if self.momentum_skip > 0 else None
        start_idx = -(self.momentum_lookback)
        
        if end_idx is not None:
            recent_prices = prices.iloc[end_idx]
            old_prices = prices.iloc[start_idx]
        else:
            recent_prices = prices.iloc[-1]
            old_prices = prices.iloc[start_idx]
        
        momentum = (recent_prices / old_prices.replace(0, np.nan)) - 1
        momentum = self._winsorize(momentum)
        
        logger.info(f"Momentum signal: {momentum.notna().sum()} valid")
        return momentum
    
    def _construct_lowvol_signal(
        self,
        returns: pd.DataFrame
    ) -> pd.Series:
        """
        Construct Low Volatility signal: inverse realized volatility.
        Higher signal = lower volatility (defensive).
        """
        if len(returns) < self.volatility_lookback:
            logger.warning(f"Insufficient return history for volatility ({len(returns)} < {self.volatility_lookback})")
            return pd.Series(index=returns.columns, data=np.nan)
        
        # Realized volatility (annualized)
        recent_returns = returns.iloc[-self.volatility_lookback:]
        volatility = recent_returns.std() * np.sqrt(252)
        
        # Inverse volatility (higher = lower vol = more attractive)
        low_vol = 1 / volatility.replace(0, np.nan)
        low_vol = self._winsorize(low_vol)
        
        logger.info(f"LowVol signal: {low_vol.notna().sum()} valid")
        return low_vol
    
    def _construct_growth_signal(
        self,
        fundamentals: pd.DataFrame
    ) -> pd.Series:
        """
        Construct Growth signal: revenue and earnings growth.
        Higher growth = faster growing company.
        """
        growth_components = {}
        
        # Revenue growth (YoY)
        if 'revenue_growth' in fundamentals.columns:
            rev_growth = fundamentals['revenue_growth']
            rev_growth = self._winsorize(rev_growth)
            growth_components['RevenueGrowth'] = rev_growth
        
        # Earnings growth (YoY)
        if 'earnings_growth' in fundamentals.columns:
            earn_growth = fundamentals['earnings_growth']
            earn_growth = self._winsorize(earn_growth)
            growth_components['EarningsGrowth'] = earn_growth
        
        # Asset growth (conservative growth proxy)
        if 'asset_growth' in fundamentals.columns:
            asset_growth = fundamentals['asset_growth']
            # Negative asset growth can be good (efficient capital use)
            # But for growth signal, we want positive growth
            asset_growth = self._winsorize(asset_growth)
            growth_components['AssetGrowth'] = asset_growth
        
        # If no explicit growth fields, try to compute from fundamentals
        if not growth_components:
            logger.warning("No growth components available - using ROE as proxy")
            if 'roe' in fundamentals.columns:
                # ROE as growth proxy (sustainable growth rate)
                growth_components['ROE_Growth'] = self._winsorize(fundamentals['roe'])
        
        if not growth_components:
            return pd.Series(index=fundamentals.index, data=np.nan)
        
        # Equal-weight composite
        composite = pd.DataFrame(growth_components).rank(pct=True).mean(axis=1)
        
        logger.info(f"Growth signal: {len(growth_components)} components, {composite.notna().sum()} valid")
        return composite
    
    def _winsorize(self, series: pd.Series) -> pd.Series:
        """Apply winsorization to handle outliers."""
        if series.empty or series.isna().all():
            return series
        
        lower = series.quantile(self.winsorize_limits[0])
        upper = series.quantile(self.winsorize_limits[1])
        
        return series.clip(lower=lower, upper=upper)
    
    def _standardize_signals(self, signal_df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize all signals to z-scores using robust statistics.
        """
        standardized = pd.DataFrame(index=signal_df.index)
        
        for col in signal_df.columns:
            series = signal_df[col]
            if series.isna().all():
                standardized[col] = series
                continue
            
            # Robust z-score using median and MAD
            median = series.median()
            mad = (series - median).abs().median()
            
            if mad > 0:
                z_score = (series - median) / (1.4826 * mad)
            else:
                z_score = series - median
            
            # Clip extreme z-scores
            z_score = z_score.clip(-3, 3)
            standardized[col] = z_score
        
        return standardized
    
    def _log_signal_stats(self, signal_df: pd.DataFrame):
        """Log signal statistics and correlations."""
        logger.info("Signal Statistics:")
        for col in signal_df.columns:
            series = signal_df[col].dropna()
            if len(series) > 0:
                logger.info(f"  {col}: mean={series.mean():.3f}, std={series.std():.3f}, "
                           f"min={series.min():.3f}, max={series.max():.3f}, n={len(series)}")
        
        # Log correlation matrix
        if len(signal_df.columns) > 1:
            corr = signal_df.corr()
            logger.info("Signal Correlations:")
            for i, col1 in enumerate(corr.columns):
                for col2 in corr.columns[i+1:]:
                    logger.info(f"  {col1} vs {col2}: {corr.loc[col1, col2]:.3f}")
    
    def get_signal_returns(
        self,
        signals: pd.DataFrame,
        forward_returns: pd.Series,
        n_quantiles: int = 5
    ) -> pd.DataFrame:
        """
        Calculate signal portfolio returns for signal-level covariance estimation.
        
        For each signal, form long-only portfolio of top quantile stocks.
        Returns the portfolio return for each signal.
        
        Args:
            signals: Signal scores (ticker x signal)
            forward_returns: Forward returns for each ticker
            n_quantiles: Number of quantiles for portfolio formation
            
        Returns:
            Series with return for each signal portfolio
        """
        signal_returns = {}
        
        for signal_name in signals.columns:
            signal_scores = signals[signal_name].dropna()
            
            if len(signal_scores) < n_quantiles * 2:
                signal_returns[signal_name] = np.nan
                continue
            
            # Get top quantile stocks
            threshold = signal_scores.quantile(1 - 1/n_quantiles)
            top_stocks = signal_scores[signal_scores >= threshold].index
            
            # Equal-weight portfolio return
            valid_returns = forward_returns.loc[forward_returns.index.isin(top_stocks)].dropna()
            
            if len(valid_returns) > 0:
                signal_returns[signal_name] = valid_returns.mean()
            else:
                signal_returns[signal_name] = np.nan
        
        return pd.Series(signal_returns)
