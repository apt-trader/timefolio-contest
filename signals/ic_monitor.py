"""
INFORMATION COEFFICIENT (IC) MONITOR
====================================
Tracks and monitors the Information Coefficient for each trading signal.

IC = correlation(signal / σ, forward_return / σ)

The IC measures the predictive power of a signal. A higher IC indicates
better forecasting ability. This module:
1. Computes IC for each signal after each rebalance
2. Maintains rolling IC history
3. Provides alerts when IC drops below threshold
4. Supports IC analysis and visualization

Based on Grinold's Fundamental Law of Active Management:
SR ≈ IC × √N

Author: TimeFolio System
Date: 2025-01-14
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class ICMonitor:
    """
    Monitors Information Coefficient for trading signals.
    
    The IC is computed as the cross-sectional correlation between
    risk-adjusted signal scores and risk-adjusted forward returns.
    """
    
    def __init__(
        self,
        ic_threshold: float = 0.02,
        rolling_window: int = 26,
        history_file: Optional[str] = None
    ):
        """
        Initialize IC Monitor.
        
        Args:
            ic_threshold: Minimum acceptable IC (alert if below)
            rolling_window: Number of periods for rolling IC calculation
            history_file: Path to store IC history (JSON format)
        """
        self.ic_threshold = ic_threshold
        self.rolling_window = rolling_window
        self.history_file = history_file
        
        # IC history: {signal_name: [(date, ic_value), ...]}
        self.ic_history: Dict[str, List[Tuple[str, float]]] = {}
        
        # Load existing history if available
        if history_file and Path(history_file).exists():
            self._load_history()
        
        logger.info(f"ICMonitor initialized:")
        logger.info(f"  - IC threshold: {ic_threshold}")
        logger.info(f"  - Rolling window: {rolling_window} periods")
    
    def compute_ic(
        self,
        signals: pd.DataFrame,
        forward_returns: pd.Series,
        volatilities: Optional[pd.Series] = None
    ) -> Dict[str, float]:
        """
        Compute Information Coefficient for each signal.
        
        IC_k = cor(signal_k / σ, r_forward / σ)
        
        Args:
            signals: DataFrame with columns as signal names, index as tickers
            forward_returns: Series of forward returns (next period), index as tickers
            volatilities: Series of volatilities for risk adjustment (optional)
        
        Returns:
            Dictionary mapping signal name to IC value
        """
        # Align data
        common_tickers = signals.index.intersection(forward_returns.index)
        if len(common_tickers) < 10:
            logger.warning(f"Too few common tickers ({len(common_tickers)}) for IC calculation")
            return {}
        
        signals_aligned = signals.loc[common_tickers]
        returns_aligned = forward_returns.loc[common_tickers]
        
        # Risk adjustment (if volatilities provided)
        if volatilities is not None:
            vol_aligned = volatilities.loc[common_tickers]
            vol_aligned = vol_aligned.replace(0, np.nan).fillna(vol_aligned.median())
            returns_adjusted = returns_aligned / vol_aligned
        else:
            returns_adjusted = returns_aligned
        
        # Compute IC for each signal
        ic_values = {}
        for signal_name in signals_aligned.columns:
            signal_values = signals_aligned[signal_name]
            
            # Risk-adjust signal if volatilities provided
            if volatilities is not None:
                signal_adjusted = signal_values / vol_aligned
            else:
                signal_adjusted = signal_values
            
            # Remove NaN values
            valid_mask = ~(signal_adjusted.isna() | returns_adjusted.isna())
            if valid_mask.sum() < 10:
                logger.warning(f"Too few valid observations for signal '{signal_name}'")
                ic_values[signal_name] = np.nan
                continue
            
            # Compute Spearman rank correlation (more robust than Pearson)
            ic = signal_adjusted[valid_mask].corr(returns_adjusted[valid_mask], method='spearman')
            ic_values[signal_name] = ic
        
        return ic_values
    
    def record_ic(
        self,
        date: pd.Timestamp,
        ic_values: Dict[str, float]
    ) -> Dict[str, str]:
        """
        Record IC values and check for alerts.
        
        Args:
            date: Date of the IC measurement
            ic_values: Dictionary of signal name to IC value
        
        Returns:
            Dictionary of alerts (signal_name -> alert_message)
        """
        date_str = date.strftime('%Y-%m-%d')
        alerts = {}
        
        for signal_name, ic in ic_values.items():
            if np.isnan(ic):
                continue
            
            # Initialize history for new signals
            if signal_name not in self.ic_history:
                self.ic_history[signal_name] = []
            
            # Record IC
            self.ic_history[signal_name].append((date_str, ic))
            
            # Check threshold
            if ic < self.ic_threshold:
                alerts[signal_name] = f"IC={ic:.4f} < threshold={self.ic_threshold}"
                logger.warning(f"Signal '{signal_name}' IC below threshold: {ic:.4f}")
        
        # Save history
        if self.history_file:
            self._save_history()
        
        return alerts
    
    def get_rolling_ic(self, signal_name: str) -> Optional[float]:
        """
        Get rolling average IC for a signal.
        
        Args:
            signal_name: Name of the signal
        
        Returns:
            Rolling average IC or None if insufficient history
        """
        if signal_name not in self.ic_history:
            return None
        
        history = self.ic_history[signal_name]
        if len(history) < self.rolling_window:
            # Use all available data if less than window
            ic_values = [ic for _, ic in history if not np.isnan(ic)]
        else:
            # Use last rolling_window observations
            ic_values = [ic for _, ic in history[-self.rolling_window:] if not np.isnan(ic)]
        
        if not ic_values:
            return None
        
        return np.mean(ic_values)
    
    def get_ic_summary(self) -> pd.DataFrame:
        """
        Get summary statistics for all signals.
        
        Returns:
            DataFrame with IC statistics per signal
        """
        summary_data = []
        
        for signal_name, history in self.ic_history.items():
            if not history:
                continue
            
            ic_values = [ic for _, ic in history if not np.isnan(ic)]
            if not ic_values:
                continue
            
            rolling_ic = self.get_rolling_ic(signal_name)
            
            summary_data.append({
                'signal': signal_name,
                'latest_ic': ic_values[-1] if ic_values else np.nan,
                'rolling_ic': rolling_ic,
                'mean_ic': np.mean(ic_values),
                'std_ic': np.std(ic_values),
                'min_ic': np.min(ic_values),
                'max_ic': np.max(ic_values),
                'n_observations': len(ic_values),
                'below_threshold': sum(1 for ic in ic_values if ic < self.ic_threshold)
            })
        
        if not summary_data:
            return pd.DataFrame()
        
        return pd.DataFrame(summary_data).set_index('signal')
    
    def get_ic_timeseries(self, signal_name: str) -> pd.Series:
        """
        Get IC time series for a specific signal.
        
        Args:
            signal_name: Name of the signal
        
        Returns:
            Series with date index and IC values
        """
        if signal_name not in self.ic_history:
            return pd.Series()
        
        history = self.ic_history[signal_name]
        dates = [pd.Timestamp(d) for d, _ in history]
        values = [ic for _, ic in history]
        
        return pd.Series(values, index=dates, name=signal_name)
    
    def _save_history(self):
        """Save IC history to JSON file."""
        if not self.history_file:
            return
        
        try:
            with open(self.history_file, 'w') as f:
                json.dump(self.ic_history, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save IC history: {e}")
    
    def _load_history(self):
        """Load IC history from JSON file."""
        if not self.history_file or not Path(self.history_file).exists():
            return
        
        try:
            with open(self.history_file, 'r') as f:
                self.ic_history = json.load(f)
            logger.info(f"Loaded IC history from {self.history_file}")
        except Exception as e:
            logger.error(f"Failed to load IC history: {e}")
            self.ic_history = {}
    
    def format_ic_report(self, ic_values: Dict[str, float], date: pd.Timestamp) -> str:
        """
        Format IC values as a readable report.
        
        Args:
            ic_values: Dictionary of signal name to IC value
            date: Date of measurement
        
        Returns:
            Formatted string report
        """
        lines = [
            f"=== Information Coefficient Report ({date.strftime('%Y-%m-%d')}) ===",
            ""
        ]
        
        for signal_name, ic in sorted(ic_values.items(), key=lambda x: -x[1] if not np.isnan(x[1]) else -999):
            if np.isnan(ic):
                status = "N/A"
            elif ic >= self.ic_threshold:
                status = "OK"
            else:
                status = "⚠️ LOW"
            
            rolling = self.get_rolling_ic(signal_name)
            rolling_str = f"{rolling:.4f}" if rolling is not None else "N/A"
            
            lines.append(f"  {signal_name:12s}: IC={ic:+.4f}  Rolling={rolling_str}  [{status}]")
        
        lines.append("")
        lines.append(f"  Threshold: {self.ic_threshold}")
        
        return "\n".join(lines)


def compute_forward_returns(
    prices: pd.DataFrame,
    current_date: pd.Timestamp,
    horizon_days: int = 5
) -> pd.Series:
    """
    Compute forward returns from current date.
    
    Args:
        prices: DataFrame of prices (date x ticker)
        current_date: Current rebalance date
        horizon_days: Number of days for forward return
    
    Returns:
        Series of forward returns per ticker
    """
    # Find current and future dates
    available_dates = prices.index[prices.index >= current_date]
    if len(available_dates) < 2:
        return pd.Series()
    
    current_idx = prices.index.get_loc(available_dates[0])
    future_idx = min(current_idx + horizon_days, len(prices.index) - 1)
    
    if future_idx <= current_idx:
        return pd.Series()
    
    current_prices = prices.iloc[current_idx]
    future_prices = prices.iloc[future_idx]
    
    # Compute returns
    forward_returns = (future_prices / current_prices - 1).dropna()
    
    return forward_returns
