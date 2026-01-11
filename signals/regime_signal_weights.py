"""
REGIME-CONDITIONAL SIGNAL WEIGHTS
=================================
Adjusts signal weights based on detected market regime.

Market Regimes:
1. Bull (Risk-On): Favor Momentum, Growth
2. Bear (Risk-Off): Favor LowVol, Quality, Value
3. High Volatility: Reduce all exposures, favor LowVol
4. Low Volatility: Normal signal weights

This module provides:
1. Simple regime detection based on market indicators
2. Regime-conditional signal weight adjustments
3. Smooth transitions between regimes

Author: TimeFolio System - Phase 2
Date: 2025-01-11
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime classification."""
    BULL = "bull"
    BEAR = "bear"
    HIGH_VOL = "high_vol"
    LOW_VOL = "low_vol"
    NEUTRAL = "neutral"


class RegimeDetector:
    """
    Detects market regime based on price and volatility indicators.
    
    Uses simple, robust indicators:
    1. Trend: Price vs moving average
    2. Volatility: Current vs historical volatility
    3. Momentum: Recent market returns
    """
    
    def __init__(
        self,
        trend_window: int = 63,           # ~3 months for trend
        vol_window: int = 21,             # ~1 month for volatility
        vol_lookback: int = 252,          # 1 year for vol comparison
        momentum_window: int = 21,        # 1 month momentum
        vol_threshold_high: float = 1.5,  # 1.5x historical vol = high vol
        vol_threshold_low: float = 0.7,   # 0.7x historical vol = low vol
        trend_threshold: float = 0.02     # 2% above/below MA for trend
    ):
        """
        Initialize regime detector.
        
        Args:
            trend_window: Window for trend moving average
            vol_window: Window for current volatility
            vol_lookback: Lookback for historical volatility comparison
            momentum_window: Window for momentum calculation
            vol_threshold_high: Threshold for high volatility regime
            vol_threshold_low: Threshold for low volatility regime
            trend_threshold: Threshold for bull/bear trend detection
        """
        self.trend_window = trend_window
        self.vol_window = vol_window
        self.vol_lookback = vol_lookback
        self.momentum_window = momentum_window
        self.vol_threshold_high = vol_threshold_high
        self.vol_threshold_low = vol_threshold_low
        self.trend_threshold = trend_threshold
        
        logger.info(f"RegimeDetector initialized:")
        logger.info(f"  - Trend window: {trend_window}d")
        logger.info(f"  - Vol window: {vol_window}d")
        logger.info(f"  - Vol thresholds: low={vol_threshold_low}, high={vol_threshold_high}")
    
    def detect_regime(
        self,
        market_prices: pd.Series,
        market_returns: Optional[pd.Series] = None
    ) -> Dict[str, any]:
        """
        Detect current market regime.
        
        Args:
            market_prices: Market index prices (e.g., KOSPI)
            market_returns: Market returns (optional, will calculate if not provided)
            
        Returns:
            Dict with regime classification and indicators
        """
        if len(market_prices) < self.vol_lookback:
            logger.warning("Insufficient data for regime detection, using NEUTRAL")
            return {
                'regime': MarketRegime.NEUTRAL,
                'trend_score': 0.0,
                'vol_ratio': 1.0,
                'momentum': 0.0,
                'confidence': 0.0
            }
        
        # Calculate returns if not provided
        if market_returns is None:
            market_returns = market_prices.pct_change().dropna()
        
        # 1. Trend indicator: Price vs MA
        ma = market_prices.rolling(window=self.trend_window).mean()
        current_price = market_prices.iloc[-1]
        current_ma = ma.iloc[-1]
        trend_score = (current_price / current_ma) - 1  # Positive = above MA
        
        # 2. Volatility indicator: Current vs historical
        current_vol = market_returns.iloc[-self.vol_window:].std() * np.sqrt(252)
        historical_vol = market_returns.iloc[-self.vol_lookback:].std() * np.sqrt(252)
        vol_ratio = current_vol / historical_vol if historical_vol > 0 else 1.0
        
        # 3. Momentum indicator: Recent returns
        momentum = market_returns.iloc[-self.momentum_window:].sum()
        
        # Classify regime
        regime = self._classify_regime(trend_score, vol_ratio, momentum)
        
        # Calculate confidence (how clear is the signal)
        confidence = self._calculate_confidence(trend_score, vol_ratio, momentum)
        
        result = {
            'regime': regime,
            'trend_score': trend_score,
            'vol_ratio': vol_ratio,
            'momentum': momentum,
            'confidence': confidence,
            'current_vol': current_vol,
            'historical_vol': historical_vol
        }
        
        logger.info(f"Regime detected: {regime.value}")
        logger.info(f"  - Trend score: {trend_score:.2%}")
        logger.info(f"  - Vol ratio: {vol_ratio:.2f}")
        logger.info(f"  - Momentum: {momentum:.2%}")
        logger.info(f"  - Confidence: {confidence:.2f}")
        
        return result
    
    def _classify_regime(
        self,
        trend_score: float,
        vol_ratio: float,
        momentum: float
    ) -> MarketRegime:
        """Classify regime based on indicators."""
        
        # High volatility takes precedence
        if vol_ratio > self.vol_threshold_high:
            return MarketRegime.HIGH_VOL
        
        # Low volatility
        if vol_ratio < self.vol_threshold_low:
            return MarketRegime.LOW_VOL
        
        # Bull market: above MA and positive momentum
        if trend_score > self.trend_threshold and momentum > 0:
            return MarketRegime.BULL
        
        # Bear market: below MA and negative momentum
        if trend_score < -self.trend_threshold and momentum < 0:
            return MarketRegime.BEAR
        
        return MarketRegime.NEUTRAL
    
    def _calculate_confidence(
        self,
        trend_score: float,
        vol_ratio: float,
        momentum: float
    ) -> float:
        """Calculate confidence in regime classification (0 to 1)."""
        
        # Confidence based on how extreme the indicators are
        trend_confidence = min(abs(trend_score) / 0.10, 1.0)  # Max at 10% deviation
        vol_confidence = min(abs(vol_ratio - 1.0) / 0.5, 1.0)  # Max at 50% deviation
        momentum_confidence = min(abs(momentum) / 0.10, 1.0)  # Max at 10% monthly return
        
        # Average confidence
        confidence = (trend_confidence + vol_confidence + momentum_confidence) / 3
        
        return confidence


class RegimeSignalAdjuster:
    """
    Adjusts signal weights based on detected market regime.
    """
    
    def __init__(
        self,
        regime_adjustments: Optional[Dict[MarketRegime, Dict[str, float]]] = None,
        adjustment_strength: float = 0.5,  # How much to adjust (0 = none, 1 = full)
        smooth_transitions: bool = True,
        transition_speed: float = 0.3      # Speed of regime transition (0-1)
    ):
        """
        Initialize regime signal adjuster.
        
        Args:
            regime_adjustments: Dict of regime -> signal adjustments
            adjustment_strength: How strongly to apply adjustments
            smooth_transitions: Whether to smooth regime transitions
            transition_speed: Speed of transition between regimes
        """
        self.adjustment_strength = adjustment_strength
        self.smooth_transitions = smooth_transitions
        self.transition_speed = transition_speed
        
        # Default regime adjustments (multipliers for each signal)
        self.regime_adjustments = regime_adjustments or {
            MarketRegime.BULL: {
                'Value': 0.8,       # Reduce value in bull markets
                'Quality': 0.9,    # Slightly reduce quality
                'Momentum': 1.3,   # Increase momentum
                'LowVol': 0.7,     # Reduce defensive
                'Growth': 1.2      # Increase growth
            },
            MarketRegime.BEAR: {
                'Value': 1.2,      # Increase value in bear markets
                'Quality': 1.3,    # Increase quality
                'Momentum': 0.6,   # Reduce momentum (momentum crashes)
                'LowVol': 1.4,     # Increase defensive
                'Growth': 0.7      # Reduce growth
            },
            MarketRegime.HIGH_VOL: {
                'Value': 1.0,      # Neutral value
                'Quality': 1.2,    # Increase quality
                'Momentum': 0.5,   # Strongly reduce momentum
                'LowVol': 1.5,     # Strongly increase defensive
                'Growth': 0.8      # Reduce growth
            },
            MarketRegime.LOW_VOL: {
                'Value': 1.0,      # Neutral
                'Quality': 1.0,
                'Momentum': 1.1,   # Slightly increase momentum
                'LowVol': 0.9,     # Slightly reduce defensive
                'Growth': 1.1      # Slightly increase growth
            },
            MarketRegime.NEUTRAL: {
                'Value': 1.0,
                'Quality': 1.0,
                'Momentum': 1.0,
                'LowVol': 1.0,
                'Growth': 1.0
            }
        }
        
        # Store previous adjustments for smooth transitions
        self.previous_adjustments: Optional[Dict[str, float]] = None
        
        logger.info(f"RegimeSignalAdjuster initialized:")
        logger.info(f"  - Adjustment strength: {adjustment_strength}")
        logger.info(f"  - Smooth transitions: {smooth_transitions}")
    
    def adjust_signal_weights(
        self,
        signal_weights: pd.Series,
        regime_info: Dict[str, any]
    ) -> pd.Series:
        """
        Adjust signal weights based on regime.
        
        Args:
            signal_weights: Original signal weights from optimizer
            regime_info: Regime detection result from RegimeDetector
            
        Returns:
            Adjusted signal weights
        """
        regime = regime_info['regime']
        confidence = regime_info.get('confidence', 1.0)
        
        # Get adjustments for this regime
        adjustments = self.regime_adjustments.get(regime, {})
        
        # Apply adjustment strength and confidence
        effective_strength = self.adjustment_strength * confidence
        
        # Calculate adjusted weights
        adjusted_weights = signal_weights.copy()
        
        for signal in signal_weights.index:
            if signal in adjustments:
                multiplier = adjustments[signal]
                # Blend between 1.0 (no adjustment) and multiplier based on strength
                effective_multiplier = 1.0 + (multiplier - 1.0) * effective_strength
                adjusted_weights[signal] *= effective_multiplier
        
        # Smooth transition if enabled
        if self.smooth_transitions and self.previous_adjustments is not None:
            for signal in adjusted_weights.index:
                if signal in self.previous_adjustments:
                    prev = self.previous_adjustments[signal]
                    curr = adjusted_weights[signal]
                    adjusted_weights[signal] = (
                        self.transition_speed * curr + 
                        (1 - self.transition_speed) * prev
                    )
        
        # Re-normalize to sum to 1
        if adjusted_weights.sum() > 0:
            adjusted_weights = adjusted_weights / adjusted_weights.sum()
        
        # Store for next iteration
        self.previous_adjustments = adjusted_weights.to_dict()
        
        # Log adjustments
        logger.info(f"Signal weights adjusted for {regime.value} regime:")
        for signal in signal_weights.index:
            orig = signal_weights[signal]
            adj = adjusted_weights[signal]
            change = (adj / orig - 1) * 100 if orig > 0 else 0
            logger.info(f"  {signal}: {orig:.1%} -> {adj:.1%} ({change:+.1f}%)")
        
        return adjusted_weights
    
    def get_regime_signal_priors(
        self,
        regime: MarketRegime
    ) -> Dict[str, float]:
        """
        Get signal expected return priors based on regime.
        
        Can be used to adjust expected returns in signal optimizer.
        
        Args:
            regime: Current market regime
            
        Returns:
            Dict of signal -> expected return adjustment
        """
        # Historical signal performance by regime (annualized)
        regime_priors = {
            MarketRegime.BULL: {
                'Value': 0.05,      # Value underperforms in bull
                'Quality': 0.08,
                'Momentum': 0.15,   # Momentum outperforms
                'LowVol': 0.03,     # LowVol underperforms
                'Growth': 0.12
            },
            MarketRegime.BEAR: {
                'Value': 0.08,
                'Quality': 0.10,
                'Momentum': -0.05,  # Momentum crashes
                'LowVol': 0.12,     # LowVol outperforms
                'Growth': 0.02
            },
            MarketRegime.HIGH_VOL: {
                'Value': 0.05,
                'Quality': 0.08,
                'Momentum': -0.02,
                'LowVol': 0.10,
                'Growth': 0.03
            },
            MarketRegime.LOW_VOL: {
                'Value': 0.07,
                'Quality': 0.07,
                'Momentum': 0.10,
                'LowVol': 0.05,
                'Growth': 0.09
            },
            MarketRegime.NEUTRAL: {
                'Value': 0.06,
                'Quality': 0.07,
                'Momentum': 0.08,
                'LowVol': 0.06,
                'Growth': 0.07
            }
        }
        
        return regime_priors.get(regime, regime_priors[MarketRegime.NEUTRAL])
    
    def reset(self):
        """Reset state for fresh start."""
        self.previous_adjustments = None


def create_default_regime_system() -> Tuple[RegimeDetector, RegimeSignalAdjuster]:
    """
    Create default regime detection and adjustment system.
    
    Returns:
        Tuple of (RegimeDetector, RegimeSignalAdjuster)
    """
    detector = RegimeDetector(
        trend_window=63,
        vol_window=21,
        vol_lookback=252,
        momentum_window=21,
        vol_threshold_high=1.5,
        vol_threshold_low=0.7,
        trend_threshold=0.02
    )
    
    adjuster = RegimeSignalAdjuster(
        adjustment_strength=0.5,
        smooth_transitions=True,
        transition_speed=0.3
    )
    
    return detector, adjuster
