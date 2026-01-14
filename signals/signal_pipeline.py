"""
SIGNAL PIPELINE
===============
Main orchestration module for the signal-based portfolio construction.

Integrates:
1. SignalConstructor - Builds 5 orthogonal signals
2. SignalOptimizer - MVO on signals
3. PortfolioMapper - Maps signal weights to stock weights

This is the main entry point for the refactored system.

Author: TimeFolio System Refactor
Date: 2025-01-11
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import logging

from signals.signal_constructor import SignalConstructor
from signals.signal_optimizer import SignalOptimizer, SignalReturnEstimator
from signals.portfolio_mapper import PortfolioMapper, TurnoverManager
from signals.transaction_costs import TransactionCostModel, CostAwarePortfolioMapper, create_korean_cost_model
from signals.regime_signal_weights import RegimeDetector, RegimeSignalAdjuster, MarketRegime, create_default_regime_system
from signals.ic_monitor import ICMonitor, compute_forward_returns

logger = logging.getLogger(__name__)


class SignalPipeline:
    """
    End-to-end signal-based portfolio construction pipeline.
    
    Flow:
    1. Construct signals from raw data
    2. Estimate signal expected returns and covariance
    3. Optimize signal weights via MVO
    4. Map signal weights to stock weights
    5. Apply turnover constraints
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize signal pipeline with configuration.
        
        Args:
            config: Configuration dictionary with parameters
        """
        self.config = config or self._default_config()
        
        # Initialize components
        self.signal_constructor = SignalConstructor(
            winsorize_limits=self.config.get('winsorize_limits', (0.02, 0.98)),
            momentum_lookback=self.config.get('momentum_lookback', 252),
            momentum_skip=self.config.get('momentum_skip', 21),
            volatility_lookback=self.config.get('volatility_lookback', 63),
            min_history_days=self.config.get('min_history_days', 126)
        )
        
        self.signal_optimizer = SignalOptimizer(
            risk_aversion=self.config.get('risk_aversion', 1.0),
            min_signal_weight=self.config.get('min_signal_weight', 0.05),
            max_signal_weight=self.config.get('max_signal_weight', 0.50),
            turnover_penalty=self.config.get('signal_turnover_penalty', 0.01),
            use_shrinkage=self.config.get('use_shrinkage', True)
        )
        
        self.portfolio_mapper = PortfolioMapper(
            max_positions=self.config.get('max_positions', 15),
            max_weight_per_stock=self.config.get('max_weight_per_stock', 0.15),
            min_weight_per_stock=self.config.get('min_weight_per_stock', 0.02),
            signal_concentration=self.config.get('signal_concentration', 0.2),
            sector_max_weight=self.config.get('sector_max_weight', 0.40),
            sector_limits=self.config.get('sector_limits', {})
        )
        
        self.turnover_manager = TurnoverManager(
            max_turnover=self.config.get('max_turnover', 0.30),
            min_trade_size=self.config.get('min_trade_size', 0.01)
        )
        
        self.signal_return_estimator = SignalReturnEstimator(
            n_quantiles=self.config.get('n_quantiles', 5),
            rebalance_frequency=self.config.get('rebalance_frequency', 'W')
        )
        
        # Transaction cost model (Phase 2)
        self.cost_model = TransactionCostModel(
            commission_rate=self.config.get('commission_rate', 0.001),
            tax_rate=self.config.get('tax_rate', 0.0023),
            market_impact_coef=self.config.get('market_impact_coef', 0.1),
            use_market_impact=self.config.get('use_market_impact', False),
            min_trade_threshold=self.config.get('min_trade_threshold', 0.005)
        )
        
        self.cost_aware_mapper = CostAwarePortfolioMapper(
            cost_model=self.cost_model,
            cost_penalty_weight=self.config.get('cost_penalty_weight', 1.0)
        )
        
        # Regime detection and signal adjustment (Phase 2)
        self.use_regime_adjustment = self.config.get('use_regime_adjustment', True)
        if self.use_regime_adjustment:
            self.regime_detector = RegimeDetector(
                trend_window=self.config.get('regime_trend_window', 63),
                vol_window=self.config.get('regime_vol_window', 21),
                vol_lookback=self.config.get('regime_vol_lookback', 252),
                momentum_window=self.config.get('regime_momentum_window', 21)
            )
            self.regime_adjuster = RegimeSignalAdjuster(
                adjustment_strength=self.config.get('regime_adjustment_strength', 0.5),
                smooth_transitions=self.config.get('regime_smooth_transitions', True),
                transition_speed=self.config.get('regime_transition_speed', 0.3)
            )
        else:
            self.regime_detector = None
            self.regime_adjuster = None
        
        # Storage for historical data
        self.signals_history: Dict[pd.Timestamp, pd.DataFrame] = {}
        self.signal_returns_history: Optional[pd.DataFrame] = None
        self.current_weights: Optional[pd.Series] = None
        
        # IC Monitoring (Phase 1 enhancement)
        ic_history_file = self.config.get('ic_history_file', 'output/ic_history.json')
        self.ic_monitor = ICMonitor(
            ic_threshold=self.config.get('ic_threshold', 0.02),
            rolling_window=self.config.get('ic_rolling_window', 26),
            history_file=ic_history_file
        )
        self.last_signals_for_ic: Optional[pd.DataFrame] = None
        self.last_rebalance_date_for_ic: Optional[pd.Timestamp] = None
        
        logger.info("SignalPipeline initialized")
        logger.info(f"  - Max positions: {self.config.get('max_positions', 15)}")
        logger.info(f"  - Risk aversion: {self.config.get('risk_aversion', 1.0)}")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for the pipeline."""
        return {
            # Signal construction
            'winsorize_limits': (0.02, 0.98),
            'momentum_lookback': 252,
            'momentum_skip': 21,
            'volatility_lookback': 63,
            'min_history_days': 126,
            
            # Signal optimization
            'risk_aversion': 1.0,
            'min_signal_weight': 0.05,
            'max_signal_weight': 0.50,
            'signal_turnover_penalty': 0.01,
            'use_shrinkage': True,
            
            # Portfolio mapping
            'max_positions': 15,
            'max_weight_per_stock': 0.15,
            'min_weight_per_stock': 0.02,
            'signal_concentration': 0.2,
            'sector_max_weight': 0.40,
            
            # Turnover
            'max_turnover': 0.30,
            'min_trade_size': 0.01,
            
            # Signal return estimation
            'n_quantiles': 5,
            'rebalance_frequency': 'W',
            'min_history_periods': 26,  # Minimum 26 weeks (~6 months) for estimation
            
            # Expected return estimation
            'expected_return_method': 'shrinkage',
            'decay_factor': 0.97,
            
            # Transaction costs (Phase 2)
            'commission_rate': 0.001,        # 0.1% commission
            'tax_rate': 0.0023,              # 0.23% securities transaction tax (sells only)
            'market_impact_coef': 0.1,
            'use_market_impact': False,
            'min_trade_threshold': 0.005,    # 0.5% minimum trade
            'cost_penalty_weight': 1.0,      # Weight for cost penalty in objective
            
            # Regime detection (Phase 2)
            'use_regime_adjustment': True,
            'regime_trend_window': 63,
            'regime_vol_window': 21,
            'regime_vol_lookback': 252,
            'regime_momentum_window': 21,
            'regime_adjustment_strength': 0.5,
            'regime_smooth_transitions': True,
            'regime_transition_speed': 0.3,
            
            # IC Monitoring
            'ic_threshold': 0.02,              # Minimum acceptable IC
            'ic_rolling_window': 26,           # Rolling window for IC average
            'ic_history_file': 'output/ic_history.json',
            'ic_forward_horizon': 5            # Days for forward return calculation
        }
    
    def run(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        market_caps: pd.DataFrame,
        fundamentals: pd.DataFrame,
        sector_map: Dict[str, str],
        rebalance_date: pd.Timestamp,
        warmup_mode: bool = False,
        forbidden_tickers: Optional[set] = None,
        trading_values: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Run the full signal pipeline for a rebalance date.
        
        Args:
            prices: Price DataFrame (date x ticker)
            returns: Return DataFrame (date x ticker)
            market_caps: Market cap DataFrame (date x ticker)
            fundamentals: Fundamentals DataFrame (ticker x metrics)
            sector_map: Dict of ticker -> sector
            rebalance_date: Date for rebalancing
            warmup_mode: If True, only build signals without optimization (for history)
            forbidden_tickers: Set of tickers to exclude from portfolio selection
            trading_values: Trading value DataFrame (date x ticker) for liquidity filter
            
        Returns:
            Dict with:
            - 'weights': Final stock weights
            - 'signal_weights': Signal-level weights
            - 'signals': Signal scores
            - 'metrics': Portfolio metrics
        """
        logger.info(f"{'='*60}")
        logger.info(f"Signal Pipeline: {rebalance_date.strftime('%Y-%m-%d')}")
        logger.info(f"{'='*60}")
        
        # Step 1: Construct signals
        logger.info("Step 1: Constructing signals...")
        signals = self.signal_constructor.construct_all_signals(
            prices=prices,
            returns=returns,
            market_caps=market_caps,
            fundamentals=fundamentals,
            date=rebalance_date
        )
        
        if signals.empty:
            logger.error("Signal construction failed")
            return self._empty_result()
        
        # Store in history
        self.signals_history[rebalance_date] = signals.copy()
        
        if warmup_mode:
            logger.info("Warmup mode: signals stored, skipping optimization")
            return {
                'weights': pd.Series(),
                'signal_weights': pd.Series(),
                'signals': signals,
                'metrics': {}
            }
        
        # Step 2: Estimate signal returns and covariance
        logger.info("Step 2: Estimating signal returns and covariance...")
        
        # Update signal returns history
        self.signal_returns_history = self.signal_return_estimator.estimate_historical_returns(
            signals_history=self.signals_history,
            returns=returns
        )
        
        min_periods = self.config.get('min_history_periods', 26)
        
        if self.signal_returns_history.empty or len(self.signal_returns_history) < min_periods:
            logger.warning(f"Insufficient signal history ({len(self.signal_returns_history) if self.signal_returns_history is not None else 0} < {min_periods}), using equal signal weights")
            signal_weights = pd.Series(1.0 / len(signals.columns), index=signals.columns)
        else:
            # Estimate expected returns
            expected_returns = self.signal_optimizer.estimate_signal_expected_returns(
                historical_signal_returns=self.signal_returns_history,
                method=self.config.get('expected_return_method', 'shrinkage'),
                decay_factor=self.config.get('decay_factor', 0.97)
            )
            
            # Estimate covariance
            covariance = self.signal_optimizer.estimate_signal_covariance(
                historical_signal_returns=self.signal_returns_history
            )
            
            # Step 3: Optimize signal weights
            logger.info("Step 3: Optimizing signal weights...")
            signal_weights = self.signal_optimizer.optimize(
                expected_returns=expected_returns,
                covariance=covariance,
                previous_weights=self.signal_optimizer.previous_weights
            )
        
        if signal_weights.empty:
            logger.error("Signal optimization failed")
            return self._empty_result()
        
        # Step 3.5: Apply regime-conditional adjustments (Phase 2)
        regime_info = None
        if self.use_regime_adjustment and self.regime_detector is not None:
            logger.info("Step 3.5: Detecting market regime and adjusting signal weights...")
            
            # Get market index prices for regime detection
            # Use first column of prices as market proxy if market_prices not available
            market_proxy = prices.mean(axis=1)  # Equal-weight market proxy
            
            regime_info = self.regime_detector.detect_regime(market_proxy)
            
            # Adjust signal weights based on regime
            signal_weights = self.regime_adjuster.adjust_signal_weights(
                signal_weights=signal_weights,
                regime_info=regime_info
            )
        
        # Step 4: Map to stock weights
        logger.info("Step 4: Mapping signals to stock weights...")
        
        # Get market caps at rebalance date (use most recent valid data if holiday)
        if rebalance_date in market_caps.index:
            market_caps_at_date = market_caps.loc[rebalance_date]
            # Check if this date has mostly missing data (holiday)
            if (market_caps_at_date > 0).sum() < 100:
                # Find most recent date with valid data
                for i in range(1, min(10, len(market_caps))):
                    prev_date = market_caps.index[-(i+1)] if rebalance_date == market_caps.index[-1] else market_caps.index[market_caps.index.get_loc(rebalance_date) - i]
                    prev_caps = market_caps.loc[prev_date]
                    if (prev_caps > 0).sum() >= 100:
                        logger.info(f"Using market caps from {prev_date.strftime('%Y-%m-%d')} (rebalance date has incomplete data)")
                        market_caps_at_date = prev_caps
                        break
        else:
            market_caps_at_date = market_caps.iloc[-1]
        
        # Filter forbidden tickers from signals BEFORE portfolio mapping
        signals_filtered = signals.copy()
        if forbidden_tickers:
            forbidden_in_signals = [t for t in signals_filtered.index if t in forbidden_tickers]
            if forbidden_in_signals:
                logger.info(f"Excluding {len(forbidden_in_signals)} forbidden tickers from selection")
                signals_filtered = signals_filtered.drop(forbidden_in_signals, errors='ignore')
        
        # Filter stocks with market cap below minimum threshold (100 billion KRW)
        min_market_cap = self.config.get('min_market_cap', 100_000_000_000)  # 100B KRW default
        if min_market_cap > 0:
            valid_caps = market_caps_at_date[market_caps_at_date >= min_market_cap]
            small_cap_tickers = [t for t in signals_filtered.index if t not in valid_caps.index]
            if small_cap_tickers:
                logger.info(f"Excluding {len(small_cap_tickers)} stocks with market cap < {min_market_cap/1e9:.0f}B KRW")
                signals_filtered = signals_filtered.drop(small_cap_tickers, errors='ignore')
        
        # Filter stocks with 5-day average trading value below threshold (3 billion KRW)
        min_trading_value = self.config.get('min_avg_trading_value', 3_000_000_000)  # 3B KRW default
        if min_trading_value > 0 and trading_values is not None and not trading_values.empty:
            # Calculate 5-day average trading value (using last 5 TRADING days, not calendar days)
            # Filter out days with mostly zero values (holidays)
            trading_data = trading_values.loc[:rebalance_date]
            valid_trading_days = trading_data[(trading_data > 0).sum(axis=1) >= 100]  # Days with at least 100 stocks traded
            recent_values = valid_trading_days.tail(5)
            if len(recent_values) > 0:
                avg_5d_value = recent_values.mean()
                valid_liquidity = avg_5d_value[avg_5d_value >= min_trading_value]
                illiquid_tickers = [t for t in signals_filtered.index if t not in valid_liquidity.index]
                if illiquid_tickers:
                    logger.info(f"Excluding {len(illiquid_tickers)} stocks with 5-day avg trading value < {min_trading_value/1e9:.0f}B KRW")
                    signals_filtered = signals_filtered.drop(illiquid_tickers, errors='ignore')
        
        stock_weights = self.portfolio_mapper.map_signals_to_stocks(
            signal_weights=signal_weights,
            signal_scores=signals_filtered,
            sector_map=sector_map,
            market_caps=market_caps_at_date
        )
        
        if stock_weights.empty:
            logger.error("Portfolio mapping failed")
            return self._empty_result()
        
        # Step 5: Apply turnover constraint and transaction cost filtering
        logger.info("Step 5: Applying turnover constraint and cost filtering...")
        
        # Filter small trades that aren't worth the transaction cost
        if self.current_weights is not None:
            stock_weights = self.cost_aware_mapper.filter_small_trades(
                target_weights=stock_weights,
                current_weights=self.current_weights
            )
        
        final_weights = self.turnover_manager.apply_turnover_constraint(
            target_weights=stock_weights,
            current_weights=self.current_weights
        )
        
        # Step 6: Estimate transaction costs
        logger.info("Step 6: Estimating transaction costs...")
        if self.current_weights is not None:
            cost_estimate = self.cost_model.estimate_rebalance_cost(
                current_weights=self.current_weights,
                target_weights=final_weights
            )
            transaction_cost = cost_estimate['total_cost']
            turnover = cost_estimate['one_way_turnover']
        else:
            transaction_cost = 0.0
            turnover = 0.0
        
        # Update current weights
        self.current_weights = final_weights.copy()
        
        # Calculate metrics
        metrics = self._calculate_metrics(
            weights=final_weights,
            signal_weights=signal_weights,
            signals=signals,
            sector_map=sector_map
        )
        
        # Add transaction cost metrics
        metrics['transaction_cost'] = transaction_cost
        metrics['transaction_cost_bps'] = transaction_cost * 10000
        
        # Add regime info if available
        if regime_info is not None:
            metrics['regime'] = regime_info['regime'].value
            metrics['regime_confidence'] = regime_info['confidence']
            metrics['trend_score'] = regime_info['trend_score']
            metrics['vol_ratio'] = regime_info['vol_ratio']
        
        # Step 7: IC Monitoring - compute IC for previous period's signals
        ic_values = {}
        ic_alerts = {}
        if self.last_signals_for_ic is not None and self.last_rebalance_date_for_ic is not None:
            logger.info("Step 7: Computing Information Coefficient for previous signals...")
            
            # Compute forward returns from last rebalance date
            forward_horizon = self.config.get('ic_forward_horizon', 5)
            forward_returns = compute_forward_returns(
                prices=prices,
                current_date=self.last_rebalance_date_for_ic,
                horizon_days=forward_horizon
            )
            
            if not forward_returns.empty:
                # Compute volatilities for risk adjustment
                vol_lookback = self.config.get('volatility_lookback', 63)
                vol_end_idx = prices.index.get_loc(self.last_rebalance_date_for_ic)
                vol_start_idx = max(0, vol_end_idx - vol_lookback)
                recent_returns = returns.iloc[vol_start_idx:vol_end_idx]
                volatilities = recent_returns.std()
                
                # Compute IC
                ic_values = self.ic_monitor.compute_ic(
                    signals=self.last_signals_for_ic,
                    forward_returns=forward_returns,
                    volatilities=volatilities
                )
                
                # Record and check for alerts
                if ic_values:
                    ic_alerts = self.ic_monitor.record_ic(
                        date=self.last_rebalance_date_for_ic,
                        ic_values=ic_values
                    )
                    
                    # Log IC report
                    ic_report = self.ic_monitor.format_ic_report(ic_values, self.last_rebalance_date_for_ic)
                    for line in ic_report.split('\n'):
                        logger.info(line)
        
        # Store current signals for next IC computation
        self.last_signals_for_ic = signals.copy()
        self.last_rebalance_date_for_ic = rebalance_date
        
        # Add IC metrics
        metrics['ic_values'] = ic_values
        metrics['ic_alerts'] = ic_alerts
        if ic_values:
            metrics['ic_mean'] = np.mean([v for v in ic_values.values() if not np.isnan(v)])
        
        logger.info(f"Pipeline complete: {len(final_weights)} positions")
        
        return {
            'weights': final_weights,
            'signal_weights': signal_weights,
            'signals': signals,
            'metrics': metrics,
            'regime_info': regime_info,
            'ic_values': ic_values,
            'ic_alerts': ic_alerts
        }
    
    def warmup(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        market_caps: pd.DataFrame,
        fundamentals_history: Dict[pd.Timestamp, pd.DataFrame],
        rebalance_dates: List[pd.Timestamp]
    ):
        """
        Warmup the pipeline by building signal history.
        
        This should be called before the first live rebalance to establish
        signal return history for covariance estimation.
        
        Args:
            prices: Full price history
            returns: Full return history
            market_caps: Full market cap history
            fundamentals_history: Dict of date -> fundamentals DataFrame
            rebalance_dates: List of historical rebalance dates
        """
        logger.info(f"Warming up pipeline with {len(rebalance_dates)} historical dates")
        
        for date in rebalance_dates:
            # Get fundamentals for this date (use most recent available)
            available_dates = [d for d in fundamentals_history.keys() if d <= date]
            
            if not available_dates:
                logger.warning(f"No fundamentals available for {date}")
                continue
            
            fundamentals_date = max(available_dates)
            fundamentals = fundamentals_history[fundamentals_date]
            
            # Run in warmup mode
            self.run(
                prices=prices.loc[:date],
                returns=returns.loc[:date],
                market_caps=market_caps.loc[:date],
                fundamentals=fundamentals,
                sector_map={},  # Not needed for warmup
                rebalance_date=date,
                warmup_mode=True
            )
        
        logger.info(f"Warmup complete: {len(self.signals_history)} signal snapshots stored")
    
    def _calculate_metrics(
        self,
        weights: pd.Series,
        signal_weights: pd.Series,
        signals: pd.DataFrame,
        sector_map: Dict[str, str]
    ) -> Dict[str, Any]:
        """Calculate portfolio metrics."""
        metrics = {}
        
        # Position metrics
        metrics['n_positions'] = len(weights)
        metrics['max_weight'] = weights.max()
        metrics['min_weight'] = weights.min()
        metrics['hhi'] = (weights ** 2).sum()
        
        # Signal exposure
        for signal in signals.columns:
            signal_exposure = (weights * signals[signal].reindex(weights.index).fillna(0)).sum()
            metrics[f'exposure_{signal}'] = signal_exposure
        
        # Sector concentration
        if sector_map:
            stock_sectors = pd.Series({
                ticker: sector_map.get(ticker, 'Unknown')
                for ticker in weights.index
            })
            sector_weights = weights.groupby(stock_sectors).sum()
            metrics['max_sector_weight'] = sector_weights.max()
            metrics['n_sectors'] = len(sector_weights[sector_weights > 0])
        
        # Turnover (if previous weights exist)
        if self.turnover_manager.previous_weights is not None:
            turnover = self.turnover_manager.calculate_turnover(
                weights, self.turnover_manager.previous_weights
            )
            metrics['turnover'] = turnover
        
        return metrics
    
    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result structure."""
        return {
            'weights': pd.Series(),
            'signal_weights': pd.Series(),
            'signals': pd.DataFrame(),
            'metrics': {}
        }
    
    def get_signal_diagnostics(self) -> Dict[str, Any]:
        """
        Get diagnostic information about signal performance.
        
        Returns:
            Dict with signal statistics and correlations
        """
        if self.signal_returns_history is None or self.signal_returns_history.empty:
            return {}
        
        diagnostics = {}
        
        # Signal return statistics
        for signal in self.signal_returns_history.columns:
            returns = self.signal_returns_history[signal].dropna()
            diagnostics[f'{signal}_mean'] = returns.mean()
            diagnostics[f'{signal}_std'] = returns.std()
            diagnostics[f'{signal}_sharpe'] = returns.mean() / returns.std() * np.sqrt(52) if returns.std() > 0 else 0
            diagnostics[f'{signal}_hit_rate'] = (returns > 0).mean()
        
        # Signal correlations
        corr = self.signal_returns_history.corr()
        for i, sig1 in enumerate(corr.columns):
            for sig2 in corr.columns[i+1:]:
                diagnostics[f'corr_{sig1}_{sig2}'] = corr.loc[sig1, sig2]
        
        return diagnostics
    
    def reset(self):
        """Reset pipeline state for fresh start."""
        self.signals_history = {}
        self.signal_returns_history = None
        self.current_weights = None
        self.signal_optimizer.previous_weights = None
        self.turnover_manager.previous_weights = None
        logger.info("Pipeline state reset")


def create_pipeline_from_config(cfg, rebalance_date=None) -> SignalPipeline:
    """
    Create SignalPipeline from Config object.
    
    Args:
        cfg: Config object from config.py
        rebalance_date: Date for loading sector limits (defaults to today)
        
    Returns:
        Configured SignalPipeline
    """
    import pandas as pd
    from utils.sector_parser import parse_sector_limits_for_date
    
    # Extract from nested config dictionaries
    opt_settings = cfg.optimization_settings
    risk_mgmt = cfg.risk_management
    factor_settings = cfg.factor_settings
    signal_settings = getattr(cfg, 'signal_settings', {}) or {}
    
    # Load per-sector limits from market_sectors.csv
    if rebalance_date is None:
        rebalance_date = pd.Timestamp.now()
    sector_limits = parse_sector_limits_for_date('market_sectors.csv', rebalance_date)
    
    config = {
        # From optimization settings
        'max_positions': signal_settings.get('max_positions', opt_settings.get('max_positions', 15)),
        'max_weight_per_stock': signal_settings.get('max_weight_per_stock', opt_settings.get('individual_limit', 0.15)),
        'risk_aversion': signal_settings.get('risk_aversion', opt_settings.get('risk_aversion', 1.0)),
        'sector_max_weight': signal_settings.get('sector_max_weight', opt_settings.get('sector_limit', 0.40)),
        'sector_limits': sector_limits,  # Per-sector limits from market_sectors.csv
        
        # From signal settings (with fallbacks)
        'max_turnover': signal_settings.get('max_turnover', risk_mgmt.get('turnover_limit', 0.30)),
        'momentum_lookback': signal_settings.get('momentum_lookback', 252),
        'momentum_skip': signal_settings.get('momentum_skip', 21),
        'volatility_lookback': signal_settings.get('volatility_lookback', 63),
        'min_history_days': signal_settings.get('min_history_days', 126),
        
        # Signal optimization
        'min_signal_weight': signal_settings.get('min_signal_weight', 0.05),
        'max_signal_weight': signal_settings.get('max_signal_weight', 0.50),
        'signal_turnover_penalty': signal_settings.get('signal_turnover_penalty', 0.01),
        'use_shrinkage': signal_settings.get('use_shrinkage', True),
        'expected_return_method': signal_settings.get('expected_return_method', 'shrinkage'),
        'decay_factor': signal_settings.get('decay_factor', 0.97),
        'min_history_periods': signal_settings.get('min_history_periods', 26),
        
        # Portfolio mapping
        'min_weight_per_stock': signal_settings.get('min_weight_per_stock', 0.02),
        'signal_concentration': signal_settings.get('signal_concentration', 0.2),
        'n_quantiles': signal_settings.get('n_quantiles', 5),
        'min_trade_size': signal_settings.get('min_trade_size', 0.01),
        'rebalance_frequency': signal_settings.get('rebalance_frequency', 'W'),
        'winsorize_limits': tuple(signal_settings.get('winsorize_limits', [0.02, 0.98])),
        'min_market_cap': signal_settings.get('min_market_cap', 100_000_000_000),  # 100B KRW
        'min_avg_trading_value': signal_settings.get('min_avg_trading_value', 3_000_000_000),  # 3B KRW
        
        # Transaction costs (Phase 2)
        'commission_rate': signal_settings.get('commission_rate', 0.001),
        'tax_rate': signal_settings.get('tax_rate', 0.0023),
        'use_market_impact': signal_settings.get('use_market_impact', False),
        'market_impact_coef': signal_settings.get('market_impact_coef', 0.1),
        'min_trade_threshold': signal_settings.get('min_trade_threshold', 0.01),
        'cost_penalty_weight': signal_settings.get('cost_penalty_weight', 1.0),
        
        # Regime detection (Phase 2)
        'use_regime_adjustment': signal_settings.get('use_regime_adjustment', True),
        'regime_trend_window': signal_settings.get('regime_trend_window', 63),
        'regime_vol_window': signal_settings.get('regime_vol_window', 21),
        'regime_vol_lookback': signal_settings.get('regime_vol_lookback', 252),
        'regime_momentum_window': signal_settings.get('regime_momentum_window', 21),
        'regime_adjustment_strength': signal_settings.get('regime_adjustment_strength', 0.3),
        'regime_smooth_transitions': signal_settings.get('regime_smooth_transitions', True),
        'regime_transition_speed': signal_settings.get('regime_transition_speed', 0.3),
    }
    
    return SignalPipeline(config=config)
