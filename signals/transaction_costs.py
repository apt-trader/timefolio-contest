"""
TRANSACTION COST MODEL
======================
Models transaction costs for Korean equity market (KOSPI/KOSDAQ).

Based on actual contest transaction data:
- Commission (수수료): ~0.1% per trade (buy or sell)
- Securities Transaction Tax (세금): ~0.23% on sells only
- Total round-trip: ~0.43%

This module integrates with the signal pipeline to:
1. Penalize turnover in the optimization objective
2. Estimate expected transaction costs
3. Provide cost-aware portfolio rebalancing

Author: TimeFolio System - Phase 2
Date: 2025-01-11
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class TransactionCostModel:
    """
    Models transaction costs for Korean equity market.
    
    Cost structure:
    - Commission: Applied to both buys and sells
    - Securities Transaction Tax (STT): Applied only to sells
    - Market impact: Optional, based on trade size relative to ADV
    """
    
    def __init__(
        self,
        commission_rate: float = 0.001,      # 0.1% commission
        tax_rate: float = 0.0023,            # 0.23% securities transaction tax (sells only)
        market_impact_coef: float = 0.1,     # Market impact coefficient
        use_market_impact: bool = False,     # Whether to model market impact
        min_trade_threshold: float = 0.005   # Minimum trade size to execute (0.5%)
    ):
        """
        Initialize transaction cost model.
        
        Args:
            commission_rate: Commission rate per trade (both buy and sell)
            tax_rate: Securities transaction tax rate (sell only)
            market_impact_coef: Coefficient for market impact model
            use_market_impact: Whether to include market impact in cost
            min_trade_threshold: Minimum trade size as fraction of portfolio
        """
        self.commission_rate = commission_rate
        self.tax_rate = tax_rate
        self.market_impact_coef = market_impact_coef
        self.use_market_impact = use_market_impact
        self.min_trade_threshold = min_trade_threshold
        
        # Derived costs
        self.buy_cost = commission_rate
        self.sell_cost = commission_rate + tax_rate
        self.round_trip_cost = self.buy_cost + self.sell_cost
        
        logger.info(f"TransactionCostModel initialized:")
        logger.info(f"  - Commission: {commission_rate:.2%}")
        logger.info(f"  - Tax (sell): {tax_rate:.2%}")
        logger.info(f"  - Buy cost: {self.buy_cost:.2%}")
        logger.info(f"  - Sell cost: {self.sell_cost:.2%}")
        logger.info(f"  - Round-trip: {self.round_trip_cost:.2%}")
    
    def estimate_rebalance_cost(
        self,
        current_weights: pd.Series,
        target_weights: pd.Series,
        portfolio_value: float = 1.0,
        adv: Optional[pd.Series] = None
    ) -> Dict[str, float]:
        """
        Estimate total transaction cost for a rebalance.
        
        Args:
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights
            portfolio_value: Total portfolio value (for market impact)
            adv: Average daily volume per stock (for market impact)
            
        Returns:
            Dict with cost breakdown
        """
        # Align weights
        all_tickers = current_weights.index.union(target_weights.index)
        current = current_weights.reindex(all_tickers).fillna(0)
        target = target_weights.reindex(all_tickers).fillna(0)
        
        # Calculate trades
        trades = target - current
        buys = trades[trades > 0]
        sells = trades[trades < 0].abs()
        
        # Filter small trades
        buys = buys[buys >= self.min_trade_threshold]
        sells = sells[sells >= self.min_trade_threshold]
        
        # Calculate costs
        buy_commission = buys.sum() * self.commission_rate
        sell_commission = sells.sum() * self.commission_rate
        sell_tax = sells.sum() * self.tax_rate
        
        total_commission = buy_commission + sell_commission
        total_tax = sell_tax
        
        # Market impact (optional)
        market_impact = 0.0
        if self.use_market_impact and adv is not None:
            # Simple square-root market impact model
            # Impact = coef * sqrt(trade_value / ADV)
            for ticker in buys.index:
                if ticker in adv.index and adv[ticker] > 0:
                    trade_value = buys[ticker] * portfolio_value
                    impact = self.market_impact_coef * np.sqrt(trade_value / adv[ticker])
                    market_impact += buys[ticker] * impact
            
            for ticker in sells.index:
                if ticker in adv.index and adv[ticker] > 0:
                    trade_value = sells[ticker] * portfolio_value
                    impact = self.market_impact_coef * np.sqrt(trade_value / adv[ticker])
                    market_impact += sells[ticker] * impact
        
        total_cost = total_commission + total_tax + market_impact
        
        # Turnover metrics
        one_way_turnover = (buys.sum() + sells.sum()) / 2
        
        result = {
            'buy_value': buys.sum(),
            'sell_value': sells.sum(),
            'one_way_turnover': one_way_turnover,
            'buy_commission': buy_commission,
            'sell_commission': sell_commission,
            'total_commission': total_commission,
            'sell_tax': sell_tax,
            'market_impact': market_impact,
            'total_cost': total_cost,
            'cost_bps': total_cost * 10000,  # In basis points
            'n_buys': len(buys),
            'n_sells': len(sells)
        }
        
        logger.info(f"Rebalance cost estimate:")
        logger.info(f"  - Turnover: {one_way_turnover:.2%}")
        logger.info(f"  - Commission: {total_commission:.4%}")
        logger.info(f"  - Tax: {total_tax:.4%}")
        logger.info(f"  - Total cost: {total_cost:.4%} ({result['cost_bps']:.1f} bps)")
        
        return result
    
    def get_cost_adjusted_returns(
        self,
        expected_returns: pd.Series,
        current_weights: pd.Series,
        target_weights: pd.Series
    ) -> pd.Series:
        """
        Adjust expected returns for transaction costs.
        
        For stocks being bought: reduce expected return by buy cost
        For stocks being sold: reduce expected return by sell cost
        For unchanged positions: no adjustment
        
        Args:
            expected_returns: Raw expected returns
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights
            
        Returns:
            Cost-adjusted expected returns
        """
        # Align all series
        all_tickers = expected_returns.index
        current = current_weights.reindex(all_tickers).fillna(0)
        target = target_weights.reindex(all_tickers).fillna(0)
        
        # Calculate trade direction
        trades = target - current
        
        # Adjust returns
        adjusted_returns = expected_returns.copy()
        
        # For buys (positive trades), subtract buy cost
        buy_mask = trades > self.min_trade_threshold
        adjusted_returns[buy_mask] -= self.buy_cost
        
        # For sells (negative trades), subtract sell cost
        sell_mask = trades < -self.min_trade_threshold
        adjusted_returns[sell_mask] -= self.sell_cost
        
        return adjusted_returns
    
    def calculate_breakeven_holding_period(
        self,
        expected_alpha: float,
        turnover: float
    ) -> float:
        """
        Calculate minimum holding period to break even on transaction costs.
        
        Args:
            expected_alpha: Expected alpha per period (e.g., weekly)
            turnover: Expected one-way turnover per rebalance
            
        Returns:
            Breakeven holding period in number of periods
        """
        if expected_alpha <= 0:
            return float('inf')
        
        cost_per_rebalance = turnover * self.round_trip_cost
        breakeven_periods = cost_per_rebalance / expected_alpha
        
        return breakeven_periods
    
    def get_optimal_rebalance_frequency(
        self,
        expected_alpha_decay: float = 0.1,
        base_turnover: float = 0.20
    ) -> Dict[str, float]:
        """
        Estimate optimal rebalancing frequency given alpha decay and costs.
        
        Args:
            expected_alpha_decay: Rate at which alpha decays per period
            base_turnover: Expected turnover per rebalance
            
        Returns:
            Dict with optimal frequency analysis
        """
        # Simple model: balance alpha capture vs transaction costs
        # More frequent = more alpha captured but more costs
        # Less frequent = less costs but alpha decays
        
        cost_per_rebalance = base_turnover * self.round_trip_cost
        
        # Optimal frequency minimizes: alpha_decay * periods + cost_per_rebalance / periods
        # Derivative = alpha_decay - cost_per_rebalance / periods^2 = 0
        # periods = sqrt(cost_per_rebalance / alpha_decay)
        
        if expected_alpha_decay > 0:
            optimal_periods = np.sqrt(cost_per_rebalance / expected_alpha_decay)
        else:
            optimal_periods = float('inf')
        
        return {
            'optimal_periods': optimal_periods,
            'cost_per_rebalance': cost_per_rebalance,
            'alpha_decay_rate': expected_alpha_decay,
            'recommendation': 'weekly' if optimal_periods <= 1.5 else 'biweekly' if optimal_periods <= 3 else 'monthly'
        }


class CostAwarePortfolioMapper:
    """
    Extends PortfolioMapper with transaction cost awareness.
    """
    
    def __init__(
        self,
        cost_model: TransactionCostModel,
        cost_penalty_weight: float = 1.0
    ):
        """
        Initialize cost-aware mapper.
        
        Args:
            cost_model: TransactionCostModel instance
            cost_penalty_weight: Weight for cost penalty in objective
        """
        self.cost_model = cost_model
        self.cost_penalty_weight = cost_penalty_weight
    
    def apply_cost_penalty(
        self,
        target_weights: pd.Series,
        current_weights: pd.Series,
        expected_returns: pd.Series,
        holding_period: int = 1
    ) -> pd.Series:
        """
        Apply transaction cost penalty to expected returns.
        
        Reduces expected returns for positions that require trading,
        making the optimizer prefer lower-turnover solutions.
        
        Args:
            target_weights: Proposed target weights
            current_weights: Current portfolio weights
            expected_returns: Expected returns per stock
            holding_period: Expected holding period in weeks
            
        Returns:
            Cost-penalized expected returns
        """
        # Align indices
        all_tickers = target_weights.index.union(current_weights.index)
        target = target_weights.reindex(all_tickers).fillna(0)
        current = current_weights.reindex(all_tickers).fillna(0)
        returns = expected_returns.reindex(all_tickers).fillna(0)
        
        # Calculate trade sizes
        trades = (target - current).abs()
        
        # Cost penalty = trade_size * cost_rate / holding_period
        # Amortize cost over expected holding period
        buy_mask = target > current
        sell_mask = target < current
        
        cost_penalty = pd.Series(0.0, index=all_tickers)
        cost_penalty[buy_mask] = trades[buy_mask] * self.cost_model.buy_cost
        cost_penalty[sell_mask] = trades[sell_mask] * self.cost_model.sell_cost
        
        # Amortize over holding period
        cost_penalty = cost_penalty / holding_period
        
        # Apply penalty
        penalized_returns = returns - self.cost_penalty_weight * cost_penalty
        
        return penalized_returns.loc[target_weights.index]
    
    def filter_small_trades(
        self,
        target_weights: pd.Series,
        current_weights: pd.Series
    ) -> pd.Series:
        """
        Filter out trades that are too small to be worth the transaction cost.
        
        Args:
            target_weights: Proposed target weights
            current_weights: Current portfolio weights
            
        Returns:
            Filtered target weights
        """
        # Align indices
        all_tickers = target_weights.index.union(current_weights.index)
        target = target_weights.reindex(all_tickers).fillna(0)
        current = current_weights.reindex(all_tickers).fillna(0)
        
        # Calculate trade sizes
        trades = target - current
        
        # Keep current weight if trade is too small
        filtered = target.copy()
        small_trade_mask = trades.abs() < self.cost_model.min_trade_threshold
        filtered[small_trade_mask] = current[small_trade_mask]
        
        # Re-normalize
        if filtered.sum() > 0:
            filtered = filtered / filtered.sum()
        
        # Return only non-zero weights
        filtered = filtered[filtered > 0]
        
        n_filtered = small_trade_mask.sum()
        if n_filtered > 0:
            logger.info(f"Filtered {n_filtered} small trades below {self.cost_model.min_trade_threshold:.2%}")
        
        return filtered


def create_korean_cost_model() -> TransactionCostModel:
    """
    Create transaction cost model calibrated for Korean market.
    
    Based on actual contest transaction data:
    - Commission: ~0.1%
    - Securities Transaction Tax: ~0.23% (sells only)
    """
    return TransactionCostModel(
        commission_rate=0.001,      # 0.1%
        tax_rate=0.0023,            # 0.23%
        market_impact_coef=0.1,
        use_market_impact=False,    # Disable for now, can enable later
        min_trade_threshold=0.005   # 0.5% minimum trade
    )
