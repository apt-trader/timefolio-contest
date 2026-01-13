"""
PORTFOLIO MAPPER
================
Maps signal weights to individual stock weights.

Given:
- Signal weights from SignalOptimizer (e.g., 40% Value, 30% Momentum, etc.)
- Signal scores for each stock

Produces:
- Final stock weights (max 15 positions for contest)

The mapping process:
1. For each signal, rank stocks by signal score
2. Weight stocks within each signal portfolio
3. Combine signal portfolios using signal weights
4. Apply cardinality constraint (max 15 positions)
5. Apply position size constraints

Author: TimeFolio System Refactor
Date: 2025-01-11
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import cvxpy as cp
import logging

logger = logging.getLogger(__name__)


class PortfolioMapper:
    """
    Maps signal-level weights to stock-level weights.
    
    Handles the transition from abstract signal allocation
    to concrete stock positions suitable for trading.
    """
    
    def __init__(
        self,
        max_positions: int = 15,
        max_weight_per_stock: float = 0.15,
        min_weight_per_stock: float = 0.02,
        signal_concentration: float = 0.2,
        sector_max_weight: float = 0.40,
        sector_limits: Optional[Dict[str, float]] = None
    ):
        """
        Initialize portfolio mapper.
        
        Args:
            max_positions: Maximum number of stocks in portfolio
            max_weight_per_stock: Maximum weight for any single stock
            min_weight_per_stock: Minimum weight (below this, exclude)
            signal_concentration: Top fraction of stocks per signal (0.2 = top 20%)
            sector_max_weight: Default maximum weight per sector (fallback)
            sector_limits: Per-sector limits from market_sectors.csv {sector: limit}
        """
        self.max_positions = max_positions
        self.max_weight_per_stock = max_weight_per_stock
        self.min_weight_per_stock = min_weight_per_stock
        self.signal_concentration = signal_concentration
        self.sector_max_weight = sector_max_weight
        self.sector_limits = sector_limits or {}
        
        logger.info(f"PortfolioMapper initialized:")
        logger.info(f"  - Max positions: {max_positions}")
        logger.info(f"  - Weight bounds: [{min_weight_per_stock}, {max_weight_per_stock}]")
        logger.info(f"  - Signal concentration: top {signal_concentration:.0%}")
        if self.sector_limits:
            logger.info(f"  - Per-sector limits: {len(self.sector_limits)} sectors from market_sectors.csv")
        else:
            logger.info(f"  - Sector max weight: {sector_max_weight:.0%} (default)")
    
    def map_signals_to_stocks(
        self,
        signal_weights: pd.Series,
        signal_scores: pd.DataFrame,
        sector_map: Optional[Dict[str, str]] = None,
        market_caps: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        Map signal weights to stock weights.
        
        Args:
            signal_weights: Weights per signal (from SignalOptimizer)
            signal_scores: Signal scores per stock (ticker x signal)
            sector_map: Optional dict of ticker -> sector
            market_caps: Optional market caps for size weighting
            
        Returns:
            Stock weights (sums to 1, max positions enforced)
        """
        if signal_weights.empty or signal_scores.empty:
            logger.error("Empty signal weights or scores")
            return pd.Series()
        
        logger.info(f"Mapping {len(signal_weights)} signals to stock weights")
        
        # Step 1: Calculate composite score for each stock
        composite_scores = self._calculate_composite_scores(
            signal_weights, signal_scores
        )
        
        if composite_scores.empty:
            logger.error("No composite scores calculated")
            return pd.Series()
        
        # Step 2: Select top stocks based on composite score
        selected_stocks = self._select_top_stocks(
            composite_scores,
            n_stocks=self.max_positions * 2  # Select 2x for filtering
        )
        
        if len(selected_stocks) == 0:
            logger.error("No stocks selected")
            return pd.Series()
        
        # Step 3: Calculate initial weights (score-weighted)
        initial_weights = self._calculate_initial_weights(
            composite_scores.loc[selected_stocks],
            market_caps.loc[selected_stocks] if market_caps is not None else None
        )
        
        # Step 4: Apply constraints and finalize
        final_weights = self._apply_constraints(
            initial_weights,
            sector_map,
            composite_scores
        )
        
        # Log final portfolio
        self._log_portfolio(final_weights, signal_scores, sector_map)
        
        return final_weights
    
    def _calculate_composite_scores(
        self,
        signal_weights: pd.Series,
        signal_scores: pd.DataFrame
    ) -> pd.Series:
        """
        Calculate composite score for each stock as weighted sum of signal scores.
        """
        # Align signals
        common_signals = signal_weights.index.intersection(signal_scores.columns)
        
        if len(common_signals) == 0:
            logger.error("No common signals between weights and scores")
            return pd.Series()
        
        # Weighted sum of signal scores
        composite = pd.Series(0.0, index=signal_scores.index)
        
        for signal in common_signals:
            weight = signal_weights[signal]
            scores = signal_scores[signal].fillna(0)
            composite += weight * scores
        
        # Drop stocks with no valid scores
        composite = composite[composite != 0]
        
        logger.info(f"Composite scores: {len(composite)} stocks, "
                   f"range [{composite.min():.3f}, {composite.max():.3f}]")
        
        return composite
    
    def _select_top_stocks(
        self,
        composite_scores: pd.Series,
        n_stocks: int
    ) -> List[str]:
        """
        Select top N stocks by composite score.
        """
        # Sort by score descending
        sorted_scores = composite_scores.sort_values(ascending=False)
        
        # Select top N
        selected = sorted_scores.head(n_stocks).index.tolist()
        
        logger.info(f"Selected {len(selected)} candidate stocks")
        
        return selected
    
    def _calculate_initial_weights(
        self,
        scores: pd.Series,
        market_caps: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        Calculate initial weights based on scores.
        
        Uses score-weighted approach with optional market cap adjustment.
        """
        # Shift scores to positive
        min_score = scores.min()
        shifted_scores = scores - min_score + 0.01
        
        # Score-based weights
        weights = shifted_scores / shifted_scores.sum()
        
        # Optional: blend with market cap weights for liquidity
        if market_caps is not None and not market_caps.empty:
            cap_weights = market_caps / market_caps.sum()
            # 70% score, 30% market cap
            weights = 0.7 * weights + 0.3 * cap_weights.reindex(weights.index).fillna(0)
            weights = weights / weights.sum()
        
        return weights
    
    def _apply_constraints(
        self,
        initial_weights: pd.Series,
        sector_map: Optional[Dict[str, str]],
        composite_scores: pd.Series
    ) -> pd.Series:
        """
        Apply portfolio constraints:
        - Max positions
        - Max/min weight per stock
        - Sector constraints
        """
        weights = initial_weights.copy()
        
        # Step 1: Enforce max positions by keeping top stocks
        if len(weights) > self.max_positions:
            # Keep stocks with highest composite scores
            top_tickers = composite_scores.loc[weights.index].nlargest(self.max_positions).index
            weights = weights.loc[top_tickers]
            weights = weights / weights.sum()  # Re-normalize
        
        # Step 2: Enforce max weight constraint
        iterations = 0
        max_iterations = 10
        
        while weights.max() > self.max_weight_per_stock and iterations < max_iterations:
            excess = weights[weights > self.max_weight_per_stock] - self.max_weight_per_stock
            weights[weights > self.max_weight_per_stock] = self.max_weight_per_stock
            
            # Redistribute excess to other stocks proportionally
            remaining = weights[weights < self.max_weight_per_stock]
            if len(remaining) > 0:
                redistribution = excess.sum() * (remaining / remaining.sum())
                weights.loc[remaining.index] += redistribution
            
            iterations += 1
        
        # Step 3: Enforce min weight constraint (remove tiny positions)
        weights = weights[weights >= self.min_weight_per_stock]
        
        if len(weights) == 0:
            logger.error("All positions below minimum weight")
            return pd.Series()
        
        weights = weights / weights.sum()  # Re-normalize
        
        # Step 4: Iteratively apply all constraints until satisfied
        # Use slightly tighter internal limits to account for normalization effects
        internal_max_weight = self.max_weight_per_stock * 0.99  # 14.85% internal -> 15% after normalization
        internal_sector_buffer = 0.99
        
        max_outer_iterations = 15
        for outer_iter in range(max_outer_iterations):
            weights = weights / weights.sum()  # Normalize first
            
            # Apply sector constraints with buffer
            if sector_map is not None:
                weights = self._apply_sector_constraints(weights, sector_map, composite_scores, buffer=internal_sector_buffer)
            
            # Apply max weight constraint
            if weights.max() > internal_max_weight:
                weights = weights.clip(upper=internal_max_weight)
                weights = weights / weights.sum()
            
            # Check if all constraints are satisfied (with actual limits, not internal)
            max_weight_ok = weights.max() <= self.max_weight_per_stock
            
            sector_ok = True
            if sector_map is not None:
                stock_sectors = pd.Series({t: sector_map.get(t, 'Unknown') for t in weights.index})
                sector_weights = weights.groupby(stock_sectors).sum()
                for sector, sw in sector_weights.items():
                    limit = self.sector_limits.get(sector, self.sector_max_weight)
                    if sw > limit:
                        sector_ok = False
                        break
            
            if max_weight_ok and sector_ok:
                break
        
        # Round to reasonable precision
        weights = weights.round(4)
        
        # Remove any zero weights after rounding
        weights = weights[weights > 0]
        weights = weights / weights.sum()
        
        # Final hard enforcement - clip to exact limits (no normalization after this)
        if weights.max() > self.max_weight_per_stock:
            excess = weights.max() - self.max_weight_per_stock
            max_ticker = weights.idxmax()
            weights[max_ticker] = self.max_weight_per_stock
            # Distribute excess to smallest positions
            other_tickers = weights.index.difference([max_ticker])
            if len(other_tickers) > 0:
                weights.loc[other_tickers] += excess / len(other_tickers)
        
        return weights
    
    def _enforce_max_weight(self, weights: pd.Series) -> pd.Series:
        """
        Enforce max_weight_per_stock constraint with iterative redistribution.
        """
        max_iterations = 10
        for _ in range(max_iterations):
            if weights.max() <= self.max_weight_per_stock + 0.0001:
                break
            
            # Cap weights at max
            excess_mask = weights > self.max_weight_per_stock
            excess = weights[excess_mask] - self.max_weight_per_stock
            weights[excess_mask] = self.max_weight_per_stock
            
            # Redistribute excess proportionally to remaining stocks
            remaining_mask = weights < self.max_weight_per_stock
            if remaining_mask.sum() > 0:
                remaining = weights[remaining_mask]
                redistribution = excess.sum() * (remaining / remaining.sum())
                weights.loc[remaining.index] += redistribution
        
        return weights
    
    def _apply_sector_constraints(
        self,
        weights: pd.Series,
        sector_map: Dict[str, str],
        composite_scores: pd.Series,
        buffer: float = 1.0
    ) -> pd.Series:
        """
        Apply sector weight constraints.
        
        Args:
            buffer: Multiplier for limits (e.g., 0.99 = 99% of limit for safety margin)
        """
        # Get sector for each stock
        stock_sectors = pd.Series({
            ticker: sector_map.get(ticker, 'Unknown')
            for ticker in weights.index
        })
        
        # Iteratively apply sector constraints until all are satisfied
        max_iterations = 10
        for iteration in range(max_iterations):
            # Calculate sector weights
            sector_weights = weights.groupby(stock_sectors).sum()
            
            # Check for violations using per-sector limits (with buffer)
            violations = {}
            for sector, sector_weight in sector_weights.items():
                limit = self.sector_limits.get(sector, self.sector_max_weight) * buffer
                if sector_weight > limit + 0.0001:
                    violations[sector] = (sector_weight, limit)
            
            if len(violations) == 0:
                break
            
            if iteration == 0:
                logger.info(f"Sector constraint violations: {[(s, f'{w:.1%}>{l:.1%}') for s,(w,l) in violations.items()]}")
            
            # Reduce weights in over-concentrated sectors
            for sector, (sector_weight, limit) in violations.items():
                sector_stocks = stock_sectors[stock_sectors == sector].index
                
                # Scale down sector stocks to exactly the limit
                scale_factor = limit / sector_weight
                weights.loc[sector_stocks] *= scale_factor
            
            # Re-normalize to sum to 1
            weights = weights / weights.sum()
        
        if iteration == max_iterations - 1:
            logger.warning(f"Sector constraints not fully satisfied after {max_iterations} iterations")
        
        return weights
    
    def _log_portfolio(
        self,
        weights: pd.Series,
        signal_scores: pd.DataFrame,
        sector_map: Optional[Dict[str, str]]
    ):
        """
        Log final portfolio composition.
        """
        logger.info(f"Final portfolio: {len(weights)} positions")
        logger.info(f"Weight range: [{weights.min():.2%}, {weights.max():.2%}]")
        logger.info(f"HHI concentration: {(weights ** 2).sum():.4f}")
        
        # Log top positions
        logger.info("Top positions:")
        for ticker, weight in weights.nlargest(5).items():
            scores_str = ", ".join([
                f"{sig}={signal_scores.loc[ticker, sig]:.2f}"
                for sig in signal_scores.columns
                if ticker in signal_scores.index
            ])
            sector = sector_map.get(ticker, 'Unknown') if sector_map else 'N/A'
            logger.info(f"  {ticker}: {weight:.2%} (sector={sector}, {scores_str})")
        
        # Log sector allocation
        if sector_map:
            stock_sectors = pd.Series({
                ticker: sector_map.get(ticker, 'Unknown')
                for ticker in weights.index
            })
            sector_weights = weights.groupby(stock_sectors).sum().sort_values(ascending=False)
            
            logger.info("Sector allocation:")
            for sector, weight in sector_weights.items():
                logger.info(f"  {sector}: {weight:.2%}")
    
    def optimize_with_risk_model(
        self,
        signal_weights: pd.Series,
        signal_scores: pd.DataFrame,
        expected_returns: pd.Series,
        covariance: pd.DataFrame,
        sector_map: Optional[Dict[str, str]] = None,
        risk_aversion: float = 1.0
    ) -> pd.Series:
        """
        Alternative mapping using stock-level MVO with signal-informed expected returns.
        
        This is a hybrid approach:
        1. Use signal scores to inform expected returns
        2. Use stock-level covariance for risk
        3. Optimize at stock level with cardinality constraint
        
        Args:
            signal_weights: Signal weights from signal optimizer
            signal_scores: Signal scores per stock
            expected_returns: Stock-level expected returns (can be signal-derived)
            covariance: Stock-level covariance matrix
            sector_map: Sector mapping
            risk_aversion: Risk aversion parameter
            
        Returns:
            Optimized stock weights
        """
        # Calculate composite scores
        composite_scores = self._calculate_composite_scores(signal_weights, signal_scores)
        
        if composite_scores.empty:
            return pd.Series()
        
        # Pre-select candidate universe (top stocks by composite score)
        n_candidates = min(self.max_positions * 3, len(composite_scores))
        candidates = composite_scores.nlargest(n_candidates).index.tolist()
        
        # Filter to candidates with valid covariance data
        valid_candidates = [c for c in candidates if c in covariance.index and c in expected_returns.index]
        
        if len(valid_candidates) < self.max_positions:
            logger.warning(f"Only {len(valid_candidates)} valid candidates, using score-based mapping")
            return self.map_signals_to_stocks(signal_weights, signal_scores, sector_map)
        
        # Setup optimization
        n = len(valid_candidates)
        mu = expected_returns.loc[valid_candidates].values
        Sigma = covariance.loc[valid_candidates, valid_candidates].values
        
        # Ensure PSD
        min_eig = np.linalg.eigvalsh(Sigma).min()
        if min_eig < 0:
            Sigma = Sigma + (-min_eig + 1e-6) * np.eye(n)
        
        # Variables
        w = cp.Variable(n)
        
        # Objective
        ret = mu @ w
        risk = cp.quad_form(w, Sigma)
        objective = ret - (risk_aversion / 2) * risk
        
        # Constraints
        constraints = [
            cp.sum(w) == 1,
            w >= 0,
            w <= self.max_weight_per_stock,
        ]
        
        # Sector constraints
        if sector_map:
            sectors = list(set(sector_map.get(t, 'Unknown') for t in valid_candidates))
            for sector in sectors:
                sector_mask = np.array([
                    1.0 if sector_map.get(t, 'Unknown') == sector else 0.0
                    for t in valid_candidates
                ])
                constraints.append(sector_mask @ w <= self.sector_max_weight)
        
        # Solve
        problem = cp.Problem(cp.Maximize(objective), constraints)
        
        try:
            problem.solve(solver=cp.OSQP, verbose=False)
            
            if problem.status not in ['optimal', 'optimal_inaccurate']:
                logger.warning(f"Stock-level optimization failed: {problem.status}")
                return self.map_signals_to_stocks(signal_weights, signal_scores, sector_map)
            
            weights = pd.Series(w.value, index=valid_candidates)
            weights = weights[weights > self.min_weight_per_stock]
            
            # Enforce max positions
            if len(weights) > self.max_positions:
                weights = weights.nlargest(self.max_positions)
            
            weights = weights / weights.sum()
            
            return weights
            
        except Exception as e:
            logger.error(f"Stock-level optimization error: {e}")
            return self.map_signals_to_stocks(signal_weights, signal_scores, sector_map)


class TurnoverManager:
    """
    Manages portfolio turnover to minimize transaction costs.
    """
    
    def __init__(
        self,
        max_turnover: float = 0.30,
        min_trade_size: float = 0.01
    ):
        """
        Initialize turnover manager.
        
        Args:
            max_turnover: Maximum one-way turnover per rebalance
            min_trade_size: Minimum trade size (below this, don't trade)
        """
        self.max_turnover = max_turnover
        self.min_trade_size = min_trade_size
        self.previous_weights: Optional[pd.Series] = None
    
    def apply_turnover_constraint(
        self,
        target_weights: pd.Series,
        current_weights: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        Apply turnover constraint to target weights.
        
        Args:
            target_weights: Desired portfolio weights
            current_weights: Current portfolio weights (if None, use stored)
            
        Returns:
            Adjusted weights respecting turnover constraint
        """
        if current_weights is None:
            current_weights = self.previous_weights
        
        if current_weights is None:
            # First rebalance, no constraint
            self.previous_weights = target_weights.copy()
            return target_weights
        
        # Align indices
        all_tickers = target_weights.index.union(current_weights.index)
        target = target_weights.reindex(all_tickers).fillna(0)
        current = current_weights.reindex(all_tickers).fillna(0)
        
        # Calculate required turnover
        turnover = (target - current).abs().sum() / 2
        
        if turnover <= self.max_turnover:
            # Within limit, use target
            self.previous_weights = target_weights.copy()
            return target_weights
        
        logger.info(f"Turnover {turnover:.2%} exceeds limit {self.max_turnover:.2%}, adjusting")
        
        # Scale down changes
        scale = self.max_turnover / turnover
        adjusted = current + scale * (target - current)
        
        # Clean up small positions
        adjusted = adjusted[adjusted.abs() > self.min_trade_size]
        adjusted = adjusted / adjusted.sum()
        
        self.previous_weights = adjusted.copy()
        
        return adjusted
    
    def calculate_turnover(
        self,
        new_weights: pd.Series,
        old_weights: pd.Series
    ) -> float:
        """
        Calculate one-way turnover between two portfolios.
        """
        all_tickers = new_weights.index.union(old_weights.index)
        new = new_weights.reindex(all_tickers).fillna(0)
        old = old_weights.reindex(all_tickers).fillna(0)
        
        return (new - old).abs().sum() / 2
