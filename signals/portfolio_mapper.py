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
        sector_limits: Optional[Dict[str, float]] = None,
        small_cap_threshold: float = 1e12,  # 1 trillion KRW
        max_small_cap_weight: float = 0.30  # 30% max for stocks < 1T market cap
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
            small_cap_threshold: Market cap threshold for small-cap classification (default 1T KRW)
            max_small_cap_weight: Maximum aggregate weight for stocks below small_cap_threshold
        """
        self.max_positions = max_positions
        self.max_weight_per_stock = max_weight_per_stock
        self.min_weight_per_stock = min_weight_per_stock
        self.signal_concentration = signal_concentration
        self.sector_max_weight = sector_max_weight
        self.sector_limits = sector_limits or {}
        self.small_cap_threshold = small_cap_threshold
        self.max_small_cap_weight = max_small_cap_weight
        
        logger.info(f"PortfolioMapper initialized:")
        logger.info(f"  - Max positions: {max_positions}")
        logger.info(f"  - Weight bounds: [{min_weight_per_stock}, {max_weight_per_stock}]")
        logger.info(f"  - Signal concentration: top {signal_concentration:.0%}")
        logger.info(f"  - Small cap (<{small_cap_threshold/1e12:.0f}T) max weight: {max_small_cap_weight:.0%}")
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
        
        # Step 2: Select top stocks based on composite score (with sector and market cap constraints)
        selected_stocks = self._select_top_stocks(
            composite_scores,
            n_stocks=self.max_positions * 2,  # Select 2x for filtering
            market_caps=market_caps,
            sector_map=sector_map
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
            composite_scores,
            market_caps
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
        n_stocks: int,
        market_caps: Optional[pd.Series] = None,
        sector_map: Optional[Dict[str, str]] = None
    ) -> List[str]:
        """
        Select top N stocks by composite score, respecting sector limits and large-cap requirements.
        
        Uses a greedy selection that:
        1. Prioritizes large-cap stocks (for 30% small-cap constraint)
        2. Respects sector limits during selection (not just after)
        3. Ensures diversification across sectors
        """
        sorted_scores = composite_scores.sort_values(ascending=False)
        
        if market_caps is None or sector_map is None:
            selected = sorted_scores.head(n_stocks).index.tolist()
            logger.info(f"Selected {len(selected)} candidate stocks")
            return selected
        
        # Track sector allocation during selection
        sector_allocation = {}
        selected = []
        
        # Calculate max stocks per sector based on sector limits and max weight
        # If sector limit is 10% and max weight is 15%, we can have at most 1 stock at 10%
        # But we want some buffer, so allow up to limit / min_weight stocks
        min_weight = 0.05  # Assume minimum 5% weight per stock
        
        for ticker in sorted_scores.index:
            if len(selected) >= n_stocks:
                break
            
            sector = sector_map.get(ticker, 'Unknown')
            sector_limit = self.sector_limits.get(sector, self.sector_max_weight)
            current_sector_alloc = sector_allocation.get(sector, 0)
            
            # Check if adding this stock would potentially exceed sector limit
            # Allow up to sector_limit / min_weight stocks per sector
            max_stocks_in_sector = max(1, int(sector_limit / min_weight))
            stocks_in_sector = sum(1 for t in selected if sector_map.get(t, '') == sector)
            
            if stocks_in_sector >= max_stocks_in_sector:
                continue  # Skip - sector is full
            
            # Check market cap constraint
            is_small_cap = market_caps.get(ticker, 0) < self.small_cap_threshold
            small_cap_count = sum(1 for t in selected if market_caps.get(t, 0) < self.small_cap_threshold)
            
            # Limit small-cap stocks to ensure we can meet 30% constraint
            # With 15% max weight, we need at least 5 large-cap stocks for 70%
            # So limit small-caps to n_stocks - 5
            max_small_caps = max(2, n_stocks - 5)
            if is_small_cap and small_cap_count >= max_small_caps:
                continue  # Skip - too many small-caps
            
            selected.append(ticker)
            sector_allocation[sector] = sector_allocation.get(sector, 0) + 1
        
        # Log selection summary
        large_cap_count = sum(1 for t in selected if market_caps.get(t, 0) >= self.small_cap_threshold)
        small_cap_count = len(selected) - large_cap_count
        logger.info(f"Selected {len(selected)} candidates: {large_cap_count} large-cap (≥1T), {small_cap_count} small-cap (<1T)")
        
        # Log sector distribution
        sector_dist = {}
        for t in selected:
            s = sector_map.get(t, 'Unknown')
            sector_dist[s] = sector_dist.get(s, 0) + 1
        logger.info(f"Sector distribution: {sector_dist}")
        
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
        composite_scores: pd.Series,
        market_caps: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        Apply portfolio constraints:
        - Max positions
        - Max/min weight per stock
        - Sector constraints
        - Small-cap aggregate constraint (< 1T KRW ≤ 30%)
        """
        weights = initial_weights.copy()
        
        # Step 1: Enforce max positions by keeping top stocks
        if len(weights) > self.max_positions:
            # Keep stocks with highest composite scores
            top_tickers = composite_scores.loc[weights.index].nlargest(self.max_positions).index
            weights = weights.loc[top_tickers]
            weights = weights / weights.sum()  # Re-normalize
        
        # Step 2: Enforce max weight constraint (strict: 15% hard cap)
        weights = self._enforce_max_weight_strict(weights)
        
        # Step 3: Enforce min weight constraint (remove tiny positions)
        weights = weights[weights >= self.min_weight_per_stock]
        
        if len(weights) == 0:
            logger.error("All positions below minimum weight")
            return pd.Series()
        
        weights = weights / weights.sum()  # Re-normalize
        
        # Step 4: Enforce small-cap aggregate constraint (< 1T KRW ≤ 30%)
        if market_caps is not None:
            weights = self._apply_small_cap_constraint(weights, market_caps, composite_scores)
        
        # Step 5: Iteratively apply all constraints until satisfied
        # Order matters: small-cap first, then sector, then max weight
        internal_max_weight = self.max_weight_per_stock * 0.99
        internal_sector_buffer = 0.99
        internal_small_cap_buffer = 0.99  # Target 29.7% to ensure final is ≤30%
        
        max_outer_iterations = 20
        for outer_iter in range(max_outer_iterations):
            weights = weights / weights.sum()  # Normalize first
            
            # 1. Apply sector constraints FIRST (drop stocks from over-concentrated sectors)
            if sector_map is not None:
                weights = self._apply_sector_constraints(weights, sector_map, composite_scores, buffer=internal_sector_buffer)
            
            # 2. Apply small-cap constraint (scale down, don't drop - to preserve diversification)
            if market_caps is not None:
                small_cap_tickers = [t for t in weights.index if market_caps.get(t, float('inf')) < self.small_cap_threshold]
                if small_cap_tickers:
                    small_cap_weight = weights.loc[small_cap_tickers].sum()
                    target_small_cap = self.max_small_cap_weight * internal_small_cap_buffer
                    if small_cap_weight > target_small_cap:
                        # Scale down small-cap weights proportionally
                        scale_factor = target_small_cap / small_cap_weight
                        weights.loc[small_cap_tickers] *= scale_factor
                        weights = weights / weights.sum()
            
            # 3. Apply max weight constraint
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
                    if sw > limit + 0.001:
                        sector_ok = False
                        break
            
            small_cap_ok = True
            if market_caps is not None:
                small_cap_tickers = [t for t in weights.index if market_caps.get(t, float('inf')) < self.small_cap_threshold]
                small_cap_weight = weights.loc[small_cap_tickers].sum() if small_cap_tickers else 0
                small_cap_ok = small_cap_weight <= self.max_small_cap_weight + 0.001
            
            if max_weight_ok and sector_ok and small_cap_ok:
                break
        
        # Round to reasonable precision
        weights = weights.round(4)
        
        # Remove any zero weights after rounding
        weights = weights[weights > 0]
        weights = weights / weights.sum()
        
        # Final hard enforcement - clip to exact limits
        weights = self._enforce_max_weight_strict(weights)
        
        # Final small-cap constraint enforcement - replace small-caps with large-caps if needed
        if market_caps is not None:
            weights = self._final_small_cap_enforcement(
                weights, market_caps, composite_scores, sector_map
            )
        
        return weights
    
    def _final_small_cap_enforcement(
        self,
        weights: pd.Series,
        market_caps: pd.Series,
        composite_scores: pd.Series,
        sector_map: Optional[Dict[str, str]]
    ) -> pd.Series:
        """
        Final enforcement of small-cap constraint by replacing small-cap stocks with large-cap alternatives.
        """
        small_cap_tickers = [t for t in weights.index if market_caps.get(t, float('inf')) < self.small_cap_threshold]
        small_cap_weight = weights.loc[small_cap_tickers].sum() if small_cap_tickers else 0
        
        if small_cap_weight <= self.max_small_cap_weight:
            return weights
        
        logger.info(f"Final small-cap enforcement: {small_cap_weight:.1%} > {self.max_small_cap_weight:.0%}")
        
        # Find large-cap stocks not in portfolio
        all_large_caps = [t for t in composite_scores.index if market_caps.get(t, 0) >= self.small_cap_threshold]
        available_large_caps = [t for t in all_large_caps if t not in weights.index]
        
        if not available_large_caps:
            logger.warning("No additional large-cap stocks available for replacement")
            # Just scale down small-caps
            scale = self.max_small_cap_weight / small_cap_weight
            weights.loc[small_cap_tickers] *= scale
            weights = weights / weights.sum()
            return self._enforce_max_weight_strict(weights)
        
        # Sort available large-caps by score
        large_cap_scores = composite_scores.loc[available_large_caps].sort_values(ascending=False)
        
        # Sort small-caps by score (lowest first for replacement)
        small_cap_scores = composite_scores.loc[small_cap_tickers].sort_values()
        
        # Replace lowest-scoring small-caps with highest-scoring large-caps
        for small_ticker in small_cap_scores.index:
            if small_cap_weight <= self.max_small_cap_weight:
                break
            
            small_weight = weights.loc[small_ticker]
            
            # Find a large-cap that can be added without violating sector constraint
            replacement_found = False
            for i, large_ticker in enumerate(large_cap_scores.index):
                if sector_map is not None:
                    large_sector = sector_map.get(large_ticker, 'Unknown')
                    stock_sectors = pd.Series({t: sector_map.get(t, 'Unknown') for t in weights.index})
                    sector_weights = weights.groupby(stock_sectors).sum()
                    sector_limit = self.sector_limits.get(large_sector, self.sector_max_weight)
                    current_sector_weight = sector_weights.get(large_sector, 0)
                    
                    # Check if there's room in this sector for the weight
                    if current_sector_weight + small_weight > sector_limit:
                        continue  # Try next large-cap
                
                # Found a valid replacement
                weights = weights.drop(small_ticker)
                weights[large_ticker] = small_weight
                small_cap_weight -= small_weight
                large_cap_scores = large_cap_scores.drop(large_ticker)
                logger.info(f"Replaced small-cap {small_ticker} with large-cap {large_ticker} ({sector_map.get(large_ticker, '?')})")
                replacement_found = True
                break
            
            if not replacement_found:
                logger.warning(f"No valid large-cap replacement found for {small_ticker}")
        
        weights = weights / weights.sum()
        weights = self._enforce_max_weight_strict(weights)
        
        # Log final result
        final_small_caps = [t for t in weights.index if market_caps.get(t, float('inf')) < self.small_cap_threshold]
        final_weight = weights.loc[final_small_caps].sum() if final_small_caps else 0
        logger.info(f"Small-cap weight after replacement: {final_weight:.1%}")
        
        return weights
    
    def _enforce_max_weight_strict(self, weights: pd.Series) -> pd.Series:
        """
        Strictly enforce max_weight_per_stock constraint (15% hard cap).
        Redistributes excess to stocks below the limit.
        """
        max_iterations = 20
        for _ in range(max_iterations):
            if weights.max() <= self.max_weight_per_stock:
                break
            
            # Cap weights at max
            excess_mask = weights > self.max_weight_per_stock
            excess = weights[excess_mask] - self.max_weight_per_stock
            weights[excess_mask] = self.max_weight_per_stock
            
            # Redistribute excess proportionally to remaining stocks below limit
            remaining_mask = weights < self.max_weight_per_stock
            if remaining_mask.sum() > 0:
                remaining = weights[remaining_mask]
                redistribution = excess.sum() * (remaining / remaining.sum())
                weights.loc[remaining.index] += redistribution
            else:
                # All stocks at max - distribute equally
                weights = weights / weights.sum()
                break
        
        return weights
    
    def _apply_small_cap_constraint(
        self,
        weights: pd.Series,
        market_caps: pd.Series,
        composite_scores: pd.Series,
        sector_map: Optional[Dict[str, str]] = None
    ) -> pd.Series:
        """
        Enforce aggregate small-cap constraint: stocks with market cap < 1T KRW ≤ 30% total.
        Drops lowest-scoring small-cap stocks until constraint is satisfied.
        """
        # Identify small-cap stocks
        small_cap_tickers = [
            t for t in weights.index 
            if market_caps.get(t, float('inf')) < self.small_cap_threshold
        ]
        
        if not small_cap_tickers:
            return weights
        
        small_cap_weight = weights.loc[small_cap_tickers].sum()
        
        if small_cap_weight <= self.max_small_cap_weight:
            return weights
        
        logger.info(f"Small-cap weight {small_cap_weight:.1%} exceeds {self.max_small_cap_weight:.0%} limit, dropping lowest-scoring small-caps...")
        
        # Sort small-cap stocks by composite score (lowest first)
        small_cap_scores = composite_scores.loc[small_cap_tickers].sort_values()
        
        # Drop lowest-scoring small-cap stocks until within limit
        current_small_cap_weight = small_cap_weight
        for ticker in small_cap_scores.index:
            if current_small_cap_weight <= self.max_small_cap_weight:
                break
            current_small_cap_weight -= weights.loc[ticker]
            weights = weights.drop(ticker)
            logger.debug(f"Dropped small-cap {ticker}, remaining small-cap weight: {current_small_cap_weight:.1%}")
        
        # Re-normalize
        if len(weights) > 0:
            weights = weights / weights.sum()
        
        # Log result
        remaining_small_caps = [t for t in weights.index if market_caps.get(t, float('inf')) < self.small_cap_threshold]
        new_small_cap_weight = weights.loc[remaining_small_caps].sum() if remaining_small_caps else 0
        logger.info(f"Small-cap weight adjusted to {new_small_cap_weight:.1%} ({len(remaining_small_caps)} stocks)")
        
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
        Apply sector weight constraints by scaling down over-concentrated sectors.
        
        Uses iterative scaling approach that preserves all stocks but adjusts weights.
        
        Args:
            buffer: Multiplier for limits (e.g., 0.99 = 99% of limit for safety margin)
        """
        # Get sector for each stock
        stock_sectors = pd.Series({
            ticker: sector_map.get(ticker, 'Unknown')
            for ticker in weights.index
        })
        
        # Iteratively apply sector constraints
        max_iterations = 50  # More iterations for convergence with tight limits
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
            
            # Scale down weights in over-concentrated sectors proportionally
            for sector, (sector_weight, limit) in violations.items():
                sector_stocks = stock_sectors[stock_sectors == sector].index
                scale_factor = limit / sector_weight
                weights.loc[sector_stocks] *= scale_factor
            
            # Re-normalize to sum to 1
            weights = weights / weights.sum()
        
        if iteration == max_iterations - 1 and len(violations) > 0:
            # Log final sector weights
            final_sector_weights = weights.groupby(stock_sectors).sum()
            logger.warning(f"Sector constraints not fully satisfied after {max_iterations} iterations")
            logger.warning(f"Final sector weights: {dict(final_sector_weights.round(3))}")
        
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
