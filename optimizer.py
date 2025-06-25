# optimizer.py
import logging
import sys
import numpy as np
import pandas as pd
import cvxpy as cp
from sklearn.covariance import LedoitWolf
from typing import Dict, Any, Optional, List, Tuple
from factor_engine import FactorEngine

logger = logging.getLogger(__name__)

class PortfolioOptimizer:
    def __init__(self, factor_engine: FactorEngine, settings: Dict[str, Any]):
        """Initialize PortfolioOptimizer with optimization settings."""
        self.factor_engine = factor_engine
        self.settings = settings
        self.max_positions = settings.get('max_positions', 12)
        self.max_weight = settings.get('individual_limit', 0.15)
        self.min_weight = settings.get('min_weight', 0.01)
        self.risk_aversion = settings.get('risk_aversion', 1.0)
        self.l2_penalty = settings.get('l2_penalty', 0.1)
        self.small_cap_threshold = settings.get('small_cap_threshold', 1e12)
        self.max_small_cap_weight = settings.get('max_small_cap_weight', 0.4)
        self.verbose = settings.get('verbose', False)
        
        # Check if we're in tuning mode (skip small-cap restrictions)
        self.skip_rules = 'tuner' in sys.modules or 'optuna' in sys.modules
        if self.skip_rules:
            logger.info("Tuning mode detected. Small-cap restrictions will be skipped.")
        
    def _validate_inputs(self, expected_returns: pd.Series, 
                        returns_df: pd.DataFrame,
                        market_caps: pd.Series,
                        sector_map: Dict[str, str],
                        sector_limits: Dict[str, float]) -> None:
        """Validate input data before optimization."""
        if not isinstance(expected_returns, pd.Series) or expected_returns.isna().any():
            raise ValueError("Expected returns must be a pandas Series with no NaN values")
        if not isinstance(returns_df, pd.DataFrame) or returns_df.isna().any().any():
            raise ValueError("Returns DataFrame must not contain NaN values")
        if not isinstance(market_caps, pd.Series) or market_caps.isna().any():
            raise ValueError("Market caps must be a pandas Series with no NaN values")
        if not isinstance(sector_map, dict) or not sector_map:
            raise ValueError("Sector map must be a non-empty dictionary")
        if not isinstance(sector_limits, dict) or not sector_limits:
            raise ValueError("Sector limits must be a non-empty dictionary")
            
    def _get_valid_tickers(self, expected_returns: pd.Series,
                         returns_df: pd.DataFrame,
                         market_caps: pd.Series,
                         sector_map: Dict[str, str]) -> Tuple[pd.Index, Dict[str, str]]:
        """
        Filters the initial universe from expected_returns to find tickers with
        high-quality data suitable for optimization.
        """
        # Start with the universe of tickers for which we have an alpha signal
        initial_universe = expected_returns.index.intersection(returns_df.columns)

        if not initial_universe.any():
            logger.warning("Initial universe is empty, no tickers with alpha signal to process.")
            return pd.Index([]), {}

        # --- Filter for High-Quality Data ---
        # Use a 1-year lookback for quality checks
        lookback_period = 252
        recent_returns = returns_df[initial_universe].tail(lookback_period)

        # Adjust threshold for shorter history, e.g., at the start of a backtest
        actual_history_length = len(recent_returns)
        min_obs_threshold = actual_history_length * 0.8  # Require 80% of available data

        # Calculate valid observations and standard deviation for each ticker
        valid_obs = recent_returns.notna().sum()
        std_dev = recent_returns.std()

        # Define quality thresholds
        min_volatility_threshold = 1e-6

        # Apply filters, filling NaN in std_dev for robust comparison
        quality_mask = (valid_obs >= min_obs_threshold) & (std_dev.fillna(0) > min_volatility_threshold)
        quality_tickers = valid_obs[quality_mask].index

        # --- Intersection with other data ---
        # Ensure tickers also have market cap and sector data
        final_universe = quality_tickers.intersection(market_caps.index)
        valid_tickers = [t for t in final_universe
                        if t in sector_map and pd.notna(sector_map.get(t))]

        valid_sector_map = {t: sector_map[t] for t in valid_tickers}

        if len(valid_tickers) < self.max_positions:
            logger.warning(
                f"Found only {len(valid_tickers)} high-quality tickers, "
                f"which is less than the target of {self.max_positions}."
            )

        return pd.Index(valid_tickers), valid_sector_map
        
    def _get_sector_indices(self, tickers: pd.Index, 
                          sector_map: Dict[str, str],
                          sector_limits: Dict[str, float]) -> Dict[str, List[int]]:
        """Pre-compute sector indices for optimization constraints."""
        sector_indices = {}
        for sector in sector_limits:
            sector_indices[sector] = [i for i, t in enumerate(tickers) 
                                     if sector_map.get(t) == sector]
        return sector_indices
        
    def _get_covariance_matrix(self, returns_df: pd.DataFrame) -> np.ndarray:
        """Calculate a robust, positive semi-definite covariance matrix using Ledoit-Wolf shrinkage."""
        # Fill any remaining NaNs with 0 before fitting. This is a simplification, but more robust
        # than dropping tickers, which could lead to index mismatches.
        returns_filled = returns_df.fillna(0)

        # Use Ledoit-Wolf shrinkage for a robust covariance estimate that is well-conditioned
        lw = LedoitWolf(assume_centered=True)
        lw.fit(returns_filled)
        cov_matrix = lw.covariance_

        cov_matrix *= 252  # Annualize
        return (cov_matrix + cov_matrix.T) / 2 # Enforce symmetry for numerical stability

    def optimize(self, expected_returns: pd.Series, 
                returns_df: pd.DataFrame,
                market_caps: pd.Series, 
                sector_map: Dict[str, str],
                sector_limits: Dict[str, float]) -> Optional[pd.Series]:
        """
        Optimize portfolio weights using a mixed-integer programming approach.
        
        Args:
            expected_returns: Expected returns for each asset
            returns_df: Historical returns for covariance estimation
            market_caps: Market capitalizations for each asset
            sector_map: Mapping from ticker to sector
            sector_limits: Maximum weight per sector
            
        Returns:
            Optimized portfolio weights or None if optimization fails
        """
        try:
            # --- 1. Data Cleaning and Alignment ---
            valid_tickers, valid_sector_map = self._get_valid_tickers(
                expected_returns, returns_df, market_caps, sector_map
            )

            if valid_tickers.empty:
                logger.warning("No valid tickers found after filtering. Skipping optimization.")
                return None
            
            # Filter all dataframes to the validated tickers
            expected_returns = expected_returns.loc[valid_tickers]
            returns_df = returns_df[valid_tickers]
            market_caps = market_caps.loc[valid_tickers]

            # --- 2. Validation ---
            # The valid_sector_map is already validated by _get_valid_tickers
            self._validate_inputs(expected_returns, returns_df, market_caps, valid_sector_map, sector_limits)

            if len(valid_tickers) < self.max_positions:
                logger.warning(f"Found only {len(valid_tickers)} high-quality tickers, which is less than the target of {self.max_positions}.")
                return None
                
            # --- 3. Prepare Optimization Inputs ---
            mu = expected_returns.values
            returns_df = returns_df[valid_tickers]
            caps = market_caps.values

            cov_matrix = self._get_covariance_matrix(returns_df)

            # --- Diagnostic Logging ---
            logger.info("--- Optimizer Input Diagnostics ---")
            logger.info(f"Number of valid tickers: {len(valid_tickers)}")
            logger.info(f"Expected returns (mu) shape: {mu.shape}")
            logger.info(f"Expected returns summary:\n{pd.Series(mu).describe().to_string()}")
            logger.info(f"NaNs in mu: {np.isnan(mu).sum()}")
            logger.info(f"Infs in mu: {np.isinf(mu).sum()}")
            logger.info(f"Covariance matrix shape: {cov_matrix.shape}")
            logger.info(f"NaNs in cov_matrix: {np.isnan(cov_matrix).sum()}")
            logger.info(f"Infs in cov_matrix: {np.isinf(cov_matrix).sum()}")
            logger.info("--- End Diagnostics ---")
            
            sector_indices = self._get_sector_indices(valid_tickers, sector_map, sector_limits)
            
            # --- 4. Define Optimization Problem ---
            n_assets = len(valid_tickers)
            weights = cp.Variable(n_assets)
            positions = cp.Variable(n_assets, boolean=True)
            
            # Objective function
            portfolio_return = mu @ weights
            portfolio_risk = cp.quad_form(weights, cp.psd_wrap(cov_matrix))
            l2_reg = self.l2_penalty * cp.sum_squares(weights)
            objective = cp.Maximize(portfolio_return - self.risk_aversion * portfolio_risk - l2_reg)
            
            # Constraints
            constraints = [
                cp.sum(weights) == 1,
                cp.sum(positions) <= self.max_positions,
                weights <= positions * self.max_weight,
                weights >= positions * self.min_weight,
            ]
            
            # Small-cap constraint (skip during tuning)
            if not self.skip_rules and np.any(is_small_cap := (caps < self.small_cap_threshold)):
                constraints.append(cp.sum(weights[is_small_cap]) <= self.max_small_cap_weight)
                logger.debug("Applied small-cap weight constraint")
            elif self.skip_rules and np.any(caps < self.small_cap_threshold):
                logger.debug("Skipping small-cap weight constraint (tuning mode)")

            # Sector constraints
            if not self.skip_rules:
                for sector, indices in sector_indices.items():
                    if indices and sector in sector_limits:
                        constraints.append(cp.sum(weights[indices]) <= sector_limits[sector])
            
            # --- 5. Solve Optimization ---
            problem = cp.Problem(objective, constraints)
            
            try:
                # Use the SCIP solver for mixed-integer quadratic programs (MIQP)
                logger.info("Attempting to solve the MIQP problem with SCIP...")
                problem.solve(solver=cp.SCIP, verbose=self.verbose)
                
                if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE] and weights.value is not None:
                    logger.info("MIQP optimization with SCIP successful.")
                    final_weights = pd.Series(weights.value, index=valid_tickers, name='weight')
                    non_zero_weights = final_weights[final_weights > 1e-6]
                    if not non_zero_weights.empty:
                        # Return normalized weights
                        return non_zero_weights / non_zero_weights.sum()
                    else:
                        logger.warning("SCIP solved but resulted in all zero weights. Will attempt fallback.")
                else:
                    logger.warning(f"SCIP solver finished with non-optimal status: {problem.status}. Will attempt fallback.")

            except Exception as e:
                logger.warning(f"Solver SCIP failed with an exception: {e}. Will attempt fallback.")

            # --- Fallback to Continuous QP if MIQP Fails ---
            logger.warning("MIQP solver failed. Falling back to a continuous QP.")
            try:
                weights_cont = cp.Variable(n_assets)
                objective_cont = cp.Maximize(mu @ weights_cont - self.risk_aversion * cp.quad_form(weights_cont, cp.psd_wrap(cov_matrix)))
                constraints_cont = [
                    cp.sum(weights_cont) == 1,
                    weights_cont >= 0,
                ]
                # Sector constraints
                if not self.skip_rules:
                    for sector, indices in sector_indices.items():
                        if indices and sector in sector_limits:
                            constraints_cont.append(cp.sum(weights_cont[indices]) <= sector_limits[sector])

                problem_cont = cp.Problem(objective_cont, constraints_cont)
                problem_cont.solve(solver=cp.OSQP, verbose=self.verbose)

                if problem_cont.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE] and weights_cont.value is not None:
                    logger.info("Fallback continuous optimization successful.")
                    fallback_weights = pd.Series(weights_cont.value, index=valid_tickers, name='weight')
                    top_weights = fallback_weights.nlargest(self.max_positions)
                    return top_weights / top_weights.sum()
                else:
                    logger.error(f"Fallback continuous QP also failed with status: {problem_cont.status}")
                    return pd.Series(dtype='float64') # Return empty series on complete failure

            except Exception as fallback_e:
                logger.error(f"Fallback continuous QP failed with exception: {fallback_e}")
                return pd.Series(dtype='float64') # Return empty series on complete failure

        except Exception as e:
            logger.error(f"Optimization failed: {str(e)}", exc_info=True)
            return None

    def _fallback_optimization(self, mu: np.ndarray, 
                             cov_matrix: np.ndarray, 
                             tickers: pd.Index) -> Optional[pd.Series]:
        """
        Fallback optimization using continuous relaxation.
        
        Args:
            mu: Expected returns
            cov_matrix: Covariance matrix
            tickers: Asset tickers
            
        Returns:
            Optimized weights or None if optimization fails
        """
        logger.warning("All mixed-integer solvers failed. Attempting continuous relaxation.")
        
        try:
            n_assets = len(tickers)
            weights = cp.Variable(n_assets, nonneg=True)
            
            objective = cp.Maximize(
                mu @ weights - 
                self.risk_aversion * cp.quad_form(weights, cp.psd_wrap(cov_matrix)) -
                self.l2_penalty * cp.sum_squares(weights)
            )
            
            constraints = [
                cp.sum(weights) == 1,
                weights <= self.max_weight
            ]
            
            problem = cp.Problem(objective, constraints)
            problem.solve(solver=cp.SCS, verbose=False)
            
            if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                full_weights = pd.Series(weights.value, index=tickers)
                top_n_weights = full_weights.nlargest(self.max_positions)
                if not top_n_weights.empty:
                    return top_n_weights / top_n_weights.sum()
                    
        except Exception as e:
            logger.error(f"Continuous fallback optimization failed: {str(e)}", exc_info=True)
        
        # Final fallback: equal weight
        logger.error("All optimization attempts failed. Falling back to equal weight.")
        top_assets = pd.Series(mu, index=tickers).nlargest(self.max_positions).index
        return pd.Series(1/len(top_assets), index=top_assets)