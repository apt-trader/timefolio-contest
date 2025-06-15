# optimizer.py
import logging
import numpy as np
import pandas as pd
import cvxpy as cp
from typing import Dict, Any

# Configure logging for this module
logger = logging.getLogger(__name__)

class PortfolioOptimizer:
    """
    Performs portfolio optimization using CVXPY. It aims to maximize a
    risk-adjusted return objective subject to a variety of real-world constraints,
    including cardinality (exact number of positions).
    """
    def __init__(self, settings: Dict[str, Any]):
        """
        Initializes the PortfolioOptimizer with given settings.

        Args:
            settings (Dict[str, Any]): A dictionary of optimization parameters
                                       from the configuration.
        """
        self.settings = settings
        self.max_positions = settings.get('max_positions', 12)
        self.max_weight = settings.get('individual_limit', 0.15)
        self.min_weight = settings.get('min_weight', 0.01) # Sensible default
        self.risk_aversion = settings.get('risk_aversion', 1.0)
        self.l2_penalty = settings.get('l2_penalty', 0.1)
        # Add more settings as needed, e.g., sector constraints
        
        logger.info("PortfolioOptimizer initialized.")
        logger.info(f"Optimization settings: Max Pos={self.max_positions}, "
                    f"Max Wgt={self.max_weight}, Risk Aversion={self.risk_aversion}, "
                    f"L2 Penalty={self.l2_penalty}")


    def optimize(self,
                 expected_returns: pd.Series,
                 returns_df: pd.DataFrame,
                 market_caps: pd.Series, # Add market_caps here
                 sector_map: Dict[str, str] = None,
                 sector_limits: Dict[str, float] = None) -> pd.Series:
        """
        Runs the core portfolio optimization logic, now including the small-cap constraint.
        """
        # Align all data sources: expected returns, historical returns, and market caps
        common_tickers = expected_returns.index.intersection(returns_df.columns).intersection(market_caps.index)
        
        mu = expected_returns[common_tickers].values
        rets = returns_df[common_tickers]
        caps = market_caps[common_tickers].values
        n_assets = len(common_tickers)
        
        if n_assets == 0:
            logger.error("No valid assets remaining after alignment. Cannot optimize.")
            return pd.Series(dtype=float)

        cov_matrix = rets.cov().values * 252
        cov_matrix = cov_matrix + 1e-8 * np.eye(n_assets)

        w = cp.Variable(n_assets, name="weights")
        z = cp.Variable(n_assets, boolean=True, name="positions")

        portfolio_return = mu @ w
        portfolio_risk = cp.quad_form(w, cov_matrix)
        l2_regularization = self.l2_penalty * cp.sum_squares(w)
        objective = cp.Maximize(portfolio_return - self.risk_aversion * portfolio_risk - l2_regularization)

        constraints = [
            cp.sum(w) == 1,
            w >= 0,
            w <= z * self.max_weight,
            w >= z * self.min_weight,
            cp.sum(z) <= self.max_positions
        ]

        # --- small-cap constraint ---
        small_cap_threshold = self.settings.get('small_cap_threshold', 1_000_000_000_000) # 1T KRW
        max_small_cap_weight = self.settings.get('max_small_cap_weight', 0.40) # 40%
        
        # Create a boolean mask for which stocks are small caps
        is_small_cap = (caps < small_cap_threshold)
        
        if np.any(is_small_cap):
            # The constraint is the sum of weights for small-cap stocks
            small_cap_sum = cp.sum(w[is_small_cap])
            constraints.append(small_cap_sum <= max_small_cap_weight)
            logger.info(f"Applying small-cap constraint: Sum of weights for {np.sum(is_small_cap)} small-cap stocks <= {max_small_cap_weight:.0%}")
        else:
            logger.info("No small-cap stocks found in the universe. Skipping constraint.")

        # Add sector constraints if provided
        if sector_map and sector_limits:
            pass # Keep your existing implementation here
            unique_sectors = set(sector_map.values())
            for sector in unique_sectors:
                if sector in sector_limits:
                    sector_tickers = [t for t, s in sector_map.items() if s == sector]
                    sector_indices = [common_tickers.get_loc(t) for t in sector_tickers if t in common_tickers]
                    if sector_indices:
                        constraints.append(cp.sum(w[sector_indices]) <= sector_limits[sector])
            

        # --- solve the problem ---
        problem = cp.Problem(objective, constraints)
        
        # Make sure to update the call in main.py to pass market_caps!
        return self._solve_problem(problem, common_tickers, expected_returns)
    
    def _solve_problem(self, problem, tickers, expected_returns):
        # This is a refactoring of the original solver logic into a helper method
        try:
            problem.solve(solver=cp.ECOS_BB, verbose=False, mi_max_iters=1500, mi_abs_gap=1e-3)
            if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                logger.warning(f"MIP solver failed. Status: {problem.status}")
                return self._fallback_heuristic(expected_returns, self.max_positions)
            if problem.variables()[0].value is None: # check weight variable
                logger.warning("MIP solver finished but weights are null. Falling back.")
                return self._fallback_heuristic(expected_returns, self.max_positions)
            
            final_weights = pd.Series(problem.variables()[0].value, index=tickers, name="weight")
            final_weights[final_weights < 1e-6] = 0
            final_weights = final_weights / final_weights.sum()
            
            logger.info("Optimization successful.")
            logger.info(f"Portfolio positions: {(final_weights > 0).sum()}")
            logger.info(f"Top 5 weights:\n{final_weights.nlargest(5)}")
            
            return final_weights.sort_values(ascending=False)
        except Exception as e:
            logger.error(f"An exception occurred during optimization: {e}")
            logger.warning("Falling back to simple heuristic.")
            return self._fallback_heuristic(expected_returns, self.max_positions)

    def _fallback_heuristic(self, expected_returns: pd.Series, n_positions: int) -> pd.Series:
        """
        A simple fallback method: equal-weight the top N stocks by expected return.
        """
        pass # Keep your existing implementation here
        logger.info(f"Executing fallback: Equal-weighting top {n_positions} assets by alpha.")
        
        # Select top N tickers based on the alpha signal
        top_n_tickers = expected_returns.nlargest(n_positions).index
        
        # Assign equal weights
        weights = pd.Series(0.0, index=expected_returns.index)
        weights.loc[top_n_tickers] = 1.0 / n_positions
        
        return weights