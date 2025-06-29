import pandas as pd
import numpy as np
import statsmodels.api as sm
import cvxpy as cp
from sklearn.covariance import LedoitWolf
from config import Config
from data_manager import DataManager
import logging
from typing import Dict, Any, Optional, List, Tuple

logger = logging.getLogger(__name__)

class PortfolioOptimizer:
    """
    Handles portfolio construction and optimization.
    """
    def __init__(self, cfg: Config, dm: DataManager):
        """
        Initializes the PortfolioOptimizer.
        """
        self.cfg = cfg
        self.dm = dm
        self.model = None
        
        # Load optimization settings
        opt_cfg = self.cfg.optimization_settings
        self.max_positions = opt_cfg.get('max_positions', 50)
        self.max_weight = opt_cfg.get('max_weight', 0.05)
        self.risk_aversion = opt_cfg.get('risk_aversion', 1.0)
        self.l2_penalty = opt_cfg.get('l2_penalty', 0.0)
        self.cov_l2_alpha = opt_cfg.get('cov_l2_alpha', 0.05)
        self.skip_rules = opt_cfg.get('skip_rules', False)
        self.verbose = opt_cfg.get('verbose', False)
        
        # Load risk settings
        risk_cfg = self.cfg.risk_settings
        self.small_cap_threshold = risk_cfg.get('small_cap_threshold', 1e9)
        self.max_small_cap_weight = risk_cfg.get('max_small_cap_weight', 0.1)

        logger.info("PortfolioOptimizer initialized.")

    def set_model(self, model):
        """
        Sets the trained factor model.
        """
        self.model = model
        logger.info("Factor model has been set in the optimizer.")

    def predict_returns(self, factors: pd.DataFrame) -> pd.Series:
        """
        Predicts expected returns using the trained factor model.
        """
        if self.model is None:
            raise ValueError("Model must be set before predicting returns.")
        
        factors_with_const = sm.add_constant(factors, has_constant='add')
        model_factors = self.model.model.exog_names
        factors_aligned = factors_with_const.reindex(columns=model_factors, fill_value=0)
        predicted_returns = self.model.predict(factors_aligned)
        predicted_returns.name = 'expected_return'
        return predicted_returns

    def _get_valid_tickers(self, expected_returns: pd.Series, risk_model: pd.DataFrame, market_caps: pd.Series) -> pd.Index:
        """
        Finds the intersection of tickers that have expected returns, risk model, and market cap data.
        """
        valid_tickers = expected_returns.index.intersection(risk_model.index).intersection(market_caps.index)
        logger.info(f"Found {len(valid_tickers)} valid tickers for optimization.")
        return valid_tickers

    def _validate_inputs(self, expected_returns: pd.Series, risk_model: pd.DataFrame, market_caps: pd.Series, sector_map: Dict[str, str], sector_limits: Dict[str, float]):
        """
        Validates the inputs for the optimization.
        """
        logger.info("Validating optimizer inputs...")
        if not isinstance(expected_returns, pd.Series) or expected_returns.empty:
            raise ValueError("`expected_returns` must be a non-empty pandas Series.")
        if not isinstance(risk_model, pd.DataFrame) or risk_model.empty:
            raise ValueError("`risk_model` must be a non-empty pandas DataFrame.")
        if not isinstance(market_caps, pd.Series) or market_caps.empty:
            raise ValueError("`market_caps` must be a non-empty pandas Series.")
        if not isinstance(sector_map, dict) or not sector_map:
            raise ValueError("`sector_map` must be a non-empty dictionary.")
        if not isinstance(sector_limits, dict) or not sector_limits:
            raise ValueError("`sector_limits` must be a non-empty dictionary.")
        logger.info("Optimizer inputs validated successfully.")

    def _get_sector_indices(self, tickers: pd.Index, sector_map: Dict[str, str], sector_limits: Dict[str, float]) -> Dict[str, List[int]]:
        """
        Get a mapping from sector to the integer indices of tickers in that sector.
        """
        sector_indices = {sector: [] for sector in sector_limits}
        for i, ticker in enumerate(tickers):
            sector = sector_map.get(ticker)
            if sector in sector_indices:
                sector_indices[sector].append(i)
        return sector_indices

    def optimize(self, expected_returns: pd.Series, risk_model: pd.DataFrame, market_caps: pd.Series, sector_map: Dict[str, str], sector_limits: Dict[str, float]) -> Optional[pd.Series]:
        """
        Optimize portfolio weights using a two-step continuous optimization approach.
        1. Pre-select a candidate universe of tickers based on expected returns.
        2. Run a continuous QP optimization on the smaller universe.
        """
        try:
            self._validate_inputs(expected_returns, risk_model, market_caps, sector_map, sector_limits)
            
            # --- 1. Data Cleaning and Alignment ---
            valid_tickers = self._get_valid_tickers(expected_returns, risk_model, market_caps)

            if len(valid_tickers) == 0:
                logger.warning("No valid tickers after filtering. Cannot optimize.")
                return pd.Series(dtype=float)

            # --- 2. Pre-selection of Candidate Universe ---
            # Create a smaller candidate universe to make optimization tractable
            candidate_universe_size = self.max_positions * 5
            if len(valid_tickers) > candidate_universe_size:
                logger.info(f"Reducing optimization universe from {len(valid_tickers)} to top {candidate_universe_size} by expected returns.")
                candidate_tickers = expected_returns.loc[valid_tickers].nlargest(candidate_universe_size).index
            else:
                logger.info(f"Using all {len(valid_tickers)} valid tickers as the candidate universe.")
                candidate_tickers = valid_tickers
            
            if len(candidate_tickers) < self.max_positions:
                logger.warning(f"Candidate universe size ({len(candidate_tickers)}) is less than target positions ({self.max_positions}).")

            # --- 3. Prepare Optimization Inputs for the Candidate Universe ---
            mu = expected_returns.loc[candidate_tickers].values
            cov_matrix = risk_model.loc[candidate_tickers, candidate_tickers].values
            caps = market_caps.loc[candidate_tickers].values
            
            # Gracefully handle tickers missing from the sector map
            sector_map = {ticker: sector_map.get(ticker, 'Unknown') for ticker in candidate_tickers}
            sector_indices = self._get_sector_indices(candidate_tickers, sector_map, sector_limits)

            # --- 4. Define and Solve Continuous QP --- 
            try:
                n_assets = len(candidate_tickers)
                weights = cp.Variable(n_assets)
                
                portfolio_return = mu @ weights
                portfolio_risk = cp.quad_form(weights, cp.psd_wrap(cov_matrix))
                l2_reg = self.l2_penalty * cp.sum_squares(weights)
                objective = cp.Maximize(portfolio_return - self.risk_aversion * portfolio_risk - l2_reg)
                
                constraints = [
                    cp.sum(weights) == 1,
                    weights <= self.max_weight,
                    weights >= 0, # Remove min_weight constraint
                ]

                if not self.skip_rules and np.any(is_small_cap := (caps < self.small_cap_threshold)):
                    constraints.append(cp.sum(weights[is_small_cap]) <= self.max_small_cap_weight)

                if not self.skip_rules:
                    for sector, indices in sector_indices.items():
                        if indices and sector in sector_limits:
                            constraints.append(cp.sum(weights[indices]) <= sector_limits[sector])

                problem = cp.Problem(objective, constraints)
                logger.info(f"Solving continuous QP problem for {n_assets} assets with OSQP...")
                problem.solve(solver=cp.OSQP, verbose=self.verbose)

                if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE] and weights.value is not None:
                    # Select top N positions and re-normalize
                    raw_weights = pd.Series(weights.value, index=candidate_tickers, name='weight')
                    top_weights = raw_weights.nlargest(self.max_positions)
                    if not top_weights.empty and top_weights.sum() > 0:
                        final_weights = top_weights / top_weights.sum()
                        logger.info(f"Successfully optimized portfolio with {len(final_weights)} assets.")
                        return final_weights
                
                logger.warning(f"Continuous QP optimization failed or returned empty weights. Status: {problem.status}")

            except Exception as e:
                logger.error(f"An exception occurred during continuous QP optimization: {e}", exc_info=True)

            # --- Fallback: Equal Weight on Top N by Market Cap from the candidate universe ---
            logger.warning("Optimization failed. Falling back to equal weight on top market cap assets from candidate universe.")
            top_assets = market_caps.loc[candidate_tickers].nlargest(self.max_positions).index
            return pd.Series(1/len(top_assets), index=top_assets)

        except Exception as e:
            logger.error(f"Optimization failed: {str(e)}", exc_info=True)
            return pd.Series(dtype=float)