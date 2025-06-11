"""
Enhanced Portfolio Metrics and Optimization

This module provides advanced portfolio optimization with metrics like:
- PCR-Sharpe ratio using PCA and Ledoit-Wolf shrinkage
- Herfindahl-Hirschman Index (HHI) for weight and return concentration
- Diversification ratio and effective number of positions
- Risk-based constraints and penalties
"""

import logging
import numpy as np
import pandas as pd
import cvxpy as cp
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from typing import Dict, List, Tuple, Optional, Any
import warnings
from scipy.stats import gmean
import sys
import os

# Add project root to path for module imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.market_data import get_risk_free_rate

# Suppress specific warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Set up logging
logger = logging.getLogger("portfolio_metrics")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

def calculate_pcr_sharpe(returns: pd.DataFrame, 
                         risk_free_rate: float = 0.0,
                         explained_variance: float = 0.95) -> np.ndarray:
    """
    Calculate PCR-Sharpe ratio using PCA and Ledoit-Wolf shrinkage.
    
    Args:
        returns: DataFrame of asset returns (T x N)
        risk_free_rate: Annual risk-free rate (default: 0.0)
        explained_variance: Minimum variance to explain with PCA (0-1)
        
    Returns:
        Array of PCR-Sharpe ratios (N,)
    """
    centered_returns = returns - returns.mean()
    n_assets = len(returns.columns)
    
    # Apply Ledoit-Wolf shrinkage
    lw = LedoitWolf()
    lw.fit(centered_returns)
    sigma = lw.covariance_
    
    # Apply PCA
    pca = PCA(n_components=min(10, n_assets-1))
    pca.fit(centered_returns)
    
    # Calculate number of components to explain desired variance
    explained_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.argmax(explained_variance_ratio >= explained_variance) + 1
    n_components = max(1, min(n_components, len(explained_variance_ratio)))
    
    # Get factor loadings and returns
    components = pca.components_[:n_components].T  # N x K
    factor_returns = centered_returns @ components  # T x K
    
    # Calculate factor covariance with jitter for numerical stability
    if factor_returns.ndim == 1 or factor_returns.shape[1] == 1:
        # Handle single factor case
        factor_cov = np.atleast_2d(np.var(factor_returns, ddof=1) + 1e-10)
    else:
        factor_cov = np.cov(factor_returns, rowvar=False)
        # Add small jitter to diagonal
        jitter = 1e-10 * np.eye(factor_cov.shape[0])
        factor_cov = 0.5 * (factor_cov + factor_cov.T) + jitter
    
    # Calculate factor risk: sum((loadings @ factor_cov) * loadings, axis=1)
    factor_risk = np.sum((components @ factor_cov) * components, axis=1)
    
    # Calculate residual variance
    residuals = centered_returns - factor_returns @ components.T
    residual_var = np.sum(residuals**2, axis=0) / (len(returns) - 1)  # Unbiased estimator
    
    # Calculate PCR-Sharpe with annualization
    annual_factor = 252  # Default for daily data
    mu = returns.mean() * annual_factor - risk_free_rate  # Excess returns
    total_risk = np.sqrt(factor_risk + residual_var + 1e-10)
    
    return mu / (total_risk + 1e-10)

def calculate_diversification_metrics(weights: np.ndarray, 
                                   cov_matrix: np.ndarray,
                                   returns: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate diversification metrics.
    
    Args:
        weights: Portfolio weights (N,)
        cov_matrix: Covariance matrix (N x N)
        returns: DataFrame of asset returns (T x N)
        
    Returns:
        Dictionary containing:
        - hhi_weight: Herfindahl-Hirschman Index for weights
        - effective_n: Effective number of positions
        - diversification_ratio: Risk-weighted average vol / portfolio vol
        - hhi_return: Herfindahl-Hirschman Index for returns
    """
    # Weight HHI
    hhi_weight = (weights ** 2).sum()
    effective_n = 1.0 / hhi_weight if hhi_weight > 1e-10 else len(weights)
    
    # Diversification ratio
    weighted_vol = weights @ np.sqrt(np.diag(cov_matrix))
    port_vol = np.sqrt(weights @ cov_matrix @ weights.T)
    diversification_ratio = weighted_vol / port_vol if port_vol > 1e-8 else 0.0
    
    # Return HHI (P&L concentration)
    if len(returns) > 0:
        pl_contributions = returns * weights
        total_pl = pl_contributions.sum(axis=1)
        # Handle division by zero and NaNs
        valid_days = (total_pl.abs() > 1e-10)
        if valid_days.any():
            pl_shares = pl_contributions[valid_days].div(total_pl[valid_days], axis=0)
            hhi_return = (pl_shares ** 2).sum(axis=1).mean()
        else:
            hhi_return = 0.0
    else:
        hhi_return = 0.0
    
    return {
        'hhi_weight': hhi_weight,
        'effective_n': effective_n,
        'diversification_ratio': diversification_ratio,
        'hhi_return': hhi_return
    }

class PortfolioOptimizer:
    """Enhanced portfolio optimizer with advanced metrics and constraints."""
    
    def __init__(self, config: Dict, risk_free_rate: Optional[float] = None):
        """
        Initialize the portfolio optimizer.
        
        Args:
            config: Configuration dictionary with optimization parameters
            risk_free_rate: Optional risk-free rate (if None, will be fetched dynamically)
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.metrics = {}
        self.post_optimization_hhi = None
        
        # Set risk-free rate (fetch dynamically if not provided)
        self.risk_free_rate = (
            risk_free_rate if risk_free_rate is not None 
            else get_risk_free_rate()
        )
        self.logger.info(f"Using risk-free rate: {self.risk_free_rate:.4f}")
        
        # Initialize metrics
        self.metrics = {
            'hhi_weight': 0.0,
            'effective_n': 0.0,
            'diversification_ratio': 0.0,
            'hhi_return': 0.0,
            'pcr_sharpe': None
        }
    
    def _check_weight_feasibility(self, n_assets: int, n_positions: int, min_weight: float, max_weight: float) -> bool:
        """Check if the weight constraints are feasible."""
        # Check if min_weight * n_positions <= 1 <= max_weight * n_positions
        min_total = min_weight * n_positions
        max_total = max_weight * n_positions
        
        if min_total > 1.0 + 1e-6:  # Allow for small numerical errors
            self.logger.error(
                f"Infeasible weights: min_weight * n_positions = {min_total:.2f} > 1.0. "
                f"Either reduce min_weight below {1.0/n_positions:.4f} or reduce n_positions."
            )
            return False
            
        if max_total < 1.0 - 1e-6:
            self.logger.error(
                f"Infeasible weights: max_weight * n_positions = {max_total:.2f} < 1.0. "
                f"Either increase max_weight above {1.0/n_positions:.4f} or increase n_positions."
            )
            return False
            
        return True
        
    def optimize(self, returns: pd.DataFrame, expected_returns: pd.Series,
                  n_positions: int = 12, use_mip: bool = True) -> pd.Series:
        """
        Optimize portfolio weights with advanced metrics.
        
        Args:
            returns: Historical returns (T x N)
            expected_returns: Expected returns (N,)
            n_positions: Target number of positions
            use_mip: Whether to use mixed-integer programming for exact position count.
                    If False, uses L1 regularization and rounds small weights.
            
        Returns:
            Optimized portfolio weights
        """
        n_assets = len(returns.columns)
        
        # Get config parameters
        cfg = self.config.get('optimization', {})
        min_weight = cfg.get('min_weight', 0.05)
        max_weight = cfg.get('individual_limit', 0.15)
        
        # Check weight feasibility
        if not self._check_weight_feasibility(n_assets, n_positions, min_weight, max_weight):
            self.logger.warning("Falling back to equal-weighted portfolio due to infeasible constraints")
            return self._fallback_weights(returns, n_positions, min_weight, max_weight)
        
        # Calculate covariance with Ledoit-Wolf
        lw = LedoitWolf()
        lw.fit(returns)
        cov = lw.covariance_ * 252  # Annualize
        
        # Ensure numerical stability
        np.fill_diagonal(cov, np.diag(cov) + 1e-6)
        cov = 0.5 * (cov + cov.T)
        
        # Get config parameters
        cfg = self.config.get('optimization', {})
        min_weight = cfg.get('min_weight', 0.05)
        max_weight = cfg.get('individual_limit', 0.15)
        
        # Create CVXPY variables
        z = cp.Variable(n_assets, boolean=True)
        w = cp.Variable(n_assets)
        
        # Calculate metrics with instance's risk-free rate
        pcr_sharpe = calculate_pcr_sharpe(
            returns, 
            risk_free_rate=self.risk_free_rate / 252,  # Convert annual to daily
            explained_variance=0.95
        )
        
        # Portfolio return and risk
        portfolio_return = expected_returns.values @ w
        portfolio_risk = cp.quad_form(w, cov)
        
        # Diversification constraints - DCP compliant
        lambda_hhi = cfg.get('lambda_hhi', 10.0)
        hhi_weight_threshold = cfg.get('hhi_weight_threshold', 0.08)
        
        # DCP-compliant HHI penalty
        hhi_penalty = lambda_hhi * cp.pos(cp.sum_squares(w) - hhi_weight_threshold)
        
        # Objective function - DCP compliant
        objective = cp.Maximize(
            portfolio_return - 
            0.5 * portfolio_risk -  # Risk aversion
            hhi_penalty +  # Convex penalty for HHI
            cfg.get('sharpe_treynor_weight', 0.5) * (pcr_sharpe @ w)  # Linear term
        )
        
        # Constraints - all DCP compliant
        constraints = [
            cp.sum(w) == 1.0,
            w >= 0.0,
            w <= z * max_weight,
            w >= z * min_weight,
            cp.sum(z) == n_positions,
            cp.norm(w, 2) <= 0.4  # L2 norm constraint for diversification
        ]
        
        # Add sector constraints if available - DCP compliant
        if 'sector_limits' in self.config:
            for sector, (sector_max_weight, stocks) in self.config['sector_limits'].items():
                sector_mask = returns.columns.isin(stocks)
                if sector_mask.any():
                    constraints.append(cp.sum(w[sector_mask]) <= sector_max_weight)
        
        # Calculate Return-HHI after optimization (not in the objective)
        # This will be used for post-optimization validation
        self.post_optimization_hhi = None
        
        # Prepare solver options
        solver_kwargs = {
            'max_iters': 1000,
            'abstol': 1e-7,
            'reltol': 1e-6,
            'feastol': 1e-7,
            'abstol_inacc': 5e-5,
            'reltol_inacc': 5e-5,
            'feastol_inacc': 1e-4
        }
        
        # First try with MIP if requested
        if use_mip and n_assets > 1:  # MIP only makes sense for >1 asset
            try:
                prob = cp.Problem(objective, constraints)
                prob.solve(solver='ECOS_BB', **solver_kwargs)
                
                if w.value is None or np.any(np.isnan(w.value)):
                    raise cp.SolverError("MIP solver returned no valid solution")
                    
                # Check if we got a good solution
                weights = pd.Series(w.value, index=returns.columns)
                weights[weights < 1e-6] = 0
                weights = weights / weights.sum()
                
                # If we have too many positions, try convex relaxation
                if (weights > 1e-6).sum() > n_positions + 2:  # Allow small tolerance
                    self.logger.warning("MIP solution has too many positions, trying convex relaxation")
                    raise cp.SolverError("Too many positions in MIP solution")
                    
                return self._process_solution(weights, returns, cov, pcr_sharpe)
                
            except (cp.SolverError, ValueError) as e:
                self.logger.warning(f"MIP optimization failed: {e}")
                if n_assets <= 30:  # Only warn for small problems
                    self.logger.warning("Falling back to convex relaxation")
                use_mip = False
        
        # If MIP failed or not requested, try convex relaxation
        if not use_mip or n_assets == 1:
            try:
                # Remove integer constraints
                constraints = [c for c in constraints if not (hasattr(c, 'is_constant') and c.is_constant())]
                
                # Add L1 regularization to encourage sparsity
                l1_penalty = 0.01 * cp.norm(w, 1)
                objective = cp.Maximize(objective.expr - l1_penalty)
                
                prob = cp.Problem(objective, constraints)
                prob.solve(solver='ECOS', **solver_kwargs)
                
                if w.value is None or np.any(np.isnan(w.value)):
                    raise cp.SolverError("Convex solver returned no valid solution")
                
                # Process and return the solution
                weights = pd.Series(w.value, index=returns.columns)
                return self._process_solution(weights, returns, cov, pcr_sharpe)
                
            except Exception as e:
                self.logger.error(f"Convex optimization failed: {e}")
                return self._fallback_weights(returns, n_positions, min_weight, max_weight)
        
        # If we get here, all solvers failed
        self.logger.error("All optimization attempts failed")
        return self._fallback_weights(returns, n_positions, min_weight, max_weight)
    
    def _process_solution(self, weights: pd.Series, returns: pd.DataFrame, 
                         cov: np.ndarray, pcr_sharpe: np.ndarray) -> pd.Series:
        """Process and validate optimization solution."""
        weights[weights < 1e-6] = 0
        weights = weights / weights.sum()
        
        # Calculate metrics
        self.metrics = calculate_diversification_metrics(weights.values, cov, returns)
        self.metrics['pcr_sharpe'] = pcr_sharpe @ weights
        
        # Calculate post-optimization Return-HHI
        if len(returns) > 0:
            pl_contributions = returns @ weights
            if np.abs(pl_contributions).sum() > 1e-10:
                pl_shares = pl_contributions / pl_contributions.sum()
                self.metrics['post_opt_hhi'] = (pl_shares ** 2).sum()
            else:
                self.metrics['post_opt_hhi'] = 0.0
        
        self._log_optimization_results()
        return weights
    
    def _log_optimization_results(self) -> None:
        """Log optimization results and check constraints."""
        metrics = self.metrics
        cfg = self.config.get('optimization', {})
        
        # Format metrics
        msg = [
            "\n=== PORTFOLIO OPTIMIZATION RESULTS ===",
            "Metrics:",
            f"  - Diversification Ratio: {metrics['diversification_ratio']:.2f}",
            f"  - Weight HHI: {metrics['hhi_weight']:.4f} (threshold: {cfg.get('hhi_weight_threshold', 'N/A')})",
            f"  - Effective N: {metrics['effective_n']:.1f}",
            f"  - Return HHI: {metrics['hhi_return']:.4f} (threshold: {cfg.get('hhi_return_threshold', 'N/A')})",
            f"  - Post-Opt HHI: {metrics.get('post_opt_hhi', 0.0):.4f}",
            f"  - PCR-Sharpe: {metrics['pcr_sharpe']:.4f}"
        ]
        
        # Add configuration
        msg.extend([
            "\nConfiguration:",
            f"  - Lambda HHI: {cfg.get('lambda_hhi', 10.0):.1f}",
            f"  - Min Weight: {cfg.get('min_weight', 0.05):.2f}",
            f"  - Max Weight: {cfg.get('individual_limit', 0.15):.2f}",
            f"  - Risk Aversion: {cfg.get('risk_aversion', 1.0):.1f}"
        ])
        
        self.logger.info("\n".join(msg))
        
        # Check constraints
        violations = []
        
        # Check weight HHI constraint
        hhi_thresh = cfg.get('hhi_weight_threshold')
        if hhi_thresh is not None and metrics['hhi_weight'] > hhi_thresh:
            violations.append(f"Weight HHI {metrics['hhi_weight']:.4f} > {hhi_thresh}")
            
        # Check return HHI constraint
        return_hhi_thresh = cfg.get('hhi_return_threshold')
        if return_hhi_thresh is not None and metrics['hhi_return'] > return_hhi_thresh:
            violations.append(f"Return HHI {metrics['hhi_return']:.4f} > {return_hhi_thresh}")
            
        # Check post-optimization HHI if available
        if 'post_opt_hhi' in metrics and return_hhi_thresh is not None:
            if metrics['post_opt_hhi'] > return_hhi_thresh:
                violations.append(f"Post-opt HHI {metrics['post_opt_hhi']:.4f} > {return_hhi_thresh}")
        
        if violations:
            self.logger.warning("\nCONSTRAINT VIOLATIONS DETECTED:\n  " + 
                             "\n  ".join(violations) + "\n")
        else:
            self.logger.info("All constraints satisfied\n")
    
    def _fallback_weights(self, returns: pd.DataFrame, n_positions: int,
                        min_weight: float, max_weight: float) -> pd.Series:
        """Fallback method when optimization fails."""
        self.logger.warning(f"Using fallback weights for {n_positions} positions")
        
        # Simple equal-weighted portfolio
        weights = pd.Series(0.0, index=returns.columns)
        top_assets = returns.mean().nlargest(n_positions).index
        weights[top_assets] = 1.0 / n_positions
        
        return weights
