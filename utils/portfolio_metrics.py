# utils/portfolio_metrics.py
import logging
import numpy as np
import pandas as pd
from typing import Dict

# Configure logging for this module
logger = logging.getLogger(__name__)

def calculate_diversification_metrics(weights: np.ndarray,
                                      cov_matrix: np.ndarray,
                                      returns: pd.DataFrame) -> Dict[str, float]:
    """
    Calculates key diversification metrics for a given portfolio.

    Args:
        weights (np.ndarray): Portfolio weights array (n_assets,).
        cov_matrix (np.ndarray): Annualized covariance matrix (n_assets, n_assets).
        returns (pd.DataFrame): DataFrame of asset returns (n_periods, n_assets).

    Returns:
        A dictionary containing:
        - 'hhi_weight': Herfindahl-Hirschman Index for weight concentration.
        - 'effective_n': Effective number of positions (1 / HHI).
        - 'diversification_ratio': Ratio of weighted average volatility to portfolio volatility.
    """
    if not isinstance(weights, np.ndarray) or weights.ndim != 1:
        raise ValueError("Weights must be a 1D numpy array.")
    if not isinstance(cov_matrix, np.ndarray) or cov_matrix.ndim != 2:
        raise ValueError("Covariance matrix must be a 2D numpy array.")
    if weights.shape[0] != cov_matrix.shape[0]:
        raise ValueError("Shape mismatch between weights and covariance matrix.")

    # --- Herfindahl-Hirschman Index (HHI) for Weight Concentration ---
    hhi_weight = np.sum(weights ** 2)
    
    # --- Effective Number of Positions ---
    # Avoid division by zero if HHI is zero (though unlikely with valid weights)
    effective_n = 1.0 / hhi_weight if hhi_weight > 1e-10 else 0.0
    
    # --- Diversification Ratio ---
    # Weighted average of individual asset volatilities
    weighted_asset_vol = weights @ np.sqrt(np.diag(cov_matrix))
    
    # Total portfolio volatility
    portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)
    
    # The ratio measures the extent of risk reduction through diversification
    diversification_ratio = weighted_asset_vol / portfolio_vol if portfolio_vol > 1e-8 else 0.0
    
    metrics = {
        'hhi_weight': hhi_weight,
        'effective_n': effective_n,
        'diversification_ratio': diversification_ratio
    }
    
    logger.debug(f"Calculated diversification metrics: {metrics}")
    
    return metrics

def calculate_portfolio_performance(weights: pd.Series,
                                    returns: pd.DataFrame,
                                    risk_free_rate: float = 0.02) -> Dict[str, float]:
    """
    Calculates overall portfolio performance statistics.

    Args:
        weights (pd.Series): Portfolio weights, indexed by ticker.
        returns (pd.DataFrame): Daily returns for all assets in the portfolio.
        risk_free_rate (float): Annualized risk-free rate.

    Returns:
        A dictionary with annualized performance metrics.
    """
    portfolio_returns = returns.mul(weights, axis=1).sum(axis=1)
    
    # Annualized Return
    annualized_return = portfolio_returns.mean() * 252
    
    # Annualized Volatility
    annualized_volatility = portfolio_returns.std() * np.sqrt(252)
    
    # Annualized Sharpe Ratio
    sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility if annualized_volatility > 1e-8 else 0.0
    
    return {
        'annualized_return': annualized_return,
        'annualized_volatility': annualized_volatility,
        'sharpe_ratio': sharpe_ratio
    }