"""
RISK ATTRIBUTION
================
Portfolio risk decomposition using Barra-style factor model.

Decomposes portfolio risk into:
1. Systematic risk (from factor exposures)
2. Idiosyncratic risk (stock-specific)
3. Factor contributions (risk from each factor)

This enables:
- Understanding where portfolio risk comes from
- Factor exposure monitoring
- Risk budgeting and limits

Author: TimeFolio-Toraniko Integration
Date: 2026-01-23
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class RiskAttributor:
    """
    Decomposes portfolio risk into factor and idiosyncratic components.
    
    Uses the factor model decomposition:
    Portfolio Variance = w'Σw = w'(B'ΣfB + D)w
                       = (Bw)'Σf(Bw) + w'Dw
                       = Systematic Variance + Idiosyncratic Variance
    """
    
    def __init__(self, annualization_factor: int = 252):
        """
        Initialize risk attributor.
        
        Args:
            annualization_factor: Factor to annualize daily returns (252 for daily)
        """
        self.annualization_factor = annualization_factor
    
    def decompose_risk(
        self,
        portfolio_weights: pd.Series,
        factor_loadings: pd.DataFrame,
        factor_covariance: np.ndarray,
        idiosyncratic_variance: pd.Series
    ) -> Dict[str, float]:
        """
        Decompose portfolio risk into systematic and idiosyncratic components.
        
        Args:
            portfolio_weights: Series of portfolio weights indexed by symbol
            factor_loadings: DataFrame of factor loadings (symbols x factors)
            factor_covariance: Factor covariance matrix (n_factors x n_factors)
            idiosyncratic_variance: Series of idiosyncratic variances by symbol
            
        Returns:
            Dict containing:
            - total_risk: Portfolio volatility (annualized)
            - total_variance: Portfolio variance (annualized)
            - systematic_risk: Risk from factor exposures
            - systematic_variance: Variance from factor exposures
            - idiosyncratic_risk: Stock-specific risk
            - idiosyncratic_variance: Stock-specific variance
            - systematic_pct: Percentage of variance from systematic
            - idiosyncratic_pct: Percentage of variance from idiosyncratic
        """
        # Align all inputs to common symbols
        common_symbols = list(
            set(portfolio_weights.index) & 
            set(factor_loadings.index) & 
            set(idiosyncratic_variance.index)
        )
        
        if len(common_symbols) == 0:
            logger.warning("No common symbols found for risk decomposition")
            return self._empty_decomposition()
        
        # Extract aligned data
        w = portfolio_weights.loc[common_symbols].values
        B = factor_loadings.loc[common_symbols].values  # n_assets x n_factors
        Σf = factor_covariance  # n_factors x n_factors
        d = idiosyncratic_variance.loc[common_symbols].values  # n_assets
        
        # Normalize weights (in case they don't sum to 1)
        w = w / np.sum(np.abs(w))
        
        # Portfolio factor exposure: Bw (n_factors x 1)
        portfolio_factor_exposure = B.T @ w
        
        # Systematic variance: (Bw)'Σf(Bw)
        systematic_var = portfolio_factor_exposure @ Σf @ portfolio_factor_exposure
        
        # Idiosyncratic variance: w'Dw = Σ(w_i^2 * d_i)
        idio_var = np.sum(w**2 * d)
        
        # Total variance
        total_var = systematic_var + idio_var
        
        # Convert to risk (volatility)
        total_risk = np.sqrt(total_var)
        systematic_risk = np.sqrt(systematic_var)
        idio_risk = np.sqrt(idio_var)
        
        return {
            'total_risk': total_risk,
            'total_variance': total_var,
            'systematic_risk': systematic_risk,
            'systematic_variance': systematic_var,
            'idiosyncratic_risk': idio_risk,
            'idiosyncratic_variance': idio_var,
            'systematic_pct': systematic_var / total_var if total_var > 0 else 0,
            'idiosyncratic_pct': idio_var / total_var if total_var > 0 else 0,
            'n_assets': len(common_symbols)
        }
    
    def factor_risk_contributions(
        self,
        portfolio_weights: pd.Series,
        factor_loadings: pd.DataFrame,
        factor_covariance: np.ndarray,
        factor_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Calculate risk contribution from each factor.
        
        Marginal Contribution to Risk (MCR) for factor f:
        MCR_f = (Σf @ portfolio_factor_exposure)_f / total_systematic_risk
        
        Risk Contribution (RC) for factor f:
        RC_f = portfolio_factor_exposure_f * MCR_f
        
        Args:
            portfolio_weights: Series of portfolio weights
            factor_loadings: DataFrame of factor loadings
            factor_covariance: Factor covariance matrix
            factor_names: Optional list of factor names
            
        Returns:
            DataFrame with columns: factor, exposure, mcr, risk_contribution, pct_contribution
        """
        # Align inputs
        common_symbols = list(
            set(portfolio_weights.index) & 
            set(factor_loadings.index)
        )
        
        w = portfolio_weights.loc[common_symbols].values
        B = factor_loadings.loc[common_symbols].values
        Σf = factor_covariance
        
        w = w / np.sum(np.abs(w))
        
        # Portfolio factor exposure
        pfe = B.T @ w  # n_factors
        
        # Systematic variance and risk
        sys_var = pfe @ Σf @ pfe
        sys_risk = np.sqrt(sys_var)
        
        if sys_risk == 0:
            logger.warning("Zero systematic risk, cannot compute factor contributions")
            return pd.DataFrame()
        
        # Marginal contribution to risk
        mcr = (Σf @ pfe) / sys_risk
        
        # Risk contribution
        rc = pfe * mcr
        
        # Percentage contribution
        pct_rc = rc / sys_risk
        
        # Get factor names
        if factor_names is None:
            factor_names = factor_loadings.columns.tolist()
        
        return pd.DataFrame({
            'factor': factor_names,
            'exposure': pfe,
            'mcr': mcr,
            'risk_contribution': rc,
            'pct_contribution': pct_rc
        })
    
    def asset_risk_contributions(
        self,
        portfolio_weights: pd.Series,
        covariance_matrix: np.ndarray,
        symbols: List[str]
    ) -> pd.DataFrame:
        """
        Calculate risk contribution from each asset.
        
        Args:
            portfolio_weights: Series of portfolio weights
            covariance_matrix: Full covariance matrix (n_assets x n_assets)
            symbols: List of symbol names corresponding to covariance matrix
            
        Returns:
            DataFrame with columns: symbol, weight, mcr, risk_contribution, pct_contribution
        """
        # Align weights with covariance matrix
        w = portfolio_weights.loc[symbols].values
        Σ = covariance_matrix
        
        w = w / np.sum(np.abs(w))
        
        # Portfolio variance and risk
        port_var = w @ Σ @ w
        port_risk = np.sqrt(port_var)
        
        if port_risk == 0:
            logger.warning("Zero portfolio risk")
            return pd.DataFrame()
        
        # Marginal contribution to risk
        mcr = (Σ @ w) / port_risk
        
        # Risk contribution
        rc = w * mcr
        
        # Percentage contribution
        pct_rc = rc / port_risk
        
        return pd.DataFrame({
            'symbol': symbols,
            'weight': w,
            'mcr': mcr,
            'risk_contribution': rc,
            'pct_contribution': pct_rc
        })
    
    def tracking_error_decomposition(
        self,
        portfolio_weights: pd.Series,
        benchmark_weights: pd.Series,
        factor_loadings: pd.DataFrame,
        factor_covariance: np.ndarray,
        idiosyncratic_variance: pd.Series
    ) -> Dict[str, float]:
        """
        Decompose tracking error vs benchmark into factor and idiosyncratic components.
        
        Args:
            portfolio_weights: Portfolio weights
            benchmark_weights: Benchmark weights
            factor_loadings: Factor loadings
            factor_covariance: Factor covariance
            idiosyncratic_variance: Idiosyncratic variance
            
        Returns:
            Dict with tracking error decomposition
        """
        # Active weights
        all_symbols = list(set(portfolio_weights.index) | set(benchmark_weights.index))
        
        port_w = portfolio_weights.reindex(all_symbols, fill_value=0)
        bench_w = benchmark_weights.reindex(all_symbols, fill_value=0)
        active_w = port_w - bench_w
        
        # Decompose active risk
        return self.decompose_risk(
            active_w,
            factor_loadings.reindex(all_symbols, fill_value=0),
            factor_covariance,
            idiosyncratic_variance.reindex(all_symbols, fill_value=0)
        )
    
    def _empty_decomposition(self) -> Dict[str, float]:
        """Return empty decomposition dict."""
        return {
            'total_risk': 0.0,
            'total_variance': 0.0,
            'systematic_risk': 0.0,
            'systematic_variance': 0.0,
            'idiosyncratic_risk': 0.0,
            'idiosyncratic_variance': 0.0,
            'systematic_pct': 0.0,
            'idiosyncratic_pct': 0.0,
            'n_assets': 0
        }
    
    def generate_risk_report(
        self,
        portfolio_weights: pd.Series,
        factor_loadings: pd.DataFrame,
        factor_covariance: np.ndarray,
        idiosyncratic_variance: pd.Series,
        factor_names: Optional[List[str]] = None
    ) -> Dict:
        """
        Generate comprehensive risk report.
        
        Args:
            portfolio_weights: Portfolio weights
            factor_loadings: Factor loadings
            factor_covariance: Factor covariance
            idiosyncratic_variance: Idiosyncratic variance
            factor_names: Optional factor names
            
        Returns:
            Dict containing:
            - decomposition: Risk decomposition summary
            - factor_contributions: Factor risk contributions
            - top_factor_exposures: Top factor exposures
            - concentration_metrics: Portfolio concentration metrics
        """
        # Risk decomposition
        decomposition = self.decompose_risk(
            portfolio_weights,
            factor_loadings,
            factor_covariance,
            idiosyncratic_variance
        )
        
        # Factor contributions
        factor_contrib = self.factor_risk_contributions(
            portfolio_weights,
            factor_loadings,
            factor_covariance,
            factor_names
        )
        
        # Top factor exposures
        if not factor_contrib.empty:
            top_exposures = factor_contrib.nlargest(5, 'exposure')[['factor', 'exposure']].to_dict('records')
            top_risk_contrib = factor_contrib.nlargest(5, 'pct_contribution')[['factor', 'pct_contribution']].to_dict('records')
        else:
            top_exposures = []
            top_risk_contrib = []
        
        # Concentration metrics
        w = portfolio_weights.values
        w_abs = np.abs(w)
        hhi = np.sum(w_abs**2)  # Herfindahl-Hirschman Index
        effective_n = 1 / hhi if hhi > 0 else 0  # Effective number of positions
        
        return {
            'decomposition': decomposition,
            'factor_contributions': factor_contrib.to_dict('records') if not factor_contrib.empty else [],
            'top_factor_exposures': top_exposures,
            'top_risk_contributors': top_risk_contrib,
            'concentration': {
                'hhi': hhi,
                'effective_n': effective_n,
                'max_weight': w_abs.max() if len(w_abs) > 0 else 0,
                'n_positions': np.sum(w_abs > 0.001)
            }
        }


def create_risk_attributor(**kwargs) -> RiskAttributor:
    """Factory function to create RiskAttributor."""
    return RiskAttributor(**kwargs)
