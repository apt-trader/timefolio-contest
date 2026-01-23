"""
FACTOR MODEL
============
Barra-style factor model using Toraniko for factor return estimation.

This module provides:
1. Factor return estimation via WLS regression
2. Factor covariance matrix estimation
3. Idiosyncratic variance estimation
4. Full covariance matrix construction: Σ = B'ΣF B + D

The factor model decomposes asset returns into:
- Market factor: Cap-weighted market return
- Sector factors: GICS sector returns (constrained to sum to 0)
- Style factors: Value, Momentum, Size (and custom factors)
- Idiosyncratic: Stock-specific residual returns

Author: TimeFolio-Toraniko Integration
Date: 2026-01-23
"""

import pandas as pd
import numpy as np
import polars as pl
from typing import Dict, List, Optional, Tuple, Union
from sklearn.covariance import LedoitWolf
import logging

from toraniko.model import estimate_factor_returns
from toraniko.math import winsorize, exp_weights

logger = logging.getLogger(__name__)


class BarraFactorModel:
    """
    Barra-style characteristic factor model using Toraniko.
    
    Estimates factor returns using weighted least squares regression,
    then constructs factor covariance matrix for portfolio optimization.
    """
    
    def __init__(
        self,
        winsor_factor: float = 0.05,
        residualize_styles: bool = True,
        factor_cov_lookback: int = 252,
        factor_cov_halflife: int = 63,
        idio_var_lookback: int = 63,
        use_exponential_weighting: bool = True
    ):
        """
        Initialize factor model.
        
        Args:
            winsor_factor: Winsorization percentile for returns (0.05 = 5%)
            residualize_styles: Whether to orthogonalize styles to market+sector
            factor_cov_lookback: Lookback days for factor covariance estimation
            factor_cov_halflife: Half-life for exponential weighting of factor cov
            idio_var_lookback: Lookback days for idiosyncratic variance
            use_exponential_weighting: Use exponential decay for covariance
        """
        self.winsor_factor = winsor_factor
        self.residualize_styles = residualize_styles
        self.factor_cov_lookback = factor_cov_lookback
        self.factor_cov_halflife = factor_cov_halflife
        self.idio_var_lookback = idio_var_lookback
        self.use_exponential_weighting = use_exponential_weighting
        
        # Cached results
        self._factor_returns: Optional[pl.DataFrame] = None
        self._residuals: Optional[pl.DataFrame] = None
        self._factor_names: Optional[List[str]] = None
        self._factor_covariance: Optional[np.ndarray] = None
        self._idiosyncratic_variance: Optional[pd.Series] = None
        
        logger.info(f"BarraFactorModel initialized:")
        logger.info(f"  - Winsorization: {winsor_factor}")
        logger.info(f"  - Residualize styles: {residualize_styles}")
        logger.info(f"  - Factor cov lookback: {factor_cov_lookback}d, halflife: {factor_cov_halflife}d")
    
    def estimate_factor_returns(
        self,
        returns_df: pl.DataFrame,
        mkt_cap_df: pl.DataFrame,
        sector_df: pl.DataFrame,
        style_df: pl.DataFrame
    ) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """
        Estimate factor returns and residuals using Toraniko.
        
        Uses weighted least squares with sqrt(market_cap) as weights,
        which is a proxy for inverse idiosyncratic variance.
        
        Args:
            returns_df: | date | symbol | asset_returns |
            mkt_cap_df: | date | symbol | market_cap |
            sector_df:  | date | symbol | sector_1 | sector_2 | ... |
            style_df:   | date | symbol | style_1 | style_2 | ... |
            
        Returns:
            Tuple of (factor_returns_df, residuals_df)
            - factor_returns_df: | date | market | sector_1 | ... | style_1 | ... |
            - residuals_df: | date | symbol_1 | symbol_2 | ... |
        """
        logger.info("Estimating factor returns via WLS regression...")
        
        try:
            factor_returns, residuals = estimate_factor_returns(
                returns_df,
                mkt_cap_df,
                sector_df,
                style_df,
                winsor_factor=self.winsor_factor,
                residualize_styles=self.residualize_styles
            )
            
            # Cache results
            self._factor_returns = factor_returns
            self._residuals = residuals
            self._factor_names = [c for c in factor_returns.columns if c != 'date']
            
            n_dates = len(factor_returns)
            n_factors = len(self._factor_names)
            logger.info(f"Estimated {n_factors} factor returns over {n_dates} dates")
            logger.info(f"  Factors: {self._factor_names[:5]}..." if n_factors > 5 else f"  Factors: {self._factor_names}")
            
            return factor_returns, residuals
            
        except Exception as e:
            logger.error(f"Factor return estimation failed: {e}")
            raise
    
    def estimate_factor_covariance(
        self,
        factor_returns: Optional[pl.DataFrame] = None,
        method: str = 'ledoit_wolf'
    ) -> np.ndarray:
        """
        Estimate factor covariance matrix from historical factor returns.
        
        Args:
            factor_returns: Factor returns DataFrame (uses cached if None)
            method: 'ledoit_wolf', 'exponential', or 'sample'
            
        Returns:
            Factor covariance matrix (n_factors x n_factors)
        """
        if factor_returns is None:
            factor_returns = self._factor_returns
        
        if factor_returns is None or factor_returns.is_empty():
            raise ValueError("No factor returns available. Run estimate_factor_returns first.")
        
        # Extract factor returns as numpy array (exclude date column)
        factor_cols = [c for c in factor_returns.columns if c != 'date']
        returns_np = factor_returns.select(factor_cols).to_numpy()
        
        # Use most recent lookback period
        if len(returns_np) > self.factor_cov_lookback:
            returns_np = returns_np[-self.factor_cov_lookback:]
        
        logger.info(f"Estimating factor covariance using {method} method ({len(returns_np)} observations)")
        
        if method == 'ledoit_wolf':
            # Ledoit-Wolf shrinkage estimator
            lw = LedoitWolf().fit(returns_np)
            cov_matrix = lw.covariance_
            logger.info(f"  Ledoit-Wolf shrinkage intensity: {lw.shrinkage_:.4f}")
            
        elif method == 'exponential':
            # Exponentially weighted covariance
            weights = exp_weights(len(returns_np), self.factor_cov_halflife)
            weights = weights / weights.sum()  # Normalize
            
            # Weighted mean
            weighted_mean = np.average(returns_np, axis=0, weights=weights)
            
            # Weighted covariance
            centered = returns_np - weighted_mean
            cov_matrix = np.zeros((returns_np.shape[1], returns_np.shape[1]))
            for i in range(len(returns_np)):
                cov_matrix += weights[i] * np.outer(centered[i], centered[i])
            
        elif method == 'sample':
            # Simple sample covariance
            cov_matrix = np.cov(returns_np, rowvar=False)
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Annualize (assuming daily returns)
        cov_matrix = cov_matrix * 252
        
        self._factor_covariance = cov_matrix
        self._factor_names = factor_cols
        
        logger.info(f"Factor covariance matrix: {cov_matrix.shape}")
        return cov_matrix
    
    def estimate_idiosyncratic_variance(
        self,
        residuals: Optional[pl.DataFrame] = None
    ) -> pd.Series:
        """
        Estimate idiosyncratic variance from residuals.
        
        Args:
            residuals: Residuals DataFrame (uses cached if None)
            
        Returns:
            Series of idiosyncratic variances indexed by symbol
        """
        if residuals is None:
            residuals = self._residuals
        
        if residuals is None or residuals.is_empty():
            raise ValueError("No residuals available. Run estimate_factor_returns first.")
        
        # Get symbol columns (exclude date)
        symbol_cols = [c for c in residuals.columns if c != 'date']
        
        # Use most recent lookback period
        residuals_np = residuals.select(symbol_cols).to_numpy()
        if len(residuals_np) > self.idio_var_lookback:
            residuals_np = residuals_np[-self.idio_var_lookback:]
        
        # Calculate variance for each symbol
        idio_var = np.nanvar(residuals_np, axis=0)
        
        # Annualize
        idio_var = idio_var * 252
        
        self._idiosyncratic_variance = pd.Series(idio_var, index=symbol_cols)
        
        logger.info(f"Estimated idiosyncratic variance for {len(symbol_cols)} symbols")
        logger.info(f"  Mean idio vol: {np.sqrt(idio_var.mean()):.4f}")
        
        return self._idiosyncratic_variance
    
    def get_factor_loadings(
        self,
        sector_df: pl.DataFrame,
        style_df: pl.DataFrame,
        date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get factor loadings (exposures) for each symbol.
        
        Args:
            sector_df: Sector scores DataFrame
            style_df: Style scores DataFrame
            date: Specific date to get loadings for (latest if None)
            
        Returns:
            DataFrame with symbols as index, factors as columns
        """
        # Filter to specific date if provided
        if date is not None:
            sector_df = sector_df.filter(pl.col('date') == date)
            style_df = style_df.filter(pl.col('date') == date)
        else:
            # Use latest date
            latest_date = sector_df['date'].max()
            sector_df = sector_df.filter(pl.col('date') == latest_date)
            style_df = style_df.filter(pl.col('date') == latest_date)
        
        # Get sector columns
        sector_cols = [c for c in sector_df.columns if c not in ['date', 'symbol']]
        style_cols = [c for c in style_df.columns if c not in ['date', 'symbol']]
        
        # Merge sector and style loadings
        loadings = sector_df.join(style_df, on=['date', 'symbol'], how='inner')
        
        # Add market loading (always 1)
        loadings = loadings.with_columns(pl.lit(1).alias('market'))
        
        # Convert to pandas
        pdf = loadings.to_pandas()
        pdf = pdf.set_index('symbol')
        
        # Select factor columns in order: market, sectors, styles
        factor_cols = ['market'] + sector_cols + style_cols
        pdf = pdf[factor_cols]
        
        return pdf
    
    def construct_full_covariance(
        self,
        factor_loadings: pd.DataFrame,
        factor_covariance: Optional[np.ndarray] = None,
        idiosyncratic_variance: Optional[pd.Series] = None
    ) -> np.ndarray:
        """
        Construct full asset covariance matrix using factor model.
        
        Σ = B'ΣF B + D
        
        Where:
        - B: Factor loadings matrix (n_assets x n_factors)
        - ΣF: Factor covariance matrix (n_factors x n_factors)
        - D: Diagonal idiosyncratic variance matrix (n_assets x n_assets)
        
        Args:
            factor_loadings: DataFrame of factor loadings (symbols x factors)
            factor_covariance: Factor covariance matrix (uses cached if None)
            idiosyncratic_variance: Idiosyncratic variances (uses cached if None)
            
        Returns:
            Full covariance matrix (n_assets x n_assets)
        """
        if factor_covariance is None:
            factor_covariance = self._factor_covariance
        if idiosyncratic_variance is None:
            idiosyncratic_variance = self._idiosyncratic_variance
        
        if factor_covariance is None:
            raise ValueError("No factor covariance available. Run estimate_factor_covariance first.")
        if idiosyncratic_variance is None:
            raise ValueError("No idiosyncratic variance available. Run estimate_idiosyncratic_variance first.")
        
        # Align symbols
        common_symbols = list(set(factor_loadings.index) & set(idiosyncratic_variance.index))
        factor_loadings = factor_loadings.loc[common_symbols]
        idiosyncratic_variance = idiosyncratic_variance.loc[common_symbols]
        
        # Get matrices
        B = factor_loadings.values  # n_assets x n_factors
        Σf = factor_covariance      # n_factors x n_factors
        D = np.diag(idiosyncratic_variance.values)  # n_assets x n_assets
        
        # Construct full covariance: Σ = B @ Σf @ B' + D
        systematic_cov = B @ Σf @ B.T
        full_cov = systematic_cov + D
        
        logger.info(f"Constructed full covariance matrix: {full_cov.shape}")
        logger.info(f"  Systematic variance contribution: {np.trace(systematic_cov) / np.trace(full_cov):.2%}")
        
        return full_cov
    
    def get_factor_expected_returns(
        self,
        factor_returns: Optional[pl.DataFrame] = None,
        method: str = 'mean',
        lookback: int = 252,
        halflife: int = 63
    ) -> pd.Series:
        """
        Estimate expected factor returns.
        
        Args:
            factor_returns: Factor returns DataFrame (uses cached if None)
            method: 'mean', 'exponential', or 'shrinkage'
            lookback: Number of days to look back
            halflife: Half-life for exponential weighting
            
        Returns:
            Series of expected returns per factor
        """
        if factor_returns is None:
            factor_returns = self._factor_returns
        
        if factor_returns is None or factor_returns.is_empty():
            raise ValueError("No factor returns available.")
        
        factor_cols = [c for c in factor_returns.columns if c != 'date']
        returns_np = factor_returns.select(factor_cols).to_numpy()
        
        if len(returns_np) > lookback:
            returns_np = returns_np[-lookback:]
        
        if method == 'mean':
            expected = np.mean(returns_np, axis=0)
            
        elif method == 'exponential':
            weights = exp_weights(len(returns_np), halflife)
            weights = weights / weights.sum()
            expected = np.average(returns_np, axis=0, weights=weights)
            
        elif method == 'shrinkage':
            # Shrink towards zero (conservative)
            raw_mean = np.mean(returns_np, axis=0)
            shrinkage = 0.5  # 50% shrinkage
            expected = raw_mean * (1 - shrinkage)
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Annualize
        expected = expected * 252
        
        return pd.Series(expected, index=factor_cols)
    
    @property
    def factor_names(self) -> List[str]:
        """Get list of factor names."""
        return self._factor_names or []
    
    @property
    def factor_returns(self) -> Optional[pd.DataFrame]:
        """Get cached factor returns as Pandas DataFrame."""
        if self._factor_returns is None:
            return None
        return self._factor_returns.to_pandas().set_index('date')
    
    @property
    def residuals(self) -> Optional[pd.DataFrame]:
        """Get cached residuals as Pandas DataFrame."""
        if self._residuals is None:
            return None
        return self._residuals.to_pandas().set_index('date')


class FactorModelIntegrator:
    """
    Integrates Barra factor model with TimeFolio's signal-based optimization.
    
    Provides methods to:
    1. Estimate signal-level factor loadings
    2. Construct signal covariance using factor model
    3. Decompose portfolio risk into factor contributions
    """
    
    def __init__(self, factor_model: BarraFactorModel):
        """
        Initialize integrator.
        
        Args:
            factor_model: Configured BarraFactorModel instance
        """
        self.factor_model = factor_model
    
    def estimate_signal_factor_loadings(
        self,
        signal_portfolios: Dict[str, pd.Series],
        factor_loadings: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Estimate factor loadings for each signal portfolio.
        
        Args:
            signal_portfolios: Dict mapping signal name -> Series of stock weights
            factor_loadings: Stock-level factor loadings
            
        Returns:
            DataFrame of signal factor loadings (signals x factors)
        """
        signal_loadings = {}
        
        for signal_name, weights in signal_portfolios.items():
            # Align weights with factor loadings
            common = list(set(weights.index) & set(factor_loadings.index))
            w = weights.loc[common].values
            B = factor_loadings.loc[common].values
            
            # Normalize weights
            w = w / w.sum()
            
            # Portfolio factor loading = weighted sum of stock loadings
            signal_loading = w @ B
            signal_loadings[signal_name] = signal_loading
        
        return pd.DataFrame(signal_loadings, index=factor_loadings.columns).T
    
    def estimate_signal_covariance_from_factors(
        self,
        signal_factor_loadings: pd.DataFrame,
        factor_covariance: np.ndarray,
        signal_idio_var: Optional[pd.Series] = None
    ) -> np.ndarray:
        """
        Estimate signal covariance using factor model.
        
        Args:
            signal_factor_loadings: Signal factor loadings (signals x factors)
            factor_covariance: Factor covariance matrix
            signal_idio_var: Optional signal-level idiosyncratic variance
            
        Returns:
            Signal covariance matrix (n_signals x n_signals)
        """
        B = signal_factor_loadings.values
        Σf = factor_covariance
        
        # Systematic covariance
        signal_cov = B @ Σf @ B.T
        
        # Add idiosyncratic if provided
        if signal_idio_var is not None:
            signal_cov += np.diag(signal_idio_var.values)
        
        return signal_cov


def create_factor_model(**kwargs) -> BarraFactorModel:
    """Factory function to create BarraFactorModel."""
    return BarraFactorModel(**kwargs)
