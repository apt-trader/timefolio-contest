"""
SIGNAL OPTIMIZER
================
Performs Mean-Variance Optimization on SIGNALS, not individual stocks.

Key insight from "Introduction To Practical Portfolio Optimization.txt":
- MVO on instruments is horrendous due to low SNR (~10bp signal vs ~2% daily vol)
- MVO on signals has much higher SNR (signal portfolios are diversified)
- Signal covariance matrix is small (5x5) vs stock covariance (200x200)

This module:
1. Estimates signal expected returns from historical signal portfolio performance
2. Estimates signal covariance matrix with shrinkage
3. Performs MVO to get optimal signal weights
4. Applies constraints (long-only, turnover penalty)

Author: TimeFolio System Refactor
Date: 2025-01-11
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, TYPE_CHECKING
import cvxpy as cp
from sklearn.covariance import LedoitWolf
import logging

if TYPE_CHECKING:
    from signals.factor_model import BarraFactorModel, FactorModelIntegrator

logger = logging.getLogger(__name__)


class SignalOptimizer:
    """
    Optimizes portfolio weights at the SIGNAL level using MVO.
    
    Instead of optimizing weights for 200+ stocks directly,
    we optimize weights for 5 signals, then map to stocks.
    """
    
    def __init__(
        self,
        risk_aversion: float = 1.0,
        min_signal_weight: float = 0.05,
        max_signal_weight: float = 0.50,
        turnover_penalty: float = 0.01,
        use_shrinkage: bool = True,
        shrinkage_target: str = 'constant_correlation'
    ):
        """
        Initialize signal optimizer.
        
        Args:
            risk_aversion: Lambda parameter for risk-return tradeoff (higher = more conservative)
            min_signal_weight: Minimum weight per signal (ensures diversification)
            max_signal_weight: Maximum weight per signal (prevents concentration)
            turnover_penalty: Penalty for deviating from previous weights
            use_shrinkage: Whether to use shrinkage for covariance estimation
            shrinkage_target: Type of shrinkage ('ledoit_wolf' or 'constant_correlation')
        """
        self.risk_aversion = risk_aversion
        self.min_signal_weight = min_signal_weight
        self.max_signal_weight = max_signal_weight
        self.turnover_penalty = turnover_penalty
        self.use_shrinkage = use_shrinkage
        self.shrinkage_target = shrinkage_target
        
        # Store previous weights for turnover calculation
        self.previous_weights: Optional[pd.Series] = None
        
        logger.info(f"SignalOptimizer initialized:")
        logger.info(f"  - Risk aversion: {risk_aversion}")
        logger.info(f"  - Signal weight bounds: [{min_signal_weight}, {max_signal_weight}]")
        logger.info(f"  - Turnover penalty: {turnover_penalty}")
        logger.info(f"  - Shrinkage: {use_shrinkage} ({shrinkage_target})")
    
    def estimate_signal_expected_returns(
        self,
        historical_signal_returns: pd.DataFrame,
        method: str = 'mean',
        decay_factor: float = 0.97
    ) -> pd.Series:
        """
        Estimate expected returns for each signal.
        
        Args:
            historical_signal_returns: DataFrame of historical signal portfolio returns
                                       (date x signal)
            method: 'mean', 'ewma', or 'shrinkage'
            decay_factor: Decay factor for EWMA (closer to 1 = slower decay)
            
        Returns:
            Series of expected returns per signal
        """
        if historical_signal_returns.empty:
            logger.warning("No historical signal returns provided")
            return pd.Series()
        
        if method == 'mean':
            # Simple historical mean
            expected_returns = historical_signal_returns.mean()
            
        elif method == 'ewma':
            # Exponentially weighted moving average (more weight on recent)
            weights = np.array([decay_factor ** i for i in range(len(historical_signal_returns))])
            weights = weights[::-1]  # Reverse so recent has higher weight
            weights = weights / weights.sum()
            
            expected_returns = (historical_signal_returns.T * weights).T.sum()
            
        elif method == 'shrinkage':
            # Shrink towards grand mean (Bayes-Stein shrinkage)
            sample_mean = historical_signal_returns.mean()
            grand_mean = sample_mean.mean()
            
            # Shrinkage intensity based on estimation error
            n = len(historical_signal_returns)
            k = len(historical_signal_returns.columns)
            
            if n > k + 2:
                sample_var = historical_signal_returns.var()
                shrinkage_intensity = (k - 2) / n / ((sample_mean - grand_mean) ** 2 / sample_var + 1e-6).sum()
                shrinkage_intensity = np.clip(shrinkage_intensity, 0, 1)
            else:
                shrinkage_intensity = 0.5  # Default moderate shrinkage
            
            expected_returns = (1 - shrinkage_intensity) * sample_mean + shrinkage_intensity * grand_mean
            logger.info(f"Shrinkage intensity for expected returns: {shrinkage_intensity:.3f}")
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Annualize if weekly returns
        expected_returns_annualized = expected_returns * 52  # Weekly to annual
        
        logger.info(f"Signal expected returns (annualized):")
        for signal, ret in expected_returns_annualized.items():
            logger.info(f"  {signal}: {ret:.2%}")
        
        return expected_returns_annualized
    
    def estimate_signal_covariance(
        self,
        historical_signal_returns: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Estimate signal covariance matrix with optional shrinkage.
        
        Args:
            historical_signal_returns: DataFrame of historical signal portfolio returns
            
        Returns:
            Covariance matrix (signal x signal)
        """
        if historical_signal_returns.empty:
            logger.warning("No historical signal returns for covariance estimation")
            return pd.DataFrame()
        
        signals = historical_signal_returns.columns.tolist()
        
        if self.use_shrinkage:
            # Ledoit-Wolf shrinkage estimator
            try:
                lw = LedoitWolf()
                lw.fit(historical_signal_returns.dropna())
                cov_matrix = pd.DataFrame(
                    lw.covariance_,
                    index=signals,
                    columns=signals
                )
                logger.info(f"Ledoit-Wolf shrinkage applied, intensity: {lw.shrinkage_:.3f}")
            except Exception as e:
                logger.warning(f"Ledoit-Wolf failed: {e}, using sample covariance")
                cov_matrix = historical_signal_returns.cov()
        else:
            cov_matrix = historical_signal_returns.cov()
        
        # Annualize (weekly to annual)
        cov_matrix_annualized = cov_matrix * 52
        
        # Log correlation matrix for diagnostics
        std = np.sqrt(np.diag(cov_matrix_annualized))
        corr_matrix = cov_matrix_annualized / np.outer(std, std)
        
        logger.info("Signal correlation matrix:")
        for i, sig1 in enumerate(signals):
            corr_str = " ".join([f"{corr_matrix.iloc[i, j]:.2f}" for j in range(len(signals))])
            logger.info(f"  {sig1}: {corr_str}")
        
        return cov_matrix_annualized
    
    def optimize(
        self,
        expected_returns: pd.Series,
        covariance: pd.DataFrame,
        previous_weights: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        Perform MVO on signals.
        
        Objective: max(w'μ - λ/2 * w'Σw - τ * ||w - w_prev||)
        
        Subject to:
        - sum(w) = 1 (fully invested)
        - w >= min_weight (minimum allocation)
        - w <= max_weight (maximum allocation)
        
        Args:
            expected_returns: Expected return per signal
            covariance: Covariance matrix of signals
            previous_weights: Previous signal weights (for turnover penalty)
            
        Returns:
            Optimal signal weights
        """
        signals = expected_returns.index.tolist()
        n_signals = len(signals)
        
        if n_signals == 0:
            logger.error("No signals to optimize")
            return pd.Series()
        
        # Align expected returns and covariance
        mu = expected_returns.loc[signals].values
        Sigma = covariance.loc[signals, signals].values
        
        # Check for numerical issues
        if np.any(np.isnan(mu)) or np.any(np.isnan(Sigma)):
            logger.error("NaN values in expected returns or covariance")
            # Fall back to equal weights
            return pd.Series(1.0 / n_signals, index=signals)
        
        # Ensure covariance is positive semi-definite
        min_eigenvalue = np.linalg.eigvalsh(Sigma).min()
        if min_eigenvalue < 0:
            logger.warning(f"Covariance not PSD (min eigenvalue: {min_eigenvalue:.6f}), adding regularization")
            Sigma = Sigma + (-min_eigenvalue + 1e-6) * np.eye(n_signals)
        
        # Define optimization variables
        w = cp.Variable(n_signals)
        
        # Objective: maximize return - risk - turnover penalty
        ret = mu @ w
        risk = cp.quad_form(w, Sigma)
        
        objective = ret - (self.risk_aversion / 2) * risk
        
        # Add turnover penalty if previous weights exist
        if previous_weights is not None and self.turnover_penalty > 0:
            prev_w = previous_weights.reindex(signals).fillna(0).values
            turnover = cp.norm(w - prev_w, 1)
            objective = objective - self.turnover_penalty * turnover
        
        # Constraints
        constraints = [
            cp.sum(w) == 1,  # Fully invested
            w >= self.min_signal_weight,  # Minimum weight
            w <= self.max_signal_weight,  # Maximum weight
        ]
        
        # Solve
        problem = cp.Problem(cp.Maximize(objective), constraints)
        
        try:
            problem.solve(solver=cp.OSQP, verbose=False)
            
            if problem.status not in ['optimal', 'optimal_inaccurate']:
                logger.warning(f"Optimization status: {problem.status}, falling back to equal weights")
                optimal_weights = pd.Series(1.0 / n_signals, index=signals)
            else:
                optimal_weights = pd.Series(w.value, index=signals)
                
                # Clean up small negative weights due to numerical precision
                optimal_weights = optimal_weights.clip(lower=0)
                optimal_weights = optimal_weights / optimal_weights.sum()  # Re-normalize
                
        except Exception as e:
            logger.error(f"Optimization failed: {e}, falling back to equal weights")
            optimal_weights = pd.Series(1.0 / n_signals, index=signals)
        
        # Store for next iteration
        self.previous_weights = optimal_weights.copy()
        
        # Log results
        logger.info("Optimal signal weights:")
        for signal, weight in optimal_weights.items():
            logger.info(f"  {signal}: {weight:.1%}")
        
        # Calculate expected portfolio metrics
        port_return = (optimal_weights.values @ mu)
        port_risk = np.sqrt(optimal_weights.values @ Sigma @ optimal_weights.values)
        sharpe = port_return / port_risk if port_risk > 0 else 0
        
        logger.info(f"Expected portfolio: Return={port_return:.2%}, Risk={port_risk:.2%}, Sharpe={sharpe:.2f}")
        
        return optimal_weights
    
    def optimize_with_views(
        self,
        expected_returns: pd.Series,
        covariance: pd.DataFrame,
        views: Dict[str, float],
        view_confidence: float = 0.5
    ) -> pd.Series:
        """
        Optimize with Black-Litterman style views on signals.
        
        Args:
            expected_returns: Prior expected returns
            covariance: Covariance matrix
            views: Dict of signal -> expected return view
            view_confidence: Confidence in views (0 to 1)
            
        Returns:
            Optimal signal weights incorporating views
        """
        # Blend prior with views
        blended_returns = expected_returns.copy()
        
        for signal, view_return in views.items():
            if signal in blended_returns.index:
                prior = blended_returns[signal]
                blended_returns[signal] = (1 - view_confidence) * prior + view_confidence * view_return
                logger.info(f"View on {signal}: {prior:.2%} -> {blended_returns[signal]:.2%}")
        
        return self.optimize(blended_returns, covariance)
    
    def get_risk_contribution(
        self,
        weights: pd.Series,
        covariance: pd.DataFrame
    ) -> pd.Series:
        """
        Calculate risk contribution of each signal.
        
        Args:
            weights: Signal weights
            covariance: Covariance matrix
            
        Returns:
            Risk contribution per signal (sums to 1)
        """
        signals = weights.index.tolist()
        w = weights.values
        Sigma = covariance.loc[signals, signals].values
        
        # Portfolio variance
        port_var = w @ Sigma @ w
        
        if port_var <= 0:
            return pd.Series(1.0 / len(signals), index=signals)
        
        # Marginal risk contribution
        mrc = Sigma @ w
        
        # Risk contribution
        rc = w * mrc / np.sqrt(port_var)
        
        # Normalize to sum to 1
        rc = rc / rc.sum()
        
        return pd.Series(rc, index=signals)


class SignalReturnEstimator:
    """
    Estimates historical signal portfolio returns for covariance estimation.
    """
    
    def __init__(
        self,
        n_quantiles: int = 5,
        rebalance_frequency: str = 'W'
    ):
        """
        Initialize signal return estimator.
        
        Args:
            n_quantiles: Number of quantiles for portfolio formation
            rebalance_frequency: 'W' for weekly, 'M' for monthly
        """
        self.n_quantiles = n_quantiles
        self.rebalance_frequency = rebalance_frequency
    
    def estimate_historical_returns(
        self,
        signals_history: Dict[pd.Timestamp, pd.DataFrame],
        returns: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Estimate historical signal portfolio returns.
        
        For each rebalance date:
        1. Form long-only portfolio of top quantile stocks for each signal
        2. Calculate portfolio return until next rebalance
        
        Args:
            signals_history: Dict of date -> signal DataFrame
            returns: Stock returns DataFrame (date x ticker)
            
        Returns:
            DataFrame of signal portfolio returns (date x signal)
        """
        dates = sorted(signals_history.keys())
        
        if len(dates) < 2:
            logger.warning("Insufficient signal history for return estimation")
            return pd.DataFrame()
        
        signal_returns_list = []
        
        for i in range(len(dates) - 1):
            current_date = dates[i]
            next_date = dates[i + 1]
            
            signals = signals_history[current_date]
            
            if signals.empty:
                continue
            
            # Get returns between dates
            period_returns = returns.loc[
                (returns.index > current_date) & (returns.index <= next_date)
            ]
            
            if period_returns.empty:
                continue
            
            # Compound returns over period
            cumulative_returns = (1 + period_returns).prod() - 1
            
            # Calculate signal portfolio returns
            signal_period_returns = {}
            
            for signal_name in signals.columns:
                signal_scores = signals[signal_name].dropna()
                
                if len(signal_scores) < self.n_quantiles * 2:
                    continue
                
                # Top quantile stocks
                threshold = signal_scores.quantile(1 - 1/self.n_quantiles)
                top_stocks = signal_scores[signal_scores >= threshold].index.tolist()
                
                # Equal-weight portfolio return
                valid_stocks = [s for s in top_stocks if s in cumulative_returns.index]
                
                if valid_stocks:
                    portfolio_return = cumulative_returns.loc[valid_stocks].mean()
                    signal_period_returns[signal_name] = portfolio_return
            
            if signal_period_returns:
                signal_returns_list.append({
                    'date': next_date,
                    **signal_period_returns
                })
        
        if not signal_returns_list:
            return pd.DataFrame()
        
        signal_returns_df = pd.DataFrame(signal_returns_list).set_index('date')
        
        logger.info(f"Estimated {len(signal_returns_df)} periods of signal returns")
        logger.info(f"Signal return statistics:")
        for col in signal_returns_df.columns:
            mean_ret = signal_returns_df[col].mean()
            std_ret = signal_returns_df[col].std()
            logger.info(f"  {col}: mean={mean_ret:.2%}, std={std_ret:.2%}")
        
        return signal_returns_df


class FactorEnhancedOptimizer(SignalOptimizer):
    """
    Enhanced signal optimizer that integrates Barra-style factor model.
    
    Uses Toraniko's factor model for:
    1. Better covariance estimation via factor decomposition
    2. Risk attribution and monitoring
    3. Factor exposure constraints
    """
    
    def __init__(
        self,
        risk_aversion: float = 1.0,
        min_signal_weight: float = 0.05,
        max_signal_weight: float = 0.50,
        turnover_penalty: float = 0.01,
        use_shrinkage: bool = True,
        shrinkage_target: str = 'constant_correlation',
        use_factor_model: bool = True,
        factor_cov_method: str = 'ledoit_wolf'
    ):
        """
        Initialize factor-enhanced optimizer.
        
        Args:
            use_factor_model: Whether to use Barra factor model for covariance
            factor_cov_method: Method for factor covariance ('ledoit_wolf', 'exponential', 'sample')
        """
        super().__init__(
            risk_aversion=risk_aversion,
            min_signal_weight=min_signal_weight,
            max_signal_weight=max_signal_weight,
            turnover_penalty=turnover_penalty,
            use_shrinkage=use_shrinkage,
            shrinkage_target=shrinkage_target
        )
        
        self.use_factor_model = use_factor_model
        self.factor_cov_method = factor_cov_method
        
        # Factor model components (lazy initialization)
        self._factor_model = None
        self._factor_integrator = None
        self._risk_attributor = None
        
        logger.info(f"FactorEnhancedOptimizer initialized:")
        logger.info(f"  - Use factor model: {use_factor_model}")
        logger.info(f"  - Factor cov method: {factor_cov_method}")
    
    @property
    def factor_model(self):
        """Lazy initialization of factor model."""
        if self._factor_model is None:
            from signals.factor_model import BarraFactorModel
            self._factor_model = BarraFactorModel(
                winsor_factor=0.05,
                residualize_styles=True,
                factor_cov_lookback=252,
                factor_cov_halflife=63
            )
        return self._factor_model
    
    @property
    def factor_integrator(self):
        """Lazy initialization of factor integrator."""
        if self._factor_integrator is None:
            from signals.factor_model import FactorModelIntegrator
            self._factor_integrator = FactorModelIntegrator(self.factor_model)
        return self._factor_integrator
    
    @property
    def risk_attributor(self):
        """Lazy initialization of risk attributor."""
        if self._risk_attributor is None:
            from signals.risk_attribution import RiskAttributor
            self._risk_attributor = RiskAttributor()
        return self._risk_attributor
    
    def estimate_signal_covariance_with_factors(
        self,
        signal_portfolios: Dict[str, pd.Series],
        factor_loadings: pd.DataFrame,
        factor_covariance: np.ndarray
    ) -> pd.DataFrame:
        """
        Estimate signal covariance using factor model.
        
        Σ_signals = B_signals' @ Σ_factors @ B_signals
        
        Where B_signals is the matrix of signal factor loadings.
        
        Args:
            signal_portfolios: Dict mapping signal name -> stock weights
            factor_loadings: Stock-level factor loadings (stocks x factors)
            factor_covariance: Factor covariance matrix
            
        Returns:
            Signal covariance matrix
        """
        # Estimate signal factor loadings
        signal_loadings = self.factor_integrator.estimate_signal_factor_loadings(
            signal_portfolios,
            factor_loadings
        )
        
        # Estimate signal covariance from factor model
        signal_cov = self.factor_integrator.estimate_signal_covariance_from_factors(
            signal_loadings,
            factor_covariance
        )
        
        # Convert to DataFrame
        signals = list(signal_portfolios.keys())
        signal_cov_df = pd.DataFrame(signal_cov, index=signals, columns=signals)
        
        logger.info(f"Estimated signal covariance using factor model")
        logger.info(f"  Signal factor loadings shape: {signal_loadings.shape}")
        
        return signal_cov_df
    
    def optimize_with_factor_model(
        self,
        expected_returns: pd.Series,
        signal_portfolios: Dict[str, pd.Series],
        factor_loadings: pd.DataFrame,
        factor_covariance: np.ndarray,
        idiosyncratic_variance: pd.Series,
        previous_weights: Optional[pd.Series] = None
    ) -> Tuple[pd.Series, Dict]:
        """
        Optimize signal weights using factor model for covariance.
        
        Args:
            expected_returns: Expected return per signal
            signal_portfolios: Dict mapping signal name -> stock weights
            factor_loadings: Stock-level factor loadings
            factor_covariance: Factor covariance matrix
            idiosyncratic_variance: Stock-level idiosyncratic variance
            previous_weights: Previous signal weights
            
        Returns:
            Tuple of (optimal_weights, risk_report)
        """
        # Estimate signal covariance using factor model
        signal_cov = self.estimate_signal_covariance_with_factors(
            signal_portfolios,
            factor_loadings,
            factor_covariance
        )
        
        # Optimize
        optimal_weights = self.optimize(expected_returns, signal_cov, previous_weights)
        
        # Generate risk report
        risk_report = self._generate_factor_risk_report(
            optimal_weights,
            signal_portfolios,
            factor_loadings,
            factor_covariance,
            idiosyncratic_variance
        )
        
        return optimal_weights, risk_report
    
    def _generate_factor_risk_report(
        self,
        signal_weights: pd.Series,
        signal_portfolios: Dict[str, pd.Series],
        factor_loadings: pd.DataFrame,
        factor_covariance: np.ndarray,
        idiosyncratic_variance: pd.Series
    ) -> Dict:
        """Generate comprehensive risk report using factor model."""
        # Construct portfolio weights from signal weights
        portfolio_weights = pd.Series(0.0, index=factor_loadings.index)
        
        for signal_name, signal_weight in signal_weights.items():
            if signal_name in signal_portfolios:
                stock_weights = signal_portfolios[signal_name]
                # Normalize stock weights within signal
                stock_weights = stock_weights / stock_weights.sum()
                # Add to portfolio
                for ticker, weight in stock_weights.items():
                    if ticker in portfolio_weights.index:
                        portfolio_weights[ticker] += signal_weight * weight
        
        # Generate risk report
        risk_report = self.risk_attributor.generate_risk_report(
            portfolio_weights,
            factor_loadings,
            factor_covariance,
            idiosyncratic_variance,
            factor_names=factor_loadings.columns.tolist()
        )
        
        return risk_report
    
    def get_factor_exposures(
        self,
        signal_weights: pd.Series,
        signal_portfolios: Dict[str, pd.Series],
        factor_loadings: pd.DataFrame
    ) -> pd.Series:
        """
        Get portfolio factor exposures given signal weights.
        
        Args:
            signal_weights: Signal weights
            signal_portfolios: Dict mapping signal name -> stock weights
            factor_loadings: Stock-level factor loadings
            
        Returns:
            Series of factor exposures
        """
        # Construct portfolio weights
        portfolio_weights = pd.Series(0.0, index=factor_loadings.index)
        
        for signal_name, signal_weight in signal_weights.items():
            if signal_name in signal_portfolios:
                stock_weights = signal_portfolios[signal_name]
                stock_weights = stock_weights / stock_weights.sum()
                for ticker, weight in stock_weights.items():
                    if ticker in portfolio_weights.index:
                        portfolio_weights[ticker] += signal_weight * weight
        
        # Calculate factor exposures
        common = list(set(portfolio_weights.index) & set(factor_loadings.index))
        w = portfolio_weights.loc[common].values
        B = factor_loadings.loc[common].values
        
        exposures = B.T @ w
        
        return pd.Series(exposures, index=factor_loadings.columns)
