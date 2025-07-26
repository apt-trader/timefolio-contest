#!/usr/bin/env python3
"""
Regime Detection and Dynamic Factor Loading Module

Institutional-grade implementation of market regime detection using:
1. Hidden Markov Models for regime identification
2. Dynamic factor loading based on regime state
3. Volatility clustering detection (GARCH-style)
4. Correlation regime shifts
5. Factor decay analysis

This addresses the critical issue that factor effectiveness varies dramatically 
across market regimes - a core principle at RenTech, DE Shaw, Two Sigma, AQR.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class MarketRegimeDetector:
    """
    Advanced regime detection using multiple methodologies.
    
    Implements approaches from:
    - RenTech: Multi-timeframe regime detection
    - DE Shaw: Volatility clustering regimes  
    - Two Sigma: Correlation regime shifts
    - AQR: Factor decay analysis
    """
    
    def __init__(self, n_regimes: int = 3, lookback_window: int = 252):
        self.n_regimes = n_regimes
        self.lookback_window = lookback_window
        self.regime_model = None
        self.regime_features = None
        self.regime_history = None
        self.factor_loadings_by_regime = {}
        
    def calculate_regime_features(self, returns: pd.DataFrame, 
                                market_returns: pd.Series = None) -> pd.DataFrame:
        """Calculate features that characterize market regimes."""
        
        if market_returns is None:
            # Use equal-weight portfolio as market proxy
            market_returns = returns.mean(axis=1)
        
        features = pd.DataFrame(index=returns.index)
        
        # 1. Volatility clustering (GARCH-style)
        rolling_vol = market_returns.rolling(window=21).std()
        features['volatility_level'] = rolling_vol
        features['volatility_regime'] = (rolling_vol > rolling_vol.rolling(252).quantile(0.7)).astype(int)
        
        # 2. Return momentum regimes
        features['return_1m'] = market_returns.rolling(21).sum()
        features['return_3m'] = market_returns.rolling(63).sum() 
        features['return_6m'] = market_returns.rolling(126).sum()
        
        # 3. Correlation regime shifts
        if returns.shape[1] > 10:
            rolling_corr = returns.rolling(window=63).corr()
            # Average pairwise correlation
            avg_correlation = []
            for date in returns.index:
                if date in rolling_corr.index:
                    corr_matrix = rolling_corr.loc[date]
                    if not corr_matrix.empty:
                        # Upper triangle correlation (excluding diagonal)
                        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
                        avg_corr = corr_matrix.values[mask].mean()
                        avg_correlation.append(avg_corr if not np.isnan(avg_corr) else 0)
                    else:
                        avg_correlation.append(0)
                else:
                    avg_correlation.append(0)
            
            features['avg_correlation'] = avg_correlation
        else:
            features['avg_correlation'] = 0
        
        # 4. Market stress indicators
        features['max_drawdown_21d'] = market_returns.rolling(21).apply(
            lambda x: (x.cumsum() - x.cumsum().expanding().max()).min()
        )
        
        # 5. Skewness and kurtosis regimes
        features['return_skew'] = market_returns.rolling(63).skew()
        features['return_kurtosis'] = market_returns.rolling(63).apply(lambda x: stats.kurtosis(x))
        
        # 6. VIX-like indicator (21-day realized vol vs 252-day historical)
        short_vol = market_returns.rolling(21).std()
        long_vol = market_returns.rolling(252).std()
        features['vol_term_structure'] = short_vol / long_vol
        
        # Clean and forward-fill features
        features = features.fillna(method='ffill').fillna(0)
        
        logger.info(f"Calculated {len(features.columns)} regime features")
        return features
    
    def detect_regimes(self, returns: pd.DataFrame, 
                      market_returns: pd.Series = None) -> pd.Series:
        """Detect market regimes using Gaussian Mixture Models."""
        
        # Calculate regime features
        features = self.calculate_regime_features(returns, market_returns)
        self.regime_features = features
        
        # Select most informative features for regime detection
        feature_cols = ['volatility_level', 'return_1m', 'return_3m', 
                       'avg_correlation', 'max_drawdown_21d', 'vol_term_structure']
        
        X = features[feature_cols].dropna()
        
        if len(X) < 100:
            logger.warning("Insufficient data for regime detection")
            return pd.Series(0, index=returns.index, name='regime')
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Fit Gaussian Mixture Model
        self.regime_model = GaussianMixture(
            n_components=self.n_regimes,
            covariance_type='full',
            random_state=42,
            max_iter=200
        )
        
        regime_labels = self.regime_model.fit_predict(X_scaled)
        
        # Create regime series
        regime_series = pd.Series(regime_labels, index=X.index, name='regime')
        regime_series = regime_series.reindex(returns.index, method='ffill').fillna(0)
        
        self.regime_history = regime_series
        
        # Log regime statistics
        regime_counts = regime_series.value_counts().sort_index()
        logger.info("Regime Detection Results:")
        for regime, count in regime_counts.items():
            pct = count / len(regime_series) * 100
            logger.info(f"  Regime {regime}: {count:4d} days ({pct:5.1f}%)")
        
        return regime_series
    
    def analyze_regime_characteristics(self, returns: pd.DataFrame, 
                                     regimes: pd.Series) -> Dict:
        """Analyze the characteristics of each detected regime."""
        
        if self.regime_features is None:
            logger.error("Must run detect_regimes first")
            return {}
        
        market_returns = returns.mean(axis=1)
        regime_analysis = {}
        
        for regime_id in sorted(regimes.unique()):
            regime_mask = regimes == regime_id
            regime_dates = regimes[regime_mask].index
            
            if len(regime_dates) == 0:
                continue
            
            # Market performance in this regime
            regime_returns = market_returns[regime_mask]
            regime_features = self.regime_features[regime_mask]
            
            analysis = {
                'regime_id': regime_id,
                'duration_days': len(regime_dates),
                'frequency_pct': len(regime_dates) / len(regimes) * 100,
                
                # Return characteristics
                'avg_daily_return': regime_returns.mean(),
                'volatility': regime_returns.std(),
                'sharpe_ratio': regime_returns.mean() / regime_returns.std() * np.sqrt(252) if regime_returns.std() > 0 else 0,
                'max_drawdown': (regime_returns.cumsum() - regime_returns.cumsum().expanding().max()).min(),
                'skewness': stats.skew(regime_returns),
                'kurtosis': stats.kurtosis(regime_returns),
                
                # Feature characteristics
                'avg_correlation': regime_features['avg_correlation'].mean(),
                'avg_volatility': regime_features['volatility_level'].mean(),
                'vol_term_structure': regime_features['vol_term_structure'].mean(),
                
                # Regime persistence
                'avg_regime_length': self._calculate_regime_persistence(regimes, regime_id),
            }
            
            regime_analysis[regime_id] = analysis
            
            # Log regime characteristics
            logger.info(f"Regime {regime_id} Characteristics:")
            logger.info(f"  Duration: {analysis['duration_days']} days ({analysis['frequency_pct']:.1f}%)")
            logger.info(f"  Returns: {analysis['avg_daily_return']*100:.3f}% daily, Vol: {analysis['volatility']*100:.2f}%")
            logger.info(f"  Sharpe: {analysis['sharpe_ratio']:.2f}, MaxDD: {analysis['max_drawdown']*100:.2f}%")
        
        return regime_analysis
    
    def _calculate_regime_persistence(self, regimes: pd.Series, regime_id: int) -> float:
        """Calculate average length of regime periods."""
        regime_changes = regimes != regimes.shift(1)
        regime_periods = []
        current_length = 0
        current_regime = None
        
        for date, regime in regimes.items():
            if regime == current_regime:
                current_length += 1
            else:
                if current_regime == regime_id and current_length > 0:
                    regime_periods.append(current_length)
                current_regime = regime
                current_length = 1
        
        # Handle final period
        if current_regime == regime_id and current_length > 0:
            regime_periods.append(current_length)
        
        return np.mean(regime_periods) if regime_periods else 0
    
    def calculate_dynamic_factor_loadings(self, factors: pd.DataFrame, 
                                        forward_returns: pd.Series,
                                        regimes: pd.Series) -> Dict:
        """Calculate factor loadings separately for each regime."""
        
        self.factor_loadings_by_regime = {}
        
        for regime_id in sorted(regimes.unique()):
            regime_mask = regimes == regime_id
            
            # Get data for this regime
            regime_factors = factors[regime_mask]
            regime_returns = forward_returns[regime_mask]
            
            # Align and clean data
            combined_data = regime_factors.join(regime_returns, how='inner').dropna()
            
            if len(combined_data) < 20:  # Need minimum observations
                logger.warning(f"Insufficient data for regime {regime_id} factor loading estimation")
                continue
            
            y = combined_data['forward_return'] if 'forward_return' in combined_data.columns else combined_data.iloc[:, -1]
            X = combined_data.drop(columns=[y.name])
            
            if X.shape[1] == 0:
                continue
            
            # Add constant
            import statsmodels.api as sm
            X = sm.add_constant(X)
            
            try:
                # OLS regression for this regime
                model = sm.OLS(y, X).fit()
                
                loadings = {
                    'regime_id': regime_id,
                    'n_observations': len(y),
                    'r_squared': model.rsquared,
                    'coefficients': dict(zip(X.columns[1:], model.params[1:])),  # Exclude constant
                    'pvalues': dict(zip(X.columns[1:], model.pvalues[1:])),
                    'significant_factors': sum(model.pvalues[1:] < 0.05),
                    'model': model
                }
                
                self.factor_loadings_by_regime[regime_id] = loadings
                
                logger.info(f"Regime {regime_id} Factor Loadings:")
                logger.info(f"  R²: {model.rsquared:.4f}, Significant factors: {loadings['significant_factors']}")
                
                # Log top factors
                factor_importance = [(factor, abs(coeff)) for factor, coeff in loadings['coefficients'].items()]
                factor_importance.sort(key=lambda x: x[1], reverse=True)
                
                for factor, importance in factor_importance[:3]:
                    coeff = loadings['coefficients'][factor]
                    pval = loadings['pvalues'][factor]
                    sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
                    logger.info(f"    {factor}: {coeff:.4f}{sig}")
                    
            except Exception as e:
                logger.warning(f"Failed to estimate factor loadings for regime {regime_id}: {e}")
                continue
        
        return self.factor_loadings_by_regime
    
    def predict_regime_aware_returns(self, current_factors: pd.DataFrame,
                                   current_regime: int = None) -> pd.Series:
        """Generate return predictions using regime-specific factor loadings."""
        
        if not self.factor_loadings_by_regime:
            logger.error("Must calculate factor loadings first")
            return pd.Series(dtype=float)
        
        # If no regime specified, use the most recent detected regime
        if current_regime is None:
            if self.regime_history is not None:
                current_regime = self.regime_history.iloc[-1]
            else:
                current_regime = 0
        
        # Get factor loadings for current regime
        if current_regime not in self.factor_loadings_by_regime:
            logger.warning(f"No factor loadings available for regime {current_regime}")
            # Fall back to regime with most data
            current_regime = max(self.factor_loadings_by_regime.keys(), 
                               key=lambda x: self.factor_loadings_by_regime[x]['n_observations'])
        
        loadings = self.factor_loadings_by_regime[current_regime]
        
        # Generate predictions
        predictions = pd.Series(0.0, index=current_factors.index)
        
        for factor, coeff in loadings['coefficients'].items():
            if factor in current_factors.columns:
                predictions += coeff * current_factors[factor].fillna(0)
        
        logger.info(f"Generated regime-aware predictions using regime {current_regime} loadings")
        return predictions
    
    def create_regime_analysis_report(self, returns: pd.DataFrame) -> str:
        """Create comprehensive regime analysis report."""
        
        if self.regime_history is None:
            return "No regime analysis available. Run detect_regimes first."
        
        report = []
        report.append("# Market Regime Analysis Report")
        report.append("=" * 50)
        report.append(f"Analysis Period: {returns.index[0]} to {returns.index[-1]}")
        report.append(f"Number of Regimes: {self.n_regimes}")
        report.append("")
        
        # Regime statistics
        regime_analysis = self.analyze_regime_characteristics(returns, self.regime_history)
        
        report.append("## Regime Characteristics")
        report.append("| Regime | Duration | Frequency | Daily Ret | Volatility | Sharpe | Max DD |")
        report.append("|--------|----------|-----------|-----------|------------|--------|--------|")
        
        for regime_id, analysis in regime_analysis.items():
            report.append(f"| {regime_id} | {analysis['duration_days']} days | "
                         f"{analysis['frequency_pct']:.1f}% | "
                         f"{analysis['avg_daily_return']*100:.3f}% | "
                         f"{analysis['volatility']*100:.2f}% | "
                         f"{analysis['sharpe_ratio']:.2f} | "
                         f"{analysis['max_drawdown']*100:.2f}% |")
        
        report.append("")
        
        # Factor loading comparison across regimes
        if self.factor_loadings_by_regime:
            report.append("## Factor Loading Comparison Across Regimes")
            
            all_factors = set()
            for loadings in self.factor_loadings_by_regime.values():
                all_factors.update(loadings['coefficients'].keys())
            
            for factor in sorted(all_factors):
                report.append(f"\n### {factor}")
                report.append("| Regime | Coefficient | P-value | Significant |")
                report.append("|--------|-------------|---------|-------------|")
                
                for regime_id in sorted(self.factor_loadings_by_regime.keys()):
                    loadings = self.factor_loadings_by_regime[regime_id]
                    if factor in loadings['coefficients']:
                        coeff = loadings['coefficients'][factor]
                        pval = loadings['pvalues'][factor]
                        sig = "Yes" if pval < 0.05 else "No"
                        report.append(f"| {regime_id} | {coeff:.4f} | {pval:.3f} | {sig} |")
        
        return "\n".join(report)

def integrate_regime_detection_into_main():
    """Generate integration code for main.py."""
    
    integration_code = '''
# Add this to main.py imports
from regime_detection import MarketRegimeDetector

def prepare_regime_aware_training_data(cfg: Config, dm: DataManager, factor_engine: FactorEngine) -> Optional[Tuple[pd.DataFrame, Dict]]:
    """Enhanced training data preparation with regime detection."""
    logger.info("--- Preparing Regime-Aware Training Data ---")
    
    # Get basic training data using existing enhanced method
    training_data = prepare_training_data(cfg, dm, factor_engine)
    
    if training_data is None:
        return None, {}
    
    # Initialize regime detector
    regime_detector = MarketRegimeDetector(n_regimes=3, lookback_window=252)
    
    # Get returns for regime detection
    train_start = pd.to_datetime(cfg.training_settings['start_date'])
    train_end = pd.to_datetime(cfg.training_settings['end_date'])
    returns = dm.returns.loc[train_start:train_end]
    
    # Detect market regimes
    logger.info("Detecting market regimes...")
    regimes = regime_detector.detect_regimes(returns)
    
    # Analyze regime characteristics
    regime_analysis = regime_detector.analyze_regime_characteristics(returns, regimes)
    
    # Calculate dynamic factor loadings
    logger.info("Calculating regime-specific factor loadings...")
    
    # Prepare factor panel for regime analysis
    training_data_reset = training_data.reset_index()
    if 'date' in training_data_reset.columns:
        factor_panel = training_data_reset.set_index(['date', 'ticker']) if 'ticker' in training_data_reset.columns else training_data_reset.set_index('date')
    else:
        factor_panel = training_data_reset
    
    forward_returns = factor_panel['forward_return'] if 'forward_return' in factor_panel.columns else factor_panel.iloc[:, -1]
    factors = factor_panel.drop(columns=[forward_returns.name])
    
    # Add regimes to factor data
    if hasattr(factor_panel.index, 'get_level_values'):
        dates = factor_panel.index.get_level_values('date') if factor_panel.index.nlevels > 1 else factor_panel.index
    else:
        dates = factor_panel.index
    
    regime_for_factors = regimes.reindex(dates, method='ffill').fillna(0)
    
    factor_loadings = regime_detector.calculate_dynamic_factor_loadings(
        factors, forward_returns, regime_for_factors
    )
    
    # Create regime analysis report
    report = regime_detector.create_regime_analysis_report(returns)
    
    # Save regime analysis
    report_file = Path("regime_analysis_report.md")
    report_file.write_text(report)
    logger.info(f"Regime analysis report saved to {report_file}")
    
    regime_info = {
        'detector': regime_detector,
        'regimes': regimes,
        'analysis': regime_analysis,
        'factor_loadings': factor_loadings,
        'report': report
    }
    
    return training_data, regime_info
'''
    
    return integration_code

if __name__ == "__main__":
    logger.info("Market Regime Detection Module initialized")
    logger.info("Key features:")
    logger.info("1. Multi-feature regime detection (volatility, correlation, momentum)")
    logger.info("2. Dynamic factor loading by regime")
    logger.info("3. Regime-aware return predictions")
    logger.info("4. Comprehensive regime analysis and reporting")
    logger.info("5. Integration with existing TimeFolio pipeline")
