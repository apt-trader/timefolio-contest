#!/usr/bin/env python3
"""
Rolling Window Dynamic Factor Models

Institutional-grade rolling window factor models for capturing:
1. Time-varying factor loadings and sensitivities
2. Regime changes and structural breaks
3. Dynamic risk exposures over time
4. Model stability and consistency monitoring
5. Adaptive forecasting with recency bias

This captures the time-varying nature of factor relationships that
static models miss, essential for real-world portfolio management.
Used extensively at top quant firms for adaptive risk models.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import joblib

# INSTITUTIONAL ENHANCEMENT: Import enhanced factor systems
from factor_engine import FactorEngine
from stable_factor_engineering import StableFactorEngineer, create_institutional_grade_factors
from regime_aware_modeling import create_regime_aware_system

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class RollingWindowFactorModel:
    """
    Dynamic factor model with rolling window estimation.
    
    Key features:
    - Time-varying factor loadings
    - Multiple window sizes for different time horizons
    - Stability monitoring and regime detection
    - Ensemble of rolling models
    - Adaptive prediction with recency weighting
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        self.rolling_models = {}
        self.factor_loadings_history = {}
        self.r2_history = {}
        self.stability_metrics = {}
        self.is_trained = False
        
        # INSTITUTIONAL ENHANCEMENT: Initialize enhanced factor systems
        logger.info("Initializing institutional-grade rolling window with enhanced factors...")
        try:
            # Initialize enhanced factor engine with regime-aware modeling
            self.enhanced_factor_engine = FactorEngine(settings={'regime_aware_modeling': True})
            
            # Initialize stable factor engineer for Korean market
            self.stable_factor_engineer = StableFactorEngineer(
                stability_target=0.5,  # Target >50% stability
                korean_adjustments=True
            )
            
            self.use_enhanced_factors = True
            logger.info("✓ Enhanced factor systems initialized for rolling window analysis")
            
        except Exception as e:
            logger.warning(f"Enhanced factor systems failed to initialize: {e}")
            logger.warning("Falling back to legacy factor calculation")
            self.use_enhanced_factors = False
        
    def _default_config(self) -> Dict:
        """Default configuration for rolling window models."""
        return {
            'window_sizes': {
                'short_term': 63,    # 3 months (quarterly earnings cycle)
                'medium_term': 126,  # 6 months (semi-annual rebalancing)
                'long_term': 252,    # 1 year (annual cycle)
                'trend': 504         # 2 years (long-term trends)
            },
            'min_periods': {
                'short_term': 30,
                'medium_term': 60,
                'long_term': 120,
                'trend': 200
            },
            'overlap_threshold': 0.7,  # Minimum overlap between windows
            'stability_threshold': 0.1,  # Maximum change for "stable" factor
            'ensemble_weights': {
                'short_term': 0.4,   # Higher weight on recent periods
                'medium_term': 0.3,
                'long_term': 0.2,
                'trend': 0.1
            },
            'regularization': {
                'alpha': 0.01,
                'method': 'ridge'  # or 'lasso', 'elastic_net'
            }
        }
    
    def fit_rolling_models(self, factors: pd.DataFrame, 
                          forward_returns: pd.Series) -> Dict:
        """
        Fit rolling window models with institutional-grade enhanced factors.
        
        Uses regime-aware modeling and stable factor engineering for improved stability.
        Returns comprehensive analysis of factor dynamics over time.
        """
        logger.info("=== TRAINING INSTITUTIONAL-GRADE ROLLING WINDOW FACTOR MODELS ===")
        
        # INSTITUTIONAL ENHANCEMENT: Use enhanced factors if available
        if self.use_enhanced_factors:
            logger.info("Applying institutional-grade factor enhancements...")
            try:
                # Apply stable factor engineering to improve factor persistence
                enhanced_factors = self.stable_factor_engineer.engineer_stable_factor_suite(
                    fundamentals=factors,  # Use input factors as proxy fundamentals
                    market_caps=pd.Series(data=1e12, index=factors.columns),  # Dummy market caps
                    historical_fundamentals=None
                )
                
                if not enhanced_factors.empty:
                    # Replace legacy factors with stable engineered factors
                    X_enhanced = enhanced_factors.reindex(factors.index, method='ffill')
                    logger.info(f"Enhanced factors: {len(X_enhanced.columns)} stable factors created")
                    logger.info(f"Enhanced factor names: {list(X_enhanced.columns)}")
                    
                    # Combine with original factors (keep best of both)
                    common_cols = set(factors.columns) & set(X_enhanced.columns)
                    X_combined = factors.copy()
                    
                    # Replace common factors with enhanced versions
                    for col in common_cols:
                        if not X_enhanced[col].isna().all():
                            X_combined[col] = X_enhanced[col]
                    
                    # Add new stable factors
                    new_factors = set(X_enhanced.columns) - set(factors.columns)
                    for col in new_factors:
                        if not X_enhanced[col].isna().all():
                            X_combined[col] = X_enhanced[col]
                    
                    factors = X_combined
                    logger.info(f"✓ Enhanced factor suite: {len(factors.columns)} total factors")
                else:
                    logger.warning("Stable factor engineering produced empty results, using original factors")
                    
            except Exception as e:
                logger.error(f"Stable factor engineering failed: {e}, using original factors")
        
        # Align data
        common_idx = factors.index.intersection(forward_returns.index)
        X = factors.loc[common_idx].copy()
        y = forward_returns.loc[common_idx].copy()
        
        # Remove missing values
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]
        
        if len(X) < 200:
            logger.error("Insufficient data for rolling window analysis")
            return {}
        
        logger.info(f"Rolling window data: {len(X)} observations, {len(X.columns)} factors")
        logger.info(f"Date range: {X.index.min()} to {X.index.max()}")
        
        results = {}
        
        # Fit models for each window size
        for window_name, window_size in self.config['window_sizes'].items():
            min_periods = self.config['min_periods'][window_name]
            
            logger.info(f"Training {window_name} rolling model (window={window_size}, min_periods={min_periods})...")
            
            try:
                model_results = self._fit_single_rolling_model(
                    X, y, window_size, min_periods, window_name
                )
                results[window_name] = model_results
                
                # Log basic statistics
                if model_results and 'r2_history' in model_results:
                    r2_values = model_results['r2_history']
                    mean_r2 = np.mean(r2_values)
                    std_r2 = np.std(r2_values)
                    logger.info(f"{window_name:12s} - Mean R²: {mean_r2:.4f} ± {std_r2:.4f}")
                
            except Exception as e:
                logger.error(f"Rolling model {window_name} failed: {e}")
                results[window_name] = {'failed': True, 'error': str(e)}
        
        # Analyze factor stability across windows
        logger.info("Analyzing factor stability across time horizons...")
        stability_analysis = self._analyze_factor_stability(results)
        results['stability_analysis'] = stability_analysis
        
        # Create ensemble predictions
        logger.info("Creating ensemble rolling window predictions...")
        ensemble_results = self._create_ensemble_predictions(X, y, results)
        results['ensemble'] = ensemble_results
        
        self.rolling_models = results
        self.is_trained = True
        
        # Generate comprehensive report
        report = self._generate_rolling_analysis_report(results)
        report_path = Path("reports/rolling_window_analysis.md")
        report_path.parent.mkdir(exist_ok=True)
        report_path.write_text(report)
        logger.info(f"Rolling window analysis report saved to {report_path}")
        
        return results
    
    def _fit_single_rolling_model(self, X: pd.DataFrame, y: pd.Series,
                                 window_size: int, min_periods: int,
                                 window_name: str) -> Dict:
        """Fit a single rolling window model."""
        
        factor_loadings = []
        r2_values = []
        dates = []
        model_coefficients = {}
        
        # Initialize coefficient storage
        for factor in X.columns:
            model_coefficients[factor] = []
        
        # Rolling window estimation
        for i in range(min_periods, len(X)):
            start_idx = max(0, i - window_size)
            end_idx = i + 1
            
            # Extract window data
            X_window = X.iloc[start_idx:end_idx]
            y_window = y.iloc[start_idx:end_idx]
            
            # Remove any remaining NaNs in this window
            window_mask = ~(X_window.isna().any(axis=1) | y_window.isna())
            X_window = X_window[window_mask]
            y_window = y_window[window_mask]
            
            if len(X_window) < min_periods // 2:
                continue
            
            try:
                # Fit regularized regression
                if self.config['regularization']['method'] == 'ridge':
                    model = Ridge(alpha=self.config['regularization']['alpha'])
                elif self.config['regularization']['method'] == 'lasso':
                    model = Lasso(alpha=self.config['regularization']['alpha'])
                else:
                    model = LinearRegression()
                
                model.fit(X_window, y_window)
                
                # Store results
                dates.append(X.index[i])
                
                # Factor loadings (coefficients)
                for j, factor in enumerate(X.columns):
                    model_coefficients[factor].append(model.coef_[j])
                
                # Model performance
                y_pred = model.predict(X_window)
                r2 = r2_score(y_window, y_pred)
                r2_values.append(r2)
                
            except Exception as e:
                logger.debug(f"Rolling window fit failed at {X.index[i]}: {e}")
                continue
        
        if len(dates) == 0:
            return {'failed': True, 'error': 'No successful rolling window fits'}
        
        # Convert to DataFrame for easy analysis
        factor_loadings_df = pd.DataFrame(model_coefficients, index=dates)
        
        return {
            'factor_loadings': factor_loadings_df,
            'r2_history': r2_values,
            'dates': dates,
            'window_size': window_size,
            'min_periods': min_periods,
            'n_periods': len(dates)
        }
    
    def _analyze_factor_stability(self, results: Dict) -> Dict:
        """Analyze factor loading stability across different time horizons."""
        
        stability_metrics = {}
        
        # Calculate stability for each factor across windows
        all_factors = set()
        for window_name, window_results in results.items():
            if 'failed' not in window_results and 'factor_loadings' in window_results:
                all_factors.update(window_results['factor_loadings'].columns)
        
        for factor in all_factors:
            factor_stability = {}
            
            for window_name, window_results in results.items():
                if 'failed' not in window_results and 'factor_loadings' in window_results:
                    loadings = window_results['factor_loadings'][factor]
                    
                    if len(loadings) > 10:  # Need sufficient data
                        # Calculate stability metrics
                        mean_loading = loadings.mean()
                        std_loading = loadings.std()
                        cv = abs(std_loading / mean_loading) if mean_loading != 0 else np.inf
                        
                        # Trend analysis (simple linear trend)
                        time_trend = np.polyfit(range(len(loadings)), loadings, 1)[0]
                        
                        factor_stability[window_name] = {
                            'mean_loading': mean_loading,
                            'std_loading': std_loading,
                            'coefficient_of_variation': cv,
                            'time_trend': time_trend,
                            'is_stable': cv < self.config['stability_threshold']
                        }
            
            stability_metrics[factor] = factor_stability
        
        # Overall stability summary
        stable_factors = []
        unstable_factors = []
        
        for factor, stability_data in stability_metrics.items():
            # A factor is considered stable if it's stable across most windows
            stable_count = sum(1 for w in stability_data.values() 
                             if w.get('is_stable', False))
            total_count = len(stability_data)
            
            if total_count > 0 and stable_count / total_count >= 0.5:
                stable_factors.append(factor)
            else:
                unstable_factors.append(factor)
        
        return {
            'factor_stability': stability_metrics,
            'stable_factors': stable_factors,
            'unstable_factors': unstable_factors,
            'stability_summary': {
                'n_stable': len(stable_factors),
                'n_unstable': len(unstable_factors),
                'stability_ratio': len(stable_factors) / (len(stable_factors) + len(unstable_factors))
                                 if (len(stable_factors) + len(unstable_factors)) > 0 else 0
            }
        }
    
    def _create_ensemble_predictions(self, X: pd.DataFrame, y: pd.Series,
                                   results: Dict) -> Dict:
        """Create ensemble predictions from multiple rolling windows."""
        
        ensemble_weights = self.config['ensemble_weights']
        
        # Get the latest factor loadings from each window
        latest_loadings = {}
        window_weights = {}
        
        for window_name, window_results in results.items():
            if ('failed' not in window_results and 
                'factor_loadings' in window_results and 
                window_name in ensemble_weights):
                
                loadings_df = window_results['factor_loadings']
                if not loadings_df.empty:
                    latest_loadings[window_name] = loadings_df.iloc[-1]  # Most recent
                    window_weights[window_name] = ensemble_weights[window_name]
        
        if not latest_loadings:
            return {'failed': True, 'error': 'No valid rolling window results for ensemble'}
        
        # Normalize weights
        total_weight = sum(window_weights.values())
        normalized_weights = {k: v/total_weight for k, v in window_weights.items()}
        
        # Calculate ensemble factor loadings (weighted average)
        ensemble_loadings = {}
        all_factors = set()
        for loadings in latest_loadings.values():
            all_factors.update(loadings.index)
        
        for factor in all_factors:
            weighted_loading = 0
            total_weight = 0
            
            for window_name, loadings in latest_loadings.items():
                if factor in loadings.index:
                    weight = normalized_weights[window_name]
                    weighted_loading += weight * loadings[factor]
                    total_weight += weight
            
            if total_weight > 0:
                ensemble_loadings[factor] = weighted_loading / total_weight
        
        return {
            'ensemble_loadings': ensemble_loadings,
            'window_weights': normalized_weights,
            'contributing_windows': list(latest_loadings.keys())
        }
    
    def predict(self, factors: pd.DataFrame) -> pd.Series:
        """Generate ensemble predictions using rolling window models."""
        
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        ensemble_results = self.rolling_models.get('ensemble', {})
        if 'failed' in ensemble_results:
            logger.error("No valid ensemble model for prediction")
            return pd.Series(0, index=factors.index, name='rolling_prediction')
        
        ensemble_loadings = ensemble_results['ensemble_loadings']
        
        # Calculate predictions
        predictions = pd.Series(0, index=factors.index, name='rolling_prediction')
        
        for factor, loading in ensemble_loadings.items():
            if factor in factors.columns:
                predictions += loading * factors[factor]
        
        return predictions
    
    def get_current_factor_exposures(self) -> Dict:
        """Get current factor exposures from ensemble model."""
        
        if not self.is_trained:
            return {}
        
        ensemble_results = self.rolling_models.get('ensemble', {})
        if 'failed' in ensemble_results:
            return {}
        
        return ensemble_results.get('ensemble_loadings', {})
    
    def _generate_rolling_analysis_report(self, results: Dict) -> str:
        """Generate comprehensive rolling window analysis report."""
        
        report = []
        report.append("# Rolling Window Factor Model Analysis")
        report.append("=" * 50)
        report.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Model Performance Summary
        report.append("\n## Rolling Window Performance Summary")
        report.append("| Window | Size | Periods | Mean R² | Std R² | Status |")
        report.append("|--------|------|---------|---------|--------|--------|")
        
        for window_name, window_results in results.items():
            if window_name in ['stability_analysis', 'ensemble']:
                continue
                
            if 'failed' in window_results:
                report.append(f"| {window_name:12s} | - | - | - | - | ❌ Failed |")
            else:
                size = window_results.get('window_size', '-')
                periods = window_results.get('n_periods', '-')
                r2_history = window_results.get('r2_history', [])
                
                if r2_history:
                    mean_r2 = np.mean(r2_history)
                    std_r2 = np.std(r2_history)
                    report.append(f"| {window_name:12s} | {size:3d} | {periods:7d} | {mean_r2:7.4f} | {std_r2:6.4f} | ✅ OK |")
                else:
                    report.append(f"| {window_name:12s} | {size:3d} | {periods:7d} | - | - | ⚠️ No data |")
        
        # Factor Stability Analysis
        stability_analysis = results.get('stability_analysis', {})
        if stability_analysis:
            summary = stability_analysis.get('stability_summary', {})
            
            report.append(f"\n## Factor Stability Analysis")
            report.append(f"- **Stable factors**: {summary.get('n_stable', 0)}")
            report.append(f"- **Unstable factors**: {summary.get('n_unstable', 0)}")
            report.append(f"- **Stability ratio**: {summary.get('stability_ratio', 0):.1%}")
            
            stable_factors = stability_analysis.get('stable_factors', [])
            unstable_factors = stability_analysis.get('unstable_factors', [])
            
            if stable_factors:
                report.append(f"\n### ✅ Stable Factors")
                for factor in stable_factors[:10]:  # Top 10
                    report.append(f"- {factor}")
            
            if unstable_factors:
                report.append(f"\n### ⚠️ Unstable Factors")
                for factor in unstable_factors[:10]:  # Top 10
                    report.append(f"- {factor}")
        
        # Ensemble Results
        ensemble_results = results.get('ensemble', {})
        if 'failed' not in ensemble_results:
            report.append(f"\n## Ensemble Model")
            
            window_weights = ensemble_results.get('window_weights', {})
            if window_weights:
                report.append("### Window Weights")
                for window, weight in window_weights.items():
                    report.append(f"- {window}: {weight:.1%}")
            
            ensemble_loadings = ensemble_results.get('ensemble_loadings', {})
            if ensemble_loadings:
                report.append("\n### Current Factor Exposures")
                report.append("| Factor | Loading | Magnitude |")
                report.append("|--------|---------|-----------|")
                
                # Sort by absolute magnitude
                sorted_loadings = sorted(ensemble_loadings.items(), 
                                       key=lambda x: abs(x[1]), reverse=True)
                
                for factor, loading in sorted_loadings[:15]:  # Top 15
                    magnitude = "High" if abs(loading) > 0.1 else "Medium" if abs(loading) > 0.05 else "Low"
                    report.append(f"| {factor:15s} | {loading:8.4f} | {magnitude:9s} |")
        
        # Recommendations
        report.append(f"\n## Recommendations")
        
        if stability_analysis:
            stable_ratio = stability_analysis.get('stability_summary', {}).get('stability_ratio', 0)
            
            if stable_ratio > 0.7:
                report.append("✅ **High factor stability** - Model is suitable for production use")
            elif stable_ratio > 0.5:
                report.append("⚠️ **Moderate factor stability** - Monitor unstable factors closely")
            else:
                report.append("❌ **Low factor stability** - Consider regime detection or shorter windows")
        
        # Performance assessment
        performance_ok = True
        for window_name, window_results in results.items():
            if (window_name not in ['stability_analysis', 'ensemble'] and 
                'failed' not in window_results):
                r2_history = window_results.get('r2_history', [])
                if r2_history and np.mean(r2_history) < 0.05:
                    performance_ok = False
                    break
        
        if performance_ok:
            report.append("✅ **Good model performance** - Rolling windows show consistent explanatory power")
        else:
            report.append("⚠️ **Poor model performance** - Consider factor engineering or longer training periods")
        
        return "\n".join(report)
    
    def save_model(self, filepath: str):
        """Save the rolling window model ensemble."""
        model_data = {
            'rolling_models': self.rolling_models,
            'config': self.config,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Rolling window models saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a pre-trained rolling window model ensemble."""
        model_data = joblib.load(filepath)
        
        self.rolling_models = model_data['rolling_models']
        self.config = model_data['config']
        self.is_trained = model_data['is_trained']
        
        logger.info(f"Rolling window models loaded from {filepath}")

if __name__ == "__main__":
    logger.info("Rolling Window Factor Models initialized")
    logger.info("Key features:")
    logger.info("1. Multiple time horizons (3M, 6M, 1Y, 2Y)")
    logger.info("2. Time-varying factor loadings")
    logger.info("3. Factor stability analysis")
    logger.info("4. Ensemble predictions with recency weighting")
    logger.info("5. Regime change detection through stability monitoring")
    logger.info("6. Comprehensive reporting and diagnostics")
