#!/usr/bin/env python3
"""
Robust and Quantile Regression Models

Institutional-grade robust regression techniques for factor models:
1. Huber Regression (robust to outliers)
2. Quantile Regression (different return distribution percentiles)  
3. RANSAC (outlier-resistant fitting)
4. Theil-Sen Regression (median-based robust regression)
5. Ensemble of robust methods with uncertainty quantification
6. Regime-aware robust modeling

These methods are essential for handling the heavy tails and outliers
common in financial data that destroy ordinary least squares performance.
Used extensively at top quant firms for production risk models.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from sklearn.linear_model import (
    HuberRegressor, RANSACRegressor, TheilSenRegressor,
    QuantileRegressor, Ridge, LinearRegression
)
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import warnings
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class RobustRegressionEnsemble:
    """
    Advanced robust regression ensemble for factor models.
    
    Combines multiple robust regression techniques to handle:
    - Heavy-tailed return distributions
    - Outliers and regime changes  
    - Non-normal residuals
    - Heteroscedasticity
    - Model uncertainty quantification
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        self.models = {}
        self.quantile_models = {}
        self.feature_scaler = None
        self.training_metrics = {}
        self.is_trained = False
        self.prediction_intervals = {}
        
    def _default_config(self) -> Dict:
        """Default configuration for robust regression ensemble."""
        return {
            'huber': {
                'epsilon': 1.35,  # Standard Huber parameter
                'max_iter': 1000,
                'alpha': 0.0001,  # Light regularization
                'fit_intercept': True,
                'tol': 1e-5
            },
            'ransac': {
                'min_samples': 0.7,  # Use 70% of data minimum
                'residual_threshold': None,  # Will be set automatically
                'max_trials': 100,
                'stop_probability': 0.99,
                'random_state': 42
            },
            'theil_sen': {
                'fit_intercept': True,
                'copy_X': True,
                'max_subpopulation': 1e4,
                'n_subsamples': None,
                'max_iter': 300,
                'tol': 1e-3,
                'random_state': 42
            },
            'quantile_regression': {
                'quantiles': [0.1, 0.25, 0.5, 0.75, 0.9],  # Key quantiles
                'alpha': 0.01,  # Regularization
                'fit_intercept': True,
                'solver': 'highs'  # Efficient linear programming solver
            },
            'ensemble': {
                'cv_folds': 5,
                'weight_method': 'performance_weighted',  # or 'equal_weighted'
                'use_prediction_intervals': True,
                'confidence_level': 0.95
            }
        }
    
    def _calculate_residual_threshold(self, y: pd.Series) -> float:
        """Calculate adaptive residual threshold for RANSAC."""
        # Use median absolute deviation (robust scale estimator)
        mad = np.median(np.abs(y - np.median(y)))
        # Convert MAD to standard deviation equivalent
        sigma_equiv = 1.4826 * mad
        # Use 2.5 sigma as threshold (captures ~99% of normal data)
        return 2.5 * sigma_equiv
    
    def train_robust_models(self, factors: pd.DataFrame, 
                          forward_returns: pd.Series) -> Dict:
        """Train ensemble of robust regression models."""
        
        logger.info("=== TRAINING ROBUST REGRESSION ENSEMBLE ===")
        
        # Align data
        common_idx = factors.index.intersection(forward_returns.index)
        X = factors.loc[common_idx].copy()
        y = forward_returns.loc[common_idx].copy()
        
        # Remove missing values
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]
        
        if len(X) < 50:
            logger.error("Insufficient data for robust regression training")
            return {}
        
        logger.info(f"Training data: {len(X)} observations, {len(X.columns)} features")
        
        # Robust scaling (less sensitive to outliers than StandardScaler)
        self.feature_scaler = RobustScaler()
        X_scaled = pd.DataFrame(
            self.feature_scaler.fit_transform(X),
            index=X.index,
            columns=X.columns
        )
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=self.config['ensemble']['cv_folds'])
        
        # 1. Train Huber Regression
        logger.info("Training Huber Regression (robust to outliers)...")
        huber_model = HuberRegressor(**self.config['huber'])
        
        cv_scores = cross_val_score(huber_model, X_scaled, y, cv=tscv, scoring='r2')
        huber_model.fit(X_scaled, y)
        self.models['huber'] = huber_model
        
        train_pred = huber_model.predict(X_scaled)
        self.training_metrics['huber'] = {
            'r2_score': r2_score(y, train_pred),
            'mse': mean_squared_error(y, train_pred),
            'mae': mean_absolute_error(y, train_pred),
            'cv_r2_mean': cv_scores.mean(),
            'cv_r2_std': cv_scores.std(),
            'outlier_fraction': np.sum(huber_model.outliers_) / len(y) if hasattr(huber_model, 'outliers_') else 0
        }
        
        # 2. Train RANSAC Regression
        logger.info("Training RANSAC Regression (outlier-resistant)...")
        ransac_config = self.config['ransac'].copy()
        ransac_config['residual_threshold'] = self._calculate_residual_threshold(y)
        
        ransac_model = RANSACRegressor(
            estimator=Ridge(alpha=0.1),
            **ransac_config
        )
        
        try:
            cv_scores = cross_val_score(ransac_model, X_scaled, y, cv=tscv, scoring='r2')
            ransac_model.fit(X_scaled, y)
            self.models['ransac'] = ransac_model
            
            train_pred = ransac_model.predict(X_scaled)
            inlier_mask = ransac_model.inlier_mask_
            
            self.training_metrics['ransac'] = {
                'r2_score': r2_score(y, train_pred),
                'mse': mean_squared_error(y, train_pred),
                'mae': mean_absolute_error(y, train_pred),
                'cv_r2_mean': cv_scores.mean(),
                'cv_r2_std': cv_scores.std(),
                'inlier_fraction': np.sum(inlier_mask) / len(y),
                'residual_threshold': ransac_config['residual_threshold']
            }
            
        except Exception as e:
            logger.warning(f"RANSAC training failed: {e}")
            self.training_metrics['ransac'] = {'failed': True}
        
        # 3. Train Theil-Sen Regression (if dataset not too large)
        if len(X) <= 1000:  # Theil-Sen is O(n^2), expensive for large datasets
            logger.info("Training Theil-Sen Regression (median-based robust)...")
            theil_sen_model = TheilSenRegressor(**self.config['theil_sen'])
            
            try:
                cv_scores = cross_val_score(theil_sen_model, X_scaled, y, cv=tscv, scoring='r2')
                theil_sen_model.fit(X_scaled, y)
                self.models['theil_sen'] = theil_sen_model
                
                train_pred = theil_sen_model.predict(X_scaled)
                self.training_metrics['theil_sen'] = {
                    'r2_score': r2_score(y, train_pred),
                    'mse': mean_squared_error(y, train_pred),
                    'mae': mean_absolute_error(y, train_pred),
                    'cv_r2_mean': cv_scores.mean(),
                    'cv_r2_std': cv_scores.std()
                }
                
            except Exception as e:
                logger.warning(f"Theil-Sen training failed: {e}")
        else:
            logger.info("Skipping Theil-Sen (dataset too large)")
        
        # 4. Train Quantile Regression Models
        logger.info("Training Quantile Regression models...")
        quantiles = self.config['quantile_regression']['quantiles']
        
        for quantile in quantiles:
            try:
                qr_model = QuantileRegressor(
                    quantile=quantile,
                    alpha=self.config['quantile_regression']['alpha'],
                    fit_intercept=self.config['quantile_regression']['fit_intercept'],
                    solver=self.config['quantile_regression']['solver']
                )
                
                qr_model.fit(X_scaled, y)
                self.quantile_models[f'q_{int(quantile*100)}'] = qr_model
                
                train_pred = qr_model.predict(X_scaled)
                
                # For quantile regression, we use quantile loss instead of MSE
                quantile_loss = np.mean(np.maximum(quantile * (y - train_pred), 
                                                  (quantile - 1) * (y - train_pred)))
                
                self.training_metrics[f'quantile_{int(quantile*100)}'] = {
                    'quantile_loss': quantile_loss,
                    'mae': mean_absolute_error(y, train_pred),
                    'quantile': quantile
                }
                
            except Exception as e:
                logger.warning(f"Quantile regression (q={quantile}) failed: {e}")
        
        # 5. Calculate ensemble weights
        self._calculate_ensemble_weights()
        
        # 6. Generate prediction intervals
        if self.config['ensemble']['use_prediction_intervals']:
            self._calculate_prediction_intervals(X_scaled, y)
        
        self.is_trained = True
        
        # Log performance summary
        logger.info("=== ROBUST REGRESSION PERFORMANCE SUMMARY ===")
        for model_name, metrics in self.training_metrics.items():
            if 'failed' not in metrics and 'quantile' not in model_name:
                r2 = metrics.get('r2_score', 0)
                cv_r2 = metrics.get('cv_r2_mean', 0)
                mae = metrics.get('mae', 0)
                logger.info(f"{model_name:12s} - R²: {r2:.4f}, CV R²: {cv_r2:.4f}, MAE: {mae:.4f}")
        
        return self.training_metrics
    
    def _calculate_ensemble_weights(self):
        """Calculate weights for ensemble based on performance."""
        
        if self.config['ensemble']['weight_method'] == 'equal_weighted':
            # Equal weights
            n_models = len([m for m in self.training_metrics.keys() 
                           if 'failed' not in self.training_metrics[m] and 'quantile' not in m])
            self.ensemble_weights = {name: 1.0/n_models for name in self.models.keys()}
        
        else:  # Performance weighted
            # Weight by cross-validation R²
            cv_scores = {}
            for name, model in self.models.items():
                metrics = self.training_metrics.get(name, {})
                if 'failed' not in metrics:
                    cv_r2 = metrics.get('cv_r2_mean', 0)
                    cv_scores[name] = max(cv_r2, 0)  # Ensure non-negative
            
            # Normalize weights
            total_score = sum(cv_scores.values())
            if total_score > 0:
                self.ensemble_weights = {name: score/total_score 
                                       for name, score in cv_scores.items()}
            else:
                # Fallback to equal weights
                n_models = len(cv_scores)
                self.ensemble_weights = {name: 1.0/n_models for name in cv_scores.keys()}
        
        logger.info(f"Ensemble weights: {self.ensemble_weights}")
    
    def _calculate_prediction_intervals(self, X: pd.DataFrame, y: pd.Series):
        """Calculate prediction intervals using quantile regression."""
        
        confidence_level = self.config['ensemble']['confidence_level']
        lower_quantile = (1 - confidence_level) / 2
        upper_quantile = 1 - lower_quantile
        
        # Find closest quantile models
        available_quantiles = [float(k.split('_')[1])/100 for k in self.quantile_models.keys()]
        
        lower_q = min(available_quantiles, key=lambda x: abs(x - lower_quantile))
        upper_q = min(available_quantiles, key=lambda x: abs(x - upper_quantile))
        
        lower_model_key = f'q_{int(lower_q*100)}'
        upper_model_key = f'q_{int(upper_q*100)}'
        
        if lower_model_key in self.quantile_models and upper_model_key in self.quantile_models:
            self.prediction_intervals = {
                'lower_quantile': lower_q,
                'upper_quantile': upper_q,
                'confidence_level': confidence_level
            }
            logger.info(f"Prediction intervals: {lower_q:.1%} - {upper_q:.1%} quantiles")
    
    def predict(self, factors: pd.DataFrame) -> pd.Series:
        """Generate ensemble robust predictions."""
        
        if not self.is_trained:
            raise ValueError("Models must be trained before prediction")
        
        # Scale features
        X_scaled = pd.DataFrame(
            self.feature_scaler.transform(factors),
            index=factors.index,
            columns=factors.columns
        )
        
        # Get predictions from each model
        predictions = pd.DataFrame(index=factors.index)
        
        for model_name, model in self.models.items():
            try:
                pred = model.predict(X_scaled)
                predictions[model_name] = pred
            except Exception as e:
                logger.warning(f"Prediction failed for {model_name}: {e}")
                predictions[model_name] = 0
        
        # Ensemble prediction (weighted average)
        ensemble_pred = np.zeros(len(predictions))
        total_weight = 0
        
        for model_name, weight in self.ensemble_weights.items():
            if model_name in predictions.columns:
                ensemble_pred += weight * predictions[model_name].values
                total_weight += weight
        
        if total_weight > 0:
            ensemble_pred /= total_weight
        
        return pd.Series(ensemble_pred, index=factors.index, name='robust_ensemble_pred')
    
    def predict_with_intervals(self, factors: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
        """Generate predictions with uncertainty intervals."""
        
        # Main prediction
        main_pred = self.predict(factors)
        
        # Prediction intervals
        intervals = pd.DataFrame(index=factors.index)
        intervals['prediction'] = main_pred
        
        if self.prediction_intervals and len(self.quantile_models) > 0:
            X_scaled = pd.DataFrame(
                self.feature_scaler.transform(factors),
                index=factors.index,
                columns=factors.columns
            )
            
            # Lower bound
            lower_q = self.prediction_intervals['lower_quantile']
            lower_key = f'q_{int(lower_q*100)}'
            if lower_key in self.quantile_models:
                intervals['lower_bound'] = self.quantile_models[lower_key].predict(X_scaled)
            
            # Upper bound  
            upper_q = self.prediction_intervals['upper_quantile']
            upper_key = f'q_{int(upper_q*100)}'
            if upper_key in self.quantile_models:
                intervals['upper_bound'] = self.quantile_models[upper_key].predict(X_scaled)
            
            # Median prediction (if available)
            if 'q_50' in self.quantile_models:
                intervals['median'] = self.quantile_models['q_50'].predict(X_scaled)
        
        return main_pred, intervals
    
    def analyze_residuals(self, factors: pd.DataFrame, 
                         forward_returns: pd.Series) -> Dict:
        """Comprehensive residual analysis for model diagnostics."""
        
        predictions = self.predict(factors)
        residuals = forward_returns - predictions
        
        # Align data
        common_idx = residuals.dropna().index
        residuals = residuals.loc[common_idx]
        
        analysis = {
            'residual_stats': {
                'mean': residuals.mean(),
                'std': residuals.std(),
                'skewness': stats.skew(residuals),
                'kurtosis': stats.kurtosis(residuals),
                'jarque_bera': stats.jarque_bera(residuals)
            },
            'normality_tests': {
                'shapiro_wilk': stats.shapiro(residuals.sample(min(5000, len(residuals)))),
                'anderson_darling': stats.anderson(residuals, dist='norm')
            },
            'heteroscedasticity': {
                'breusch_pagan': self._breusch_pagan_test(factors.loc[common_idx], residuals)
            },
            'outlier_analysis': {
                'outlier_threshold': 3.0,
                'n_outliers': np.sum(np.abs(residuals) > 3 * residuals.std()),
                'outlier_fraction': np.sum(np.abs(residuals) > 3 * residuals.std()) / len(residuals)
            }
        }
        
        return analysis
    
    def _breusch_pagan_test(self, X: pd.DataFrame, residuals: pd.Series) -> Dict:
        """Simple Breusch-Pagan test for heteroscedasticity."""
        try:
            from scipy.stats import chi2
            
            # Auxiliary regression: squared residuals on factors
            y_aux = residuals ** 2
            X_with_const = X.copy()
            X_with_const['const'] = 1
            
            # Simple regression
            aux_model = LinearRegression()
            aux_model.fit(X_with_const, y_aux)
            
            # R-squared from auxiliary regression
            y_pred_aux = aux_model.predict(X_with_const)
            ss_res = np.sum((y_aux - y_pred_aux) ** 2)
            ss_tot = np.sum((y_aux - np.mean(y_aux)) ** 2)
            r_squared_aux = 1 - (ss_res / ss_tot)
            
            # Test statistic
            n = len(residuals)
            lm_statistic = n * r_squared_aux
            p_value = 1 - chi2.cdf(lm_statistic, df=len(X.columns))
            
            return {
                'lm_statistic': lm_statistic,
                'p_value': p_value,
                'heteroscedastic': p_value < 0.05
            }
            
        except Exception as e:
            logger.warning(f"Breusch-Pagan test failed: {e}")
            return {'failed': True}
    
    def generate_diagnostics_report(self, factors: pd.DataFrame,
                                  forward_returns: pd.Series) -> str:
        """Generate comprehensive diagnostics report."""
        
        # Residual analysis
        residual_analysis = self.analyze_residuals(factors, forward_returns)
        
        report = []
        report.append("# Robust Regression Diagnostics Report")
        report.append("=" * 50)
        
        # Model performance
        report.append("\n## Model Performance Summary")
        for model_name, metrics in self.training_metrics.items():
            if 'failed' not in metrics and 'quantile' not in model_name:
                r2 = metrics.get('r2_score', 0)
                cv_r2 = metrics.get('cv_r2_mean', 0)
                mae = metrics.get('mae', 0)
                report.append(f"**{model_name}**: R² = {r2:.4f}, CV R² = {cv_r2:.4f}, MAE = {mae:.4f}")
        
        # Ensemble weights
        report.append(f"\n## Ensemble Weights")
        for model, weight in self.ensemble_weights.items():
            report.append(f"- {model}: {weight:.3f}")
        
        # Residual diagnostics
        report.append(f"\n## Residual Analysis")
        res_stats = residual_analysis['residual_stats']
        report.append(f"- **Mean**: {res_stats['mean']:.6f}")
        report.append(f"- **Std Dev**: {res_stats['std']:.4f}")
        report.append(f"- **Skewness**: {res_stats['skewness']:.3f}")
        report.append(f"- **Kurtosis**: {res_stats['kurtosis']:.3f}")
        
        jb_stat, jb_pval = res_stats['jarque_bera']
        report.append(f"- **Jarque-Bera**: {jb_stat:.2f} (p = {jb_pval:.4f})")
        
        # Outlier analysis
        outlier_info = residual_analysis['outlier_analysis']
        report.append(f"- **Outliers**: {outlier_info['n_outliers']} ({outlier_info['outlier_fraction']:.1%})")
        
        # Heteroscedasticity test
        if 'heteroscedasticity' in residual_analysis:
            bp_test = residual_analysis['heteroscedasticity']['breusch_pagan']
            if 'failed' not in bp_test:
                report.append(f"- **Breusch-Pagan Test**: LM = {bp_test['lm_statistic']:.2f} (p = {bp_test['p_value']:.4f})")
                if bp_test['heteroscedastic']:
                    report.append("  ⚠️ **Heteroscedasticity detected**")
                else:
                    report.append("  ✅ **Homoscedastic residuals**")
        
        return "\n".join(report)
    
    def save_model(self, filepath: str):
        """Save the trained robust regression ensemble."""
        model_data = {
            'models': self.models,
            'quantile_models': self.quantile_models,
            'feature_scaler': self.feature_scaler,
            'ensemble_weights': self.ensemble_weights,
            'prediction_intervals': self.prediction_intervals,
            'training_metrics': self.training_metrics,
            'config': self.config
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Robust regression ensemble saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a pre-trained robust regression ensemble."""
        model_data = joblib.load(filepath)
        
        self.models = model_data['models']
        self.quantile_models = model_data.get('quantile_models', {})
        self.feature_scaler = model_data['feature_scaler']
        self.ensemble_weights = model_data.get('ensemble_weights', {})
        self.prediction_intervals = model_data.get('prediction_intervals', {})
        self.training_metrics = model_data['training_metrics']
        self.config = model_data['config']
        self.is_trained = True
        
        logger.info(f"Robust regression ensemble loaded from {filepath}")

if __name__ == "__main__":
    logger.info("Robust Regression Ensemble Module initialized")
    logger.info("Available robust methods:")
    logger.info("1. Huber Regression - robust to outliers with tunable threshold")
    logger.info("2. RANSAC - outlier-resistant iterative fitting")  
    logger.info("3. Theil-Sen - median-based robust regression")
    logger.info("4. Quantile Regression - models different parts of return distribution")
    logger.info("5. Ensemble weighting - combines all methods optimally")
    logger.info("6. Prediction intervals - uncertainty quantification")
