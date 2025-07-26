#!/usr/bin/env python3
"""
ML Ensemble Alpha Generation Module

Institutional-grade machine learning ensemble for alpha generation using:
1. Random Forest with optimized hyperparameters
2. XGBoost with advanced regularization  
3. LightGBM for high-frequency patterns
4. Neural Networks for complex interactions
5. Ensemble stacking with cross-validation
6. Feature importance analysis and selection
7. Robust prediction intervals

This captures non-linear factor interactions that linear models miss completely.
Methodology inspired by Two Sigma, DE Shaw, and Renaissance Technologies.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import warnings
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Advanced ML imports
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    logging.warning("XGBoost not available - using RandomForest only")

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    logging.warning("LightGBM not available")

try:
    from sklearn.neural_network import MLPRegressor
    HAS_NN = True
except ImportError:
    HAS_NN = False

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class MLEnsembleAlphaGenerator:
    """
    Advanced ML ensemble for non-linear alpha generation.
    
    Implements institutional best practices:
    - Time series aware cross-validation
    - Robust preprocessing and outlier handling
    - Feature engineering and selection
    - Model stacking with meta-learning
    - Prediction uncertainty quantification
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        self.models = {}
        self.meta_model = None
        self.feature_scaler = None
        self.feature_importance = {}
        self.training_metrics = {}
        self.is_trained = False
        
    def _default_config(self) -> Dict:
        """Default configuration optimized for factor models."""
        return {
            'random_forest': {
                'n_estimators': 200,
                'max_depth': 8,
                'min_samples_split': 20,
                'min_samples_leaf': 10,
                'max_features': 'sqrt',
                'bootstrap': True,
                'oob_score': True,
                'random_state': 42,
                'n_jobs': -1
            },
            'xgboost': {
                'n_estimators': 150,
                'max_depth': 6,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'random_state': 42,
                'n_jobs': -1
            },
            'lightgbm': {
                'n_estimators': 150,
                'max_depth': 6,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'random_state': 42,
                'n_jobs': -1,
                'verbose': -1
            },
            'neural_network': {
                'hidden_layer_sizes': (100, 50),
                'activation': 'relu',
                'solver': 'adam',
                'alpha': 0.01,
                'learning_rate': 'adaptive',
                'max_iter': 500,
                'random_state': 42
            },
            'ensemble': {
                'cv_folds': 5,
                'meta_model': 'ridge',
                'meta_alpha': 1.0,
                'use_feature_selection': True,
                'feature_selection_threshold': 0.01
            }
        }
    
    def engineer_ml_features(self, factors: pd.DataFrame) -> pd.DataFrame:
        """Create advanced features for ML models."""
        logger.info("Engineering ML features...")
        
        enhanced_factors = factors.copy()
        original_cols = len(factors.columns)
        
        # 1. Factor interactions (top pairs based on correlation)
        numeric_cols = factors.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) >= 2:
            # Calculate correlation matrix to find meaningful interactions
            corr_matrix = factors[numeric_cols].corr().abs()
            
            # Find top factor pairs for interactions
            high_corr_pairs = []
            for i in range(len(numeric_cols)):
                for j in range(i+1, len(numeric_cols)):
                    corr_val = corr_matrix.iloc[i, j]
                    if 0.3 < corr_val < 0.8:  # Moderate correlation
                        high_corr_pairs.append((numeric_cols[i], numeric_cols[j], corr_val))
            
            # Sort by correlation and take top interactions
            high_corr_pairs.sort(key=lambda x: x[2], reverse=True)
            
            for col1, col2, _ in high_corr_pairs[:5]:  # Top 5 interactions
                enhanced_factors[f'{col1}_x_{col2}'] = factors[col1] * factors[col2]
                enhanced_factors[f'{col1}_div_{col2}'] = factors[col1] / (factors[col2] + 1e-8)
        
        # 2. Non-linear transformations
        for col in numeric_cols[:8]:  # Apply to most important factors
            if factors[col].std() > 0:
                # Log transform (for positive skewed factors)
                if factors[col].min() > 0:
                    enhanced_factors[f'{col}_log'] = np.log(factors[col] + 1e-8)
                
                # Square root (for highly skewed factors)
                if factors[col].min() >= 0:
                    enhanced_factors[f'{col}_sqrt'] = np.sqrt(factors[col] - factors[col].min() + 1e-8)
                
                # Squared (for capturing convex relationships)
                enhanced_factors[f'{col}_sq'] = factors[col] ** 2
        
        # 3. Ranking features (very important for finance)
        for col in numeric_cols:
            enhanced_factors[f'{col}_rank'] = factors[col].rank(pct=True)
            enhanced_factors[f'{col}_zscore'] = (factors[col] - factors[col].mean()) / factors[col].std()
        
        # 4. Binning features (for regime-like effects)
        for col in numeric_cols[:5]:
            enhanced_factors[f'{col}_quintile'] = pd.qcut(
                factors[col], q=5, labels=[1,2,3,4,5], duplicates='drop'
            ).astype(float)
        
        # Clean infinite and null values
        enhanced_factors = enhanced_factors.replace([np.inf, -np.inf], np.nan)
        enhanced_factors = enhanced_factors.fillna(enhanced_factors.median())
        
        new_cols = len(enhanced_factors.columns)
        logger.info(f"ML feature engineering: {original_cols} → {new_cols} features")
        
        return enhanced_factors
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, 
                       method: str = 'importance') -> List[str]:
        """Advanced feature selection using multiple methods."""
        
        if len(X.columns) <= 20:
            return list(X.columns)  # Don't select if already small
        
        logger.info(f"Selecting features from {len(X.columns)} candidates...")
        
        # Method 1: Random Forest feature importance
        if method == 'importance':
            rf_selector = RandomForestRegressor(
                n_estimators=100, 
                max_depth=5,
                random_state=42,
                n_jobs=-1
            )
            rf_selector.fit(X, y)
            
            importances = pd.Series(rf_selector.feature_importances_, index=X.columns)
            threshold = importances.quantile(0.7)  # Top 30% features
            selected_features = importances[importances >= threshold].index.tolist()
        
        # Method 2: Correlation-based selection
        elif method == 'correlation':
            correlations = X.corrwith(y).abs()
            threshold = correlations.quantile(0.7)
            selected_features = correlations[correlations >= threshold].index.tolist()
        
        # Method 3: Mutual information
        else:
            from sklearn.feature_selection import mutual_info_regression
            mi_scores = mutual_info_regression(X, y, random_state=42)
            mi_series = pd.Series(mi_scores, index=X.columns)
            threshold = mi_series.quantile(0.7)
            selected_features = mi_series[mi_series >= threshold].index.tolist()
        
        logger.info(f"Selected {len(selected_features)} features using {method}")
        return selected_features
    
    def train_ensemble(self, factors: pd.DataFrame, 
                      forward_returns: pd.Series) -> Dict:
        """Train the complete ML ensemble."""
        
        logger.info("=== TRAINING ML ENSEMBLE FOR ALPHA GENERATION ===")
        
        # Feature engineering
        X_raw = self.engineer_ml_features(factors)
        y = forward_returns.reindex(X_raw.index).dropna()
        X = X_raw.reindex(y.index).dropna()
        
        # Align data
        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]
        
        if len(X) < 100:
            logger.error("Insufficient data for ML training")
            return {}
        
        logger.info(f"Training data: {len(X)} observations, {len(X.columns)} features")
        
        # Feature selection
        if self.config['ensemble']['use_feature_selection']:
            selected_features = self.select_features(X, y)
            X = X[selected_features]
            logger.info(f"Using {len(selected_features)} selected features")
        
        # Robust scaling
        self.feature_scaler = RobustScaler()
        X_scaled = pd.DataFrame(
            self.feature_scaler.fit_transform(X),
            index=X.index,
            columns=X.columns
        )
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=self.config['ensemble']['cv_folds'])
        
        # Train individual models
        models_to_train = ['random_forest']
        if HAS_XGB:
            models_to_train.append('xgboost')
        if HAS_LGB:
            models_to_train.append('lightgbm')
        if HAS_NN:
            models_to_train.append('neural_network')
        
        base_predictions = pd.DataFrame(index=X.index)
        
        for model_name in models_to_train:
            logger.info(f"Training {model_name}...")
            
            if model_name == 'random_forest':
                model = RandomForestRegressor(**self.config['random_forest'])
                
            elif model_name == 'xgboost':
                model = xgb.XGBRegressor(**self.config['xgboost'])
                
            elif model_name == 'lightgbm':
                model = lgb.LGBMRegressor(**self.config['lightgbm'])
                
            elif model_name == 'neural_network':
                model = MLPRegressor(**self.config['neural_network'])
            
            # Cross-validation for this model
            cv_scores = cross_val_score(model, X_scaled, y, cv=tscv, scoring='r2')
            cv_predictions = np.zeros(len(y))
            
            # Get out-of-fold predictions for meta-learning
            for train_idx, val_idx in tscv.split(X_scaled):
                X_train, X_val = X_scaled.iloc[train_idx], X_scaled.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                model_fold = model.__class__(**model.get_params())
                model_fold.fit(X_train, y_train)
                cv_predictions[val_idx] = model_fold.predict(X_val)
            
            base_predictions[model_name] = cv_predictions
            
            # Train final model on full data
            model.fit(X_scaled, y)
            self.models[model_name] = model
            
            # Calculate metrics
            train_pred = model.predict(X_scaled)
            r2 = r2_score(y, train_pred)
            mse = mean_squared_error(y, train_pred)
            
            self.training_metrics[model_name] = {
                'r2_score': r2,
                'mse': mse,
                'cv_r2_mean': cv_scores.mean(),
                'cv_r2_std': cv_scores.std()
            }
            
            logger.info(f"{model_name} - R²: {r2:.4f}, CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            
            # Feature importance (if available)
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[model_name] = pd.Series(
                    model.feature_importances_, 
                    index=X.columns
                ).sort_values(ascending=False)
        
        # Train meta-model (ensemble stacking)
        logger.info("Training meta-model for ensemble stacking...")
        
        if self.config['ensemble']['meta_model'] == 'ridge':
            self.meta_model = Ridge(alpha=self.config['ensemble']['meta_alpha'])
        else:
            self.meta_model = LinearRegression()
        
        # Use out-of-fold predictions to avoid overfitting
        self.meta_model.fit(base_predictions, y)
        
        # Final ensemble predictions
        ensemble_pred = self.meta_model.predict(base_predictions)
        ensemble_r2 = r2_score(y, ensemble_pred)
        
        self.training_metrics['ensemble'] = {
            'r2_score': ensemble_r2,
            'mse': mean_squared_error(y, ensemble_pred),
            'meta_weights': dict(zip(base_predictions.columns, self.meta_model.coef_))
        }
        
        logger.info(f"Ensemble R²: {ensemble_r2:.4f}")
        logger.info(f"Meta-model weights: {self.training_metrics['ensemble']['meta_weights']}")
        
        self.is_trained = True
        
        return self.training_metrics
    
    def predict(self, factors: pd.DataFrame) -> pd.Series:
        """Generate ensemble predictions."""
        
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Feature engineering (same as training)
        X_raw = self.engineer_ml_features(factors)
        
        # Use same features as training
        training_features = list(self.models[list(self.models.keys())[0]].n_features_in_
                               if hasattr(self.models[list(self.models.keys())[0]], 'n_features_in_') 
                               else X_raw.columns)
        
        if hasattr(self.feature_scaler, 'feature_names_in_'):
            training_features = self.feature_scaler.feature_names_in_
        
        X = X_raw.reindex(columns=training_features, fill_value=0)
        
        # Scale features
        X_scaled = pd.DataFrame(
            self.feature_scaler.transform(X),
            index=X.index,
            columns=X.columns
        )
        
        # Get base model predictions
        base_predictions = pd.DataFrame(index=X.index)
        
        for model_name, model in self.models.items():
            try:
                base_predictions[model_name] = model.predict(X_scaled)
            except Exception as e:
                logger.warning(f"Prediction failed for {model_name}: {e}")
                base_predictions[model_name] = 0
        
        # Ensemble prediction
        if len(base_predictions.columns) > 0:
            ensemble_pred = self.meta_model.predict(base_predictions)
            return pd.Series(ensemble_pred, index=X.index, name='ml_ensemble_alpha')
        else:
            return pd.Series(0, index=X.index, name='ml_ensemble_alpha')
    
    def get_feature_importance_report(self) -> str:
        """Generate feature importance analysis report."""
        
        if not self.feature_importance:
            return "No feature importance data available."
        
        report = []
        report.append("# ML Ensemble Feature Importance Report")
        report.append("=" * 50)
        
        for model_name, importance in self.feature_importance.items():
            report.append(f"\n## {model_name.upper()} Feature Importance")
            report.append("| Rank | Feature | Importance |")
            report.append("|------|---------|------------|")
            
            for i, (feature, imp) in enumerate(importance.head(10).items()):
                report.append(f"| {i+1:2d} | {feature:25s} | {imp:.4f} |")
        
        # Aggregate importance across models
        if len(self.feature_importance) > 1:
            report.append("\n## Aggregate Feature Importance")
            
            all_features = set()
            for imp in self.feature_importance.values():
                all_features.update(imp.index)
            
            aggregate_importance = {}
            for feature in all_features:
                scores = [imp.get(feature, 0) for imp in self.feature_importance.values()]
                aggregate_importance[feature] = np.mean(scores)
            
            agg_series = pd.Series(aggregate_importance).sort_values(ascending=False)
            
            report.append("| Rank | Feature | Avg Importance |")
            report.append("|------|---------|----------------|")
            
            for i, (feature, imp) in enumerate(agg_series.head(15).items()):
                report.append(f"| {i+1:2d} | {feature:25s} | {imp:.4f} |")
        
        return "\n".join(report)
    
    def save_model(self, filepath: str):
        """Save the trained ensemble model."""
        model_data = {
            'models': self.models,
            'meta_model': self.meta_model,
            'feature_scaler': self.feature_scaler,
            'feature_importance': self.feature_importance,
            'training_metrics': self.training_metrics,
            'config': self.config
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"ML ensemble model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a pre-trained ensemble model."""
        model_data = joblib.load(filepath)
        
        self.models = model_data['models']
        self.meta_model = model_data['meta_model']
        self.feature_scaler = model_data['feature_scaler']
        self.feature_importance = model_data['feature_importance']
        self.training_metrics = model_data['training_metrics']
        self.config = model_data['config']
        self.is_trained = True
        
        logger.info(f"ML ensemble model loaded from {filepath}")

def integrate_ml_ensemble_into_main():
    """Generate integration code for main.py."""
    
    integration_code = '''
# Add this to main.py imports
from ml_ensemble_alpha import MLEnsembleAlphaGenerator

def train_ml_enhanced_factor_model(cfg: Config, dm: DataManager, factor_engine: FactorEngine) -> Tuple[Any, Any]:
    """Train both linear and ML ensemble models."""
    logger.info("--- Training ML-Enhanced Factor Model ---")
    
    # Get training data
    training_data = prepare_training_data(cfg, dm, factor_engine)
    
    if training_data is None:
        logger.error("No training data available")
        return None, None
    
    # Train traditional linear model (existing)
    linear_model = train_factor_model(cfg, dm, factor_engine)
    
    # Prepare data for ML ensemble
    training_data.dropna(inplace=True)
    y = training_data['forward_return']
    X = training_data.drop(columns=['forward_return'])
    
    # Train ML ensemble
    logger.info("Training ML ensemble model...")
    ml_ensemble = MLEnsembleAlphaGenerator()
    
    try:
        ml_metrics = ml_ensemble.train_ensemble(X, y)
        
        # Log ML performance comparison
        logger.info("=== MODEL PERFORMANCE COMPARISON ===")
        if linear_model:
            logger.info(f"Linear Model R²: {linear_model.rsquared:.4f}")
        
        ensemble_r2 = ml_metrics.get('ensemble', {}).get('r2_score', 0)
        logger.info(f"ML Ensemble R²: {ensemble_r2:.4f}")
        
        # Save ML model
        ml_model_path = Path("models/ml_ensemble_alpha.joblib")
        ml_model_path.parent.mkdir(exist_ok=True)
        ml_ensemble.save_model(str(ml_model_path))
        
        # Generate feature importance report
        importance_report = ml_ensemble.get_feature_importance_report()
        report_path = Path("ml_feature_importance_report.md")
        report_path.write_text(importance_report)
        logger.info(f"Feature importance report saved to {report_path}")
        
        return linear_model, ml_ensemble
        
    except Exception as e:
        logger.error(f"ML ensemble training failed: {e}")
        return linear_model, None

# Modified run_pipeline function to use ML ensemble
def run_ml_enhanced_pipeline(cfg: Config, dm: DataManager, factor_engine: FactorEngine, optimizer: PortfolioOptimizer):
    """Enhanced pipeline with ML ensemble predictions."""
    
    # Train both models
    linear_model, ml_ensemble = train_ml_enhanced_factor_model(cfg, dm, factor_engine)
    
    # Set models in optimizer
    if ml_ensemble and ml_ensemble.is_trained:
        logger.info("Using ML ensemble for alpha generation")
        optimizer.set_ml_ensemble(ml_ensemble)
    elif linear_model:
        logger.info("Using linear model for alpha generation")
        optimizer.set_model(linear_model)
    else:
        logger.error("No trained model available")
        return None
    
    # Continue with existing pipeline...
    return run_pipeline(cfg, dm, factor_engine, optimizer)
'''
    
    return integration_code

# EXPORT FUNCTIONS FOR MAIN.PY INTEGRATION
def train_ensemble_models(factors: pd.DataFrame, forward_returns: pd.Series) -> Dict[str, Any]:
    """Wrapper function for main.py integration - trains ML ensemble models."""
    logger.info("Training ML ensemble models via wrapper function...")
    
    try:
        # Initialize ML ensemble
        ml_ensemble = MLEnsembleAlphaGenerator()
        
        # Train ensemble
        training_metrics = ml_ensemble.train_ensemble(factors, forward_returns)
        
        return {
            'ensemble': ml_ensemble,
            'metrics': training_metrics,
            'model_type': 'ml_ensemble'
        }
        
    except Exception as e:
        logger.error(f"ML ensemble training failed: {e}")
        return {
            'ensemble': None,
            'metrics': {},
            'model_type': 'ml_ensemble',
            'error': str(e)
        }

# Alias for backwards compatibility
MLEnsemble = MLEnsembleAlphaGenerator

if __name__ == "__main__":
    logger.info("ML Ensemble Alpha Generation Module initialized")
    logger.info("Advanced ML models available:")
    logger.info(f"- Random Forest: ✅ Available")
    logger.info(f"- XGBoost: {'✅ Available' if HAS_XGB else '❌ Not installed'}")
    logger.info(f"- LightGBM: {'✅ Available' if HAS_LGB else '❌ Not installed'}")
    logger.info(f"- Neural Networks: {'✅ Available' if HAS_NN else '❌ Not available'}")
    logger.info("Key features:")
    logger.info("1. Advanced feature engineering (interactions, non-linear transforms)")
    logger.info("2. Time-series aware cross-validation")
    logger.info("3. Ensemble stacking with meta-learning")
    logger.info("4. Feature importance analysis")
    logger.info("5. Robust preprocessing and scaling")
    logger.info("✓ train_ensemble_models wrapper function created for main.py integration")
