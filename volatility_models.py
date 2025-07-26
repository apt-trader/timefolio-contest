#!/usr/bin/env python3
"""
Advanced Volatility Modeling Module

Institutional-grade volatility modeling for factor models:
1. GARCH(1,1) for volatility clustering
2. GJR-GARCH for asymmetric volatility (leverage effect)
3. EGARCH for exponential volatility modeling
4. Dynamic correlation modeling (DCC-GARCH)
5. Volatility forecasting and risk budgeting

Essential for institutional risk management and portfolio optimization.
Captures volatility clustering, mean reversion, and leverage effects.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from scipy import stats
import warnings
from pathlib import Path
from datetime import datetime
import joblib

# Advanced volatility modeling imports
try:
    from arch import arch_model
    HAS_ARCH = True
except ImportError:
    HAS_ARCH = False
    logging.warning("ARCH package not available - using simplified volatility models")

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class VolatilityModelEnsemble:
    """Advanced volatility modeling ensemble for factor models."""
    
    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        self.volatility_models = {}
        self.fitted_models = {}
        self.is_trained = False
        
    def _default_config(self) -> Dict:
        """Default configuration for volatility models."""
        return {
            'garch_models': {
                'GARCH': {'p': 1, 'q': 1, 'mean': 'constant', 'dist': 'normal'},
                'GJR-GARCH': {'p': 1, 'o': 1, 'q': 1, 'mean': 'constant', 'dist': 't'},
                'EGARCH': {'p': 1, 'q': 1, 'mean': 'constant', 'dist': 'skewt'}
            },
            'forecast_horizon': 21,
            'min_observations': 252,
            'correlation_window': 252
        }
    
    def fit_volatility_models(self, returns: pd.DataFrame) -> Dict:
        """Fit comprehensive volatility models."""
        logger.info("=== FITTING ADVANCED VOLATILITY MODELS ===")
        
        returns = returns.dropna()
        if len(returns) < self.config['min_observations']:
            logger.error(f"Insufficient data: {len(returns)}")
            return {}
        
        logger.info(f"Volatility data: {len(returns)} obs, {len(returns.columns)} assets")
        
        results = {}
        
        # Market returns for volatility modeling
        market_returns = returns.mean(axis=1) * 100
        
        # 1. Fit GARCH models
        if HAS_ARCH:
            logger.info("Fitting GARCH models...")
            garch_results = self._fit_garch_models(market_returns)
            results['garch_models'] = garch_results
        else:
            logger.info("Fitting simplified volatility models...")
            simple_results = self._fit_simple_models(market_returns)
            results['simple_models'] = simple_results
        
        # 2. Dynamic correlations
        logger.info("Computing dynamic correlations...")
        corr_results = self._compute_dynamic_correlations(returns)
        results['correlations'] = corr_results
        
        # 3. Generate forecasts
        logger.info("Generating volatility forecasts...")
        forecasts = self._generate_forecasts(results, market_returns)
        results['forecasts'] = forecasts
        
        # 4. Diagnostics
        diagnostics = self._run_diagnostics(market_returns, results)
        results['diagnostics'] = diagnostics
        
        self.volatility_models = results
        self.is_trained = True
        
        # Save report
        report = self._generate_report(results)
        report_path = Path("reports/volatility_analysis_report.md")
        report_path.parent.mkdir(exist_ok=True)
        report_path.write_text(report)
        logger.info(f"Volatility analysis report saved to {report_path}")
        
        return results
    
    def _fit_garch_models(self, returns: pd.Series) -> Dict:
        """Fit GARCH models using ARCH package."""
        results = {}
        
        for model_name, config in self.config['garch_models'].items():
            logger.info(f"Fitting {model_name}...")
            
            try:
                if model_name == 'GARCH':
                    model = arch_model(returns, mean=config['mean'], vol='GARCH', 
                                     p=config['p'], q=config['q'], dist=config['dist'])
                elif model_name == 'GJR-GARCH':
                    model = arch_model(returns, mean=config['mean'], vol='GARCH',
                                     p=config['p'], o=config['o'], q=config['q'], dist=config['dist'])
                elif model_name == 'EGARCH':
                    model = arch_model(returns, mean=config['mean'], vol='EGARCH',
                                     p=config['p'], q=config['q'], dist=config['dist'])
                
                fitted = model.fit(disp='off', show_warning=False)
                
                results[model_name] = {
                    'model': fitted,
                    'aic': fitted.aic,
                    'bic': fitted.bic,
                    'volatility': fitted.conditional_volatility
                }
                
                logger.info(f"{model_name} - AIC: {fitted.aic:.2f}")
                
            except Exception as e:
                logger.error(f"{model_name} failed: {e}")
                results[model_name] = {'failed': True}
        
        return results
    
    def _fit_simple_models(self, returns: pd.Series) -> Dict:
        """Simple volatility models when ARCH not available."""
        results = {}
        
        # EWMA models
        for name, decay in [('short', 0.94), ('medium', 0.97), ('long', 0.99)]:
            ewma_vol = returns.ewm(alpha=1-decay).std()
            results[f'EWMA_{name}'] = {
                'volatility': ewma_vol,
                'decay': decay,
                'mean_vol': ewma_vol.mean()
            }
        
        # Rolling volatility
        for window in [21, 63, 126, 252]:
            roll_vol = returns.rolling(window).std()
            results[f'Rolling_{window}D'] = {
                'volatility': roll_vol,
                'window': window,
                'mean_vol': roll_vol.mean()
            }
        
        return results
    
    def _compute_dynamic_correlations(self, returns: pd.DataFrame) -> Dict:
        """Compute dynamic correlation matrices."""
        window = self.config['correlation_window']
        
        # Select subset for efficiency
        max_assets = min(15, len(returns.columns))
        assets = returns.columns[:max_assets]
        data = returns[assets]
        
        correlations = []
        dates = []
        
        for i in range(window, len(data)):
            window_data = data.iloc[i-window:i]
            corr_matrix = window_data.corr()
            correlations.append(corr_matrix.values)
            dates.append(data.index[i])
        
        if correlations:
            corr_tensor = np.stack(correlations)
            
            # Average correlation time series
            avg_corrs = []
            for matrix in corr_tensor:
                upper_tri = matrix[np.triu_indices(len(matrix), k=1)]
                avg_corrs.append(np.mean(upper_tri))
            
            avg_corr_series = pd.Series(avg_corrs, index=dates)
            
            return {
                'correlation_matrices': corr_tensor,
                'dates': dates,
                'avg_correlation': avg_corr_series,
                'assets': assets.tolist(),
                'stats': {
                    'mean_corr': avg_corr_series.mean(),
                    'corr_vol': avg_corr_series.std(),
                    'min_corr': avg_corr_series.min(),
                    'max_corr': avg_corr_series.max()
                }
            }
        
        return {}
    
    def _generate_forecasts(self, results: Dict, returns: pd.Series) -> Dict:
        """Generate volatility forecasts."""
        forecasts = {}
        horizon = self.config['forecast_horizon']
        
        # GARCH forecasts
        if 'garch_models' in results:
            for name, model_data in results['garch_models'].items():
                if 'failed' not in model_data:
                    try:
                        fitted = model_data['model']
                        forecast = fitted.forecast(horizon=horizon)
                        forecasts[f'{name}_forecast'] = {
                            'volatility': np.sqrt(forecast.variance.iloc[-1, :])
                        }
                    except Exception as e:
                        logger.warning(f"Forecast failed for {name}: {e}")
        
        # Simple model forecasts
        if 'simple_models' in results:
            for name, model_data in results['simple_models'].items():
                if 'volatility' in model_data:
                    current_vol = model_data['volatility'].iloc[-1]
                    forecasts[f'{name}_forecast'] = {
                        'volatility': current_vol
                    }
        
        # Ensemble forecast
        if forecasts:
            vol_values = [f['volatility'] for f in forecasts.values() 
                         if isinstance(f.get('volatility'), (float, int, np.number))]
            
            if vol_values:
                ensemble_vol = np.mean(vol_values)
                ensemble_std = np.std(vol_values)
                
                forecasts['ensemble'] = {
                    'volatility': ensemble_vol,
                    'uncertainty': ensemble_std,
                    'n_models': len(vol_values)
                }
                
                logger.info(f"Ensemble volatility forecast: {ensemble_vol:.4f} ± {ensemble_std:.4f}")
        
        return forecasts
    
    def _run_diagnostics(self, returns: pd.Series, results: Dict) -> Dict:
        """Run volatility model diagnostics."""
        diagnostics = {}
        
        # Basic statistics
        diagnostics['return_stats'] = {
            'mean': returns.mean(),
            'std': returns.std(),
            'skewness': stats.skew(returns.dropna()),
            'kurtosis': stats.kurtosis(returns.dropna()),
            'jarque_bera': stats.jarque_bera(returns.dropna())
        }
        
        # Volatility clustering test (simplified)
        squared_returns = returns ** 2
        try:
            from sklearn.linear_model import LinearRegression
            
            lags = 5
            y = squared_returns[lags:].values
            X = np.column_stack([squared_returns[i:-lags+i].values for i in range(lags)])
            
            model = LinearRegression().fit(X, y)
            r2 = model.score(X, y)
            
            diagnostics['arch_test'] = {
                'r_squared': r2,
                'arch_effects': r2 > 0.05
            }
            
        except Exception as e:
            logger.warning(f"ARCH test failed: {e}")
        
        return diagnostics
    
    def _generate_report(self, results: Dict) -> str:
        """Generate comprehensive volatility report."""
        report = []
        report.append("# Advanced Volatility Modeling Report")
        report.append("=" * 50)
        report.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Model summary
        report.append("\n## Volatility Models")
        
        if 'garch_models' in results:
            report.append("### GARCH Models")
            report.append("| Model | AIC | BIC | Status |")
            report.append("|-------|-----|-----|--------|")
            
            for name, data in results['garch_models'].items():
                if 'failed' in data:
                    report.append(f"| {name:12s} | - | - | ❌ Failed |")
                else:
                    aic = data['aic']
                    bic = data['bic']
                    report.append(f"| {name:12s} | {aic:7.2f} | {bic:7.2f} | ✅ OK |")
        
        if 'simple_models' in results:
            report.append("### Simple Models")
            for name, data in results['simple_models'].items():
                mean_vol = data.get('mean_vol', 0)
                report.append(f"- **{name}**: Mean Vol = {mean_vol:.4f}")
        
        # Forecasts
        if 'forecasts' in results:
            forecasts = results['forecasts']
            if 'ensemble' in forecasts:
                ens = forecasts['ensemble']
                report.append(f"\n## Volatility Forecast")
                report.append(f"**Ensemble**: {ens['volatility']:.4f} ± {ens['uncertainty']:.4f}")
        
        # Diagnostics
        if 'diagnostics' in results:
            diag = results['diagnostics']
            if 'return_stats' in diag:
                stats_data = diag['return_stats']
                report.append(f"\n## Return Statistics")
                report.append(f"- Mean: {stats_data['mean']:.4f}")
                report.append(f"- Std: {stats_data['std']:.4f}")
                report.append(f"- Skewness: {stats_data['skewness']:.3f}")
                report.append(f"- Kurtosis: {stats_data['kurtosis']:.3f}")
        
        # Correlations
        if 'correlations' in results and 'stats' in results['correlations']:
            corr_stats = results['correlations']['stats']
            report.append(f"\n## Dynamic Correlations")
            report.append(f"- Mean: {corr_stats['mean_corr']:.3f}")
            report.append(f"- Volatility: {corr_stats['corr_vol']:.3f}")
            report.append(f"- Range: {corr_stats['min_corr']:.3f} to {corr_stats['max_corr']:.3f}")
        
        return "\n".join(report)
    
    def get_volatility_forecast(self) -> Optional[float]:
        """Get ensemble volatility forecast."""
        if not self.is_trained:
            return None
        
        forecasts = self.volatility_models.get('forecasts', {})
        ensemble = forecasts.get('ensemble', {})
        return ensemble.get('volatility')
    
    def save_model(self, filepath: str):
        """Save volatility models."""
        joblib.dump({
            'volatility_models': self.volatility_models,
            'config': self.config,
            'is_trained': self.is_trained
        }, filepath)
        logger.info(f"Volatility models saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load volatility models."""
        data = joblib.load(filepath)
        self.volatility_models = data['volatility_models']
        self.config = data['config']
        self.is_trained = data['is_trained']
        logger.info(f"Volatility models loaded from {filepath}")

if __name__ == "__main__":
    logger.info("Advanced Volatility Modeling Module initialized")
    logger.info("Available models:")
    logger.info("1. GARCH(1,1) - Standard volatility clustering")
    logger.info("2. GJR-GARCH - Asymmetric volatility (leverage effect)")
    logger.info("3. EGARCH - Exponential volatility modeling")
    logger.info("4. Dynamic correlation modeling")
    logger.info("5. Ensemble volatility forecasting")
