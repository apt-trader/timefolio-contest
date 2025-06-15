# factor_engine.py
import logging
import numpy as np
import pandas as pd
from typing import Dict, List
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import xgboost as xgb

logger = logging.getLogger(__name__)

class FactorEngine:
    def __init__(self, settings: Dict):
        self.settings = settings
        self.mom_windows = settings.get('mom_windows', [20, 60, 120])
        self.vol_window = settings.get('vol_window', 20)
        self.rsi_window = settings.get('rsi_window', 14)
        self.prediction_model = settings.get('prediction_model', 'pcr') # 'pcr' or 'xgb'
        logger.info(f"FactorEngine initialized. Prediction model: {self.prediction_model.upper()}")

    def _cross_sectional_rank(self, series: pd.Series) -> pd.Series:
        """Ranks data cross-sectionally and scales it from -1 to 1."""
        return series.rank(pct=True).sub(0.5).mul(2).fillna(0)

    def calculate_all_factors(self,
                              prices: pd.DataFrame,
                              volumes: pd.DataFrame,
                              market_index: pd.Series,
                              fundamental_data: pd.DataFrame,
                              market_caps: pd.Series,
                              macro_data: pd.DataFrame) -> pd.DataFrame:
        """Main entry point to calculate and combine all factors."""
        logger.info("Starting calculation of all factor types.")
        returns = prices.pct_change()
        
        # Calculate each factor. Note: they are now single Series representing the latest signal.
        factors = {
            'Momentum': self._calculate_vol_adj_momentum(prices, returns),
            'Reversion': self._calculate_mean_reversion(prices),
            'Liquidity': self._calculate_liquidity(prices, volumes),
            'Low_Beta': self._calculate_beta(returns, market_index),
            'Value': self._calculate_value_factors(fundamental_data, market_caps),
            'Quality': self._calculate_quality_factors(fundamental_data),
            'Profitability': self._calculate_profitability_factors(fundamental_data),
            'Macro_Regime': self._calculate_macro_regime_factor(macro_data)
        }

        # Combine all factor series into a single DataFrame
        combined_factors = pd.DataFrame(factors).dropna(axis=1, how='all')
        
        # The Macro_Regime is a single value, broadcast it to all tickers
        if 'Macro_Regime' in combined_factors.columns:
            macro_signal = combined_factors['Macro_Regime'].iloc[0]
            combined_factors['Macro_Regime'] = macro_signal
        
        logger.info(f"Factor calculation complete. {len(combined_factors.columns)} factors calculated for {len(combined_factors)} tickers.")
        return combined_factors.fillna(0)

    # --- Technical Factor Methods (return latest signal as Series) ---
    def _calculate_vol_adj_momentum(self, prices, returns):
        vol = returns.rolling(self.vol_window).std()
        mom = prices.pct_change(self.mom_windows[-1]) # Use longest window for signal
        adj_mom = mom.iloc[-1] / (vol.iloc[-1] + 1e-6)
        return self._cross_sectional_rank(adj_mom)

    def _calculate_mean_reversion(self, prices):
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(self.rsi_window).mean()
        loss = -delta.where(delta < 0, 0).rolling(self.rsi_window).mean()
        rs = gain.iloc[-1] / (loss.iloc[-1] + 1e-6)
        rsi = 100 - (100 / (1 + rs))
        return self._cross_sectional_rank(-rsi) # Lower RSI is better

    def _calculate_liquidity(self, prices, volumes):
        avg_daily_value = (prices * volumes).rolling(20).mean()
        return self._cross_sectional_rank(avg_daily_value.iloc[-1])
        
    def _calculate_beta(self, returns, market_index):
        market_returns = market_index.pct_change()
        rolling_cov = returns.rolling(60).cov(market_returns)
        beta = rolling_cov / (market_returns.rolling(60).var() + 1e-6)
        return self._cross_sectional_rank(-beta.iloc[-1]) # Lower beta is better

    # --- Fundamental Factor Methods ---
    def _calculate_value_factors(self, fundamental_data, market_caps):
        if fundamental_data.empty: return pd.Series()
        data = fundamental_data.join(market_caps.rename('MarketCap'), how='inner')
        data['BPR'] = data['Equity'] / data['MarketCap']
        data['EPR'] = data['NetIncome'] / data['MarketCap']
        return self._cross_sectional_rank(data['BPR']) + self._cross_sectional_rank(data['EPR'])

    def _calculate_quality_factors(self, fundamental_data):
        if fundamental_data.empty: return pd.Series()
        data = fundamental_data.copy()
        data['ROE'] = data['NetIncome'] / data['Equity']
        data['Leverage'] = data['Liabilities'] / data['Assets']
        return self._cross_sectional_rank(data['ROE']) + self._cross_sectional_rank(-data['Leverage'])

    def _calculate_profitability_factors(self, fundamental_data):
        if fundamental_data.empty: return pd.Series()
        data = fundamental_data.copy()
        data['GrossProfitability'] = data['GrossProfit'] / data['Assets']
        return self._cross_sectional_rank(data['GrossProfitability'])
        
    def _calculate_macro_regime_factor(self, macro_data):
        if macro_data.empty: return pd.Series({'Macro_Regime': 0})
        yield_spread = macro_data['us10y2y'].iloc[0]
        macro_signal = 1 if yield_spread > 0.001 else -1 # Risk-on vs Risk-off
        logger.info(f"Macro Regime (10y-2y spread: {yield_spread:.3f}): {'Risk-On' if macro_signal > 0 else 'Risk-Off'}")
        return pd.Series({'Macro_Regime': macro_signal})

    # --- Prediction Model ---
    def predict_returns(self, factors: pd.DataFrame) -> pd.Series:
        """Combines all factors into a final alpha signal."""
        # Simple equal-weighted combination for now. Can be replaced with a model.
        logger.info("Combining factors into a final alpha signal.")
        # We interact the macro regime with other factors
        if 'Macro_Regime' in factors.columns:
            macro_signal = factors['Macro_Regime'].iloc[0]
            # In risk-off, we might favor Quality and Low_Beta more
            risk_off_weights = {'Value': 0.2, 'Quality': 0.4, 'Low_Beta': 0.4, 'Momentum': 0.0, 'Reversion': 0.0}
            risk_on_weights = {'Value': 0.2, 'Quality': 0.2, 'Low_Beta': 0.0, 'Momentum': 0.4, 'Reversion': 0.2}
            
            weights = risk_off_weights if macro_signal < 0 else risk_on_weights
            
            # Drop profitability if it's not adding value, or give it a weight
            factor_cols = [col for col in ['Momentum', 'Reversion', 'Low_Beta', 'Value', 'Quality'] if col in factors.columns]
            
            # Apply weights
            weighted_factors = pd.Series(0.0, index=factors.index)
            for factor, weight in weights.items():
                if factor in factor_cols and weight > 0:
                    weighted_factors += factors[factor] * weight
            
            return weighted_factors
            
        else: # Fallback if no macro data
            return factors.mean(axis=1)