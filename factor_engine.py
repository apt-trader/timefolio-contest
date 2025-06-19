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
        self.mom_windows = settings.get('mom_windows', [21, 63, 252]) # Use business days
        self.vol_window = settings.get('vol_window', 21)
        self.rsi_window = settings.get('rsi_window', 14)
        logger.info(f"FactorEngine initialized.")

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
        
        # --- The 'Momentum' key now calls the new composite function ---
        factors = {
            'Momentum': self._calculate_momentum_factors(prices), # UPGRADED
            'Reversion': self._calculate_mean_reversion(prices),
            'Liquidity': self._calculate_liquidity(prices, volumes),
            'Low_Beta': self._calculate_beta(returns, market_index),
            'Value': self._calculate_value_factors(fundamental_data, market_caps),
            'Quality': self._calculate_quality_factors(fundamental_data),
            'Profitability': self._calculate_profitability_factors(fundamental_data),
            'Macro_Regime': self._calculate_macro_regime_factor(macro_data)
        }

        combined_factors = pd.DataFrame(factors).dropna(axis=1, how='all')
        
        if 'Macro_Regime' in combined_factors.columns:
            macro_signal = combined_factors['Macro_Regime'].iloc[0]
            combined_factors['Macro_Regime'] = macro_signal
        
        logger.info(f"Factor calculation complete. {len(combined_factors.columns)} factors calculated for {len(combined_factors)} tickers.")
        return combined_factors.fillna(0)

    # --- UPGRADED MOMENTUM METHOD ---
    def _calculate_momentum_factors(self, prices: pd.DataFrame) -> pd.Series:
        """
        Calculates a robust, composite momentum factor from multiple signals.
        1. Short-Term Reversal (1-Month): Negative signal.
        2. Classic Momentum (12-Month): Positive signal.
        3. Momentum Acceleration: Change in the 12-month momentum.
        """
        logger.debug("Calculating multi-dimensional momentum factor.")
        
        # Ensure we have enough data
        if len(prices) < 252 + 63: # 1 year + 3 months
            logger.warning("Not enough price data for full momentum calculation. Skipping.")
            return pd.Series(0, index=prices.columns)

        # 1. Short-Term Reversal (1-Month)
        # We rank this negatively because we want to bet against recent winners.
        reversal_1m = prices.pct_change(self.mom_windows[0]).iloc[-1]
        reversal_signal = self._cross_sectional_rank(-reversal_1m)

        # 2. Classic Momentum (12-Month)
        momentum_12m = prices.pct_change(self.mom_windows[2]).iloc[-1]
        momentum_signal = self._cross_sectional_rank(momentum_12m)
        
        # 3. Momentum Acceleration
        # Calculate 12-month momentum series over the last 3 months
        momentum_12m_series = prices.pct_change(self.mom_windows[2]).rolling(window=self.mom_windows[1]).mean()
        # The 'acceleration' is the slope of this series. We use a simple diff as a proxy.
        acceleration = momentum_12m_series.iloc[-1] - momentum_12m_series.iloc[-2]
        acceleration_signal = self._cross_sectional_rank(acceleration)

        # Combine the signals. The weights are a strategic choice.
        # Here, we give most importance to the classic trend, followed by its acceleration.
        # The reversal signal is used as a small hedge against momentum crashes.
        composite_momentum = (
            momentum_signal * 0.50 +
            acceleration_signal * 0.35 +
            reversal_signal * 0.15
        )
        
        return self._cross_sectional_rank(composite_momentum)

    # def _calculate_vol_adj_momentum(self, prices, returns):
    #     vol = returns.rolling(self.vol_window).std()
    #     mom = prices.pct_change(self.mom_windows[-1]) # Use longest window for signal
    #     adj_mom = mom.iloc[-1] / (vol.iloc[-1] + 1e-6)
    #     return self._cross_sectional_rank(adj_mom)

    def _calculate_mean_reversion(self, prices):
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(self.rsi_window).mean()
        loss = -delta.where(delta < 0, 0).rolling(self.rsi_window).mean()
        # Ensure loss is not zero to avoid division by zero
        rs = gain.iloc[-1] / (loss.iloc[-1] + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        return self._cross_sectional_rank(-rsi) # Lower RSI is a buy signal

    def _calculate_liquidity(self, prices, volumes):
        avg_daily_value = (prices * volumes).rolling(20).mean()
        return self._cross_sectional_rank(avg_daily_value.iloc[-1])
        
    def _calculate_beta(self, returns, market_index):
        if market_index.empty: return pd.Series(0, index=returns.columns)
        market_returns = market_index.pct_change().dropna()
        # Align returns with market returns
        aligned_returns, aligned_market = returns.align(market_returns, join='inner', axis=0)
        if len(aligned_returns) < 60: return pd.Series(0, index=returns.columns)
        
        rolling_cov = aligned_returns.rolling(60).cov(aligned_market)
        beta = rolling_cov / (aligned_market.rolling(60).var() + 1e-8)
        return self._cross_sectional_rank(-beta.iloc[-1]) # Lower beta is better

    def _calculate_value_factors(self, fundamental_data: pd.DataFrame, market_caps: pd.Series) -> pd.Series:
        """
        Calculates a composite value factor from BPR, EPR, PSR, and CF/P.
        """
        logger.debug("Calculating robust composite value factor.")
        if fundamental_data.empty or market_caps.empty:
            logger.warning("Fundamental data or market caps are empty. Skipping value factor.")
            return pd.Series()

        # Join with market caps to get a per-stock price proxy
        data = fundamental_data.join(market_caps.rename('MarketCap'), how='inner')
        if 'MarketCap' not in data.columns or data['MarketCap'].isna().all():
            return pd.Series()
        
        # Calculate the four value ratios (Yields). Higher is better.
        # Use .get() to safely access columns that might be missing for some stocks
        data['BPR'] = data.get('Equity', 0) / data['MarketCap']
        data['EPR'] = data.get('NetIncome', 0) / data['MarketCap']
        data['SPR'] = data.get('Sales', 0) / data['MarketCap'] # Sales-to-Price Ratio
        data['CFPR'] = data.get('OperatingCF', 0) / data['MarketCap'] # CashFlow-to-Price Ratio

        # Rank each factor individually. This normalizes their scales.
        bpr_rank = self._cross_sectional_rank(data['BPR'])
        epr_rank = self._cross_sectional_rank(data['EPR'])
        spr_rank = self._cross_sectional_rank(data['SPR'])
        cfpr_rank = self._cross_sectional_rank(data['CFPR'])
        
        # Combine the ranked factors into a single composite score
        # We can assign different weights if we have a view on which is more important
        composite_value = bpr_rank + epr_rank + spr_rank + cfpr_rank
        
        return self._cross_sectional_rank(composite_value)

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
        if macro_data.empty or 'us10y2y' not in macro_data.columns or macro_data['us10y2y'].isna().all():
            logger.warning("Macro data is unavailable. Defaulting to neutral regime (0).")
            return pd.Series({'Macro_Regime': 0})
        
        yield_spread = macro_data['us10y2y'].iloc[0]
        macro_signal = 1 if yield_spread > 0.001 else -1 # Risk-on vs Risk-off
        logger.info(f"Macro Regime (10y-2y spread: {yield_spread:.3f}): {'Risk-On' if macro_signal > 0 else 'Risk-Off'}")
        return pd.Series({'Macro_Regime': macro_signal})

    # --- Prediction Model ---
    def predict_returns(self, factors: pd.DataFrame) -> pd.Series:
        """Combines all factors into a final alpha signal."""
        logger.info("Combining factors into a final alpha signal.")
        if 'Macro_Regime' in factors.columns:
            macro_signal = factors['Macro_Regime'].iloc[0]
            risk_off_weights = {'Value': 0.2, 'Quality': 0.4, 'Low_Beta': 0.4, 'Momentum': 0.0, 'Reversion': 0.0}
            risk_on_weights = {'Value': 0.2, 'Quality': 0.2, 'Low_Beta': 0.0, 'Momentum': 0.4, 'Reversion': 0.2}
            
            weights = risk_off_weights if macro_signal < 0 else risk_on_weights
            
            factor_cols = [col for col in ['Momentum', 'Reversion', 'Low_Beta', 'Value', 'Quality'] if col in factors.columns]
            
            weighted_factors = pd.Series(0.0, index=factors.index)
            for factor, weight in weights.items():
                if factor in factor_cols and weight > 0:
                    weighted_factors += factors[factor] * weight
            
            return weighted_factors
        else:
            return factors.mean(axis=1)