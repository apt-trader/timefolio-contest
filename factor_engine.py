# factor_engine.py
import logging
import numpy as np
import pandas as pd
from typing import Dict, List
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import xgboost as xgb
from tqdm import tqdm

logger = logging.getLogger(__name__)

class FactorEngine:
    def __init__(self, settings: Dict):
        self.settings = settings
        self.mom_windows = settings.get('mom_windows', [21, 63, 252])
        self.vol_window = settings.get('vol_window', 21)
        self.rsi_window = settings.get('rsi_window', 14)
        self.prediction_model_type = settings.get('prediction_model', 'pcr')
        self.n_pca_components = settings.get('n_pca_components', 5)
        self.xgb_params = settings.get('xgb_params', {})
        self.prediction_target_days = settings.get('prediction_target_days', 21)
        logger.info(f"FactorEngine initialized. Prediction model: {self.prediction_model_type.upper()}")

    def _cross_sectional_rank(self, series: pd.Series) -> pd.Series:
        """Ranks data cross-sectionally from -1 to 1."""
        return series.rank(pct=True).sub(0.5).mul(2)

    def calculate_all_factors(self, prices: pd.DataFrame, volumes: pd.DataFrame,
                              market_caps: pd.DataFrame,
                              fundamental_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Calculates all factors in a point-in-time (PIT) manner for the entire history.
        This is computationally intensive and designed to create a training set for the ML model.
        """
        logger.info("Calculating all historical factors for model training...")
        returns = prices.pct_change()
        all_factors_list = []
        
        # We need at least one year of data to calculate factors
        for i in tqdm(range(252, len(prices)), desc="Calculating Historical Factors"):
            current_date = prices.index[i]
            
            # --- Point-in-Time Data Slicing ---
            prices_slice = prices.iloc[:i+1]
            volumes_slice = volumes.iloc[:i+1]
            returns_slice = returns.iloc[:i+1]
            market_caps_slice = market_caps.loc[current_date]
            
            # Find the latest fundamental data available *as of this date*
            # This is the core of a true point-in-time backtest
            pit_fundamentals_list = []
            for ticker in prices.columns:
                if ticker_df := fundamental_data.get(ticker):
                    # Find the latest report date <= current date
                    latest_pit_report = ticker_df[ticker_df.index <= current_date]
                    if not latest_pit_report.empty:
                        pit_fundamentals_list.append(latest_pit_report.iloc[-1])
            
            if not pit_fundamentals_list: continue
            
            pit_fundamentals_df = pd.DataFrame(pit_fundamentals_list)
            
            # --- Factor Calculation for this single day ---
            factors = {
                'Momentum': self._calculate_momentum_factors(prices_slice),
                'Value': self._calculate_value_factors(pit_fundamentals_df, market_caps_slice),
                'Quality': self._calculate_quality_factors(pit_fundamentals_df),
                'Investment': self._calculate_investment_factor(pit_fundamentals_df),
                'Profitability': self._calculate_profitability_factors(pit_fundamentals_df),
                'Low_Beta': self._calculate_beta(returns_slice, prices_slice.mean(axis=1)),
            }
            daily_factors = pd.DataFrame(factors).fillna(0)
            daily_factors['date'] = current_date
            all_factors_list.append(daily_factors)

        if not all_factors_list:
            logger.error("Could not generate any historical factors.")
            return pd.DataFrame()

        # Combine all daily factor dataframes into one large dataframe
        return pd.concat(all_factors_list).reset_index().rename(columns={'index':'ticker'}).set_index(['date', 'ticker'])

    # --- Factor Calculation Methods ---
    def _calculate_momentum_factors(self, prices: pd.DataFrame) -> pd.Series:
        if len(prices) < 252 + 63: return pd.Series(0, index=prices.columns)
        reversal_1m = self._cross_sectional_rank(-prices.pct_change(self.mom_windows[0]).iloc[-1])
        momentum_12m = self._cross_sectional_rank(prices.pct_change(self.mom_windows[2]).iloc[-1])
        accel_series = prices.pct_change(self.mom_windows[2]).rolling(self.mom_windows[1]).mean()
        acceleration = self._cross_sectional_rank(accel_series.iloc[-1] - accel_series.iloc[-2])
        return self._cross_sectional_rank(momentum_12m * 0.5 + acceleration * 0.35 + reversal_1m * 0.15)

    def _calculate_value_factors(self, fundamental_data: pd.DataFrame, market_caps: pd.Series) -> pd.Series:
        if fundamental_data.empty: return pd.Series()
        data = fundamental_data.join(market_caps.rename('MarketCap'), how='inner')
        if 'MarketCap' not in data or data['MarketCap'].le(0).all(): return pd.Series()
        data = data[data['MarketCap'] > 0].copy()
        ratios = pd.DataFrame({
            'BPR': data.get('Equity', 0) / data['MarketCap'], 'EPR': data.get('NetIncome', 0) / data['MarketCap'],
            'SPR': data.get('Sales', 0) / data['MarketCap'], 'CFPR': data.get('OperatingCF', 0) / data['MarketCap']
        }).replace([np.inf,-np.inf], 0).fillna(0)
        return self._cross_sectional_rank(ratios.apply(self._cross_sectional_rank, axis=0).sum(axis=1))

    def _calculate_quality_factors(self, fundamental_data: pd.DataFrame) -> pd.Series:
        if fundamental_data.empty: return pd.Series()
        data = fundamental_data.copy()
        data['ROE'] = data.get('NetIncome') / data.get('Equity')
        data['Leverage'] = data.get('Liabilities') / data.get('Assets')
        # Note: ROE stability is too complex for this simplified PIT loop and would be pre-calculated
        # in a more advanced system. We use the core ROE and Leverage here.
        return self._cross_sectional_rank(data['ROE']) - self._cross_sectional_rank(data['Leverage'])

    def _calculate_investment_factor(self, fundamental_data: pd.DataFrame) -> pd.Series:
        if 'Assets' not in fundamental_data.columns or 'Assets_PY' not in fundamental_data.columns:
            return pd.Series()
        # Asset growth requires previous year's assets, which a proper PIT data fetch would provide.
        # This assumes the data manager provides a column 'Assets_PY'
        asset_growth = (fundamental_data['Assets'] / fundamental_data['Assets_PY']) - 1
        return self._cross_sectional_rank(-asset_growth) # Lower growth is better

    def _calculate_profitability_factors(self, fundamental_data: pd.DataFrame) -> pd.Series:
        if 'GrossProfit' not in fundamental_data or 'Assets' not in fundamental_data: return pd.Series()
        return self._cross_sectional_rank(fundamental_data['GrossProfit'] / fundamental_data['Assets'])

    def _calculate_beta(self, returns: pd.DataFrame, market_index: pd.Series) -> pd.Series:
        aligned_returns, aligned_market = returns.align(market_index.pct_change().dropna(), join='inner', axis=0)
        if len(aligned_returns) < 60: return pd.Series(0, index=returns.columns)
        rolling_cov = aligned_returns.rolling(60).cov(aligned_market)
        beta = rolling_cov / (aligned_market.rolling(60).var() + 1e-8)
        return self._cross_sectional_rank(-beta.iloc[-1])
    
    def _calculate_liquidity(self, prices, volumes):
        return self._cross_sectional_rank((prices * volumes).rolling(20).mean().iloc[-1])

    def predict_returns(self, factors: pd.DataFrame, returns: pd.DataFrame) -> pd.Series:
        """
        Trains a model on historical data to predict future returns from the latest factors.
        """
        logger.info(f"Training predictive model ({self.prediction_model_type.upper()}) to generate alpha signal...")
        
        y = returns.shift(-self.prediction_target_days).rolling(self.prediction_target_days).sum().stack()
        
        # Align factors (X) and target (y)
        training_data = factors.join(y.rename('forward_return')).dropna()
        if training_data.empty:
            logger.error("No training data available after aligning factors and forward returns.")
            return pd.Series()

        X_train = training_data.drop(columns='forward_return')
        y_train = training_data['forward_return']
        
        X_latest = factors.loc[factors.index.get_level_values('date') == factors.index.get_level_values('date').max()]

        # MODEL SELECTION LOGIC
        if self.prediction_model_type == 'xgb':
            logger.info("Using XGBoost Regressor.")
            model = xgb.XGBRegressor(objective='reg:squarederror', **self.xgb_params, n_jobs=-1)
            model.fit(X_train, y_train)
            predictions = model.predict(X_latest)
        else: # Default to PCR
            logger.info("Using Principal Component Regression (PCR).")
            scaler = StandardScaler()
            pca = PCA(n_components=self.n_pca_components)
            model = LinearRegression()
            
            X_train_scaled = scaler.fit_transform(X_train)
            X_train_pca = pca.fit_transform(X_train_scaled)
            model.fit(X_train_pca, y_train)

            X_latest_scaled = scaler.transform(X_latest)
            X_latest_pca = pca.transform(X_latest_scaled)
            predictions = model.predict(X_latest_pca)
        
        alpha_signal = pd.Series(predictions, index=X_latest.index.get_level_values('ticker'))
        logger.info("Successfully generated final alpha signal from trained model.")
        return self._cross_sectional_rank(alpha_signal)