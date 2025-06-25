import pandas as pd
import numpy as np
import statsmodels.api as sm
from typing import Dict, List, Any
import logging
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

class FactorEngine:
    """
    Calculates a variety of financial factors and uses a Principal Component Regression
    model to generate a unified alpha signal.
    """

    def __init__(self, settings: Dict[str, Any] = None):
        """
        Initializes the FactorEngine.

        Args:
            settings (Dict[str, Any]): Configuration settings for the factor engine.
        """
        self.settings = settings if settings is not None else {}
        self.small_cap_threshold = self.settings.get('small_cap_threshold', 1_000_000_000_000)
        self.model: Pipeline = None
        self.feature_names: List[str] = None
        self.factor_df = pd.DataFrame()
        self.latest_market_caps: pd.DataFrame = None

    def _cross_sectional_rank(self, series: pd.Series) -> pd.Series:
        """
        Ranks a series cross-sectionally and scales it from -1 to 1.
        """
        return series.rank(pct=True).mul(2).sub(1)

    def _validate_and_align_factor(self, factor_values: pd.Series, expected_index: pd.Index, factor_name: str) -> pd.Series:
        """
        Validates and aligns a factor Series to the expected index.
        """
        if factor_values is None or factor_values.empty:
            return pd.Series(0, index=expected_index)
        
        factor_values = pd.to_numeric(factor_values, errors='coerce')
        aligned = factor_values.reindex(expected_index)
        aligned.replace([np.inf, -np.inf], np.nan, inplace=True)
        return aligned.fillna(0)

    def _calculate_momentum_factors(self, prices: pd.DataFrame) -> pd.Series:
        if prices.shape[0] < 252: return pd.Series(0, index=prices.columns)
        returns = prices.pct_change()
        momentum_12m = (prices.shift(21) / prices.shift(252) - 1).iloc[-1]
        mom_6m = prices.pct_change(126)
        mom_accel = mom_6m.ffill().pct_change(63).iloc[-1]
        vol = returns.rolling(252, min_periods=126).std().iloc[-1] * np.sqrt(252)
        vol_scaled_momentum = momentum_12m / (vol.replace(0, np.nan).add(1e-6))
        composite = (momentum_12m.fillna(0) + mom_accel.fillna(0) + vol_scaled_momentum.fillna(0)) / 3
        return self._cross_sectional_rank(composite)

    def _calculate_value_factors(self, fundamentals: pd.DataFrame, latest_prices: pd.Series) -> pd.Series:
        """Calculate a composite value factor. 
        Due to sparse income statement data, this factor now primarily relies on Book-to-Price.
        """
        def safe_get(df, key):
            return df[key] if key in df else pd.Series(0, index=df.index)

        # Ensure prices are not zero to avoid division errors
        prices = latest_prices.replace(0, 1e-6).reindex(fundamentals.index).fillna(1e-6)

        # Book-to-Price (B2P) is the most reliable value metric with available data.
        b2p = safe_get(fundamentals, 'book_value_per_share') / prices
        
        # Other metrics like E/P, S/P, CF/P have low coverage and are excluded.
        # The composite is now just the ranked B2P.
        composite = b2p.fillna(0)
        return self._cross_sectional_rank(composite)

    def _calculate_quality_factors(self, fundamentals: pd.DataFrame, historical_fundamentals: Dict[str, pd.DataFrame]) -> pd.Series:
        """Calculates a composite quality factor based on balance sheet health.
        Relies on leverage, as ROE has poor data coverage.
        """
        
        def safe_get(df, key):
            return df[key] if key in df else pd.Series(0, index=df.index)

        # Leverage is a reliable quality metric based on high-coverage balance sheet data.
        leverage = safe_get(fundamentals, 'debt_to_equity')
        
        # Lower leverage is higher quality, so we rank the negative of the leverage ratio.
        # A small epsilon is added to avoid division by zero for companies with no debt.
        composite = -leverage.fillna(0)
        return self._cross_sectional_rank(composite)

    def _calculate_investment_factor(self, historical_fundamentals: Dict[str, pd.DataFrame]) -> pd.Series:
        """Calculates the investment factor based on asset growth. Lower growth is ranked higher."""
        tickers = list(historical_fundamentals.keys())
        if not historical_fundamentals or not tickers:
            return pd.Series(dtype=np.float64)

        hist_df = pd.concat(historical_fundamentals, names=['ticker', 'date']).reset_index()
        
        if 'total_assets' not in hist_df.columns:
            # If total_assets data is missing, return a neutral factor
            return pd.Series(0, index=tickers)

        # Calculate asset growth for each ticker
        asset_growth = hist_df.groupby('ticker')['total_assets'].apply(
            lambda x: (x.iloc[-1] / x.iloc[-2] - 1) if len(x) > 1 and x.iloc[-2] != 0 else 0
        )
        
        # Reindex to ensure all tickers are present, filling missing ones with 0
        asset_growth = asset_growth.reindex(tickers).fillna(0)

        # Lower growth is better, so we rank the negative of asset growth
        return self._cross_sectional_rank(-asset_growth)

    def _calculate_profitability_factors(self, fundamentals: pd.DataFrame) -> pd.Series:
        """Calculates a composite profitability factor.
        Uses Asset Turnover as a proxy for profitability due to sparse income statement data.
        """
        def safe_get(df, key):
            return df[key] if key in df else pd.Series(0, index=df.index)

        # Total Assets data is reliable.
        total_assets = safe_get(fundamentals, 'total_assets').replace(0, 1e-6)
        
        # Sales data has ~55% coverage, which is the best available for a profitability proxy.
        sales = safe_get(fundamentals, 'sales_per_share') * safe_get(fundamentals, 'shares_outstanding')
        
        # Asset Turnover = Sales / Total Assets. Higher is better.
        asset_turnover = sales / total_assets
        
        composite = asset_turnover.fillna(0)
        return self._cross_sectional_rank(composite)

    def _calculate_volatility_factor(self, prices: pd.DataFrame) -> pd.Series:
        if prices.shape[0] < 252: return pd.Series(0, index=prices.columns)
        returns = prices.pct_change()
        volatility = returns.rolling(252, min_periods=126).std().iloc[-1] * np.sqrt(252)
        return self._cross_sectional_rank(-volatility) # Lower volatility is better

    def _calculate_ivol_factor(self, prices: pd.DataFrame, market_prices: pd.Series = None) -> pd.Series:
        if market_prices is None or prices.shape[0] < 60: return pd.Series(0, index=prices.columns)
        returns = prices.pct_change().dropna()
        market_returns = market_prices.pct_change().dropna()
        common_dates = returns.index.intersection(market_returns.index)
        returns, market_returns = returns.loc[common_dates], market_returns.loc[common_dates]

        ivol_vals = {}
        lr = LinearRegression()
        X = market_returns.values.reshape(-1, 1)
        for ticker in prices.columns:
            if ticker in returns.columns and not returns[ticker].isnull().all():
                y = returns[ticker].values
                try:
                    lr.fit(X, y)
                    residuals = y - lr.predict(X)
                    ivol_vals[ticker] = np.std(residuals) * np.sqrt(252)
                except Exception:
                    ivol_vals[ticker] = np.nan
            else:
                ivol_vals[ticker] = np.nan
        return self._cross_sectional_rank(-pd.Series(ivol_vals)) # Lower IVOL is better

    def _calculate_volume_momentum(self, volumes: pd.DataFrame) -> pd.Series:
        if volumes.shape[0] < 252: return pd.Series(0, index=volumes.columns)
        avg_vol_12m = volumes.rolling(252, min_periods=126).mean()
        avg_vol_1m = volumes.rolling(21, min_periods=10).mean()
        volume_momentum = (avg_vol_1m.iloc[-1] / avg_vol_12m.iloc[-1].add(1e-6)) - 1
        return self._cross_sectional_rank(volume_momentum)

    def _calculate_volume_trend(self, volumes: pd.DataFrame) -> pd.Series:
        if volumes.shape[0] < 60: return pd.Series(0, index=volumes.columns)
        log_volumes = np.log1p(volumes.iloc[-60:])
        time_trend = np.arange(len(log_volumes))
        slopes = {ticker: np.polyfit(time_trend, log_volumes[ticker].dropna(), 1)[0] 
                  for ticker in log_volumes.columns if len(log_volumes[ticker].dropna()) > 1}
        return self._cross_sectional_rank(pd.Series(slopes))

    def _calculate_liquidity_factor(self, prices: pd.DataFrame, volumes: pd.DataFrame) -> pd.Series:
        if prices.shape[0] < 63: return pd.Series(0, index=prices.columns)
        avg_dollar_volume = (prices * volumes).iloc[-63:].mean()
        return self._cross_sectional_rank(avg_dollar_volume)

    def _calculate_volume_volatility(self, volumes: pd.DataFrame) -> pd.Series:
        if volumes.shape[0] < 126: return pd.Series(0, index=volumes.columns)
        log_volumes = np.log1p(volumes)
        volume_vol = log_volumes.rolling(126, min_periods=63).std().iloc[-1]
        return self._cross_sectional_rank(-volume_vol) # Lower is better

    def calculate_factors_for_date(self, prices: pd.DataFrame, volumes: pd.DataFrame, market_caps: pd.DataFrame, fundamentals: pd.DataFrame,
                             historical_fundamentals: Dict[str, pd.DataFrame], market_prices: pd.Series = None) -> pd.DataFrame:
        """Calculates all factors for a given date and returns a DataFrame."""
        # Cache market caps for the predict step
        self.latest_market_caps = market_caps.iloc[-1] if not market_caps.empty else pd.Series(dtype=float)

        tickers = prices.columns.tolist()
        logger.info(f"Calculating factors for {len(tickers)} tickers.")
        expected_index = pd.Index(tickers)

        # Align fundamentals to the same index as prices
        fundamentals = fundamentals.reindex(tickers)

        factors = {}
        factor_methods = {
            'momentum': self._calculate_momentum_factors, 'value': self._calculate_value_factors,
            'quality': self._calculate_quality_factors, 'investment': self._calculate_investment_factor,
            'profitability': self._calculate_profitability_factors, 'volatility': self._calculate_volatility_factor,
            'ivol': self._calculate_ivol_factor, 'volume_momentum': self._calculate_volume_momentum,
            'volume_trend': self._calculate_volume_trend, 'liquidity': self._calculate_liquidity_factor,
            'volume_volatility': self._calculate_volume_volatility
        }
        method_args = {
            'momentum': (prices,), 'value': (fundamentals, prices.iloc[-1]),
            'quality': (fundamentals, historical_fundamentals), 'investment': (historical_fundamentals,),
            'profitability': (fundamentals,), 'volatility': (prices,),
            'ivol': (prices, market_prices), 'volume_momentum': (volumes,),
            'volume_trend': (volumes,), 'liquidity': (prices, volumes),
            'volume_volatility': (volumes,)
        }
        for name, method in factor_methods.items():
            logger.debug(f"Calculating factor: {name}")
            try:
                raw_factor = method(*method_args[name])
                factors[name] = self._validate_and_align_factor(raw_factor, expected_index, name)
                if factors[name].isnull().all():
                    logger.warning(f"Factor '{name}' is all NaN after calculation and alignment.")
            except Exception as e:
                logger.error(f"Error calculating factor '{name}': {e}", exc_info=True)
                factors[name] = pd.Series(0, index=expected_index)
        
        factors_df = pd.DataFrame(factors, index=expected_index).fillna(0)
        logger.info("Factor calculation complete.")
        return factors_df

    def train(self, factor_panel: pd.DataFrame, forward_returns: pd.Series):
        """
        Trains a Principal Component Regression model on historical factor data.
        """
        logger.info("Training Principal Component Regression model...")
        n_components = self.settings.get('n_pca_components', 5)

        aligned_factors, aligned_returns = factor_panel.align(forward_returns, join='inner', axis=0)
        aligned_factors.dropna(inplace=True)
        aligned_returns = aligned_returns.loc[aligned_factors.index]

        if aligned_factors.empty:
            logger.error("Factor and return data do not align or are empty. Model cannot be trained.")
            return

        self.feature_names = aligned_factors.columns.tolist()
        X = aligned_factors
        y = aligned_returns

        if len(X) < n_components * 10:
            logger.error(f"Not enough data to train a reliable model (have {len(X)} rows). Aborting training.")
            return

        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=n_components)),
            ('regressor', LinearRegression())
        ])
        
        self.model.fit(X, y)
        logger.info(f"PCR Model trained successfully. PCA explained variance: {self.model.named_steps['pca'].explained_variance_ratio_.sum():.2%}")

    def predict(self, latest_factors: pd.DataFrame) -> pd.Series:
        """
        Uses the trained PCR model to predict expected returns, neutralizes them
        against market cap, and normalizes them using cross-sectional z-scoring.
        """
        if self.model is None:
            raise RuntimeError("Model must be trained before making predictions.")

        if latest_factors.empty:
            logger.warning("latest_factors is empty. Cannot predict.")
            return pd.Series(dtype=float)

        latest_factors.dropna(how='all', inplace=True)
        if latest_factors.empty:
            return pd.Series(dtype=float)

        X = latest_factors.reindex(columns=self.feature_names, fill_value=0)
        raw_scores = self.model.predict(X)
        scores_series = pd.Series(raw_scores, index=X.index, name='alpha')

        # --- Market Cap Neutralization ---
        logger.info("Performing market-cap neutralization on alpha scores...")
        if self.latest_market_caps is None or self.latest_market_caps.empty:
            logger.warning("Market cap data not available. Skipping neutralization.")
            neutralized_scores = scores_series
        else:
            market_caps = self.latest_market_caps.reindex(scores_series.index)
            logger.info(f"Market cap summary for neutralization:\n{market_caps.describe()}")
            combined_data = pd.concat([scores_series, market_caps], axis=1)
            combined_data.columns = ['alpha', 'market_cap']
            combined_data.dropna(inplace=True)

            if combined_data.empty or len(combined_data) < 2:
                 logger.warning("Not enough data for neutralization after merging with market caps. Skipping.")
                 neutralized_scores = scores_series
            else:
                combined_data['is_small_cap'] = (combined_data['market_cap'] < self.small_cap_threshold).astype(int)
                logger.info(f"Small-cap indicator distribution (1=Small, 0=Large):\n{combined_data['is_small_cap'].value_counts(normalize=True)}")
                if combined_data['is_small_cap'].nunique() < 2:
                    logger.warning("No variance in small-cap indicator. Skipping neutralization.")
                    neutralized_scores = scores_series.reindex(combined_data.index)
                else:
                    X_neut = sm.add_constant(combined_data['is_small_cap'])
                    model = sm.OLS(combined_data['alpha'], X_neut).fit()
                    neutralized_scores = model.resid
                    logger.info("Neutralized alpha scores against small-cap factor.")
        
        final_neutralized_scores = neutralized_scores.reindex(scores_series.index).fillna(0)

        # --- Cross-Sectional Z-Scoring ---
        final_scores = self._normalize_scores(final_neutralized_scores)
        
        return final_scores

    def _normalize_scores(self, scores_series: pd.Series) -> pd.Series:
        """
        Helper to normalize raw alpha scores into a z-score.
        """
        mean = scores_series.mean()
        std = scores_series.std()

        if std > 1e-9:
            normalized_scores = (scores_series - mean) / std
        else:
            normalized_scores = scores_series - mean

        logger.info(
            f"Normalized expected returns. "
            f"Original Mean: {mean:.4f}, Std: {std:.4f}. "
            f"New Mean: {normalized_scores.mean():.4f}, Std: {normalized_scores.std():.4f}"
        )
        return normalized_scores
