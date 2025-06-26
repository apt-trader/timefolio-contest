import pandas as pd
import numpy as np
import statsmodels.api as sm
from typing import Dict, List, Any, Tuple
import logging
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

def _safe_get(df: pd.DataFrame, key: str, index) -> pd.Series:
    """Safely get a column from a dataframe, returning a series of NaNs if not found."""
    if key in df:
        return df[key]
    return pd.Series(np.nan, index=index)

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
        self.latest_market_caps: pd.Series = None

    def _winsorize(self, series: pd.Series, limits=(0.05, 0.05)) -> pd.Series:
        """Limits extreme values to reduce the impact of outliers."""
        return series.clip(lower=series.quantile(limits[0]), upper=series.quantile(1 - limits[1]))

    def _standardize(self, series: pd.Series) -> pd.Series:
        """Scales a series to have a mean of 0 and a standard deviation of 1."""
        std = series.std()
        if std == 0:
            return series - series.mean()
        return (series - series.mean()) / std

    def _calculate_value_factors(self, fundamentals: pd.DataFrame, market_caps: pd.Series) -> Tuple[pd.Series, ...]:
        """Calculates B/P, E/P, S/P, and CF/P."""
        mcaps = market_caps.replace(0, np.nan)
        book_value = _safe_get(fundamentals, 'total_equity', market_caps.index)
        earnings = _safe_get(fundamentals, 'net_income', market_caps.index)
        sales = _safe_get(fundamentals, 'revenue', market_caps.index)
        cash_flow = _safe_get(fundamentals, 'operating_cash_flow', market_caps.index)
        
        b2p = book_value / mcaps
        e2p = earnings / mcaps
        s2p = sales / mcaps
        c2p = cash_flow / mcaps
        return b2p, e2p, s2p, c2p

    def _calculate_quality_factors(self, fundamentals: pd.DataFrame, historical_fundamentals: Dict[str, pd.DataFrame]) -> Tuple[pd.Series, ...]:
        """Calculates ROE, Financial Leverage, and ROE Stability."""
        roe = _safe_get(fundamentals, 'roe', fundamentals.index)
        leverage = _safe_get(fundamentals, 'debt_to_equity', fundamentals.index)
        
        roe_stability_values = {}
        for ticker, df in historical_fundamentals.items():
            if 'roe' in df.columns and len(df['roe'].dropna()) >= 2:
                roe_stability_values[ticker] = df['roe'].std()
        roe_stability = pd.Series(roe_stability_values, name='roe_stability')

        # Higher leverage and instability are bad, so we negate them.
        return roe, -leverage, -roe_stability

    def _calculate_profitability_factors(self, fundamentals: pd.DataFrame) -> Tuple[pd.Series, ...]:
        """Calculates Gross Profitability (GPA), Operating Margin, and Net Margin."""
        total_assets = _safe_get(fundamentals, 'total_assets', fundamentals.index).replace(0, np.nan)
        revenue = _safe_get(fundamentals, 'revenue', fundamentals.index).replace(0, np.nan)
        
        gross_profit = _safe_get(fundamentals, 'gross_profit', fundamentals.index)
        operating_income = _safe_get(fundamentals, 'operating_income', fundamentals.index)
        net_income = _safe_get(fundamentals, 'net_income', fundamentals.index)

        gpa = gross_profit / total_assets
        opm = operating_income / revenue
        npm = net_income / revenue
        return gpa, opm, npm

    def _calculate_momentum_factors(self, prices: pd.DataFrame) -> Tuple[pd.Series, ...]:
        """Calculates 12M momentum, 6M acceleration, and volatility-scaled momentum."""
        if prices.shape[0] < 252:
            return pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float)
        
        returns = prices.pct_change()
        # 12-month momentum, skipping most recent month
        mom12m = prices.pct_change(252 - 21).iloc[-1]
        
        # 6-month acceleration
        mom6m_late = prices.pct_change(126 - 21).iloc[-1]
        mom6m_early = prices.pct_change(126 - 21).iloc[-126]
        acceleration = mom6m_late - mom6m_early
        
        # Volatility-scaled momentum
        vol_scaled_mom = mom12m / returns.rolling(252).std().iloc[-1].replace(0, np.nan)
        
        return mom12m, acceleration, vol_scaled_mom

    def _calculate_investment_factors(self, historical_fundamentals: Dict[str, pd.DataFrame]) -> Tuple[pd.Series, ...]:
        """Calculates Asset Growth and CAPEX growth."""
        tickers = list(historical_fundamentals.keys())
        if not tickers:
            return pd.Series(dtype=float), pd.Series(dtype=float)

        asset_growth_vals = {}
        capex_growth_vals = {}
        
        for ticker, df in historical_fundamentals.items():
            # Asset Growth
            if 'total_assets' in df.columns and len(df) > 1 and df['total_assets'].iloc[-2] != 0:
                asset_growth_vals[ticker] = (df['total_assets'].iloc[-1] / df['total_assets'].iloc[-2]) - 1
            # CAPEX Growth
            if 'capex' in df.columns and len(df) > 1 and df['capex'].iloc[-2] != 0:
                 capex_growth_vals[ticker] = (df['capex'].iloc[-1] / df['capex'].iloc[-2]) - 1

        asset_growth = pd.Series(asset_growth_vals)
        capex_growth = pd.Series(capex_growth_vals)

        # Lower asset growth is considered better
        return -asset_growth, capex_growth

    def calculate_factors_for_date(self, prices: pd.DataFrame, volumes: pd.DataFrame, market_caps: pd.DataFrame, fundamentals: pd.DataFrame,
                             historical_fundamentals: Dict[str, pd.DataFrame], market_prices: pd.Series = None) -> pd.DataFrame:
        """Calculates all factors for a given date and returns a DataFrame."""
        self.latest_market_caps = market_caps.iloc[-1] if not market_caps.empty else pd.Series(dtype=float)
        
        # Align fundamentals to the same index as prices
        fundamentals = fundamentals.reindex(prices.columns)

        b2p, e2p, s2p, c2p = self._calculate_value_factors(fundamentals, self.latest_market_caps)
        roe, lev, roe_stab = self._calculate_quality_factors(fundamentals, historical_fundamentals)
        gpa, opm, npm = self._calculate_profitability_factors(fundamentals)
        mom, acc, vol_mom = self._calculate_momentum_factors(prices)
        inv, capex_g = self._calculate_investment_factors(historical_fundamentals)

        all_factors = pd.DataFrame({
            'B2P': b2p, 'E2P': e2p, 'S2P': s2p, 'C2P': c2p,
            'ROE': roe, 'Leverage': lev, 'ROE_Stability': roe_stab,
            'GPA': gpa, 'OPM': opm, 'NPM': npm,
            'Momentum': mom, 'Acceleration': acc, 'Vol_Momentum': vol_mom,
            'Investment': inv, 'CAPEX_Growth': capex_g
        })

        # Align all factors to the master price index, then clean and standardize
        all_factors = all_factors.reindex(prices.columns)
        all_factors = all_factors.apply(self._winsorize, axis=0).apply(self._standardize, axis=0)
        
        final_factors = all_factors.fillna(0)
        logger.info(f"Successfully calculated {final_factors.shape[1]} factors for {final_factors.shape[0]} tickers.")
        return final_factors

    def train(self, factor_panel: pd.DataFrame, forward_returns: pd.Series):
        """
        Trains a Principal Component Regression model on historical factor data.
        """
        logger.info("Training Principal Component Regression model...")
        n_components = self.settings.get('n_pca_components', 5)

        aligned_factors, aligned_returns = factor_panel.align(forward_returns, join='inner', axis=0)
        
        # Drop rows where the return is NaN, and then align factors to the remaining returns
        aligned_returns.dropna(inplace=True)
        aligned_factors = aligned_factors.loc[aligned_returns.index].dropna(how='all')
        
        if aligned_factors.empty or len(aligned_factors) < n_components:
            logger.error(f"Not enough data to train. Have {len(aligned_factors)} samples, need at least {n_components} for PCA. Aborting.")
            return

        self.feature_names = aligned_factors.columns.tolist()
        X = aligned_factors
        y = aligned_returns.loc[X.index]

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
            combined_data = pd.DataFrame({'alpha': scores_series, 'market_cap': market_caps}).dropna()

            if len(combined_data) < 2:
                 logger.warning("Not enough data for neutralization. Skipping.")
                 neutralized_scores = scores_series
            else:
                combined_data['is_small_cap'] = (combined_data['market_cap'] < self.small_cap_threshold).astype(int)
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