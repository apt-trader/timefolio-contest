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

def _safe_get(data, key: str, index) -> pd.Series:
    """Safely get a column/value from DataFrame or Series, returning a Series."""
    if isinstance(data, pd.DataFrame):
        if key in data:
            return data[key]
        return pd.Series(np.nan, index=index)
    elif isinstance(data, pd.Series):
        if key in data.index:
            # Return single value as Series with the provided index
            return pd.Series(data[key], index=index)
        return pd.Series(np.nan, index=index)
    else:
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

    def _calculate_value_factors(self, fundamentals: pd.DataFrame, market_caps: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """Calculates value factors: B/P, E/P, S/P, C/P."""
        print(f"[DEBUG] _calculate_value_factors input type: {type(fundamentals)}")
        print(f"[DEBUG] _calculate_value_factors input shape: {getattr(fundamentals, 'shape', 'N/A')}")
        
        if fundamentals.empty or (hasattr(fundamentals, 'columns') and fundamentals.columns.empty):
            logger.warning("Value factors: Fundamentals data is missing or empty. Returning NaNs.")
            nan_series = pd.Series(np.nan, index=market_caps.index)
            return nan_series, nan_series, nan_series, nan_series

        # If fundamentals is a DataFrame with multiple rows, take the most recent (last) row
        if isinstance(fundamentals, pd.DataFrame):
            print(f"[DEBUG] DataFrame with {len(fundamentals)} rows, {len(fundamentals.columns)} columns")
            if len(fundamentals) > 1:
                fundamentals = fundamentals.iloc[-1]  # Take the last row as a Series
                print(f"[DEBUG] Took last row, now type: {type(fundamentals)}")
            elif len(fundamentals) == 1:
                fundamentals = fundamentals.iloc[0]  # Take the single row as a Series
                print(f"[DEBUG] Took single row, now type: {type(fundamentals)}")
            else:
                # Empty DataFrame
                fundamentals = pd.Series(index=market_caps.index, dtype=float)
        
        # Now fundamentals should be a Series, align with market_caps
        if isinstance(fundamentals, pd.Series):
            print(f"[DEBUG] Series shape: {fundamentals.shape}")
            print(f"[DEBUG] Market caps shape: {market_caps.shape}")
            
            # Reindex fundamentals to match market_caps index
            fundamentals = fundamentals.reindex(market_caps.index)
            print(f"[DEBUG] After reindex, fundamentals shape: {fundamentals.shape}")
            
            # Calculate ratios - these should be 1D Series
            total_equity = fundamentals.get('total_equity')
            net_income = fundamentals.get('net_income')
            revenue = fundamentals.get('revenue')
            operating_cash_flow = fundamentals.get('operating_cash_flow')
            
            print(f"[DEBUG] total_equity type: {type(total_equity)}, shape: {getattr(total_equity, 'shape', 'N/A')}")
            
            # Handle case where .get() returns a scalar or Series
            if np.isscalar(total_equity) or total_equity is None:
                b2p = pd.Series(total_equity if total_equity is not None else np.nan, index=market_caps.index) / market_caps
            else:
                b2p = total_equity / market_caps
                
            if np.isscalar(net_income) or net_income is None:
                e2p = pd.Series(net_income if net_income is not None else np.nan, index=market_caps.index) / market_caps
            else:
                e2p = net_income / market_caps
                
            if np.isscalar(revenue) or revenue is None:
                s2p = pd.Series(revenue if revenue is not None else np.nan, index=market_caps.index) / market_caps
            else:
                s2p = revenue / market_caps
                
            if np.isscalar(operating_cash_flow) or operating_cash_flow is None:
                c2p = pd.Series(operating_cash_flow if operating_cash_flow is not None else np.nan, index=market_caps.index) / market_caps
            else:
                c2p = operating_cash_flow / market_caps
                
            print(f"[DEBUG] Final b2p shape: {getattr(b2p, 'shape', 'N/A')}")
            
        else:
            logger.warning("Value factors: Unexpected fundamentals data structure. Returning NaNs.")
            nan_series = pd.Series(np.nan, index=market_caps.index)
            return nan_series, nan_series, nan_series, nan_series
        
        return b2p, e2p, s2p, c2p

    def _calculate_quality_factors(self, fundamentals: pd.DataFrame, historical_fundamentals: Dict[str, pd.DataFrame], date: pd.Timestamp) -> Tuple[pd.Series, ...]:
        """Calculates ROE, Financial Leverage, and ROE Stability on a point-in-time basis."""
        roe = _safe_get(fundamentals, 'roe', fundamentals.index)
        leverage = _safe_get(fundamentals, 'debt_to_equity', fundamentals.index)
        
        roe_stability_values = {}
        for ticker, df in historical_fundamentals.items():
            # Filter for point-in-time data
            pit_df = df[df.index.date <= date.date()]
            if 'roe' in pit_df.columns and len(pit_df['roe'].dropna()) >= 4:  # Use at least 4 quarters for stable std dev
                roe_stability_values[ticker] = pit_df['roe'].std()
        # Reindex to ensure the series aligns with the full universe for the given date.
        roe_stability = pd.Series(roe_stability_values, name='roe_stability').reindex(fundamentals.index)

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
        """
        Calculates standard 12-1M momentum, 6M acceleration, and volatility-scaled momentum.
        Momentum is calculated as the return from 12 months ago to 1 month ago to avoid
        the short-term reversal effect, which is a well-documented market anomaly.
        """
        # Define lookback periods in trading days
        month_period = 21
        six_month_period = 126
        year_period = 252
        
        # Ensure there's enough data for the longest lookback
        if len(prices) < year_period + month_period:
            nan_series = pd.Series(np.nan, index=prices.columns)
            logger.warning(f"Not enough price data ({len(prices)} days) to calculate momentum, need {year_period + month_period}. Returning NaNs.")
            return nan_series, nan_series, nan_series

        # --- 12-1 Month Momentum ---
        # Price from 1 month ago (t-1M)
        price_t_minus_1 = prices.iloc[-month_period]
        # Price from 12 months ago (t-12M)
        price_t_minus_12 = prices.iloc[-year_period]
        
        # Calculate momentum, handling potential zero prices
        mom12m = (price_t_minus_1 / price_t_minus_12.replace(0, np.nan)) - 1

        # --- Volatility-Scaled Momentum ---
        # Volatility is calculated over the past year's log returns
        log_returns = np.log(prices.replace(0, np.nan)).diff()
        vol_returns = log_returns.iloc[-year_period:] # Use the last year of returns for vol calc
        annualized_vol = vol_returns.std() * np.sqrt(252)
        vol_scaled_mom = mom12m / annualized_vol.replace(0, np.nan)

        # --- 6-Month Acceleration (also lagged) ---
        # Price from 7 months ago (t-7M)
        price_t_minus_7 = prices.iloc[- (6 * month_period + month_period)]
        
        # Recent 6-month momentum (from t-7M to t-1M)
        late_6m_ret = (price_t_minus_1 / price_t_minus_7.replace(0, np.nan)) - 1
        
        # Check if there's enough data for prior momentum
        if len(prices) > year_period + six_month_period:
            # Price from 13 months ago (t-13M)
            price_t_minus_13 = prices.iloc[- (12 * month_period + month_period)]
            # Prior 6-month momentum (from t-13M to t-7M)
            early_6m_ret = (price_t_minus_7 / price_t_minus_13.replace(0, np.nan)) - 1
            acceleration = late_6m_ret - early_6m_ret
        else:
            acceleration = pd.Series(np.nan, index=prices.columns)
            
        return mom12m, acceleration, vol_scaled_mom

    def _calculate_investment_factors(self, historical_fundamentals: Dict[str, pd.DataFrame], date: pd.Timestamp, index: pd.Index) -> Tuple[pd.Series, ...]:
        """Calculates point-in-time Asset Growth and CAPEX growth."""
        if not historical_fundamentals:
            nan_series = pd.Series(np.nan, index=index)
            return nan_series, nan_series

        asset_growth_vals = {}
        capex_growth_vals = {}
        
        for ticker, df in historical_fundamentals.items():
            # Filter for point-in-time data
            pit_df = df[df.index.date <= date.date()]
            
            # Asset Growth
            if 'total_assets' in pit_df.columns:
                assets = pit_df['total_assets'].dropna()
                if len(assets) > 1 and assets.iloc[-2] != 0:
                    asset_growth_vals[ticker] = (assets.iloc[-1] / assets.iloc[-2]) - 1
            # CAPEX Growth
            if 'capex' in pit_df.columns:
                capex = pit_df['capex'].dropna()
                if len(capex) > 1 and capex.iloc[-2] != 0:
                     capex_growth_vals[ticker] = (capex.iloc[-1] / capex.iloc[-2]) - 1

        asset_growth = pd.Series(asset_growth_vals).reindex(index)
        capex_growth = pd.Series(capex_growth_vals).reindex(index)

        # Lower asset growth is considered better
        return -asset_growth, capex_growth

    def calculate_factors_for_date(self, date: pd.Timestamp, prices: pd.DataFrame, volumes: pd.DataFrame, market_caps: pd.Series, fundamentals: pd.DataFrame,
                             historical_fundamentals: Dict[str, pd.DataFrame], market_prices: pd.Series = None) -> pd.DataFrame:
        """Calculates all factors for a given date and returns a DataFrame."""
        logger.debug(f"[{date.date()}] Calculating factors for {len(prices.columns)} tickers.")
        self.latest_market_caps = market_caps
        
        # Debug fundamentals structure
        logger.debug(f"Fundamentals shape before processing: {fundamentals.shape}")
        logger.debug(f"Fundamentals index: {fundamentals.index[:5] if len(fundamentals.index) > 0 else 'Empty'}")
        logger.debug(f"Fundamentals columns: {fundamentals.columns[:5] if len(fundamentals.columns) > 0 else 'Empty'}")
        
        # If fundamentals has date index, get the most recent available data for each ticker
        if isinstance(fundamentals.index, pd.DatetimeIndex):
            # Get data up to the current date
            available_data = fundamentals[fundamentals.index <= date]
            if not available_data.empty:
                # Take the most recent row for each ticker
                fundamentals = available_data.iloc[-1]
                logger.debug(f"Using most recent fundamental data from: {available_data.index[-1]}")
            else:
                logger.warning(f"No fundamental data available for date {date}")
                fundamentals = pd.Series(index=prices.columns, dtype=float)
        else:
            # If not date-indexed, assume it's already properly formatted
            # Ensure we have the right columns (tickers)
            fundamentals = fundamentals.reindex(prices.columns)
            
        logger.debug(f"Fundamentals shape after processing: {getattr(fundamentals, 'shape', 'Series')}")
        
        # Ensure fundamentals is a Series for value factor calculation
        if isinstance(fundamentals, pd.DataFrame):
            if len(fundamentals) == 1:
                # Single row DataFrame - convert to Series
                fundamentals = fundamentals.iloc[0]
            elif len(fundamentals) > 1:
                # Multiple rows - take the last row as the most recent
                fundamentals = fundamentals.iloc[-1]
            else:
                # Empty DataFrame - create empty Series
                fundamentals = pd.Series(index=prices.columns, dtype=float)
                
        logger.debug(f"Fundamentals converted to Series shape: {getattr(fundamentals, 'shape', 'N/A')}")
        
        # Ensure market_caps is a 1D Series for the current date
        if isinstance(self.latest_market_caps, pd.DataFrame):
            # If market_caps is 2D, get the data for the current date
            if date in self.latest_market_caps.index:
                current_market_caps = self.latest_market_caps.loc[date]
            else:
                # Use the most recent available date
                available_dates = self.latest_market_caps.index[self.latest_market_caps.index <= date]
                if len(available_dates) > 0:
                    current_market_caps = self.latest_market_caps.loc[available_dates[-1]]
                else:
                    current_market_caps = self.latest_market_caps.iloc[-1]
        else:
            # Already a 1D Series
            current_market_caps = self.latest_market_caps
            
        print(f"[DEBUG] Current market caps shape: {getattr(current_market_caps, 'shape', 'N/A')}")

        b2p, e2p, s2p, c2p = self._calculate_value_factors(fundamentals, current_market_caps)
        roe, lev, roe_stab = self._calculate_quality_factors(fundamentals, historical_fundamentals, date)
        gpa, opm, npm = self._calculate_profitability_factors(fundamentals)
        mom, acc, vol_mom = self._calculate_momentum_factors(prices)
        inv, capex_g = self._calculate_investment_factors(historical_fundamentals, date, prices.columns)

        logger.info(f"Debug factor shapes: "
                    f"B2P: {getattr(b2p, 'shape', 'N/A')}, "
                    f"E2P: {getattr(e2p, 'shape', 'N/A')}, "
                    f"S2P: {getattr(s2p, 'shape', 'N/A')}, "
                    f"C2P: {getattr(c2p, 'shape', 'N/A')}, "
                    f"ROE: {getattr(roe, 'shape', 'N/A')}, "
                    f"Leverage: {getattr(lev, 'shape', 'N/A')}, "
                    f"ROE_Stability: {getattr(roe_stab, 'shape', 'N/A')}, "
                    f"GPA: {getattr(gpa, 'shape', 'N/A')}, "
                    f"OPM: {getattr(opm, 'shape', 'N/A')}, "
                    f"NPM: {getattr(npm, 'shape', 'N/A')}, "
                    f"Momentum: {getattr(mom, 'shape', 'N/A')}, "
                    f"Acceleration: {getattr(acc, 'shape', 'N/A')}, "
                    f"Vol_Momentum: {getattr(vol_mom, 'shape', 'N/A')}, "
                    f"Investment: {getattr(inv, 'shape', 'N/A')}, "
                    f"CAPEX_Growth: {getattr(capex_g, 'shape', 'N/A')}")

        all_factors = pd.DataFrame({
            'B2P': b2p, 'E2P': e2p, 'S2P': s2p, 'C2P': c2p,
            'ROE': roe, 'Leverage': lev, 'ROE_Stability': roe_stab,
            'GPA': gpa, 'OPM': opm, 'NPM': npm,
            'Momentum': mom, 'Acceleration': acc, 'Vol_Momentum': vol_mom,
            'Investment': inv, 'CAPEX_Growth': capex_g
        })

        # Align all factors to the master price index, then clean and standardize
        all_factors = all_factors.reindex(prices.columns)
        all_factors.dropna(how='all', inplace=True) # Drop tickers where all factors are NaN

        # --- Diagnostic Logging ---
        logger.debug(f"[{date.date()}] Raw factor non-null counts:\n" + str(all_factors.notna().sum()))

        if all_factors.empty:
            logger.warning(f"[{date.date()}] Factor DataFrame is empty because no factors could be calculated for any ticker.")
            return all_factors

        # Drop columns that are entirely NaN
        initial_cols = set(all_factors.columns)
        all_factors.dropna(axis=1, how='all', inplace=True)
        final_cols = set(all_factors.columns)
        dropped_cols = initial_cols - final_cols
        if dropped_cols:
            logger.warning(f"[{date.date()}] Dropped fully null factor columns: {sorted(list(dropped_cols))}")

        # Impute missing values with the cross-sectional median for that day.
        # This is more robust than filling with 0, especially before standardization.
        imputed_factors = all_factors.apply(lambda x: x.fillna(x.median()), axis=0)

        # After imputation, some columns might still be all NaN if all values were NaN to begin with.
        imputed_factors.dropna(axis=1, how='all', inplace=True)
        imputed_factors.dropna(axis=0, how='all', inplace=True)
        
        if imputed_factors.empty:
            logger.warning(f"[{date.date()}] Factor DataFrame is empty after imputation. All factors were NaN.")
            return imputed_factors

        # Now, winsorize and standardize.
        processed_factors = imputed_factors.apply(self._winsorize, axis=0).apply(self._standardize, axis=0)
        
        # As a final fallback, fill any remaining NaNs with 0. This can happen if a column had only one non-NaN value.
        final_factors = processed_factors.fillna(0)
        
        logger.info(f"Successfully calculated {final_factors.shape[1]} factors for {final_factors.shape[0]} tickers on {date.date()}.")
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
        final_scores = self._normalize_alpha_scores(final_neutralized_scores)
        
        return final_scores

    def _normalize_alpha_scores(self, scores_series: pd.Series) -> pd.Series:
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