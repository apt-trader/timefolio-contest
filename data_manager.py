# data_manager.py
import logging
import sqlite3
import sys
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union
import os
from pathlib import Path
from config import Config
from fetchers.krx_fetcher import KRXDataFetcher
from fetchers.financial_fetcher import FinancialsFetcher
from fetchers.macro_fetcher import MacroFetcher
from compliance_filters import ComplianceFilter

logger = logging.getLogger(__name__)

class DataManager:
    def __init__(self, cfg: Config, backtest_mode: bool = False):
        self.cfg = cfg
        self.db_path = self.cfg.db_path
        self.start_date = pd.to_datetime(self.cfg.start_date)
        self.end_date = pd.to_datetime(self.cfg.end_date)
        self.conn = self._create_connection()
        self.backtest_mode = backtest_mode
        self._apply_db_optimizations()
        self._krx = KRXDataFetcher(db_path=self.db_path)
        self._fin = FinancialsFetcher(api_key=os.getenv('DART_API_KEY'), db_path=self.db_path)
        self._macro = MacroFetcher(fred_api_key=os.getenv('FRED_API_KEY'))
        self.tickers: List[str] = []; self.sector_map: Dict[str, str] = {}
        # Dataframes to be populated by load_data()
        self.prices, self.volumes, self.market_caps, self.returns = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        self.latest_fundamentals, self.historical_fundamentals = pd.DataFrame(), {}
        self.macro_data, self.market_prices = pd.DataFrame(), pd.DataFrame()

    def _create_connection(self):
        """Creates a database connection."""
        try:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            logger.info(f"Successfully connected to database at {self.db_path}")
            return conn
        except sqlite3.Error as e:
            logger.error(f"Error connecting to database: {e}")
            raise

    def _normalize_company_name(self, name: str) -> str:
        """
        Normalizes company names to facilitate merging between different data sources.
        - Converts to lowercase
        - Removes common corporate suffixes and punctuation
        - Removes all whitespace
        """
        if not isinstance(name, str):
            return ''
        name = name.lower()
        # List of suffixes and patterns to remove
        suffixes_to_remove = [
            '(주)', '주식회사', 'co.,ltd.', 'co,ltd', 'co., ltd.', 'ltd.', 'inc.', 
            ',', '.', '&', ' co'
        ]
        for suffix in suffixes_to_remove:
            name = name.replace(suffix, '')
        
        # Remove all whitespace characters
        name = ''.join(name.split())
        return name

    def _build_sector_map(self) -> bool:
        """
        Builds the sector map by merging listing info with the sector universe file.
        Uses normalized company names to ensure a robust match.
        """
        try:
            logger.info("Building sector map...")
            # Use get_listing_info() to ensure we have the full, clean data
            self.listing_info = self.get_listing_info()
            if self.listing_info.empty:
                logger.error("Cannot build sector map: listing_info is empty.")
                return False

            sector_file_path = Path(self.cfg.data_settings['stock_universe_file'])
            if not sector_file_path.exists():
                logger.error(f"Sector universe file not found at: {sector_file_path}")
                return False
            
            # Correctly read the CSV with all 4 columns and utf-8 encoding.
            sector_df = pd.read_csv(
                sector_file_path,
                dtype={'섹터코드': str, '섹터명': str, '종목코드': str, '종목명': str},
                encoding='utf-8'
            )

            # The CSV is malformed. The actual content is: '섹터명' -> ticker, '종목코드' -> company name
            sector_df.rename(columns={'종목코드': 'company_name', '섹터명': 'code', '섹터코드': 'sector_code'}, inplace=True)

            # Clean the ticker code by removing the leading 'A'.
            sector_df['code'] = sector_df['code'].str.replace('A', '', regex=False)

            # Normalize company names for a robust merge.
            self.listing_info['normalized_name'] = self.listing_info['name'].apply(self._normalize_company_name)
            sector_df['normalized_name'] = sector_df['company_name'].apply(self._normalize_company_name)

            # Merge on the normalized name.
            merged_df = pd.merge(self.listing_info, sector_df, on='normalized_name', how='inner')

            if merged_df.empty:
                logger.warning("Sector map is empty after merging. Check for name mismatches between listing info and sector file.")
                # For debugging, let's see what the normalized names look like from both sources.
                logger.info("Sample normalized names from listing_info (DB):")
                logger.info(self.listing_info[['name', 'normalized_name']].head(10))
                logger.info("Sample normalized names from sector_df (CSV):")
                logger.info(sector_df[['company_name', 'normalized_name', 'code']].head(10))
                self.sector_map = {}
            else:
                # Use 'code_x' because 'code' column is duplicated after merge
                self.sector_map = merged_df.set_index('code_x')['sector_code'].to_dict()
                logger.info(f"Successfully built sector map with {len(self.sector_map)} entries.")
            
            return True
        except Exception as e:
            logger.error(f"Failed to build sector map: {e}", exc_info=True)
            return False

    def load_data(self) -> bool:
        """Orchestrates the loading of all necessary data."""
        logger.info("--- Starting Data Loading Sequence ---")
        try:
            # Step 0: Build the sector map. This is a prerequisite for everything.
            logger.info("Step 0: Building sector map...")
            if not self._build_sector_map():
                logger.critical("Step 0 FAILED: Could not build sector map.")
                return False
            logger.info("Step 0 SUCCESS: Sector map built.")

            # Step 1: Determine stock universe.
            logger.info("Step 1: Determining stock universe...")
            # In backtest mode, the initial universe is all tickers that have a sector mapping.
            # The compliance filters will then be applied for each rebalance date.
            self.tickers = list(self.sector_map.keys())
            if not self.tickers:
                logger.critical("Step 1 FAILED: No tickers found after building sector map. The universe is empty.")
                return False
            logger.info(f"Step 1 SUCCESS: Initial universe loaded with {len(self.tickers)} tickers with sector data.")

            # Step 2: Loading market data (prices, volumes, caps)...
            logger.info("Step 2: Loading market data (prices, volumes, caps)...")
            if not self._load_market_data():
                logger.critical("Step 2 FAILED: Market data loading.")
                return False
            logger.info("Step 2 SUCCESS: Market data loaded.")

            # Step 3: Loading market index data (KOSPI)...
            logger.info("Step 3: Loading market index data (KOSPI)...")
            if not self._load_market_index_data():
                logger.critical("Step 3 FAILED: Market index data loading.")
                return False
            logger.info("Step 3 SUCCESS: Market index data loaded.")

            # Step 4: Fetching all historical financial statement data...
            logger.info("Step 4: Fetching all historical financial statement data using the new bulk fetcher...")
            self.financials_df = self._fin.get_historical_financials_from_db(self.tickers)
            if not self.financials_df.empty:
                # Ensure ticker is a zero-padded string for consistency
                self.financials_df['ticker'] = self.financials_df['ticker'].astype(str).str.zfill(6)
                
                # The new fetcher guarantees 'report_date' column.
                if 'report_date' not in self.financials_df.columns:
                    logger.error("CRITICAL: Historical fundamentals are missing the 'report_date' column even after fetching.")
                    self.historical_fundamentals = {}
                    self.latest_fundamentals = pd.DataFrame()
                else:
                    self.financials_df['report_date'] = pd.to_datetime(self.financials_df['report_date'], errors='coerce')
                    self.financials_df.dropna(subset=['report_date'], inplace=True)
                    
                    # Group by ticker to create a dictionary of historical dataframes
                    self.historical_fundamentals = {
                        ticker: group.drop(columns='ticker', errors='ignore').set_index('report_date').sort_index()
                        for ticker, group in self.financials_df.groupby('ticker')
                    }
                    self.latest_fundamentals = self.financials_df.sort_values('report_date').groupby('ticker').tail(1).set_index('ticker')

                    logger.info(f"Loaded historical fundamentals for {len(self.historical_fundamentals)} tickers.")
                    logger.info(f"Latest fundamentals available for {len(self.latest_fundamentals)} tickers.")
            else:
                logger.warning("No historical fundamentals found for any tickers in the universe.")
                self.historical_fundamentals = {}
                self.latest_fundamentals = pd.DataFrame()
            logger.info("Step 4 SUCCESS: Historical financial data processed.")

            # Step 5: Fetching macroeconomic data...
            logger.info("Step 5: Fetching macroeconomic data...")
            macro_query = f"SELECT * FROM macro_data WHERE date BETWEEN '{self.start_date.strftime('%Y-%m-%d')}' AND '{self.end_date.strftime('%Y-%m-%d')}'"
            self.macro_data = pd.read_sql(macro_query, self.conn, index_col='date', parse_dates=['date'])
            logger.info("Step 5 SUCCESS: Macroeconomic data fetched and loaded.")

            if self.prices.empty:
                logger.error("No market data loaded for the specified universe and date range.")
                return False

            # Step 6: Aligning and processing final dataframes...
            logger.info("Step 6: Aligning and processing final dataframes...")
            self._align_and_process_data()
            self._calculate_and_fill_returns()
            logger.info("Step 6 SUCCESS: Data aligned and returns calculated.")
            
            self._log_data_quality()
            logger.info("--- Data Loading Sequence Complete ---")
            return True
        except Exception as e:
            logger.exception(f"A critical error occurred during the data loading sequence: {e}")
            return False

    def _align_and_process_data(self):
        """
        Aligns all time-series data to a common date index and ticker columns.
        This ensures consistency before any calculations are performed.
        """
        logger.info("Aligning all dataframes to a common index...")
        if self.prices.empty:
            logger.error("Cannot align data, prices dataframe is empty.")
            return

        master_index = self.prices.index
        master_columns = self.prices.columns

        self.prices = self.prices.reindex(index=master_index, columns=master_columns).ffill()
        self.volumes = self.volumes.reindex(index=master_index, columns=master_columns).fillna(0)
        self.market_caps = self.market_caps.reindex(index=master_index, columns=master_columns).ffill()
        self.market_prices = self.market_prices.reindex(master_index).ffill()
        self.macro_data = self.macro_data.reindex(master_index).ffill()

        self.prices.bfill(inplace=True)
        self.market_caps.bfill(inplace=True)
        self.market_prices.bfill(inplace=True)
        self.macro_data.bfill(inplace=True)
        logger.info("Data alignment complete.")

    def _calculate_and_fill_returns(self):
        """Calculates daily returns and handles infinite values."""
        logger.info("Calculating daily returns...")
        if self.prices.empty:
            logger.error("Cannot calculate returns, prices dataframe is empty.")
            self.returns = pd.DataFrame()
            return

        self.returns = self.prices.pct_change().fillna(0)
        self.returns.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.returns.fillna(0, inplace=True)
        logger.info("Daily returns calculated and cleaned.")

    def _log_data_quality(self):
        """Logs summary statistics and data quality metrics."""
        logger.info("--- Data Quality Report ---")
        if not self.prices.empty:
            total_datapoints = self.prices.size
            missing_prices = self.prices.isnull().sum().sum()
            logger.info(f"Price Data: {self.prices.shape[0]} days, {self.prices.shape[1]} tickers.")
            logger.info(f"Missing Price Points: {missing_prices} ({missing_prices/total_datapoints:.2%})")
        else:
            logger.warning("Price data is empty.")
        if not self.latest_fundamentals.empty:
            logger.info(f"Latest Fundamentals: {self.latest_fundamentals.shape[0]} tickers, {self.latest_fundamentals.shape[1]} fields.")
            missing_fundamentals = self.latest_fundamentals.isnull().sum()
            if (missing_fundamentals > 0).any():
                logger.debug(f"Missing fundamental data points per field:\n{missing_fundamentals[missing_fundamentals > 0]}")
        else:
            logger.warning("Latest fundamentals data is empty.")
        logger.info("--- End of Report ---")

    def _load_market_data(self) -> bool:
        """Load market data with validation and cleaning."""
        logger.info("Loading market data...")
        
        try:
            if not hasattr(self, 'tickers') or not self.tickers:
                logger.error("No tickers available to load market data")
                return False
                
            placeholders = ','.join(['?'] * len(self.tickers))
            query = f"""
                SELECT code, date, close, volume, market_cap
                FROM daily_prices 
                WHERE date BETWEEN ? AND ? AND code IN ({placeholders})
            """
            
            params = [self.start_date.strftime('%Y-%m-%d'), self.end_date.strftime('%Y-%m-%d')] + self.tickers
            df = pd.read_sql_query(query, self.conn, params=params, parse_dates=['date'], dtype={'code': str})
            
            if df.empty:
                logger.warning("No market data found in the database for the specified date range and tickers. This may be expected for some periods.")
                # Initialize empty dataframes with correct columns/index to prevent downstream errors
                self.prices = pd.DataFrame(columns=self.tickers, index=pd.date_range(self.start_date, self.end_date))
                self.volumes = pd.DataFrame(columns=self.tickers, index=pd.date_range(self.start_date, self.end_date))
                self.market_caps = pd.DataFrame(columns=self.tickers, index=pd.date_range(self.start_date, self.end_date))
                self.returns = pd.DataFrame(columns=self.tickers, index=pd.date_range(self.start_date, self.end_date))
                return True
                    
            # Pivot data to wide format with tickers as columns
            price_df = df.pivot(index='date', columns='code', values='close')
            volume_df = df.pivot(index='date', columns='code', values='volume')
            market_cap_df = df.pivot(index='date', columns='code', values='market_cap')

            # --- Diagnostic Check for Market Cap ---
            if market_cap_df.isnull().all().all():
                logger.critical("Market cap data is entirely NULL after being loaded from the database. "
                                "This indicates a problem with the data source or the ingestion process. "
                                "The pipeline cannot continue without valid market cap data.")
                # To prevent silent failure downstream, we treat this as a fatal error.
                return False

            # --- Data Cleaning and Validation ---
            full_date_range = pd.date_range(start=self.start_date, end=self.end_date, freq='D')

            # Reindex to ensure all dates and all tickers from the universe are present.
            price_df = price_df.reindex(index=full_date_range, columns=self.tickers)
            volume_df = volume_df.reindex(index=full_date_range, columns=self.tickers)
            market_cap_df = market_cap_df.reindex(index=full_date_range, columns=self.tickers)

            # Apply a robust filling strategy and assign to self.
            self.prices = price_df.ffill().bfill().fillna(0)
            self.volumes = volume_df.ffill().bfill().fillna(0)
            self.market_caps = market_cap_df.ffill().bfill().fillna(0)

            # Calculate returns on the fully cleaned price data
            # Calculate returns, cap extreme losses, and handle NaNs
            returns = self.prices.pct_change()
            # Cap returns at -99.99% to prevent -100% which wipes out portfolio value
            returns.clip(lower=-0.9999, inplace=True)
            # Replace infinite values that can occur from price going from 0 to non-zero
            returns.replace([np.inf, -np.inf], np.nan, inplace=True)
            self.returns = returns.fillna(0)
                    
            logger.info(f"Loaded market data for {len(self.prices.columns)} tickers from {self.start_date.date()} to {self.end_date.date()}")
            logger.info(f"  - Price data shape: {self.prices.shape}")
            logger.info(f"  - Volume data shape: {self.volumes.shape}")
            logger.info(f"  - Market cap data shape: {self.market_caps.shape}")
            
            return True
                    
        except Exception as e:
            logger.error(f"Error loading market data: {e}", exc_info=True)
            return False

    def _load_market_index_data(self) -> bool:
        """
        Fetches historical data for the market index (KOSPI) for the entire period.
        """
        logger.info("Fetching historical KOSPI data for the full period...")
        try:
            # yfinance can be sensitive to date types. Pass as 'YYYY-MM-DD' strings for robustness.
            start_str = self.start_date.strftime('%Y-%m-%d')
            # yfinance end date is exclusive, so add a day before formatting.
            end_str = (self.end_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
            kospi = yf.download(
                '^KS11',
                start=start_str,
                end=end_str,
                progress=False,
                auto_adjust=True
            )
            if kospi.empty:
                logger.error("Failed to download KOSPI data. Momentum factors cannot be calculated.")
                return False
            
            self.market_prices = kospi[['Close']]
            self.market_prices.rename(columns={'Close': 'kospi_close'}, inplace=True)
            logger.info(f"Successfully loaded KOSPI data from {self.start_date.date()} to {self.end_date.date()}.")
            return True

        except Exception as e:
            logger.error(f"An error occurred while fetching KOSPI data: {e}", exc_info=True)
            return False

    def _apply_db_optimizations(self, run_vacuum: bool = False):
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("PRAGMA journal_mode=WAL;"); conn.execute("PRAGMA cache_size = -80000;")
                conn.execute("PRAGMA synchronous = NORMAL;"); conn.execute("PRAGMA temp_store = MEMORY;")
                if run_vacuum: conn.execute("VACUUM;")
        except sqlite3.Error as e: logger.error(f"Failed to apply DB optimizations: {e}")

    def get_listing_info(self) -> pd.DataFrame:
        """Fetches the complete listing information from the local database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Check if the table exists first
                table_exists = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table' AND name='listing_info'", conn)
                if table_exists.empty:
                    logger.error("'listing_info' table not found. Please run the fetcher with --fetch-listings.")
                    return pd.DataFrame()
                
                df = pd.read_sql("SELECT code, name, market, listing_date FROM listing_info", conn, parse_dates=['listing_date'], dtype={'code': str})
                logger.info(f"Successfully loaded {len(df)} records from 'listing_info' table.")
                return df
        except Exception as e:
            logger.error(f"Failed to load listing info from database: {e}", exc_info=True)
            return pd.DataFrame()

    def _get_universe_at_date(self, date_str: str, min_history_days: int = 252) -> pd.DataFrame:
        """
        Helper to get all stocks with a minimum trading history at a given date.
        Returns a DataFrame with the stock 'code' as the index.
        """
        end_date = pd.to_datetime(date_str)

        # 1. Find the most recent actual trading day in the DB on or before the requested date
        actual_date_query = f"SELECT MAX(date) FROM daily_prices WHERE date <= '{end_date.strftime('%Y-%m-%d')}'"
        actual_date_result = pd.read_sql_query(actual_date_query, self.conn)
        if actual_date_result.empty or actual_date_result.iloc[0, 0] is None:
            logger.warning(f"No trading data found on or before {date_str}.")
            return pd.DataFrame()
        actual_date = pd.to_datetime(actual_date_result.iloc[0, 0])
        
        logger.info(f"Building universe based on trading date: {actual_date.date()} (requested: {date_str})")

        # 2. Define lookback window
        lookback_start_date = actual_date - pd.DateOffset(days=min_history_days)
        query_start_date = actual_date - pd.DateOffset(days=min_history_days * 1.5) # Safe buffer
        logger.info(f"Lookback window for history check: {lookback_start_date.date()} to {actual_date.date()}")

        # 3. Find stocks that are tradable on actual_date AND have existed for long enough
        query = f"""
            SELECT T1.code, T1.name, T1.market, T1.first_date
            FROM (
                SELECT code, name, market, MIN(date) as first_date, COUNT(date) as date_count
                FROM daily_prices
                WHERE date BETWEEN '{query_start_date.strftime('%Y-%m-%d')}' AND '{actual_date.strftime('%Y-%m-%d')}'
                GROUP BY code, name, market
            ) AS T1
            JOIN listing_info T2 ON T1.code = T2.code
            WHERE T1.date_count >= {min_history_days}
            AND T2.listing_date <= '{lookback_start_date.strftime('%Y-%m-%d')}'
        """
        df = pd.read_sql_query(query, self.conn, dtype={'code': str})

        if df.empty:
            logger.warning(f"No stocks found on {actual_date.date()} with at least {min_history_days} days of history.")
            return pd.DataFrame()

        logger.info(f"Found {len(df)} stocks with sufficient trading history.")
        
        # 4. Set the stock code as the index for easier merging later
        return df.set_index('code')

    def run_compliance_filters_for_date(self, date: pd.Timestamp) -> Tuple[List[str], Dict[str, str]]:
        """Runs all compliance filters for a specific point in time."""
        end_date = pd.to_datetime(date)
        is_tuning = 'tuner' in sys.modules or 'optuna' in sys.modules

        # 1. Get base universe with sufficient history
        initial_universe_df = self._get_universe_at_date(end_date.strftime('%Y-%m-%d'), min_history_days=252)
        if initial_universe_df.empty:
            return [], {}
        
        logger.info(f"Found {len(initial_universe_df)} stocks with sufficient history.")

        # --- DEBUG LOGGING START ---
        if not initial_universe_df.empty:
            logger.info(f"Debug: Sample of initial universe index: {initial_universe_df.index.tolist()[:5]}")
        if self.sector_map:
            logger.info(f"Debug: Sample of sector_map keys: {list(self.sector_map.keys())[:5]}")
        # --- DEBUG LOGGING END ---

        # 2. Load all necessary data for filtering
        full_listing_info = self.get_listing_info()
        if full_listing_info.empty:
            return [], {}

        # 3. Merge base universe with listing info to get sectors and listing dates
        universe_with_listings = initial_universe_df.merge(
            full_listing_info[['code', 'listing_date']],
            left_index=True,
            right_on='code',
            how='left'
        )

        # 4. Load and apply sector mapping from the pre-loaded full map
        # Map sectors using the index, which contains the cleaned codes from the initial universe query.
        universe_with_listings['sector'] = universe_with_listings['code'].map(self.sector_map)

        # --- DEBUG LOGGING START ---
        non_null_sectors = universe_with_listings['sector'].notna().sum()
        logger.info(f"Debug: Found {non_null_sectors} stocks with a valid sector mapping.")
        logger.info(f"Debug: Universe size before sector dropna: {len(universe_with_listings)}")
        # --- DEBUG LOGGING END ---

        # 5. Filter out stocks with no sector
        initial_universe = universe_with_listings.dropna(subset=['sector'])
        logger.info(f"Debug: Universe size after sector dropna: {len(initial_universe)}")
        logger.info(f"Universe size after sector mapping: {len(initial_universe)} stocks.")

        # 6. Fetch price/volume data for the liquidity filter directly from the database
        lookback_days = 90
        start_dt = (end_date - pd.DateOffset(days=lookback_days)).strftime('%Y-%m-%d')
        end_dt = end_date.strftime('%Y-%m-%d')
        
        ticker_list = initial_universe['code'].tolist()
        if not ticker_list:
            logger.warning(f"Universe for date {end_date.date()} is empty before fetching market data.")
            return [], {}

        with sqlite3.connect(self.db_path) as conn:
            placeholders = ','.join(['?'] * len(ticker_list))
            query = f"SELECT date, code, close, volume FROM daily_prices WHERE date BETWEEN ? AND ? AND code IN ({placeholders})"
            params = [start_dt, end_dt] + ticker_list
            price_volume_df = pd.read_sql(query, conn, params=params, parse_dates=['date'], dtype={'code': str})

        if price_volume_df.empty:
            logger.warning(f"No market data found in DB for the filtered universe for date {end_date.date()}.")
            return [], {}

        prices = price_volume_df.pivot(index='date', columns='code', values='close')
        volumes = price_volume_df.pivot(index='date', columns='code', values='volume')

        # 7. Apply all compliance filters
        compliance_filter = ComplianceFilter(skip_rules=is_tuning)
        filter_results = compliance_filter.apply_all_filters(
            prices=prices,
            volumes=volumes,
            listings_info=initial_universe,  # Pass the merged dataframe
            end_date=end_date
        )

        # 8. Exclude filtered tickers and finalize the universe
        forbidden_tickers = filter_results['all']
        final_tickers = sorted([t for t in initial_universe['code'].tolist() if t not in forbidden_tickers])
        final_sector_map = initial_universe[initial_universe['code'].isin(final_tickers)].set_index('code')['sector'].to_dict()

        logger.info(f"Final compliant universe for {end_date.date()} contains {len(final_tickers)} tickers.")
        return final_tickers, final_sector_map

    def _log_data_quality(self) -> None:
        """Log data quality metrics for debugging and monitoring."""
        logger.info("=== Data Quality Report ===")
        logger.info(f"Total tickers with complete data: {len(self.tickers)}")
        
        # Market data stats
        if not self.prices.empty:
            missing_prices = self.prices.isna().sum().sum()
            total_prices = self.prices.size
            logger.info(f"Price data: {missing_prices/total_prices*100:.2f}% missing values")
            
        # Sector distribution
        if self.sector_map:
            sector_dist = pd.Series(self.sector_map).value_counts()
            logger.info(f"Sector distribution:\n{sector_dist}")
            
        # Fundamental data coverage
        if not self.latest_fundamentals.empty:
            logger.info(f"Latest fundamentals coverage: {len(self.latest_fundamentals)}/{len(self.tickers)} tickers")

    def _load_full_macro_data(self) -> pd.DataFrame:
        """Load macroeconomic data from the database."""
        self._macro.fetch_and_store(self.cfg.end_date)
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql("SELECT * FROM macro_data", conn, index_col='date', parse_dates=['date'])

    def get_sector_mappings(self) -> Dict[str, str]:
        """Get the mapping of tickers to their sectors."""
        return self.sector_map