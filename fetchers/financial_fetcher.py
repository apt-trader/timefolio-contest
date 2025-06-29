# fetchers/financial_fetcher.py
import os, sys, ssl, time, random, argparse, logging, sqlite3
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import OpenDartReader
from dotenv import load_dotenv
from tqdm import tqdm
import signal
from .corp_code_cache import CorpCodeCache

load_dotenv()

logs_dir = Path(__file__).parent.parent / 'logs'
logs_dir.mkdir(exist_ok=True, parents=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(logs_dir / 'financial_fetcher.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FinancialsFetcher:
    def __init__(self, api_key: Optional[str], db_path: str, db_only_mode: bool = False):
        self.db_only_mode = db_only_mode
        self.dart = None
        if not self.db_only_mode:
            if api_key:
                try:
                    self.dart = OpenDartReader(api_key)
                except ValueError as e:
                    logger.error(f"Failed to initialize OpenDartReader: {e}")
                    # This can happen if the API key is invalid or quota is exceeded on init
                    self.dart = None
            else:
                logger.warning("DART API key not provided. Fetcher will not be able to make live API calls.")

        self.db_path = db_path
        self.corp_code_cache = CorpCodeCache()
        self._init_db_structure()
        self.ticker_to_corp_code_map = self._load_company_mapping()
        logger.info(f"FinancialsFetcher initialized for DB at {db_path}.")

    def _create_connection(self) -> sqlite3.Connection:
        try:
            conn = sqlite3.connect(self.db_path, timeout=10, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous = NORMAL;")
            return conn
        except sqlite3.Error as e:
            logger.critical(f"FATAL: DB connection to {self.db_path} failed. Error: {e}")
            raise

    def _init_db_structure(self):
        # This schema is the single source of truth.
        schema = """CREATE TABLE financials (
            corp_code TEXT, ticker TEXT, report_date TEXT, report_code TEXT, account_nm TEXT, 
            account_id TEXT, account_detail TEXT, thstrm_nm TEXT, thstrm_amount TEXT, 
            frmtrm_nm TEXT, frmtrm_amount TEXT, bfefrmtrm_nm TEXT, bfefrmtrm_amount TEXT, 
            PRIMARY KEY (corp_code, report_date, report_code, account_id)
        );"""
        try:
            with self._create_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='financials'")
                table_exists = cursor.fetchone()

                if not table_exists:
                    # Fresh setup, just create the table.
                    cursor.execute(schema)
                else:
                    # Table exists, check if migration is needed.
                    cursor.execute("PRAGMA table_info(financials)")
                    columns = [info[1] for info in cursor.fetchall()]
                    if 'report_date' not in columns:
                        logger.info("Legacy 'financials' schema detected. Starting robust migration...")
                        # 1. Rename old table
                        cursor.execute("ALTER TABLE financials RENAME TO financials_old")
                        # 2. Create new table with the correct schema
                        cursor.execute(schema)
                        # 3. Copy data, mapping old columns to new ones
                        cursor.execute("PRAGMA table_info(financials_old)")
                        old_cols = [info[1] for info in cursor.fetchall()]
                        
                        new_cols_list = ['corp_code', 'ticker', 'report_date', 'report_code', 'account_nm', 'account_id', 'account_detail', 'thstrm_nm', 'thstrm_amount', 'frmtrm_nm', 'frmtrm_amount', 'bfefrmtrm_nm', 'bfefrmtrm_amount']
                        old_cols_select = []
                        for col in new_cols_list:
                            if col == 'report_date': old_cols_select.append('rcept_dt' if 'rcept_dt' in old_cols else 'NULL')
                            elif col == 'ticker': old_cols_select.append('ticker' if 'ticker' in old_cols else 'NULL')
                            else: old_cols_select.append(f'"{col}"' if col in old_cols else 'NULL')
                        
                        insert_sql = f"INSERT INTO financials ({', '.join(new_cols_list)}) SELECT {', '.join(old_cols_select)} FROM financials_old;"
                        logger.debug(f"Executing data migration: {insert_sql}")
                        cursor.execute(insert_sql)
                        # 4. Drop old table
                        cursor.execute("DROP TABLE financials_old")
                        logger.info("Schema migration completed successfully.")

                # Always ensure indices exist on the (now correct) table.
                conn.execute("CREATE INDEX IF NOT EXISTS idx_financials_ticker_date ON financials(ticker, report_date);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_financials_corp_code_date ON financials(corp_code, report_date);")
                conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Failed to initialize or migrate 'financials' table: {e}", exc_info=True)
            raise

    def _load_company_mapping(self) -> Dict[str, str]:
        logger.info("Loading company ticker-to-corp_code mapping from database...")
        try:
            with self._create_connection() as conn:
                if pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table' AND name='company_mapping'", conn).empty:
                    logger.warning("'company_mapping' table not found. Ticker lookups will rely solely on API calls.")
                    return {}
                # Corrected query to use the actual column name '종목코드' and alias it to 'code'.
                df = pd.read_sql_query("SELECT \"종목코드\" as code, corp_code FROM company_mapping", conn)
                df['code'] = df['code'].astype(str).str.zfill(6)
                mapping = df.set_index('code')['corp_code'].to_dict()
                logger.info(f"Successfully loaded {len(mapping)} company mappings.")
                return mapping
        except Exception as e:
            logger.error(f"Failed to load company mapping from DB: {e}. Proceeding without it.", exc_info=True)
            return {}

    def find_corp_code(self, identifier: str) -> Optional[str]:
        if not identifier: return None
        if identifier in self.ticker_to_corp_code_map: return self.ticker_to_corp_code_map[identifier]
        cached_code = self.corp_code_cache.get(identifier)
        if cached_code is not None: return cached_code
        if not self.dart:
            logger.warning(f"DART API not configured. Cannot fetch corp_code for new ticker '{identifier}'.")
            return None
        logger.debug(f"Corp code for '{identifier}' not in local map or cache. Falling back to API.")
        def handler(signum, frame): raise TimeoutError(f"API call for {identifier} timed out")
        signal.signal(signal.SIGALRM, handler)
        signal.alarm(10)
        try:
            time.sleep(random.uniform(0.5, 1.0))
            corp_code = self.dart.find_corp_code(identifier)
            self.corp_code_cache.set(identifier, corp_code)
            return corp_code
        except Exception as e:
            logger.error(f"Failed to find corp_code for '{identifier}' via API: {e}")
            self.corp_code_cache.set(identifier, None)
            return None
        finally:
            signal.alarm(0)

    def get_historical_financials_from_db(self, tickers: List[str]) -> pd.DataFrame:
        if not tickers: return pd.DataFrame()
        logger.info(f"Querying all historical financials for {len(tickers)} tickers from DB.")
        # Query directly by ticker for simplicity and robustness.
        placeholders = ','.join(['?'] * len(tickers))
        query = f"SELECT * FROM financials WHERE ticker IN ({placeholders})"
        try:
            with self._create_connection() as conn:
                df = pd.read_sql_query(query, conn, params=tickers)


            if df.empty:
                logger.warning("No financial data found in DB for the requested corp_codes.")
                return pd.DataFrame()

            # --- Data Cleaning and Type Conversion ---
            # 1. Strip whitespace from ticker column to ensure consistency.
            df['ticker'] = df['ticker'].astype(str).str.strip()
            
            # 2. Robustly convert 'report_date' from Unix timestamp (stored as text) to datetime.
            df['report_date'] = pd.to_datetime(pd.to_numeric(df['report_date'], errors='coerce'), unit='s')
            
            # 3. Handle None values in report_code (replace with a default value)
            df['report_code'] = df['report_code'].fillna('11013')  # Default to consolidated statement code
            
            # 4. Drop any rows where the date conversion or other critical data for pivot is missing.
            df.dropna(subset=['report_date', 'ticker', 'account_nm', 'thstrm_amount'], inplace=True)

            # --- Final Data Preparation for Pivot ---
            logger.info(f"[DEBUG] Pre-conversion DataFrame shape: {df.shape}")
            logger.info(f"[DEBUG] Sample thstrm_amount values: {df['thstrm_amount'].head(10).tolist()}")
            
            # 1. Convert the value column to a numeric type BEFORE pivoting. This is the critical fix.
            df['thstrm_amount'] = pd.to_numeric(df['thstrm_amount'].astype(str).str.replace(',', ''), errors='coerce')
            logger.info(f"[DEBUG] After numeric conversion, non-null count: {df['thstrm_amount'].notna().sum()}")
            logger.info(f"[DEBUG] After numeric conversion, null count: {df['thstrm_amount'].isna().sum()}")
            
            df.dropna(subset=['thstrm_amount'], inplace=True) # Drop rows where conversion failed
            logger.info(f"[DEBUG] After dropna, DataFrame shape: {df.shape}")
            
            if df.empty:
                logger.error("[DEBUG] DataFrame is empty after numeric conversion - all values failed to convert!")
                return pd.DataFrame()
            
            # 2. Check for and handle duplicates before pivot
            logger.info(f"[DEBUG] Unique account_nm values: {df['account_nm'].nunique()}")
            logger.info(f"[DEBUG] Sample account_nm values: {df['account_nm'].unique()[:10].tolist()}")
            
            # Check for duplicates that would cause pivot to fail
            duplicates = df.duplicated(subset=['corp_code', 'ticker', 'report_date', 'report_code', 'account_nm'], keep=False)
            if duplicates.any():
                logger.warning(f"[DEBUG] Found {duplicates.sum()} duplicate rows, removing duplicates...")
                df = df.drop_duplicates(subset=['corp_code', 'ticker', 'report_date', 'report_code', 'account_nm'], keep='first')
                logger.info(f"[DEBUG] After removing duplicates, shape: {df.shape}")
            
            # 3. Analyze data before pivot
            logger.info(f"[DEBUG] Unique combinations of index cols: {df[['corp_code', 'ticker', 'report_date', 'report_code']].drop_duplicates().shape[0]}")
            logger.info(f"[DEBUG] Sample data before pivot:")
            logger.info(f"[DEBUG] {df[['corp_code', 'ticker', 'report_date', 'report_code', 'account_nm', 'thstrm_amount']].head(10)}")
            
            # 4. Perform the pivot operation.
            try:
                pivot_df = df.pivot_table(index=['corp_code', 'ticker', 'report_date', 'report_code'], columns='account_nm', values='thstrm_amount', aggfunc='first').reset_index()
                pivot_df.columns.name = None
                logger.info(f"[DEBUG] Post-pivot DataFrame shape: {pivot_df.shape}")
                
                if len(pivot_df) == 0:
                    logger.error("[DEBUG] Pivot returned empty DataFrame despite valid input data!")
                    logger.info(f"[DEBUG] Trying alternative pivot without reset_index...")
                    alt_pivot = df.pivot_table(index=['corp_code', 'ticker', 'report_date', 'report_code'], columns='account_nm', values='thstrm_amount', aggfunc='first')
                    logger.info(f"[DEBUG] Alternative pivot shape: {alt_pivot.shape}")
                    logger.info(f"[DEBUG] Alternative pivot index: {alt_pivot.index[:5] if len(alt_pivot) > 0 else 'Empty'}")
                    
            except Exception as pivot_error:
                logger.error(f"[DEBUG] Pivot operation failed: {pivot_error}")
                logger.info(f"[DEBUG] Sample of data that failed to pivot: {df.head()}")
                return pd.DataFrame()
            
            # 5. Map Korean account names to English column names for factor engine compatibility
            logger.info(f"[DEBUG] Before mapping, sample columns: {list(pivot_df.columns)[:10]}")
            pivot_df = self._map_korean_to_english_columns(pivot_df)
            logger.info(f"[DEBUG] After column mapping, DataFrame shape: {pivot_df.shape}")
            if not pivot_df.empty:
                english_cols = [col for col in pivot_df.columns if col not in ['corp_code', 'ticker', 'report_date', 'report_code']]
                logger.info(f"[DEBUG] Available English columns: {english_cols[:10]}...")
                logger.info(f"[DEBUG] Total English columns found: {len(english_cols)}")
                # Check for key columns needed by factor engine
                key_cols = ['total_assets', 'total_equity', 'revenue', 'net_income', 'operating_cash_flow']
                found_key_cols = [col for col in key_cols if col in pivot_df.columns]
                logger.info(f"[DEBUG] Key factor engine columns found: {found_key_cols}")
            else:
                logger.error(f"[DEBUG] pivot_df is empty after column mapping!")
            logger.info(f"Successfully retrieved and processed {len(pivot_df)} historical financial reports.")
            return pivot_df
        except Exception as e:
            logger.error(f"Error fetching historical financials from DB: {e}", exc_info=True)
            return pd.DataFrame()

    def _map_korean_to_english_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Maps Korean financial statement account names to English column names expected by the factor engine."""
        # Mapping dictionary for Korean account names to English equivalents
        korean_to_english = {
            # Balance Sheet Items
            '자산총계': 'total_assets',
            '유동자산': 'current_assets', 
            '비유동자산': 'non_current_assets',
            '자본총계': 'total_equity',
            '부채총계': 'total_liabilities',
            '유동부채': 'current_liabilities',
            '비유동부채': 'non_current_liabilities',
            '자본금': 'capital_stock',
            '이익잉여금': 'retained_earnings',
            '유형자산': 'tangible_assets',
            '무형자산': 'intangible_assets',
            '현금및현금성자산': 'cash_and_equivalents',
            
            # Income Statement Items
            '매출액': 'revenue',
            '당기순이익': 'net_income',
            '영업이익': 'operating_income',
            '매출총이익': 'gross_profit',
            '영업활동현금흐름': 'operating_cash_flow',
            '법인세차감전순이익': 'pre_tax_income',
            '법인세비용차감전순이익': 'pre_tax_income',
            
            # Cash Flow Statement Items
            '영업활동으로 인한 현금흐름': 'operating_cash_flow',
            '투자활동으로 인한 현금흐름': 'investing_cash_flow',
            '투자활동현금흐름': 'investing_cash_flow',
            '2. 투자활동으로 인한 현금유출액': 'capex',
            '재무활동현금흐름': 'financing_cash_flow',
            '재무활동으로 인한 현금흐름': 'financing_cash_flow',
            
            # Alternative names that might appear
            '수익': 'revenue',
            '총자산': 'total_assets',
            '순이익': 'net_income',
            '총매출': 'revenue',
            '당기순이익(손실)': 'net_income',
            '영업에서 창출된 현금흐름': 'operating_cash_flow',
            '영업으로부터 창출된 현금흐름': 'operating_cash_flow'
        }
        
        # Rename columns using the mapping
        df_mapped = df.rename(columns=korean_to_english)
        
        # Handle duplicate columns by consolidating them
        # Find columns that appear multiple times
        duplicate_cols = df_mapped.columns[df_mapped.columns.duplicated()].unique()
        
        if len(duplicate_cols) > 0:
            logger.info(f"[DEBUG] Found duplicate columns after mapping: {list(duplicate_cols)}")
            
            # For each duplicate column, consolidate by taking the first non-null value
            for col in duplicate_cols:
                # Get all columns with this name
                dup_data = df_mapped[col]
                if isinstance(dup_data, pd.DataFrame):
                    # Combine columns by taking first non-null value across columns
                    consolidated = dup_data.fillna(method='ffill', axis=1).iloc[:, -1]
                    # Drop all duplicate columns
                    df_mapped = df_mapped.drop(columns=[col])
                    # Add back the consolidated column
                    df_mapped[col] = consolidated
        
        # Log the mapping results
        mapped_cols = [col for col in korean_to_english.values() if col in df_mapped.columns]
        if mapped_cols:
            logger.info(f"[DEBUG] Successfully mapped columns: {mapped_cols}")
        else:
            logger.warning(f"[DEBUG] No Korean columns were mapped to English. Available columns: {list(df.columns)}")
        
        # Calculate derived financial ratios needed by factor engine
        self._calculate_financial_ratios(df_mapped)
        
        return df_mapped
    
    def _calculate_financial_ratios(self, df: pd.DataFrame) -> None:
        """Calculate financial ratios like ROE and debt-to-equity that are needed by factor engine."""
        try:
            # Ensure we have a clean index to avoid alignment issues
            df.reset_index(drop=True, inplace=True)
            
            # Calculate ROE = Net Income / Total Equity
            if 'net_income' in df.columns and 'total_equity' in df.columns:
                # Ensure columns are Series, not DataFrame (in case of duplicates)
                net_income_col = df['net_income']
                total_equity_col = df['total_equity']
                
                if isinstance(net_income_col, pd.DataFrame):
                    net_income_col = net_income_col.iloc[:, 0]  # Take first column
                if isinstance(total_equity_col, pd.DataFrame):
                    total_equity_col = total_equity_col.iloc[:, 0]  # Take first column
                
                # Convert to numeric and handle zeros/nulls
                net_income = pd.to_numeric(net_income_col, errors='coerce').fillna(0)
                total_equity = pd.to_numeric(total_equity_col, errors='coerce')
                total_equity = total_equity.replace(0, np.nan)
                
                # Calculate ROE directly without pandas alignment
                df['roe'] = net_income / total_equity
                logger.info(f"[DEBUG] Calculated ROE for {df['roe'].notna().sum()} records")
            
            # Calculate Debt-to-Equity Ratio = Total Liabilities / Total Equity  
            if 'total_liabilities' in df.columns and 'total_equity' in df.columns:
                # Ensure columns are Series, not DataFrame
                total_liabilities_col = df['total_liabilities']
                total_equity_col = df['total_equity']
                
                if isinstance(total_liabilities_col, pd.DataFrame):
                    total_liabilities_col = total_liabilities_col.iloc[:, 0]
                if isinstance(total_equity_col, pd.DataFrame):
                    total_equity_col = total_equity_col.iloc[:, 0]
                
                total_liabilities = pd.to_numeric(total_liabilities_col, errors='coerce').fillna(0)
                total_equity = pd.to_numeric(total_equity_col, errors='coerce')
                total_equity = total_equity.replace(0, np.nan)
                
                df['debt_to_equity'] = total_liabilities / total_equity
                logger.info(f"[DEBUG] Calculated debt_to_equity for {df['debt_to_equity'].notna().sum()} records")
            
            # Set CAPEX as absolute value of investing cash outflows (if negative, make positive)
            if 'capex' in df.columns:
                capex_col = df['capex']
                if isinstance(capex_col, pd.DataFrame):
                    capex_col = capex_col.iloc[:, 0]
                    
                capex = pd.to_numeric(capex_col, errors='coerce')
                df['capex'] = np.abs(capex)
                df.loc[df['capex'] == 0, 'capex'] = np.nan  # Set zero values to NaN
                logger.info(f"[DEBUG] Processed CAPEX for {df['capex'].notna().sum()} records")
                
        except Exception as e:
            logger.warning(f"[DEBUG] Error calculating financial ratios: {e}")
            import traceback
            traceback.print_exc()

    def fetch_and_save_financial_reports_for_year(self, year: int, report_type: str) -> int:
        """
        Fetches financial reports for all listed companies for a given year and report type,
        respecting API rate limits.
{{ ... }}
        Args:
            year (int): The business year.
            report_type (str): The report code (e.g., '11011' for Annual).

        Returns:
            int: The number of successfully fetched and saved reports.
        """
        if self.db_only_mode or not self.dart:
            logger.error("Fetcher is in DB-only mode or DART API is not configured. Cannot fetch new financials.")
            return 0

        corp_codes = self._get_all_corp_codes()
        if not corp_codes:
            logger.warning("No corporation codes found to fetch data for.")
            return 0

        logger.info(f"Starting financial report fetch for {len(corp_codes)} companies for {year} ({report_type})...")
        success_count = 0
        total_count = len(corp_codes)
        
        # Rate limiting parameters
        MAX_CALLS_PER_MINUTE = 900  # Safety margin below 1000
        call_count = 0
        start_time = time.time()

        for i, (corp_code, stock_code) in enumerate(corp_codes):
            try:
                # Check if data already exists
                if self._check_if_data_exists(stock_code, year, report_type):
                    logger.debug(f"[{i+1}/{total_count}] Skipping {stock_code} for {year} ({report_type}): Data already exists.")
                    # This doesn't count as an API call, but we count it as a success if it exists
                    success_count += 1
                    continue

                # Check rate limit before making a call
                if call_count >= MAX_CALLS_PER_MINUTE:
                    elapsed_time = time.time() - start_time
                    if elapsed_time < 60:
                        sleep_time = 60 - elapsed_time
                        logger.info(f"Rate limit reached. Pausing for {sleep_time:.2f} seconds...")
                        time.sleep(sleep_time)
                    # Reset counter and timer
                    call_count = 0
                    start_time = time.time()

                logger.debug(f"[{i+1}/{total_count}] Fetching report for {corp_code} ({stock_code})...")
                df = self.dart.finstate(corp_code, year, report_type)
                call_count += 1 # Increment after the API call

                if df is not None and not df.empty:
                    self._save_report(df, stock_code, year, report_type)
                    success_count += 1
                    logger.debug(f"Successfully saved report for {stock_code}")
                else:
                    logger.warning(f"No financial data returned for {corp_code} ({stock_code}) for the period.")

            except Exception as e:
                # DART API may return error for companies with no data for the requested period
                logger.error(f"Failed to fetch or save report for {corp_code} ({stock_code}): {e}")

        logger.info(f"Completed fetch for {year} ({report_type}). Successfully processed {success_count}/{total_count} reports.")
        return success_count

    def _fetch_and_store_for_ticker(self, ticker: str, year: int, report_code: str) -> bool:
        corp_code = self.find_corp_code(ticker)
        if not corp_code:
            logger.warning(f"Skipping {ticker} for {year} {report_code}: could not find corp_code.")
            return False
        if not self.dart:
            logger.error("DART API not configured, cannot fetch new data.")
            return False
        try:
            logger.debug(f"Fetching {year} {report_code} for {ticker} (corp_code: {corp_code})")
            fs_df = self.dart.finstate(corp_code, year, reprt_code=report_code)
            if fs_df is None or fs_df.empty:
                logger.info(f"No data returned from API for {ticker} ({corp_code}) for {year} {report_code}.")
                return False
            fs_df['ticker'] = str(ticker).zfill(6)
            self.store_financials_df(fs_df, year=year, report_code=report_code)
            return True
        except Exception as e:
            logger.error(f"Failed to fetch/store for {ticker} ({corp_code}) {year}: {e}")
            return False

    def store_financials_df(self, df: pd.DataFrame, year: int, report_code: str):
        if df.empty: return
        # Standardize column names before storing
        # Check for a date column and standardize it. If none exists, we can't store it.
        if 'rcept_dt' in df.columns:
            df = df.rename(columns={'rcept_dt': 'report_date'})
        # If date is missing, create a proxy date to avoid data loss.
        if 'report_date' not in df.columns:
            ticker_info = df['ticker'].iloc[0] if 'ticker' in df.columns and not df.empty else 'N/A'
            logger.warning(f"DataFrame is missing 'rcept_dt'. Using year-end as a proxy report_date. Ticker: {ticker_info}")
            if report_code == '11011':  # Annual report
                df['report_date'] = pd.to_datetime(f'{year}-12-31')
            else:
                logger.error(f"Cannot determine proxy date for report_code: {report_code}. Skipping {ticker_info}.")
                return

        required_cols = {'corp_code', 'report_date', 'reprt_code', 'account_nm'}
        if not required_cols.issubset(df.columns):
            logger.error(f"Dataframe is missing one of required columns for storage: {required_cols - set(df.columns)}")
            return
        try:
            with self._create_connection() as conn:
                # Use a temporary table for robust, safe insertion
                df.to_sql('financials_temp', conn, if_exists='replace', index=False)
                
                # Get columns from the destination table (financials)
                main_table_cols = {col[1] for col in conn.execute("PRAGMA table_info(financials)").fetchall()}
                # Get columns from the source temp table
                temp_table_cols = {col[1] for col in conn.execute("PRAGMA table_info(financials_temp)").fetchall()}

                # Find the common columns to prevent schema mismatch errors
                common_cols = list(main_table_cols.intersection(temp_table_cols))
                cols_str = ', '.join(f'"{c}"' for c in common_cols)
                
                if not cols_str:
                    logger.error("No common columns found between source and destination tables. Aborting insert.")
                    return

                # Insert only the common columns, replacing existing rows on conflict.
                insert_sql = f"""INSERT OR REPLACE INTO financials ({cols_str}) 
                                 SELECT {cols_str} FROM financials_temp;"""

                cursor = conn.cursor()
                cursor.execute(insert_sql)
                cursor.execute("DROP TABLE financials_temp;")
                conn.commit()
            logger.info(f"Successfully stored/updated {len(df)} financial records.")
        except Exception as e:
            logger.error(f"Error storing financials to DB: {e}", exc_info=True)

    def fetch_financials_for_tickers(self, tickers: List[str], start_year: int, end_year: int):
        if not self.dart:
            logger.error("Cannot fetch financials: DART API key not set.")
            return
        logger.info(f"Starting financial data fetch for {len(tickers)} tickers from {start_year} to {end_year}.")
        report_codes = {'11013': 'Q1', '11012': 'Q2', '11014': 'Q3', '11011': 'Annual'}
        for ticker in tqdm(tickers, desc="Fetching Financials"):
            for year in range(start_year, end_year + 1):
                for code, name in report_codes.items():
                    self._fetch_and_store_for_ticker(ticker, year, code)
                    time.sleep(random.uniform(0.8, 1.2))
