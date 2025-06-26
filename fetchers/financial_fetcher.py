# fetchers/financial_fetcher.py
import os
import sys
import ssl
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
from collections import deque

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm
from dart_fss import set_api_key, get_corp_list, fs

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import certifi
import sqlite3
import logging
import argparse
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta
from collections import deque, defaultdict

import urllib3
import requests
import OpenDartReader
import yaml
from dotenv import load_dotenv
import numpy as np
from tqdm import tqdm

load_dotenv()
# Configure logging
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

# Configure SSL context for better certificate handling
ssl_context = ssl.create_default_context(cafile=certifi.where())
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configure requests session with retry strategy
session = requests.Session()
retry = requests.adapters.Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[500, 502, 503, 504]
)
adapter = requests.adapters.HTTPAdapter(max_retries=retry)
session.mount('http://', adapter)
session.mount('https://', adapter)

try:
    config_path = Path(__file__).parent.parent / "config/config.yaml"
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    universe_file = Path(__file__).parent.parent / cfg['data_settings']['stock_universe_file']
except FileNotFoundError:
    logger.error(f"Config file not found at {config_path}.")
    universe_file = None

class RateLimiter:
    def __init__(self, max_per_minute: int = 900, max_per_day: int = 9500):
        self.max_per_minute, self.max_per_day = max_per_minute, max_per_day
        self.calls_minute, self.calls_day = deque(), deque()
    def wait(self):
        now = time.time()
        while self.calls_minute and self.calls_minute[0] < now - 60: self.calls_minute.popleft()
        while self.calls_day and self.calls_day[0] < now - 86400: self.calls_day.popleft()
        if len(self.calls_day) >= self.max_per_day: raise RuntimeError(f"Daily API limit ({self.max_per_day}) reached.")
        if len(self.calls_minute) >= self.max_per_minute:
            time.sleep(60.1 - (now - self.calls_minute[0]))
        self.calls_minute.append(time.time()); self.calls_day.append(time.time())

class FinancialsFetcher:
    def __init__(self, api_key: str, db_path: str = str(Path(__file__).parent.parent / "db" / "krx_data.db"), verify_ssl: bool = True):
        if not api_key:
            raise ValueError("DART API key is required.")
            
        self.api_key = api_key
        self.verify_ssl = verify_ssl
        
        # Configure cache directory
        self.cache_dir = Path(__file__).parent / 'docs_cache'
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize OpenDartReader
        try:
            # Create a requests session with retry strategy
            session = requests.Session()
            retry_strategy = requests.adapters.Retry(
                total=3,
                backoff_factor=1,
                status_forcelist=[500, 502, 503, 504]
            )
            adapter = requests.adapters.HTTPAdapter(max_retries=retry_strategy)
            session.mount('http://', adapter)
            session.mount('https://', adapter)
            
            # Configure SSL verification
            if not self.verify_ssl:
                session.verify = False
                logger.warning("SSL verification is disabled. This is not recommended for production use.")
            
            # Initialize OpenDartReader with the configured session
            self.dart = OpenDartReader(api_key)
            # Set the session directly on the client's session attribute if available
            if hasattr(self.dart, 'session'):
                self.dart.session = session
                
            logger.info("Successfully initialized OpenDartReader")
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenDartReader: {e}")
            raise
        
        # Database setup
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.execute('PRAGMA journal_mode=WAL')  # Better concurrency
        self.corp_code_cache: Dict[str, Optional[str]] = {}
        self._init_financials_table()
        self.existing_columns = self._get_existing_columns()
        self.rate_limiter = RateLimiter()
        
        logger.info(f"FinancialFetcher initialized with SSL verification {'enabled' if verify_ssl else 'disabled'}")

    def _init_financials_table(self):
        try:
            with self.conn:
                self.conn.execute("""
                    CREATE TABLE IF NOT EXISTS financials (
                        rcept_no TEXT, reprt_code TEXT, bsns_year TEXT, corp_code TEXT, sj_div TEXT, sj_nm TEXT, fs_div TEXT, fs_nm TEXT, fs_sn TEXT,
                        account_id TEXT, account_nm TEXT, account_detail TEXT, thstrm_nm TEXT, thstrm_amount REAL, thstrm_add_amount REAL, thstrm_q_nm TEXT, 
                        thstrm_q_amount REAL, frmtrm_nm TEXT, frmtrm_amount REAL, frmtrm_add_amount REAL, frmtrm_q_nm TEXT, frmtrm_q_amount REAL,
                        bfefrmtrm_nm TEXT, bfefrmtrm_amount REAL, ord INTEGER, currency TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, PRIMARY KEY (rcept_no, account_id, fs_div)
                    )""")
                self.conn.execute("CREATE INDEX IF NOT EXISTS idx_financials_corp_code_year ON financials(corp_code, bsns_year)")
        except sqlite3.Error as e: logger.critical(f"FATAL: DB schema init failed. Error: {e}"); raise

    def _get_existing_columns(self) -> list:
        return [row[1] for row in self.conn.execute("PRAGMA table_info(financials)").fetchall()]

    def find_corp_code(self, identifier: str) -> Optional[str]:
        if identifier in self.corp_code_cache: return self.corp_code_cache[identifier]
        try:
            self.rate_limiter.wait(); corp_code = self.dart.find_corp_code(identifier)
            self.corp_code_cache[identifier] = corp_code; return corp_code
        except Exception: self.corp_code_cache[identifier] = None; return None

    def fetch_financials(self, corp_code: str, year: int, report_type: str = '11011', max_retries: int = 3):
        """Fetch financial data with retry logic and improved error handling.
        
        Args:
            corp_code: Company code
            year: Fiscal year
            report_type: Report type code (default: '11011' for annual report)
            max_retries: Maximum number of retry attempts (default: 3)
            
        Returns:
            bool: True if data was successfully fetched and stored, False otherwise
        """
        # Skip if data already exists
        if self.conn.execute(
            "SELECT 1 FROM financials WHERE corp_code = ? AND bsns_year = ? AND reprt_code = ? LIMIT 1",
            (corp_code, str(year), report_type)
        ).fetchone():
            logger.debug(f"Data already exists for {corp_code} {year} (report: {report_type})")
            return True

        last_exception = None
        for attempt in range(max_retries):
            try:
                # Try with CFS (Consolidated) first
                self.rate_limiter.wait()
                logger.debug(f"Attempt {attempt + 1}/{max_retries}: Fetching CFS data for {corp_code} {year}")
                
                try:
                    # Try with SSL verification first
                    df = self.dart.finstate_all(corp_code, year, report_type, fs_div='CFS')
                    logger.debug(f"CFS data retrieved for {corp_code} {year}")
                except Exception as cfs_error:
                    logger.debug(f"CFS fetch failed for {corp_code} {year}, trying OFS. Error: {cfs_error}")
                    df = None
                
                # If CFS failed or empty, try OFS (Separate)
                if not isinstance(df, pd.DataFrame) or df.empty:
                    self.rate_limiter.wait()
                    logger.debug(f"Trying OFS data for {corp_code} {year}")
                    df = self.dart.finstate_all(corp_code, year, report_type, fs_div='OFS')
                
                # If we got valid data, store it and return
                if isinstance(df, pd.DataFrame) and not df.empty:
                    logger.info(f"Successfully retrieved financial data for {corp_code} {year}")
                    self._store_financials(df)
                    return True
                    
                logger.warning(f"No data returned for {corp_code} {year} (attempt {attempt + 1}/{max_retries})")
                    
            except requests.exceptions.SSLError as ssl_err:
                last_exception = ssl_err
                logger.warning(f"SSL Error (attempt {attempt + 1}/{max_retries}) for {corp_code} {year}: {ssl_err}")
                if attempt == max_retries - 1:  # Last attempt
                    logger.error(f"Max retries reached for {corp_code} {year} due to SSL errors. Giving up.")
                time.sleep(2 ** attempt)  # Exponential backoff
                
            except requests.exceptions.RequestException as req_err:
                last_exception = req_err
                logger.error(f"Request error for {corp_code} {year} (attempt {attempt + 1}/{max_retries}): {req_err}")
                if attempt == max_retries - 1:
                    logger.error(f"Max retries reached for {corp_code} {year} due to request errors.")
                time.sleep(2 ** attempt)  # Exponential backoff
                
            except Exception as e:
                last_exception = e
                logger.error(f"Unexpected error processing {corp_code} {year} (attempt {attempt + 1}/{max_retries}): {e}", 
                            exc_info=True)
                if attempt == max_retries - 1:
                    logger.error(f"Max retries reached for {corp_code} {year} due to unexpected errors.")
                time.sleep(2 ** attempt)  # Exponential backoff
        
        # If we get here, all attempts failed
        logger.error(f"Failed to fetch data for {corp_code} {year} after {max_retries} attempts. "
                    f"Last error: {str(last_exception)[:200]}")
        return False

    def _store_financials(self, df: pd.DataFrame) -> bool:
        """Process and store financial data in the database.
        
        Args:
            df: DataFrame containing financial data to store
            
        Returns:
            bool: True if storage was successful, False otherwise
        """
        if not isinstance(df, pd.DataFrame) or df.empty:
            logger.warning("No data to store: Empty or invalid DataFrame provided")
            return False
            
        try:
            # Validate and prepare data
            valid_cols = [c for c in df.columns if c in self.existing_columns]
            if not valid_cols:
                logger.error("No valid columns found in the DataFrame to store")
                return False
                
            df_to_store = df[valid_cols].copy()
            
            # Convert numeric columns
            numeric_cols = ['thstrm_amount', 'frmtrm_amount', 'bfefrmtrm_amount', 'ord']
            for col in numeric_cols:
                if col in df_to_store:
                    df_to_store[col] = pd.to_numeric(df_to_store[col], errors='coerce')
            
            # Define primary key and update columns
            pk_cols = ['rcept_no', 'account_id', 'fs_div']
            missing_pk_cols = [col for col in pk_cols if col not in df_to_store.columns]
            if missing_pk_cols:
                logger.error(f"Missing required primary key columns: {missing_pk_cols}")
                return False
                
            update_cols = [f'"{c}" = excluded."{c}"' 
                         for c in df_to_store.columns 
                         if c not in pk_cols]
            
            # Prepare SQL query
            columns_str = ','.join(f'"{c}"' for c in df_to_store.columns)
            placeholders = ','.join(['?'] * len(df_to_store.columns))
            updates_str = ','.join(update_cols)
            
            sql = f"""
                INSERT INTO financials ({columns_str})
                VALUES ({placeholders})
                ON CONFLICT({','.join(pk_cols)})
                DO UPDATE SET {updates_str}
            """
            
            # Prepare data for batch insert
            data = df_to_store.where(pd.notna(df_to_store), None)
            records = data.to_records(index=False).tolist()
            
            # Execute in transaction
            with self.conn:
                cursor = self.conn.cursor()
                cursor.executemany(sql, records)
                
                # Log success
                corp_codes = df_to_store['corp_code'].unique()
                years = df_to_store['bsns_year'].unique()
                logger.info(
                    f"Stored {len(records)} financial records for "
                    f"{len(corp_codes)} companies for year(s) {', '.join(map(str, sorted(years)))}"
                )
                
                return True
                
        except sqlite3.Error as db_err:
            logger.error(f"Database error while storing financial data: {db_err}", exc_info=True)
            if hasattr(self, 'conn'):
                self.conn.rollback()
            return False
            
        except Exception as e:
            logger.error(f"Unexpected error while storing financial data: {e}", exc_info=True)
            if hasattr(self, 'conn'):
                self.conn.rollback()
            return False

    def get_latest_fundamentals(self, tickers: List[str], as_of_date: str) -> pd.DataFrame:
        """Load the latest fundamental data for the given tickers as of a specific date.
        
        Args:
            tickers: List of stock ticker symbols
            as_of_date: Date in 'YYYY-MM-DD' format to get data as of
            
        Returns:
            DataFrame containing the latest fundamental data for each ticker
        """
        logger.info(f"Loading latest fundamentals for {len(tickers)} tickers as of {as_of_date}.")
        
        # Define all required account mappings with multiple possible field names
        account_map = {
            # Basic financials
            'Assets': ['자산총계', 'ifrs-full_Assets', 'dart_TotalAssets'],
            'Liabilities': ['부채총계', 'ifrs-full_Liabilities', 'dart_TotalLiabilities'],
            'Equity': ['자본총계', 'ifrs-full_Equity', 'dart_TotalEquity'],
            'NetIncome': ['당기순이익', '당기순이익(손실)', 'ifrs-full_ProfitLoss', 'dart_NetIncomeLoss'],
            'GrossProfit': ['매출총이익', 'ifrs-full_GrossProfit', 'dart_GrossProfit'],
            'Sales': [
                '매출액', 
                '수익(매출액)', 
                'ifrs-full_Revenue', 
                'dart_Revenue', 
                'ifrs-full_RevenueFromContractsWithCustomers'
            ],
            'OperatingIncome': [
                '영업이익', 
                '영업이익(손실)', 
                'ifrs-full_ProfitLossFromOperatingActivities', 
                'dart_OperatingIncomeLoss'
            ],
            'OperatingCF': [
                '영업활동으로인한현금흐름', 
                'ifrs-full_CashFlowsFromUsedInOperatingActivities', 
                'dart_CashFlowsFromUsedInOperatingActivities'
            ],
            'CostOfGoodsSold': ['매출원가', 'ifrs-full_CostOfSales', 'dart_CostOfSales'],
            'InvestingCF': [
                '투자활동으로인한현금흐름', 
                'ifrs-full_CashFlowsFromUsedInInvestingActivities'
            ],
            'FinancingCF': [
                '재무활동으로인한현금흐름', 
                'ifrs-full_CashFlowsFromUsedInFinancingActivities'
            ],
            'TotalDebt': [
                '총차입금', 
                'ifrs-full_ShorttermBorrowings', 
                'ifrs-full_LongtermBorrowings'
            ],
            'CashAndEquivalents': ['현금및현금성자산', 'ifrs-full_CashAndCashEquivalents'],
            'CurrentAssets': ['유동자산', 'ifrs-full_CurrentAssets'],
            'CurrentLiabilities': ['유동부채', 'ifrs-full_CurrentLiabilities'],
            'AccountsReceivable': ['매출채권', 'ifrs-full_TradeAndOtherCurrentReceivables'],
            'TotalInventory': ['재고자산', 'ifrs-full_Inventories']
        }
        
        try:
            # Get corporation codes for all tickers
            corp_codes_map = {ticker: self.find_corp_code(ticker) for ticker in tickers}
            valid_corp_codes = [c for c in list(corp_codes_map.values()) if c is not None]
            
            if not valid_corp_codes:
                logger.warning("No valid corporation codes found for the provided tickers.")
                return pd.DataFrame()
                
            ticker_lookup = {v: k for k, v in corp_codes_map.items()}
            
            # Get all possible account fields to query
            all_fields = []
            field_mapping = {}
            
            # Create a mapping of actual DB fields to our standard field names
            for field_name, possible_fields in account_map.items():
                for field in possible_fields:
                    field_mapping[field] = field_name
                    all_fields.append(field)
            
            if not all_fields:
                logger.warning("No account fields found to query.")
                return pd.DataFrame()
            
            # Build and execute the SQL query
            placeholders = ','.join(['?'] * len(valid_corp_codes))
            field_placeholders = ','.join(['?'] * len(all_fields))
            
            sql = f"""
            WITH LatestReports AS (
                SELECT corp_code, MAX(rcept_no) AS rcept_no 
                FROM financials 
                WHERE corp_code IN ({placeholders})
                  AND bsns_year <= ? 
                  AND reprt_code = '11011' 
                GROUP BY corp_code
            ) 
            SELECT f.corp_code, f.account_id, f.thstrm_amount 
            FROM financials f 
            JOIN LatestReports lr ON f.corp_code = lr.corp_code AND f.rcept_no = lr.rcept_no 
            WHERE f.account_id IN ({field_placeholders})
            """
            
            params = valid_corp_codes + [as_of_date[:4]] + all_fields
            
            with self.conn:
                raw_df = pd.read_sql_query(sql, self.conn, params=params)

            if raw_df.empty:
                logger.warning("No financial data found for the given tickers and date range.")
                return pd.DataFrame()
                
            # Process the raw data into a clean DataFrame
            result_data = []
            
            for (corp_code, account_id), group in raw_df.groupby(['corp_code', 'account_id']):
                if account_id in field_mapping:
                    standard_field = field_mapping[account_id]
                    amount = group['thstrm_amount'].iloc[0] if not group.empty else None
                    if amount is not None:
                        result_data.append({
                            'ticker': ticker_lookup.get(corp_code, corp_code),
                            'field': standard_field,
                            'value': amount
                        })
            
            if not result_data:
                logger.warning("No valid financial data found after processing.")
                return pd.DataFrame()
                
            # Convert to a pivot table with tickers as index and fields as columns
            result_df = pd.DataFrame(result_data).pivot(
                index='ticker',
                columns='field',
                values='value'
            ).reset_index()
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error in get_latest_fundamentals: {e}", exc_info=True)
            return pd.DataFrame()

    def _process_and_validate_fundamentals(self, raw_df: pd.DataFrame, account_map: Dict, ticker_lookup: Dict) -> pd.DataFrame:
        """Process raw financial data, calculate ratios, and validate."""
        # Pivot the data to have accounts as columns
        df = raw_df.pivot(index='corp_code', columns='account_id', values='thstrm_amount')

        # Create a reverse mapping from DB field names to our standard names
        reverse_mapping = {field: std_name for std_name, fields in account_map.items() for field in fields}

        # Rename columns to our standard names, handling duplicates by keeping the first
        df = df.rename(columns=reverse_mapping)
        df = df.groupby(level=0, axis=1).first()

        # Calculate financial ratios
        df = self._calculate_financial_ratios(df)

        # Add ticker symbols and clean up index
        df['ticker'] = df.index.map(ticker_lookup)
        df = df.reset_index(drop=True).set_index('ticker')

        logger.info(f"Successfully processed fundamentals for {len(df)} tickers.")
        return df

    def _calculate_financial_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate key financial ratios safely."""
        try:
            # Ensure columns exist before calculation
            if 'NetIncome' in df and 'Equity' in df:
                df['ROE'] = df['NetIncome'] / df['Equity'].replace(0, np.nan)
            if 'NetIncome' in df and 'Assets' in df:
                df['ROA'] = df['NetIncome'] / df['Assets'].replace(0, np.nan)
            if 'GrossProfit' in df and 'Sales' in df:
                df['GrossMargin'] = df['GrossProfit'] / df['Sales'].replace(0, np.nan)
            if 'OperatingIncome' in df and 'Sales' in df:
                df['OperatingMargin'] = df['OperatingIncome'] / df['Sales'].replace(0, np.nan)
            if 'TotalDebt' in df and 'Equity' in df:
                df['DebtToEquity'] = df['TotalDebt'] / df['Equity'].replace(0, np.nan)
            if 'CurrentAssets' in df and 'CurrentLiabilities' in df:
                df['CurrentRatio'] = df['CurrentAssets'] / df['CurrentLiabilities'].replace(0, np.nan)
            return df
        except Exception as e:
            logger.error(f"Error calculating financial ratios: {e}")
            return df

    def __del__(self):
        try:
            if hasattr(self, 'conn'):
                self.conn.close()
        except Exception as e:
            logger.debug(f"Error closing database connection: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch DART financial statements.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-i", "--identifier", help="Single stock code")
    group.add_argument("-a", "--all", action="store_true", help="Fetch for all tickers in universe file.")
    parser.add_argument("-s", "--start-year", required=True, type=int, help="Start year (YYYY)")
    parser.add_argument("-e", "--end-year", required=True, type=int, help="End year (YYYY)")
    parser.add_argument("--universe-file", type=str, default="universe.csv",
                        help="Path to universe file (default: universe.csv)")
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Get API key from environment
    api_key = os.getenv('DART_API_KEY')
    if not api_key:
        logger.critical("DART_API_KEY environment variable not found")
        sys.exit(1)
    
    # Initialize fetcher
    fetcher = FinancialsFetcher(api_key=api_key)
    
    # Get tickers to process
    if args.all:
        universe_file = Path(args.universe_file)
        if not universe_file.exists():
            logger.critical(f"Universe file not found: {universe_file}")
            sys.exit(1)
            
        try:
            tickers = pd.read_csv(universe_file)['종목코드']\
                .astype(str)\
                .str.lstrip('A')\
                .str.zfill(6)\
                .tolist()
        except Exception as e:
            logger.error(f"Error reading universe file: {e}")
            sys.exit(1)
    else:
        tickers = [args.identifier]
    
    # Process each year and ticker
    for year in range(args.start_year, args.end_year + 1):
        for ticker in tqdm(tickers, desc=f"Processing Year {year}", unit="ticker"):
            corp_code = fetcher.find_corp_code(ticker)
            if corp_code:
                fetcher.fetch_financials(corp_code, year)