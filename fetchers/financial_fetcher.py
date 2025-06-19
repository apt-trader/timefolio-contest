# financials_fetcher.py
# Fetch, store, and compute financial-statement–based factors via OpenDARTReader (DART OpenAPI)

import os
import sys  # Import sys to handle exit
import sqlite3
import logging
import time
from datetime import datetime
import pandas as pd
from OpenDartReader.dart import OpenDartReader
from dotenv import load_dotenv
import argparse
import yaml
from pathlib import Path
from typing import Optional, List
from tqdm import tqdm
from collections import deque

# Load environment variables from .env file at the start of the script
load_dotenv()

logger = logging.getLogger(__name__)

# Load sector universe file from config
try:
    config_path = Path(__file__).parent.parent / "config/config.yaml"
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    universe_file = Path(__file__).parent.parent / cfg['data_settings']['stock_universe_file']
except FileNotFoundError:
    logger.error(f"Config file not found at {config_path}. Please ensure it exists.")
    universe_file = None # Handle gracefully if config is missing

class RateLimiter:
    """Simple token-bucket style rate limiter to prevent exceeding OpenDARTReader limits."""
    def __init__(self, max_per_minute: int = 900, max_per_day: int = 9500):
        self.max_per_minute = max_per_minute
        self.max_per_day = max_per_day
        self.calls_minute = deque()
        self.calls_day = deque()

    def wait(self):
        now = time.time()
        while self.calls_minute and now - self.calls_minute[0] > 60:
            self.calls_minute.popleft()
        while self.calls_day and now - self.calls_day[0] > 86400:
            self.calls_day.popleft()
        
        if len(self.calls_day) >= self.max_per_day:
            raise RuntimeError(f"Daily OpenDARTReader API call limit ({self.max_per_day}) reached.")
        
        if len(self.calls_minute) >= self.max_per_minute:
            sleep_time = 60.1 - (now - self.calls_minute[0])
            logger.info(f"Minute rate limit reached. Sleeping for {sleep_time:.2f} seconds.")
            time.sleep(sleep_time)
            now = time.time()
            while self.calls_minute and now - self.calls_minute[0] > 60:
                self.calls_minute.popleft()

        self.calls_minute.append(now)
        self.calls_day.append(now)

class FinancialsFetcher:
    """
    A. Uses DART OpenAPI to pull corporate filings and financials.
    B. Persists into a SQLite 'financials' table with a fixed, robust schema.
    C. Provides methods to retrieve structured fundamental data.
    """
    def __init__(self, api_key: str, db_path: str = str(Path(__file__).parent.parent / "krx_data.db")):
        """Initialize FinancialsFetcher with an API key and database path."""
        if not api_key:
            raise ValueError("DART API key is required.")
            
        self.api_key = api_key
        self.dart = OpenDartReader(self.api_key)
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._init_financials_table()
        self.existing_columns = self._get_existing_columns()
        self.rate_limiter = RateLimiter()

    def _init_financials_table(self):
        """Initialize SQLite schema for raw financial data."""
        try:
            with self.conn:
                self.conn.execute("""
                    CREATE TABLE IF NOT EXISTS financials (
                        rcept_no TEXT, reprt_code TEXT, bsns_year TEXT, corp_code TEXT,
                        sj_div TEXT, sj_nm TEXT, fs_div TEXT, fs_nm TEXT, fs_sn TEXT,
                        account_id TEXT, account_nm TEXT, account_detail TEXT,
                        thstrm_nm TEXT, thstrm_amount REAL, thstrm_add_amount REAL,
                        thstrm_q_nm TEXT, thstrm_q_amount REAL,
                        frmtrm_nm TEXT, frmtrm_amount REAL, frmtrm_add_amount REAL,
                        frmtrm_q_nm TEXT, frmtrm_q_amount REAL,
                        bfefrmtrm_nm TEXT, bfefrmtrm_amount REAL,
                        ord INTEGER, currency TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (rcept_no, account_id, fs_div)
                    )""")
                self.conn.execute("CREATE INDEX IF NOT EXISTS idx_financials_corp_code_year ON financials(corp_code, bsns_year)")
                self.conn.execute("CREATE INDEX IF NOT EXISTS idx_financials_account_id ON financials(account_id)")
                self.conn.execute("""
                CREATE TRIGGER IF NOT EXISTS update_financials_timestamp
                AFTER UPDATE ON financials FOR EACH ROW
                BEGIN
                    UPDATE financials SET updated_at = CURRENT_TIMESTAMP WHERE rowid = NEW.rowid;
                END;""")
        except sqlite3.Error as e:
            logger.critical(f"FATAL: Could not create or verify database schema. Error: {e}")
            raise

    def _get_existing_columns(self) -> list:
        """Return a list of existing column names in the financials table."""
        cursor = self.conn.execute("PRAGMA table_info(financials)")
        return [row[1] for row in cursor.fetchall()]

    def find_corp_code(self, identifier: str) -> Optional[str]:
        """Find DART corporation code by company name or stock code."""
        try:
            self.rate_limiter.wait()
            return self.dart.find_corp_code(identifier)
        except Exception as e:
            logger.error(f"Could not find corporation code for identifier '{identifier}': {e}")
            return None

    def fetch_financials(self, corp_code: str, year: int, report_type: str = '11011'):
        """Fetch financial statements for a given corporation and year."""
        logger.info(f"Processing financials for corp_code={corp_code}, year={year}, report_type={report_type}")
        
        self.rate_limiter.wait()
        
        cursor = self.conn.cursor()
        cursor.execute("SELECT 1 FROM financials WHERE corp_code = ? AND bsns_year = ? AND reprt_code = ? LIMIT 1", (corp_code, str(year), report_type))
        if cursor.fetchone():
            logger.info(f"Data for {corp_code} {year} ({report_type}) already exists. Skipping fetch.")
            return

        try:
            logger.debug(f"Fetching CFS (Consolidated) for {corp_code} {year}...")
            fs_all = self.dart.finstate_all(corp_code, year, report_type, fs_div='CFS')
            
            if isinstance(fs_all, pd.DataFrame) and fs_all.empty:
                logger.info(f"No CFS data for {corp_code} {year}, trying OFS (Separate)...")
                self.rate_limiter.wait()
                fs_all = self.dart.finstate_all(corp_code, year, report_type, fs_div='OFS')
                
            if isinstance(fs_all, pd.DataFrame) and not fs_all.empty:
                self._store_financials(fs_all)
                
        except Exception as e:
            logger.error(f"Error processing {corp_code} {year}: {e}", exc_info=True)
            
        logger.warning(f"No financial data found for {corp_code} {year} (report_type: {report_type})")

    def _store_financials(self, df: pd.DataFrame) -> bool:
        """Store financial data in SQLite database using a fixed schema."""
        if df is None or df.empty:
            logger.warning("Store operation cancelled: DataFrame is empty.")
            return False
            
        try:
            valid_cols = [col for col in df.columns if col in self.existing_columns]
            df_to_store = df[valid_cols].copy()

            for col in ['thstrm_amount', 'frmtrm_amount', 'bfefrmtrm_amount', 'thstrm_q_amount', 'frmtrm_q_amount']:
                if col in df_to_store.columns:
                    df_to_store[col] = pd.to_numeric(df_to_store[col], errors='coerce')
            
            pk_cols = ['rcept_no', 'account_id', 'fs_div']
            update_cols = [col for col in df_to_store.columns if col not in pk_cols]
            
            sql = f"""
            INSERT INTO financials ({', '.join(f'"{c}"' for c in df_to_store.columns)})
            VALUES ({', '.join(['?'] * len(df_to_store.columns))})
            ON CONFLICT({', '.join(pk_cols)}) DO UPDATE SET
            {', '.join(f'"{c}" = excluded."{c}"' for c in update_cols)}, updated_at = CURRENT_TIMESTAMP
            """
            
            records = df_to_store.where(pd.notna(df_to_store), None).to_records(index=False).tolist()
            with self.conn:
                self.conn.executemany(sql, records)
            
            logger.info(f"Successfully stored/updated {len(records)} financial records.")
            return True
            
        except Exception:
            logger.exception(f"FATAL: Unexpected error storing financials.")
            self.conn.rollback()
            return False
    
    def get_latest_fundamentals(self, tickers: List[str], as_of_date: str) -> pd.DataFrame:
        """
        Queries the database for the latest available annual financial statements
        for a given list of tickers and pivots the data into a usable format.
        """
        logger.info(f"Loading latest fundamentals for {len(tickers)} tickers as of {as_of_date}.")
        
        # --- Added Sales and OperatingCF ---
        account_map = {
            'Assets': 'ifrs-full_Assets',
            'Liabilities': 'ifrs-full_Liabilities',
            'Equity': 'ifrs-full_Equity',
            'NetIncome': 'ifrs-full_ProfitLoss',
            'GrossProfit': 'ifrs-full_GrossProfit',
            'Sales': 'ifrs-full_Revenue',  # Added for Sales-to-Price
            'OperatingCF': 'ifrs-full_CashFlowsFromUsedInOperatingActivities' # Added for CF-to-Price
        }
        account_ids = list(account_map.values())

        # Create a mapping from stock code to DART corp_code
        corp_codes_map = {ticker: self.find_corp_code(ticker) for ticker in tickers}
        valid_corp_codes = [c for c in corp_codes_map.values() if c is not None]
        if not valid_corp_codes:
            logger.warning("No valid DART corp_codes found for the given tickers.")
            return pd.DataFrame()

        ticker_lookup = {v: k for k, v in corp_codes_map.items()}
        
        # The SQL query remains the same, it just gets more account_ids
        sql = f"""
        WITH LatestReports AS (
            SELECT corp_code, MAX(rcept_no) AS rcept_no
            FROM financials
            WHERE corp_code IN ({','.join(['?'] * len(valid_corp_codes))})
              AND bsns_year < ? AND reprt_code = '11011'
            GROUP BY corp_code
        )
        SELECT f.corp_code, f.account_id, f.thstrm_amount
        FROM financials f JOIN LatestReports lr ON f.corp_code = lr.corp_code AND f.rcept_no = lr.rcept_no
        WHERE f.account_id IN ({','.join(['?'] * len(account_ids))})
        """
        params = valid_corp_codes + [as_of_date[:4]] + account_ids
        
        try:
            with self.conn:
                raw_df = pd.read_sql_query(sql, self.conn, params=params)
        except Exception as e:
            logger.error(f"Database query for fundamentals failed: {e}")
            return pd.DataFrame()

        if raw_df.empty:
            logger.warning("No fundamental data found in the database for the given tickers and date.")
            return pd.DataFrame()

        pivot_df = raw_df.pivot_table(
            index='corp_code', columns='account_id', values='thstrm_amount'
        ).rename(columns={v: k for k, v in account_map.items()})

        pivot_df.index = pivot_df.index.map(ticker_lookup)
        
        logger.info(f"Successfully pivoted fundamental data for {len(pivot_df)} tickers.")
        return pivot_df

    def __del__(self):
        try:
            self.conn.close()
        except Exception:
            pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch DART financial statements.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-i", "--identifier", help="Single stock code (e.g., '005930')")
    group.add_argument("-a", "--all", action="store_true", help="Fetch for all tickers in the universe file.")
    parser.add_argument("-s", "--start-year", required=True, type=int, help="Start year (YYYY)")
    parser.add_argument("-e", "--end-year", required=True, type=int, help="End year (YYYY)")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    args = parser.parse_args()
    
    logging.basicConfig(level=getattr(logging, args.log_level), format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    api_key = os.getenv('DART_API_KEY')
    if not api_key:
        logger.critical("DART_API_KEY not found in environment. Please set it in your .env file or environment variables.")
        sys.exit(1)
        
    fetcher = FinancialsFetcher(api_key=api_key)

    if args.all:
        if universe_file and universe_file.exists():
            df = pd.read_csv(universe_file)
            tickers = df['종목코드'].astype(str).str.lstrip('A').str.zfill(6).tolist()
            logger.info(f"Loaded {len(tickers)} tickers from {universe_file}")
        else:
            logger.critical(f"Universe file not found or configured. Cannot proceed with --all.")
            tickers = []
    else:
        tickers = [args.identifier]

    for year in range(args.start_year, args.end_year + 1):
        for ticker in tqdm(tickers, desc=f"Processing Year {year}"):
            try:
                corp_code = fetcher.find_corp_code(ticker)
                if corp_code:
                    fetcher.fetch_financials(corp_code, year, report_type='11011')
            except Exception as e:
                logger.error(f"Unexpected error processing ticker {ticker} for year {year}: {e}")