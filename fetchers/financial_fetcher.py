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
    def __init__(self, api_key: str, db_path: str):
        self.dart = OpenDartReader(api_key) if api_key else None
        if not api_key: logger.warning("DART API key not provided. FinancialsFetcher will not be able to fetch new data.")
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
            # Use the standardized 'report_date' column for pivoting
            pivot_df = df.pivot_table(index=['corp_code', 'ticker', 'report_date', 'report_code'], columns='account_nm', values='thstrm_amount', aggfunc='first').reset_index()
            pivot_df.columns.name = None
            value_cols = [col for col in pivot_df.columns if col not in ['corp_code', 'ticker', 'report_date', 'report_code']]
            for col in value_cols:
                pivot_df[col] = pd.to_numeric(pivot_df[col].astype(str).str.replace(',', ''), errors='coerce')
            logger.info(f"Successfully retrieved and processed {len(pivot_df)} historical financial reports.")
            return pivot_df
        except Exception as e:
            logger.error(f"Error fetching historical financials from DB: {e}", exc_info=True)
            return pd.DataFrame()

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
