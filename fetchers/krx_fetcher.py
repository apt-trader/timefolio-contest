#fetchers/krx_fetcer.py
import time
import sqlite3
import logging
import random
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Union, Optional
from io import BytesIO
import pandas as pd
import numpy as np
import requests
from requests.adapters import HTTPAdapter, Retry
from tqdm import tqdm

# Configure logging
logs_dir = Path(__file__).parent.parent / 'logs'
logs_dir.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(logs_dir / 'krx_fetcher.log')
    ]
)
logger = logging.getLogger('krx_fetcher')

# Constants
GEN_OTP_URL = "https://data.krx.co.kr/comm/fileDn/GenerateOTP/generate.cmd"
DOWNLOAD_URL = "https://data.krx.co.kr/comm/fileDn/download_csv/download.cmd"
DEFAULT_HEADERS = {
    "Referer": "https://data.krx.co.kr/contents/MDC/MDI/mdiLoader",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"
}

class KRXDataFetcher:
    """
    A class to fetch and manage KRX market data using the OTP-based download system.
    """
    def __init__(self, cache_dir: str = None, db_path: str = str(Path(__file__).parent.parent / "db" / "krx_data.db"), 
                 throttle: float = 0.5, max_retries: int = 3):
        if cache_dir is None:
            self.cache_dir = Path(__file__).parent / 'krx_cache'
        else:
            self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.throttle = throttle
        self.max_retries = max_retries
        self.last_request = 0
        self.session = self._create_session()
        self._init_database()
    
    def _create_session(self) -> requests.Session:
        session = requests.Session()
        retry_strategy = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        session.headers.update(DEFAULT_HEADERS)
        return session
    
    def _throttle(self):
        """Enforces request throttling with randomized jitter."""
        elapsed = time.time() - self.last_request
        if elapsed < self.throttle:
            time.sleep(self.throttle - elapsed)
        
        jitter = random.uniform(0.5, 1.5)
        time.sleep(jitter)
        self.last_request = time.time()
    
    def _init_database(self):
        with self.conn:
            # Table for daily market data
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS daily_prices (
                code TEXT, date DATE, open REAL, high REAL, low REAL, close REAL,
                volume INTEGER, value INTEGER, market_cap REAL, market TEXT, name TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, PRIMARY KEY (code, date)
            )""")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_daily_prices_code ON daily_prices (code)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_daily_prices_date ON daily_prices (date)")
            
            # Table for listing information (IPO dates)
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS listing_info (
                code TEXT PRIMARY KEY, name TEXT, market TEXT, listing_date DATE,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_listing_info_date ON listing_info (listing_date)")
    
    def _get_otp(self, bld: str, **kwargs) -> Optional[str]:
        """Generic OTP generator for different KRX endpoints."""
        payload = {
            "bld": bld,
            "csvxls_isNo": "false",
            "name": "fileDown",
            "url": bld
        }
        payload.update(kwargs)

        for attempt in range(self.max_retries + 1):
            try:
                self._throttle()
                logger.debug(f"Requesting OTP for {bld} with params {kwargs} (attempt {attempt+1})")
                response = self.session.post(GEN_OTP_URL, data=payload, timeout=30)
                response.raise_for_status()
                otp = response.text.strip()
                if not otp: raise ValueError("Empty OTP received")
                return otp
            except (requests.RequestException, ValueError) as e:
                if attempt == self.max_retries:
                    logger.error(f"Failed to get OTP after {self.max_retries+1} attempts: {e}")
                    return None
                logger.warning(f"Attempt {attempt+1} to get OTP failed, retrying... Error: {e}")
                time.sleep(attempt + 1)

    def _download_data_raw(self, otp: str) -> Optional[bytes]:
        """Helper function to download the raw bytes of the CSV file with retries."""
        if not otp: return None
        for attempt in range(self.max_retries + 1):
            try:
                self._throttle()
                logger.debug(f"Downloading raw data with OTP: {otp} (attempt {attempt+1})")
                response = self.session.post(DOWNLOAD_URL, data={"code": otp}, timeout=60)
                response.raise_for_status()
                content = response.content

                # Check for HTML error page in response, which indicates a server-side issue
                if content.strip().lower().startswith(b'<html>') or content.strip().lower().startswith(b'<!doctype html'):
                    raise ValueError("Received HTML error page instead of CSV data.")

                if not content:
                    logger.warning("Received empty file, likely a non-trading day.")
                    return None
                return content # Success

            except (requests.RequestException, ValueError) as e:
                if attempt == self.max_retries:
                    logger.error(f"Failed to download raw data after {self.max_retries+1} attempts: {e}")
                    return None
                logger.warning(f"Attempt {attempt+1} to download raw data failed, retrying... Error: {e}")
                time.sleep(2 ** attempt) # Exponential backoff

    def fetch_daily_data(self, date, markets: List[str] = None) -> Dict[str, pd.DataFrame]:
        """Fetches, processes, and cleans daily data, using a file cache."""
        if markets is None: markets = ["STK", "KSQ"]
        
        date_str = date.strftime('%Y%m%d') if isinstance(date, datetime) else date
        date_dt = datetime.strptime(date_str, '%Y%m%d')
            
        results = {}
        for mkt in markets:
            df = pd.DataFrame()
            try:
                cache_file = self.cache_dir / f"{mkt}-{date_str}.csv"
                
                if cache_file.exists():
                    logger.info(f"Using cached file for {mkt} on {date_str}")
                    df = pd.read_csv(cache_file, encoding='EUC-KR', dtype={'종목코드': str})
                else:
                    logger.info(f"Fetching {mkt} data for {date_str} from network")
                    otp = self._get_otp("dbms/MDC/STAT/standard/MDCSTAT01501", mktId=mkt, trdDd=date_str, share="1", money="1")
                    raw_bytes = self._download_data_raw(otp)
                    
                    if raw_bytes:
                        df = pd.read_csv(BytesIO(raw_bytes), encoding='EUC-KR', dtype={'종목코드': str})
                        # Safeguard: Ensure cache directory exists right before writing.
                        self.cache_dir.mkdir(exist_ok=True, parents=True)
                        with open(cache_file, 'wb') as f:
                            f.write(raw_bytes)
                    else:
                        logger.warning(f"No data downloaded for {mkt} on {date_str}.")
                        df = pd.DataFrame()
                
                if df.empty:
                    results[mkt] = df
                    continue

                # Correctly map KRX columns. '상장주식수' is listed shares. Market cap must be calculated.
                column_map = {
                    '종목코드': 'code', '종목명': 'name', '시장구분': 'market_name',
                    '시가': 'open', '고가': 'high', '저가': 'low', '종가': 'close',
                    '거래량': 'volume', '거래대금': 'value', '상장주식수': 'listed_shares'
                }
                df.rename(columns=column_map, inplace=True)
                df['date'] = date_dt
                df['market'] = mkt
                df['code'] = df['code'].str.zfill(6)

                # Clean numeric columns by removing commas
                numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'value', 'listed_shares']
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = df[col].astype(str).str.replace(',', '', regex=False)
                        df[col] = pd.to_numeric(df[col], errors='coerce')

                # Calculate market cap manually: close * listed_shares
                if 'close' in df.columns and 'listed_shares' in df.columns:
                    df['market_cap'] = df['close'] * df['listed_shares']
                else:
                    df['market_cap'] = 0.0

                final_columns = ['code', 'name', 'date', 'market', 'open', 'high', 'low', 'close', 'volume', 'value', 'market_cap']
                df = df.reindex(columns=final_columns)
                df.fillna({'open': 0.0, 'high': 0.0, 'low': 0.0, 'close': 0.0, 'market_cap': 0.0, 'volume': 0, 'value': 0}, inplace=True)
                df[['volume', 'value']] = df[['volume', 'value']].astype(int)

                results[mkt] = df
                logger.info(f"Processed {len(df)} rows for {mkt} on {date_str}")

            except Exception as e:
                logger.error(f"Critical error processing {mkt} data for {date_str}: {e}", exc_info=True)
                results[mkt] = pd.DataFrame()

        return results

    def fetch_historical_data(self, start_date: Union[str, datetime], 
                             end_date: Union[str, datetime] = None,
                             markets: List[str] = None, force_refetch: bool = False) -> pd.DataFrame:
        if end_date is None: end_date = datetime.now()
        if isinstance(start_date, str): start_date = datetime.strptime(start_date, '%Y%m%d')
        if isinstance(end_date, str): end_date = datetime.strptime(end_date, '%Y%m%d')

        existing_dates = set()
        if not force_refetch:
            try:
                existing_dates = set(d.strftime('%Y%m%d') for d in self.get_available_dates())
            except Exception as e:
                logger.warning(f"Could not fetch existing dates from DB: {e}")

        date_range = pd.bdate_range(start_date, end_date)
        # This method is now deprecated in favor of batch processing inside fetch_historical_data
        pass

    def fetch_historical_data(self, start_date: Union[str, datetime], 
                             end_date: Union[str, datetime] = None,
                             markets: List[str] = None, force_refetch: bool = False) -> pd.DataFrame:
        if end_date is None: end_date = datetime.now()
        if isinstance(start_date, str): start_date = datetime.strptime(start_date, '%Y%m%d')
        if isinstance(end_date, str): end_date = datetime.strptime(end_date, '%Y%m%d')

        existing_dates = set()
        if not force_refetch:
            try:
                existing_dates = set(d.strftime('%Y%m%d') for d in self.get_available_dates())
            except Exception as e:
                logger.warning(f"Could not fetch existing dates from DB: {e}")

        date_range = pd.bdate_range(start_date, end_date)
        total_rows_processed = 0
        for date in tqdm(date_range, desc="Fetching Historical Market Data"):
            date_str = date.strftime('%Y%m%d')
            if date_str in existing_dates:
                logger.debug(f"Date {date_str} already exists in database. Skipping.")
                continue
            
            daily_results = self.fetch_daily_data(date, markets)
            daily_df = pd.concat(daily_results.values(), ignore_index=True)
            
            if not daily_df.empty:
                rows_added = self.update_database(daily_df)
                total_rows_processed += rows_added
        
        logger.info(f"Completed historical fetch. Total rows processed: {total_rows_processed}")
        # Return an empty dataframe as the data is now in the DB.
        return pd.DataFrame()

    def update_database(self, df: pd.DataFrame) -> int:
        """Updates the daily_prices table in the database. Returns number of rows processed."""
        if df.empty:
            return 0

        rows_to_process = len(df)
        try:
            # Use INSERT OR REPLACE to handle both new and existing records efficiently.
            update_query = """
            INSERT OR REPLACE INTO daily_prices (code, date, open, high, low, close, volume, value, market_cap, market, name)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            # Ensure date is in 'YYYY-MM-DD' format for SQLite
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
            
            # Replace numpy's NaN with None for SQLite compatibility (to store as NULL)
            df_for_db = df[['code', 'date', 'open', 'high', 'low', 'close', 'volume', 'value', 'market_cap', 'market', 'name']].copy()
            df_for_db.replace({np.nan: None}, inplace=True)
            update_data = [tuple(x) for x in df_for_db.to_numpy()]

            with self.conn:
                self.conn.executemany(update_query, update_data)
            
            # logger.info(f"Successfully updated/inserted {rows_to_process} rows.")
            return rows_to_process
        except Exception as e:
            logger.error(f"Failed to update database for {rows_to_process} rows: {e}", exc_info=True)
            return 0

    def get_available_dates(self) -> pd.DatetimeIndex:
        with self.conn:
            df = pd.read_sql("SELECT DISTINCT date FROM daily_prices ORDER BY date", self.conn, parse_dates=['date'])
        return pd.DatetimeIndex(df['date'])

    def fetch_listing_info(self) -> Optional[pd.DataFrame]:
        """Fetches the complete list of stocks and their listing dates from KRX."""
        logger.info("Fetching complete listing information from KRX...")
        try:
            # This endpoint provides the full list of listed stocks.
            # The `share='1'` parameter is crucial.
            otp = self._get_otp("dbms/MDC/STAT/standard/MDCSTAT01901", mktId='ALL', share='1')
            raw_bytes = self._download_data_raw(otp)
            if not raw_bytes:
                logger.error("Received no data from listing info endpoint.")
                return None

            # The data is encoded in cp949 as discovered from research.
            df = pd.read_csv(BytesIO(raw_bytes), encoding='cp949', dtype={'단축코드': str})
            
            # Map columns to a standard format based on logged output.
            column_map = {
                '단축코드': 'code',
                '한글 종목약명': 'name',
                '시장구분': 'market',
                '상장일': 'listing_date'
            }
            df.rename(columns=column_map, inplace=True)
            
            # Clean up data.
            df['code'] = df['code'].str.zfill(6)
            df['listing_date'] = pd.to_datetime(df['listing_date'], errors='coerce')
            
            # Select and validate final columns.
            final_cols = ['code', 'name', 'market', 'listing_date']
            df = df.reindex(columns=final_cols).dropna(subset=['listing_date'])
            
            logger.info(f"Successfully fetched and processed {len(df)} listings.")
            return df
        except Exception as e:
            logger.error(f"Failed to fetch listing info: {e}", exc_info=True)
            return None

    def update_listing_info_in_db(self, df: pd.DataFrame) -> int:
        """Updates the listing_info table in the database."""
        if df.empty: return 0
        df_to_save = df.copy()
        df_to_save['listing_date'] = pd.to_datetime(df_to_save['listing_date']).dt.strftime('%Y-%m-%d')
        
        with self.conn:
            cursor = self.conn.cursor()
            try:
                cursor.execute("BEGIN TRANSACTION")
                placeholders = ', '.join(['?'] * len(df_to_save.columns))
                columns_str = ', '.join(f'"{c}"' for c in df_to_save.columns)
                update_clause = ', '.join([f'"{c}"=excluded."{c}"' for c in ['name', 'market', 'listing_date']])
                sql = f"INSERT INTO listing_info ({columns_str}) VALUES ({placeholders}) ON CONFLICT(code) DO UPDATE SET {update_clause}, updated_at=CURRENT_TIMESTAMP"
                
                cursor.executemany(sql, df_to_save.to_records(index=False).tolist())
                count = cursor.rowcount
                cursor.execute("COMMIT")
                logger.info(f"Successfully updated/inserted {count} rows in the listing_info table.")
                return count
            except Exception as e:
                cursor.execute("ROLLBACK")
                logger.error(f"Listing info database update failed: {e}", exc_info=True)
                raise

    def __del__(self):
        if self.conn: self.conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fetch KRX market data')
    parser.add_argument('-s', '--start-date', help='Start date (YYYYMMDD)')
    parser.add_argument('-e', '--end-date', help='End date (YYYYMMDD)')
    parser.add_argument('--market', default='ALL', choices=['STK', 'KSQ', 'ALL'], help='Market to fetch (STK, KSQ, ALL)')
    parser.add_argument('--update-db', action='store_true', help='Update the database with fetched data.')
    parser.add_argument('--force-refetch', action='store_true', help='Force re-fetch of data even if it exists in the database.')
    parser.add_argument('--fetch-listings', action='store_true', help='Fetch and update the listing information (IPO dates).')
    args = parser.parse_args()

    fetcher = KRXDataFetcher()

    if args.fetch_listings:
        listings_df = fetcher.fetch_listing_info()
        if listings_df is not None and not listings_df.empty and args.update_db:
            fetcher.update_listing_info_in_db(listings_df)
        elif listings_df is not None:
            print("\n--- Fetched Listing Info Sample ---")
            print(listings_df.head())

    if args.start_date and args.end_date:
        markets = ['STK', 'KSQ'] if args.market == 'ALL' else [args.market]
        df = fetcher.fetch_historical_data(args.start_date, args.end_date, markets, force_refetch=args.force_refetch)
        
        if df.empty:
            logger.info("No new data to update in the database.")
        elif args.update_db:
            fetcher.update_database(df)
        else:
            print("\n--- Fetched Data Sample ---")
            print((df[['code', 'name', 'date', 'open', 'close', 'volume', 'value', 'market_cap']].head()))
    
    if not args.fetch_listings and not (args.start_date and args.end_date):
        parser.print_help()
        logger.warning("Please specify an action: either --fetch-listings or both --start-date and --end-date.")

    logger.info("Done!")