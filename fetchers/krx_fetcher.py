"""
KRX Data Fetcher Module

This module provides a robust, polite interface for fetching historical market data from KRX.
It handles throttling with jitter, file-level caching, error recovery, and data persistence.
"""
#
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
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS daily_prices (
                code TEXT, date DATE, open REAL, high REAL, low REAL, close REAL,
                volume INTEGER, value INTEGER, market_cap REAL, market TEXT, name TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, PRIMARY KEY (code, date)
            )""")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_daily_prices_code ON daily_prices (code)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_daily_prices_date ON daily_prices (date)")
    
    def _get_otp(self, mkt: str, date: str) -> Optional[str]:
        payload = {
            "bld": "dbms/MDC/STAT/standard/MDCSTAT01501", "mktId": mkt, "trdDd": date,
            "share": "1", "money": "1", "csvxls_isNo": "false", "name": "fileDown",
            "url": "dbms/MDC/STAT/standard/MDCSTAT01501"
        }
        for attempt in range(self.max_retries + 1):
            try:
                self._throttle()
                logger.debug(f"Requesting OTP for {mkt} on {date} (attempt {attempt+1})")
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
        """Helper function to download the raw bytes of the CSV file."""
        if not otp: return None
        try:
            self._throttle()
            logger.debug(f"Downloading raw data with OTP: {otp}")
            response = self.session.post(DOWNLOAD_URL, data={"code": otp}, timeout=60, stream=True)
            response.raise_for_status()
            content = response.content
            if not content:
                logger.warning("Received empty file, likely a non-trading day.")
                return None
            return content
        except Exception as e:
            logger.error(f"Failed to download raw data: {e}")
            return None

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
                    otp = self._get_otp(mkt, date_str)
                    raw_bytes = self._download_data_raw(otp)
                    
                    if raw_bytes:
                        df = pd.read_csv(BytesIO(raw_bytes), encoding='EUC-KR', dtype={'종목코드': str})
                        with open(cache_file, 'wb') as f:
                            f.write(raw_bytes)
                    else:
                        logger.warning(f"No data downloaded for {mkt} on {date_str}.")
                        df = pd.DataFrame()
                
                if df.empty:
                    results[mkt] = df
                    continue

                column_map = {'종목코드': 'code', '종목명': 'name', '시장구분': 'market_name', '시가': 'open', '고가': 'high', '저가': 'low', '종가': 'close', '거래량': 'volume', '거래대금': 'value', '시가총액': 'market_cap'}
                df.rename(columns=column_map, inplace=True)
                df['date'] = date_dt
                df['market'] = mkt
                df['code'] = df['code'].str.zfill(6)
                
                numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'value', 'market_cap']
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
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
        all_data = []

        for date in tqdm(date_range, desc="Fetching Historical Market Data"):
            date_str = date.strftime('%Y%m%d')
            if date_str in existing_dates:
                logger.debug(f"Date {date_str} already exists in database. Skipping.")
                continue
            
            daily_results = self.fetch_daily_data(date, markets)
            for df in daily_results.values():
                if not df.empty:
                    all_data.append(df)

        if not all_data:
            logger.warning("No new data was fetched for the given period.")
            return pd.DataFrame()

        return pd.concat(all_data, ignore_index=True)

    def update_database(self, df: pd.DataFrame) -> int:
        if df.empty: return 0
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        db_columns = ['code', 'date', 'open', 'high', 'low', 'close', 'volume', 'value', 'market_cap', 'name', 'market']
        df_to_save = df.reindex(columns=db_columns)
        with self.conn:
            cursor = self.conn.cursor()
            try:
                cursor.execute("BEGIN TRANSACTION")
                placeholders = ', '.join(['?'] * len(df_to_save.columns))
                columns_str = ', '.join(f'"{c}"' for c in df_to_save.columns)
                update_cols = [c for c in df_to_save.columns if c not in ('code', 'date')]
                update_clause = ', '.join([f'"{c}"=excluded."{c}"' for c in update_cols])
                sql = f"INSERT INTO daily_prices ({columns_str}) VALUES ({placeholders}) ON CONFLICT(code, date) DO UPDATE SET {update_clause}, updated_at=CURRENT_TIMESTAMP"
                cursor.executemany(sql, df_to_save.to_records(index=False).tolist())
                count = cursor.rowcount
                cursor.execute("COMMIT")
                logger.info(f"Successfully updated/inserted {count} rows in the database.")
                return count
            except Exception as e:
                cursor.execute("ROLLBACK")
                logger.error(f"Database update failed: {e}", exc_info=True)
                raise

    def get_available_dates(self) -> pd.DatetimeIndex:
        with self.conn:
            df = pd.read_sql("SELECT DISTINCT date FROM daily_prices ORDER BY date", self.conn, parse_dates=['date'])
        return pd.DatetimeIndex(df['date'])

    def __del__(self):
        if self.conn: self.conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fetch KRX market data')
    parser.add_argument('-s', '--start-date', required=True, help='Start date (YYYYMMDD)')
    parser.add_argument('-e', '--end-date', required=True, help='End date (YYYYMMDD)')
    parser.add_argument('--market', default='ALL', choices=['STK', 'KSQ', 'ALL'], help='Market to fetch (STK, KSQ, ALL)')
    parser.add_argument('--update-db', action='store_true', help='Update the database with fetched data.')
    parser.add_argument('--force-refetch', action='store_true', help='Force re-fetch of data even if it exists in the database.')
    args = parser.parse_args()

    fetcher = KRXDataFetcher()
    markets = ['STK', 'KSQ'] if args.market == 'ALL' else [args.market]
    
    df = fetcher.fetch_historical_data(args.start_date, args.end_date, markets, force_refetch=args.force_refetch)
    
    if df.empty:
        logger.info("No new data to update in the database.")
    elif args.update_db:
        fetcher.update_database(df)
    else:
        print("\n--- Fetched Data Sample ---")
        print(df[['code', 'name', 'date', 'open', 'close', 'volume', 'value', 'market_cap']].head())
    
    logger.info("Done!")