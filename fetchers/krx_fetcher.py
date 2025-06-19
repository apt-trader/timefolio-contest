"""
KRX Data Fetcher Module

This module provides a robust interface for fetching historical market data from KRX
using their OTP-based download system. It handles throttling, error recovery,
and data persistence.
"""
#
import time
import sqlite3
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Union
from io import BytesIO
import pandas as pd
import requests
from requests.adapters import HTTPAdapter, Retry

# Ensure logs directory exists in the project root
logs_dir = Path('logs')
logs_dir.mkdir(exist_ok=True)

# Set up logging
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
    
    This class handles:
    - Session management with retries
    - Request throttling
    - Data caching
    - Error handling and recovery
    - Database persistence
    """
    
    def __init__(self, cache_dir: str = None, db_path: str = str(Path(__file__).parent.parent / "krx_data.db"), 
                 throttle: float = 0.5, max_retries: int = 3):
        # Set cache directory to be within the fetchers directory if not specified
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
        """Create and configure a requests session with retry logic."""
        session = requests.Session()
        retry_strategy = Retry(
            total=3, backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        session.headers.update(DEFAULT_HEADERS)
        return session
    
    def _throttle(self):
        """Enforce request throttling to respect rate limits."""
        elapsed = time.time() - self.last_request
        if elapsed < self.throttle:
            time.sleep(self.throttle - elapsed)
        self.last_request = time.time()
    
    def _init_database(self):
        """Initialize the SQLite database with required tables."""
        with self.conn:
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS daily_prices (
                code TEXT, date DATE,
                open REAL, high REAL, low REAL, close REAL,
                volume INTEGER, value INTEGER, market_cap REAL,
                market TEXT, name TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (code, date)
            )
            """)
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_daily_prices_code ON daily_prices (code)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_daily_prices_date ON daily_prices (date)")
    
    def _get_otp(self, mkt: str, date: str) -> str:
        """Get OTP for data download."""
        payload = {
            "bld": "dbms/MDC/STAT/standard/MDCSTAT01501", 
            "mktId": mkt,
            "trdDd": date,
            "share": "1",
            "money": "1",
            "csvxls_isNo": "false",
            "name": "fileDown",
            "url": "dbms/MDC/STAT/standard/MDCSTAT01501"
        }
        
        for attempt in range(self.max_retries + 1):
            try:
                self._throttle()
                logger.debug(f"Requesting OTP for {mkt} on {date} (attempt {attempt + 1})")
                response = self.session.post(GEN_OTP_URL, data=payload, timeout=30)
                response.raise_for_status()
                otp = response.text.strip()
                if not otp:
                    raise ValueError("Empty OTP received")
                logger.debug(f"Successfully received OTP '{otp}' for {mkt} on {date}")
                return otp
            except (requests.RequestException, ValueError) as e:
                if attempt == self.max_retries:
                    logger.error(f"Failed to get OTP after {self.max_retries+1} attempts: {e}")
                    raise
                logger.warning(f"Attempt {attempt + 1} to get OTP failed, retrying... Error: {e}")
                time.sleep(attempt + 1)
    
    def _download_data(self, otp: str) -> pd.DataFrame:
        """
        Download data using OTP and return a raw DataFrame.
        Handles non-trading days by returning an empty DataFrame.
        """
        try:
            self._throttle()
            logger.debug(f"Downloading data with OTP: {otp}")
            response = self.session.post(DOWNLOAD_URL, data={"code": otp}, timeout=60, stream=True)
            response.raise_for_status()
            
            # Return the raw DataFrame from the CSV content
            return pd.read_csv(BytesIO(response.content), encoding='EUC-KR')

        except pd.errors.EmptyDataError:
            # This is an expected condition for non-trading days (holidays)
            logger.warning("Received empty data, which is expected for a non-trading day. Skipping.")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"An unexpected error occurred while downloading data: {e}", exc_info=True)
            raise

    def fetch_daily_data(self, date, markets: List[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Fetch, process, and clean daily data for all tickers in specified markets.
        This function is now the single source of truth for data processing.
        """
        if markets is None:
            markets = ["STK", "KSQ"]
            
        if isinstance(date, datetime):
            date_str = date.strftime('%Y%m%d')
            date_dt = date
        else:
            date_str = date
            date_dt = datetime.strptime(date_str, '%Y%m%d')
            
        results = {}
        for mkt in markets:
            try:
                logger.info(f"Fetching {mkt} data for {date_str}")
                otp = self._get_otp(mkt, date_str)
                df = self._download_data(otp)
                
                if df.empty:
                    logger.warning(f"No data returned for {mkt} on {date_str} (likely a holiday).")
                    continue
                
                # FIX: Consolidate all data cleaning and processing here.
                
                # 1. Define the authoritative column map
                column_map = {
                    '종목코드': 'code', 
                    '종목명': 'name', 
                    '시장구분': 'market_name', 
                    '시가': 'open', 
                    '고가': 'high', 
                    '저가': 'low', 
                    '종가': 'close', 
                    '거래량': 'volume', 
                    '거래대금': 'value', 
                    '시가총액': 'market_cap'
                }
                df.rename(columns=column_map, inplace=True)

                # 2. Add metadata columns
                df['date'] = date_dt
                df['market'] = mkt
                
                # 3. Standardize stock code format
                df['code'] = df['code'].astype(str).str.zfill(6)
                
                # 4. Convert all numeric columns, coercing errors to NaN
                numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'value', 'market_cap']
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # 5. Select and order the final set of columns
                final_columns = ['code', 'name', 'date', 'market', 'open', 'high', 'low', 'close', 'volume', 'value', 'market_cap']
                df = df.reindex(columns=final_columns)

                # 6. Fill any NaN values that resulted from coercion (e.g., for suspended stocks)
                for col in ['open', 'high', 'low', 'close', 'market_cap']:
                    df[col] = df[col].fillna(0.0)
                for col in ['volume', 'value']:
                     df[col] = df[col].fillna(0).astype(int)

                results[mkt] = df
                logger.info(f"Fetched and processed {len(df)} rows for {mkt} on {date_str}")
            except Exception as e:
                logger.error(f"Critical error processing {mkt} data for {date_str}: {e}", exc_info=True)
                if mkt in results:
                    del results[mkt]
        return results

    def fetch_historical_data(self, start_date: Union[str, datetime], 
                             end_date: Union[str, datetime] = None,
                             markets: List[str] = None, force_refetch: bool = False) -> pd.DataFrame:
        """Fetch historical data for a date range."""
        if end_date is None: end_date = datetime.now()
        if isinstance(start_date, str): start_date = datetime.strptime(start_date, '%Y%m%d')
        if isinstance(end_date, str): end_date = datetime.strptime(end_date, '%Y%m%d')

        existing_dates = set()
        if not force_refetch:
            existing_dates = set(d.strftime('%Y%m%d') for d in self.get_available_dates())

        date_range = pd.bdate_range(start_date, end_date)
        all_data = []

        for date in date_range:
            date_str = date.strftime('%Y%m%d')
            if date_str in existing_dates:
                logger.info(f"Date {date_str} already exists in database, skipping incremental fetch.")
                continue
            try:
                results = self.fetch_daily_data(date, markets)
                all_data.extend(results.values())
            except Exception as e:
                logger.error(f"Error processing {date_str}: {e}")
                continue

        if not all_data:
            logger.warning("No new data was fetched for the given period.")
            return pd.DataFrame()

        return pd.concat(all_data, ignore_index=True)
    
    def update_database(self, df: pd.DataFrame) -> int:
        """Update the SQLite database with new data."""
        if df.empty: return 0
        
        required_columns = ['code', 'date', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"DataFrame is missing required columns for DB update.")
        
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        
        db_columns = ['code', 'date', 'open', 'high', 'low', 'close', 'volume', 'value', 'market_cap', 'name', 'market']
        df_to_save = df.reindex(columns=db_columns)
        
        with self.conn:
            cursor = self.conn.cursor()
            try:
                cursor.execute("BEGIN TRANSACTION")
                placeholders = ', '.join(['?'] * len(df_to_save.columns))
                columns_str = ', '.join(f'"{col}"' for col in db_columns)
                update_cols = [col for col in db_columns if col not in ('code', 'date')]
                update_clause = ', '.join([f'"{col}"=excluded."{col}"' for col in update_cols])
                
                sql = f"INSERT INTO daily_prices ({columns_str}) VALUES ({placeholders}) ON CONFLICT(code, date) DO UPDATE SET {update_clause}, updated_at=CURRENT_TIMESTAMP"
                
                cursor.executemany(sql, df_to_save.to_records(index=False).tolist())
                count = cursor.rowcount
                cursor.execute("COMMIT")
                logger.info(f"Successfully updated/inserted {count} rows in the database.")
                return count
            except Exception as e:
                cursor.execute("ROLLBACK")
                logger.error(f"Database update failed: {e}")
                raise
    
    def get_available_dates(self) -> pd.DatetimeIndex:
        """Get all dates with available data in the database."""
        df = pd.read_sql("SELECT DISTINCT date FROM daily_prices ORDER BY date", self.conn, parse_dates=['date'])
        return pd.DatetimeIndex(df['date'])
    
    def get_stock_data(self, code: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Get historical data for a specific stock."""
        code = str(code).strip().zfill(6)
        query = "SELECT * FROM daily_prices WHERE code = ?"
        params = [code]
        if start_date:
            query += " AND date >= ?"
            params.append(f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:8]}")
        if end_date:
            query += " AND date <= ?"
            params.append(f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:8]}")
        query += " ORDER BY date"
        
        df = pd.read_sql(query, self.conn, params=params, parse_dates=['date'], index_col='date')
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'value', 'market_cap']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df

    def __del__(self):
        if self.conn: self.conn.close()

def main():
    """Example usage of the KRXDataFetcher."""
    import argparse
    parser = argparse.ArgumentParser(description='Fetch KRX market data')
    parser.add_argument('-s', '--start-date', required=True, help='Start date (YYYYMMDD)')
    parser.add_argument('-e', '--end-date', required=True, help='End date (YYYYMMDD)')
    parser.add_argument('--market', default='ALL', choices=['STK', 'KSQ', 'ALL'], help='Market to fetch (STK, KSQ, ALL)')
    parser.add_argument('--update-db', action='store_true', help='Update the database with fetched data.')
    parser.add_argument('--force-refetch', action='store_true', help='Force re-fetch of data even if it exists in the database.')
    args = parser.parse_args()

    fetcher = KRXDataFetcher()
    markets = ['STK', 'KSQ'] if args.market == 'ALL' else [args.market]
    
    logger.info(f"Fetching data from {args.start_date} to {args.end_date} for markets: {', '.join(markets)}")
    df = fetcher.fetch_historical_data(args.start_date, args.end_date, markets, force_refetch=args.force_refetch)
    
    if df.empty:
        logger.info("No new data to update in the database.")
        return
    
    if args.update_db:
        fetcher.update_database(df)
    else:
        print("\n--- Fetched Data Sample ---")
        print(df[['code', 'name', 'date', 'open', 'close', 'volume', 'value', 'market_cap']].head())
    
    logger.info("Done!")

if __name__ == "__main__":
    main()