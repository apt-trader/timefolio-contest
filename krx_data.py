# krx_data.py
import pandas as pd
import requests
from datetime import datetime, timedelta
from pathlib import Path
import time
import logging
from typing import Optional, List, Dict, Any
from io import BytesIO
import sqlite3
from contextlib import contextmanager
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('krx_data')

# Constants
BASE_URL = "https://data.krx.co.kr/comm/fileDn"
GEN_OTP_URL = f"{BASE_URL}/GenerateOTP/generate.cmd"
DOWNLOAD_URL = f"{BASE_URL}/download_csv/download.cmd"
HEADERS = {
    "Referer": "https://data.krx.co.kr/",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
}

# Database setup
DB_PATH = Path("krx_data.db")

@contextmanager
def get_db_connection():
    """Context manager for SQLite database connections"""
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.row_factory = sqlite3.Row
        yield conn
    finally:
        conn.close()

def init_db():
    """Initialize the SQLite database"""
    with get_db_connection() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS krx_daily (
            code TEXT,
            date DATE,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            value INTEGER,
            mkt TEXT,
            name TEXT,
            PRIMARY KEY (code, date)
        )
        """)
        conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_krx_daily_code ON krx_daily (code);
        """)
        conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_krx_daily_date ON krx_daily (date);
        """)

class KRXDataFetcher:
    def __init__(self, cache_dir: str = "krx_cache", throttle: float = 0.25):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.throttle = throttle
        self.last_request = 0
        
    def _throttle(self):
        """Enforce request throttling"""
        elapsed = time.time() - self.last_request
        if elapsed < self.throttle:
            time.sleep(self.throttle - elapsed)
        self.last_request = time.time()
        
    def _get_otp(self, mkt: str, date: str) -> str:
        """Get OTP for data download"""
        params = {
            "mktId": mkt,
            "trdDd": date,
            "money": "1",
            "csvxls_isNo": "false",
            "name": "fileDown",
            "url": "dbms/MDC/STAT/standard/MDCSTAT03901"
        }
        self._throttle()
        try:
            response = self.session.get(GEN_OTP_URL, params=params, timeout=10)
            response.raise_for_status()
            return response.text.strip()
        except requests.RequestException as e:
            logger.error(f"Failed to get OTP for {mkt} {date}: {e}")
            raise
            
    def _download_data(self, otp: str) -> pd.DataFrame:
        """Download and parse CSV data using OTP"""
        try:
            response = self.session.post(DOWNLOAD_URL, data={"code": otp}, timeout=30)
            response.raise_for_status()
            
            # Handle EUC-KR encoding and parse CSV
            df = pd.read_csv(BytesIO(response.content), encoding='EUC-KR')
            
            # Standardize column names
            column_map = {
                '종목코드': 'code',
                '종목명': 'name',
                '시가': 'open',
                '고가': 'high',
                '저가': 'low',
                '종가': 'close',
                '거래량': 'volume',
                '거래대금': 'value',
                '시가총액': 'market_cap',
                '상장주식수': 'listed_shares'
            }
            df = df.rename(columns=column_map)
            
            # Convert numeric columns
            numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'value', 'market_cap', 'listed_shares']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to download/parse data: {e}")
            raise
            
    def fetch_daily_universe(self, date: str, markets: List[str] = None) -> Dict[str, pd.DataFrame]:
        """Fetch daily data for all tickers in specified markets"""
        if markets is None:
            markets = ["STK", "KSQ"]  # KOSPI and KOSDAQ
            
        results = {}
        for mkt in markets:
            try:
                otp = self._get_otp(mkt, date)
                df = self._download_data(otp)
                df['date'] = pd.to_datetime(date)
                df['mkt'] = mkt
                results[mkt] = df
                logger.info(f"Fetched {len(df)} rows for {mkt} on {date}")
            except Exception as e:
                logger.error(f"Error fetching {mkt} on {date}: {e}")
                continue
                
        return results
        
    def fetch_historical_data(self, start_date: str, end_date: str = None, 
                            markets: List[str] = None, max_retries: int = 2) -> pd.DataFrame:
        """Fetch historical data for a date range"""
        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')
            
        date_range = pd.date_range(start_date, end_date, freq='B')
        all_data = []
        
        for date in date_range:
            date_str = date.strftime('%Y%m%d')
            for attempt in range(max_retries + 1):
                try:
                    results = self.fetch_daily_universe(date_str, markets)
                    for mkt, df in results.items():
                        all_data.append(df)
                    break  # Success, move to next date
                except Exception as e:
                    if attempt == max_retries:
                        logger.error(f"Failed to fetch {date_str} after {max_retries} attempts: {e}")
                    else:
                        logger.warning(f"Attempt {attempt + 1} failed for {date_str}, retrying...")
                        time.sleep(1)  # Back off before retry
                        
        if not all_data:
            return pd.DataFrame()
            
        return pd.concat(all_data, ignore_index=True)
        
    def update_database(self, df: pd.DataFrame) -> int:
        """Update SQLite database with new data"""
        if df.empty:
            return 0
            
        with get_db_connection() as conn:
            df.to_sql('krx_daily', conn, if_exists='append', index=False)
            return len(df)

def main():
    # Initialize database
    init_db()
    
    # Create fetcher with 300ms throttle (be nice to KRX servers)
    fetcher = KRXDataFetcher(throttle=0.3)
    
    # Example: Fetch last 30 trading days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=60)  # Go back extra days to account for weekends/holidays
    
    logger.info(f"Fetching data from {start_date.date()} to {end_date.date()}")
    
    # Fetch data
    df = fetcher.fetch_historical_data(
        start_date=start_date.strftime('%Y%m%d'),
        end_date=end_date.strftime('%Y%m%d'),
        markets=['STK', 'KSQ']
    )
    
    if not df.empty:
        # Update database
        count = fetcher.update_database(df)
        logger.info(f"Updated {count} rows in database")
    else:
        logger.warning("No data was fetched")

if __name__ == "__main__":
    main()