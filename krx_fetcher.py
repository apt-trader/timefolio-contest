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

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('krx_fetcher.log')
    ]
)
logger = logging.getLogger('krx_fetcher')

# Constants
BASE_URL = "https://data.krx.co.kr/comm/fileDn"
GEN_OTP_URL = f"{BASE_URL}/GenerateOTP/generate.cmd"
DOWNLOAD_URL = f"{BASE_URL}/download_csv/download.cmd"
DEFAULT_HEADERS = {
    "Referer": "https://data.krx.co.kr/contents/MDC/MDI/mdiLoader",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
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
    
    def __init__(self, cache_dir: str = "krx_cache", db_path: str = str(Path(__file__).parent / "krx_data.db"), 
                 throttle: float = 0.25, max_retries: int = 3):
        """
        Initialize the KRX data fetcher.
        
        Args:
            cache_dir: Directory to store cached data files
            db_path: Path to SQLite database file
            throttle: Minimum seconds between requests
            max_retries: Maximum number of retry attempts for failed requests
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.db_path = db_path
        # Persistent SQLite connection
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.throttle = throttle
        self.max_retries = max_retries
        self.last_request = 0
        
        # Set up requests session with retry strategy
        self.session = self._create_session()
        
        # Initialize database
        self._init_database()
    
    def _create_session(self) -> requests.Session:
        """Create and configure a requests session with retry logic."""
        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[408, 429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
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
                volume INTEGER, value INTEGER, market_cap REAL, -- ADDED market_cap
                market TEXT, name TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (code, date)
            )
            """)
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_daily_prices_code ON daily_prices (code)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_daily_prices_date ON daily_prices (date)")
            self.conn.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")
    
    def _get_otp(self, mkt: str, date: str) -> str:
        """
        Get OTP for data download.
        
        Args:
            mkt: Market identifier ('STK' for KOSPI, 'KSQ' for KOSDAQ)
            date: Trading date in 'YYYYMMDD' format
            
        Returns:
            str: One-time password for download
        """
        params = {
            "mktId": mkt,
            "trdDd": date,
            "money": "1",
            "csvxls_isNo": "false",
            "name": "fileDown",
            "url": "dbms/MDC/STAT/standard/MDCSTAT03901"
        }
        
        for attempt in range(self.max_retries + 1):
            try:
                self._throttle()
                logger.debug(f"Requesting OTP for {mkt} on {date} (attempt {attempt + 1})")
                
                response = self.session.get(
                    GEN_OTP_URL,
                    params=params,
                    timeout=30
                )
                response.raise_for_status()
                
                otp = response.text.strip()
                if not otp:
                    raise ValueError("Empty OTP received")
                    
                logger.debug(f"Successfully received OTP for {mkt} on {date}")
                return otp
                
            except (requests.RequestException, ValueError) as e:
                if attempt == self.max_retries:
                    logger.error(f"Failed to get OTP after {self.max_retries} attempts: {e}")
                    raise
                logger.warning(f"Attempt {attempt + 1} failed, retrying...")
                time.sleep(1)  # Back off before retry
    
    def _download_data(self, otp: str) -> pd.DataFrame:
        """
        Download and parse market data using OTP.
        
        Args:
            otp: One-time password from _get_otp()
            
        Returns:
            pd.DataFrame: Parsed market data
        """
        try:
            self._throttle()
            logger.debug("Downloading data with OTP")
            
            response = self.session.post(
                DOWNLOAD_URL,
                data={"code": otp},
                timeout=60,
                stream=True
            )
            response.raise_for_status()
            
            # Read and parse the CSV data
            df = pd.read_csv(BytesIO(response.content), encoding='EUC-KR')
            
            # Standardize column names
            column_map = {
                '종목코드': 'code',
                '종목명': 'name',
                '시장구분': 'market_name',
                '업종명': 'sector',
                '시가': 'open',
                '고가': 'high',
                '저가': 'low',
                '종가': 'close',
                '거래량': 'volume',
                '거래대금': 'value',
                '등락률': 'change_rate',
                '시가총액': 'market_cap',
                '상장주식수': 'listed_shares',
                '시가총액비중': 'market_share',
                '상장주식수': 'listed_shares',
                '외국인보유수량': 'foreign_holdings',
                '외국인한도수량': 'foreign_limit',
                '외국인한도소진율': 'foreign_limit_ratio',
                '대용가': 'par_value',
                'PER': 'per',
                'ROE': 'roe'
            }
            
            # Rename columns that exist in the dataframe
            df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})
            
            # Convert numeric columns
            numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'value', 'market_cap', 'listed_shares', 'foreign_holdings', 'foreign_limit', 'foreign_limit_ratio', 'par_value', 'per', 'roe', 'change_rate']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(
                        df[col].astype(str).str.replace(',', ''), 
                        errors='coerce'
                    )
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to download/parse data: {e}")
            raise
    
    def fetch_daily_data(self, date, markets: List[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Fetch daily data for all tickers in specified markets.
        
        Args:
            date: Trading date (datetime or 'YYYYMMDD' string)
            markets: List of market identifiers (default: ['STK', 'KSQ'])
            
        Returns:
            Dict mapping market codes to DataFrames with required columns:
            - code: Stock code
            - name: Stock name
            - open: Opening price
            - high: Highest price
            - low: Lowest price
            - close: Closing price
            - volume: Trading volume
            - value: Trading value
            - date: Trading date
            - market: Market identifier (STK/KSQ)
        """
        if markets is None:
            markets = ["STK", "KSQ"]  # KOSPI and KOSDAQ
            
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
                    logger.warning(f"No data returned for {mkt} on {date_str}")
                    continue
                
                # Ensure we have all required columns with proper types
                required_columns = {
                    'code': str,
                    'name': str,
                    'open': float,
                    'high': float,
                    'low': float,
                    'close': float,
                    'volume': int,
                    'value': int,
                    'date': 'datetime64[ns]',
                    'market': str
                }
                
                # Add missing columns with default values
                for col, dtype in required_columns.items():
                    if col not in df.columns:
                        if dtype == str:
                            df[col] = ''
                        elif dtype == float:
                            df[col] = 0.0
                        elif dtype == int:
                            df[col] = 0
                            
                # Ensure proper data types
                df['code'] = df['code'].astype(str).str.strip()
                df['name'] = df['name'].astype(str).str.strip()
                
                for col in ['open', 'high', 'low', 'close']:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                    
                df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0).astype(int)
                df['value'] = pd.to_numeric(df.get('value', 0), errors='coerce').fillna(0).astype(int)
                
                # Add metadata
                df['date'] = date_dt
                df['market'] = mkt
                
                # Select and order columns
                final_columns = ['code', 'name', 'open', 'high', 'low', 'close', 'volume', 'value', 'date', 'market']
                df = df[final_columns].copy()

                results[mkt] = df
                logger.info(f"Fetched {len(df)} rows for {mkt} on {date_str}")
            except Exception as e:
                logger.error(f"Error fetching {mkt} data for {date_str}: {e}", exc_info=True)
                if mkt in results:
                    del results[mkt]
                
        return results
    
    def fetch_historical_data(self, start_date: Union[str, datetime], 
                             end_date: Union[str, datetime] = None,
                             markets: List[str] = None) -> pd.DataFrame:
        """
        Fetch historical data for a date range.
        
        Args:
            start_date: Start date (inclusive)
            end_date: End date (inclusive, defaults to today)
            markets: List of market identifiers (default: ['STK', 'KSQ'])
            
        Returns:
            Combined DataFrame with data for all dates and markets
        """
        if end_date is None:
            end_date = datetime.now()

        # Convert to datetime objects if they're strings
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y%m%d')
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y%m%d')

        # Skip dates already present in database for incremental updates
        existing_dates = set(d.strftime('%Y%m%d') for d in self.get_available_dates())

        # Generate business days in the range
        date_range = pd.bdate_range(start_date, end_date)
        all_data = []

        for date in date_range:
            date_str = date.strftime('%Y%m%d')
            if date_str in existing_dates:
                logger.info(f"Date {date_str} already exists in database, skipping incremental fetch")
                continue
            try:
                results = self.fetch_daily_data(date, markets)
                for df in results.values():
                    all_data.append(df)
            except Exception as e:
                logger.error(f"Error processing {date}: {e}")
                continue

        if not all_data:
            logger.warning("No data was fetched")
            return pd.DataFrame()

        return pd.concat(all_data, ignore_index=True)
    
    def update_database(self, df: pd.DataFrame) -> int:
        """Update the SQLite database with new data, including market_cap."""
        if df.empty:
            return 0
            
        # Add market_cap to the list of required columns
        required_columns = ['code', 'date', 'open', 'high', 'low', 'close', 'volume', 'market_cap']
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"DataFrame is missing required columns for DB update: {missing_cols}")
        
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        
        # Define all columns to be saved
        db_columns = ['code', 'date', 'open', 'high', 'low', 'close', 'volume', 'value', 'market_cap', 'name', 'market']
        df_to_save = df.reindex(columns=db_columns) # Ensure all columns are present, filling missing with NaN
        
        with self.conn:
            cursor = self.conn.cursor()
            cursor.execute("BEGIN TRANSACTION")
            try:
                placeholders = ', '.join(['?'] * len(df_to_save.columns))
                columns_str = ', '.join(f'"{col}"' for col in df_to_save.columns)
                update_clause = ', '.join([f'"{col}"=excluded."{col}"' for col in df_to_save.columns if col not in ('code', 'date')])
                
                sql = f"INSERT INTO daily_prices ({columns_str}) VALUES ({placeholders}) ON CONFLICT(code, date) DO UPDATE SET {update_clause}"
                
                cursor.executemany(sql, df_to_save.to_records(index=False).tolist())
                count = cursor.rowcount
                cursor.execute("COMMIT")
                logger.debug(f"Updated {count} rows in database for date {df['date'].iloc[0]}")
                return count
            except Exception as e:
                cursor.execute("ROLLBACK")
                logger.error(f"Error updating database: {e}")
                raise
    
    def get_available_dates(self) -> pd.DatetimeIndex:
        """Get all dates with available data in the database."""
        with self.conn:
            conn = self.conn
            df = pd.read_sql(
                "SELECT DISTINCT date FROM daily_prices ORDER BY date",
                conn,
                parse_dates=['date']
            )
        return pd.DatetimeIndex(df['date'])
    
    def get_stock_data(self, code: str, start_date: str = None, 
                      end_date: str = None) -> pd.DataFrame:
        """
        Get historical data for a specific stock.
        
        Args:
            code: Stock code (with or without leading zeros)
            start_date: Start date (inclusive) in 'YYYYMMDD' format
            end_date: End date (inclusive) in 'YYYYMMDD' format
            
        Returns:
            DataFrame with historical data for the stock, indexed by date
        """
        # Ensure code is a string and strip any leading/trailing whitespace
        code = str(code).strip()
        
        # Build the query
        query = """
        SELECT * FROM daily_prices 
        WHERE code = ?
        """
        params = [code]
        
        # Add date filters if provided
        if start_date:
            # Convert YYYYMMDD to YYYY-MM-DD format for SQL
            start_date_formatted = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:8]}"
            query += " AND date >= ?"
            params.append(start_date_formatted)
            
        if end_date:
            # Convert YYYYMMDD to YYYY-MM-DD format for SQL
            end_date_formatted = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:8]}"
            query += " AND date <= ?"
            params.append(end_date_formatted)
            
        query += " ORDER BY date"
        
        try:
            with self.conn:
                conn = self.conn
                df = pd.read_sql(
                    query,
                    conn,
                    params=params,
                    parse_dates=['date'],
                    index_col='date'
                )
            
            # Ensure we have the expected columns
            expected_columns = ['code', 'name', 'open', 'high', 'low', 'close', 'volume', 'value', 'market_cap', 'name', 'market']
            for col in expected_columns:
                if col not in df.columns:
                    df[col] = None
            
            # Convert numeric columns to appropriate types
            numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'value']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Ensure code is string type
            if 'code' in df.columns:
                df['code'] = df['code'].astype(str).str.strip()
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching stock data for {code}: {str(e)}")
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=expected_columns + ['date']).set_index('date')


    def __del__(self):
        try:
            self.conn.close()
        except Exception:
            pass


def main():
    """Example usage of the KRXDataFetcher."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Fetch KRX market data')
    parser.add_argument('-s', '--start-date', required=True,
                        type=lambda s: datetime.strptime(s, '%Y%m%d'),
                        help='Start date (YYYYMMDD)')
    parser.add_argument('-e', '--end-date', required=True,
                        type=lambda s: datetime.strptime(s, '%Y%m%d'),
                        help='End date (YYYYMMDD)')
    parser.add_argument('--market', default='ALL', choices=['STK', 'KSQ', 'ALL'], 
                       help='Market to fetch (STK=KOSPI, KSQ=KOSDAQ, ALL=both)')
    parser.add_argument('--update-db', action='store_true', help='Update database')
    args = parser.parse_args()

    # Normalize to datetime
    start_date = args.start_date
    end_date   = args.end_date
    
    # Initialize fetcher
    fetcher = KRXDataFetcher(throttle=0.3)
    
    # Set markets
    markets = ['STK', 'KSQ'] if args.market == 'ALL' else [args.market]
    
    # Fetch data
    logger.info(f"Fetching data from {start_date.date()} to {end_date.date()} for markets: {', '.join(markets)}")
    df = fetcher.fetch_historical_data(start_date, end_date, markets)
    
    if df.empty:
        logger.warning("No data was fetched")
        return
    
    # Update database if requested
    if args.update_db:
        fetcher.update_database(df)
    else:
        # Just show a preview
        print(f"\nFetched {len(df)} rows of data")
        print("\nSample data:")
        print(df[['code', 'name', 'open', 'high', 'low', 'close', 'volume']].head())
    
    logger.info("Done!")


if __name__ == "__main__":
    main()
