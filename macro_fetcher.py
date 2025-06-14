import sqlite3
import logging
import os
import time
import random
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from functools import wraps
from dotenv import load_dotenv

import pandas as pd
import yfinance as yf
from fredapi import Fred
from requests.exceptions import RequestException

# Load environment variables
load_dotenv()
FRED_API_KEY = os.getenv('FRED_API_KEY')
if not FRED_API_KEY:
    raise ValueError("FRED_API_KEY not found in .env file")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent / "krx_data.db"

def rate_limited(max_per_second: float = 0.2):
    """Decorator to limit the number of calls to a function per second."""
    min_interval = 1.0 / max_per_second
    
    def decorate(func):
        last_called = [0.0]
        
        @wraps(func)
        def wrapped(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            if elapsed < min_interval:
                sleep_time = min_interval - elapsed
                time.sleep(sleep_time)
            result = func(*args, **kwargs)
            last_called[0] = time.time()
            return result
        return wrapped
    return decorate


def retry_with_backoff(retries: int = 3, backoff_in_seconds: float = 1.0):
    """Decorator to retry a function with exponential backoff."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retry_count = 0
            while retry_count < retries:
                try:
                    return func(*args, **kwargs)
                except RequestException as e:
                    retry_count += 1
                    if retry_count == retries:
                        logger.error(f"Max retries ({retries}) reached. Last error: {str(e)}")
                        raise
                    sleep_time = backoff_in_seconds * (2 ** (retry_count - 1)) + random.uniform(0, 1)
                    logger.warning(f"Request failed (attempt {retry_count}/{retries}). Retrying in {sleep_time:.2f}s...")
                    time.sleep(sleep_time)
        return wrapper
    return decorator


class MacroFetcher:
    def __init__(self, fred_api_key: str):
        self.fred = Fred(api_key=fred_api_key)
        self.db_path = str(DB_PATH)
        # Persistent SQLite connection
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._init_macro_table()
        # Track last API call time for rate limiting
        self._last_api_call = 0
        self._min_interval = 5.0  # Minimum seconds between API calls

    def _init_macro_table(self):
        """Initialize the macro_data table if it doesn't exist"""
        with self.conn:
            # Check if table exists and has the correct schema
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='macro_data'
            """)
            table_exists = cursor.fetchone() is not None
            
            if table_exists:
                # Check if table has the expected columns
                cursor.execute("PRAGMA table_info(macro_data)")
                columns = [col[1] for col in cursor.fetchall()]
                expected_columns = {'date', 'eq_mom_1m', 'eq_mom_3m', 'vix', 
                                  'us10y', 'us2y', 'us3m', 'us10y3m', 'us10y2y'}
                
                if not expected_columns.issubset(columns):
                    # Table exists but has wrong schema - recreate it
                    logger.info("Old schema detected, recreating macro_data table")
                    cursor.execute("DROP TABLE macro_data")
                    table_exists = False
            
            if not table_exists:
                # Create new table with updated schema
                cursor.execute("""
                CREATE TABLE macro_data (
                    date DATE PRIMARY KEY,
                    eq_mom_1m REAL,
                    eq_mom_3m REAL,
                    vix REAL,
                    us10y REAL,
                    us2y REAL,
                    us3m REAL,
                    us10y3m REAL,
                    us10y2y REAL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """)
                logger.info("Created new macro_data table with updated schema")
            
            self.conn.commit()

    @rate_limited(max_per_second=0.2)  # 1 call per 5 seconds
    @retry_with_backoff(retries=5, backoff_in_seconds=2.0)
    def _safe_fred_request(self, series_id: str, as_of_date: str) -> float:
        """Make a single FRED API request with error handling"""
        try:
            # Add a small jitter to avoid hitting rate limits
            time.sleep(random.uniform(0.1, 0.5))
            
            # Get the series data
            data = self.fred.get_series(series_id, as_of_date, as_of_date)
            
            if data.empty or pd.isna(data.iloc[0]):
                logger.warning(f"No data found for {series_id} on {as_of_date}")
                return None
                
            return float(data.iloc[0])
            
        except Exception as e:
            logger.error(f"Error fetching {series_id} for {as_of_date}: {str(e)}")
            raise
    
    def _fetch_fred_data(self, series_id: str, as_of_date: str) -> Optional[float]:
        """Helper method to fetch data from FRED API with rate limiting and retries"""
        try:
            # Add a small jitter to avoid hitting rate limits
            time.sleep(random.uniform(0.1, 0.5))
            
            # Get the series data
            return self._safe_fred_request(series_id, as_of_date)
        except Exception as e:
            logger.error(f"Failed to fetch {series_id} after retries: {str(e)}")
            return None
    
    @rate_limited(max_per_second=0.5)  # 2 calls per second for yfinance
    def _fetch_kospi_momentum(self, as_of_date: str) -> Tuple[Optional[float], Optional[float]]:
        """
        Fetch KOSPI momentum data
        
        Args:
            as_of_date: Date in 'YYYY-MM-DD' format
            
        Returns:
            Tuple of (1-month momentum, 3-month momentum) as percentages
            Returns (None, None) if data is not available
        """
        try:
            # First try to get data from the database
            try:
                kospi = pd.read_sql(
                    """
                    SELECT date, close FROM daily_prices
                    WHERE code='KOSPI' AND date <= ?
                    ORDER BY date DESC LIMIT 121
                    """,
                    self.conn,
                    params=[as_of_date]
                )
                
                if not kospi.empty:
                    kospi['date'] = pd.to_datetime(kospi['date'])
                    kospi = kospi.sort_values('date').set_index('date')
                    
                    latest_close = kospi.iloc[-1]['close']
                    
                    # Calculate 1M momentum (21 trading days)
                    mom_1m = (latest_close / kospi.iloc[-min(22, len(kospi))]['close'] - 1) * 100 \
                             if len(kospi) >= 2 else None
                            
                    # 3M momentum (63 trading days)
                    mom_3m = (latest_close / kospi.iloc[-min(63, len(kospi))]['close'] - 1) * 100 \
                             if len(kospi) >= 2 else None
                    
                    return mom_1m, mom_3m
                    
            except (sqlite3.OperationalError, pd.errors.DatabaseError) as e:
                logger.warning(f"Could not fetch KOSPI data from database: {str(e)}")
                
            # Fallback: Try to fetch recent KOSPI data from yfinance
            try:
                logger.info("Fetching KOSPI data from yfinance...")
                end_date = (pd.to_datetime(as_of_date) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
                start_date = (pd.to_datetime(as_of_date) - pd.Timedelta(days=180)).strftime('%Y-%m-%d')
                
                kospi = yf.download('^KS11', start=start_date, end=end_date, progress=False, auto_adjust=True)
                if kospi.empty:
                    raise ValueError("No KOSPI data found from yfinance")
                
                kospi = kospi[['Close']].rename(columns={'Close': 'close'})
                kospi = kospi.sort_index()
                
                latest_close = kospi.iloc[-1]['close']
                
                # Calculate 1M momentum (21 trading days)
                mom_1m = (latest_close / kospi.iloc[-min(22, len(kospi))]['close'] - 1) * 100 \
                         if len(kospi) >= 2 else None
                        
                # 3M momentum (63 trading days)
                mom_3m = (latest_close / kospi.iloc[-min(63, len(kospi))]['close'] - 1) * 100 \
                         if len(kospi) >= 2 else None
                
                return mom_1m, mom_3m
                
            except Exception as yf_error:
                logger.warning(f"Could not fetch KOSPI data from yfinance: {str(yf_error)}")
                return None, None
                
        except Exception as e:
            logger.error(f"Error in _fetch_kospi_momentum: {str(e)}")
            return None, None

    @rate_limited(max_per_second=0.5)  # 2 calls per second for yfinance
    def _fetch_vix(self, as_of_date: str) -> float:
        """Fetch VIX data from Yahoo Finance"""
        try:
            # Add random jitter before starting
            time.sleep(random.uniform(0.1, 0.5))
            logger.info("Fetching VIX data from Yahoo Finance...")
            end_date = (pd.to_datetime(as_of_date) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
            vix_data = yf.Ticker('^VIX').history(start=as_of_date, end=end_date)
            
            if vix_data.empty:
                raise ValueError("No VIX data available for the specified date")
                
            return float(vix_data['Close'].iloc[0])
            
        except Exception as e:
            logger.error(f"Error fetching VIX data: {str(e)}")
            raise

    def _fetch_bond_momentum(self, as_of_date: str) -> Tuple[float, float]:
        """Fetch 10-year Treasury yield momentum over 1m and 3m from FRED."""
        # 30-day and 90-day lookbacks
        date_dt = pd.to_datetime(as_of_date)
        start_1m = (date_dt - timedelta(days=30)).strftime('%Y-%m-%d')
        start_3m = (date_dt - timedelta(days=90)).strftime('%Y-%m-%d')
        # 1-month momentum
        series_1m = self.fred.get_series('DGS10', start_1m, as_of_date)
        bond_mom_1m = float(series_1m.iloc[-1] / series_1m.iloc[0] - 1)
        # 3-month momentum
        series_3m = self.fred.get_series('DGS10', start_3m, as_of_date)
        bond_mom_3m = float(series_3m.iloc[-1] / series_3m.iloc[0] - 1)
        return bond_mom_1m, bond_mom_3m

    def fetch_and_store(self, as_of_date: str) -> bool:
        """
        Fetch macroeconomic data and store it in the database
        
        Args:
            as_of_date: Date in 'YYYY-MM-DD' format
            
        Returns:
            bool: True if operation was successful, False otherwise
        """
        logger.info(f"Starting macro data update for {as_of_date}")
        # Skip if entry already exists for incremental updates
        cursor = self.conn.cursor()
        cursor.execute("SELECT 1 FROM macro_data WHERE date = ?", (as_of_date,))
        if cursor.fetchone():
            logger.info(f"Macro data for {as_of_date} already exists, skipping incremental update")
            return True
        # Validate date format
        try:
            as_of_date_dt = pd.to_datetime(as_of_date).date()
        except Exception as e:
            logger.error(f"Invalid date format: {as_of_date}. Expected 'YYYY-MM-DD'", exc_info=True)
            return False

        try:
            # 1) Fetch FRED data
            dgs10 = self._fetch_fred_data('DGS10', as_of_date)
            dgs2  = self._fetch_fred_data('DGS2',  as_of_date)
            try:
                # Try to fetch KOSPI momentum data (this is optional)
                try:
                    eq_mom_1m, eq_mom_3m = self._fetch_kospi_momentum(as_of_date)
                    if eq_mom_1m is not None and eq_mom_3m is not None:
                        logger.info(f"Fetched KOSPI momentum: 1M={float(eq_mom_1m.iloc[0]) if hasattr(eq_mom_1m, 'iloc') else float(eq_mom_1m):.2f}%, 3M={float(eq_mom_3m.iloc[0]) if hasattr(eq_mom_3m, 'iloc') else float(eq_mom_3m):.2f}%")
                    else:
                        logger.warning("Could not fetch KOSPI momentum data")
                except Exception as e:
                    logger.warning(f"Error fetching KOSPI momentum: {str(e)}")
                    eq_mom_1m, eq_mom_3m = None, None
                    
                # Fetch VIX data
                vix = self._fetch_vix(as_of_date)
                if vix is not None:
                    logger.info(f"Fetched VIX: {vix:.2f}")
                    
                # FRED data - skip weekends and holidays
                date_dt = pd.to_datetime(as_of_date)
                if date_dt.weekday() >= 5:  # Skip weekends (5=Saturday, 6=Sunday)
                    logger.info(f"Skipping {as_of_date} (weekend)")
                    return True
                    
                # Check if it's a US market holiday (no FRED updates on holidays)
                holiday_dates = [
                    '2022-01-17', '2022-02-21', '2022-04-15', '2022-05-30',
                    '2022-06-20', '2022-07-04', '2022-09-05', '2022-11-24',
                    '2022-12-26',  # 2022 holidays
                    '2023-01-02', '2023-01-16', '2023-02-20', '2023-04-07',
                    '2023-05-29', '2023-06-19', '2023-07-04', '2023-09-04',
                    '2023-11-23', '2023-12-25'  # 2023 holidays
                ]
                
                if as_of_date in holiday_dates:
                    logger.info(f"Skipping {as_of_date} (US market holiday)")
                    return True
                    
                logger.info(f"Fetching FRED data for {as_of_date}")
                us10y = self._fetch_fred_data('DGS10', as_of_date)  # 10-Year Treasury Constant Maturity Rate
                us2y = self._fetch_fred_data('DGS2', as_of_date)    # 2-Year Treasury Constant Maturity Rate
                us3m = self._fetch_fred_data('DGS3MO', as_of_date)  # 3-Month Treasury Bill
                
                if us10y is not None and us3m is not None:
                    us10y3m = us10y - us3m  # 10Y-3M Spread
                    logger.info(f"10Y-3M Spread: {us10y3m:.2f}%")
                    
                if us10y is not None and us2y is not None:
                    us10y2y = us10y - us2y  # 10Y-2Y Spread
                    logger.info(f"10Y-2Y Spread: {us10y2y:.2f}%")
                
                # Prepare data for insertion, ensuring all values are native Python types
                data = {
                    'date': as_of_date,
                    'eq_mom_1m': float(eq_mom_1m) if eq_mom_1m is not None else None,
                    'eq_mom_3m': float(eq_mom_3m) if eq_mom_3m is not None else None,
                    'vix': float(vix) if vix is not None else None,
                    'us10y': float(us10y) if us10y is not None else None,
                    'us2y': float(us2y) if us2y is not None else None,
                    'us3m': float(us3m) if us3m is not None else None,
                    'us10y3m': float(us10y3m) if us10y3m is not None else None,
                    'us10y2y': float(us10y2y) if us10y2y is not None else None
                }
                
                # Store in database
                with self.conn:
                    cursor = self.conn.cursor()
                    # First, delete any existing entry for this date
                    cursor.execute("DELETE FROM macro_data WHERE date = ?", (as_of_date,))
                    
                    # Insert new data
                    cursor.execute("""
                    INSERT INTO macro_data (
                        date, eq_mom_1m, eq_mom_3m, vix, 
                        us10y, us2y, us3m, us10y3m, us10y2y
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        data['date'],
                        data['eq_mom_1m'],
                        data['eq_mom_3m'],
                        data['vix'],
                        data['us10y'],
                        data['us2y'],
                        data['us3m'],
                        data['us10y3m'],
                        data['us10y2y']
                    ))
                    self.conn.commit()
                
                logger.info(f"Successfully updated macro data for {as_of_date}")
                return True
                
            except Exception as e:
                logger.error(f"Error during data fetching: {str(e)}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to update macro data for {date_str}: {str(e)}")
            return False

if __name__ == "__main__":
    if not FRED_API_KEY:
        raise ValueError("FRED_API_KEY environment variable is not set")
        
    parser = argparse.ArgumentParser(
        description="Fetch macroeconomic data over a date range."
    )
    parser.add_argument(
        "--start-date", "-s",
        required=True,
        help="Start date in YYYY-MM-DD format"
    )
    parser.add_argument(
        "--end-date", "-e",
        required=True,
        help="End date in YYYY-MM-DD format"
    )
    args = parser.parse_args()

    fetcher = MacroFetcher(FRED_API_KEY)
    # Generate daily dates between start and end
    for dt in pd.date_range(start=args.start_date, end=args.end_date, freq='D'):
        date_str = dt.strftime('%Y-%m-%d')
        try:
            fetcher.fetch_and_store(date_str)
        except Exception as e:
            logger.error(f"Error on {date_str}: {e}")
    def __del__(self):
        try:
            self.conn.close()
        except:
            pass