# fetchers/macro_fetcher.py

import sqlite3
import logging
import os
import time
import random
import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional
from functools import wraps
from dotenv import load_dotenv

import pandas as pd
import yfinance as yf
from fredapi import Fred
from requests.exceptions import RequestException
import pandas_market_calendars as mcal

# Load environment variables
load_dotenv()
FRED_API_KEY = os.getenv('FRED_API_KEY')
if not FRED_API_KEY:
    raise ValueError("FRED_API_KEY not found in .env file or environment variables")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent / "db" / "krx_data.db"

def rate_limited(max_per_second: float):
    """Decorator to limit function calls per second."""
    min_interval = 1.0 / max_per_second
    def decorate(func):
        last_called = [0.0]
        @wraps(func)
        def wrapped(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
            result = func(*args, **kwargs)
            last_called[0] = time.time()
            return result
        return wrapped
    return decorate

def retry_with_backoff(retries: int = 3, backoff_in_seconds: float = 2.0):
    """Decorator to retry a function with exponential backoff."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for i in range(retries):
                try:
                    return func(*args, **kwargs)
                except RequestException as e:
                    if i == retries - 1:
                        logger.error(f"Max retries ({retries}) reached. Last error: {e}")
                        raise
                    sleep_time = backoff_in_seconds * (2 ** i) + random.uniform(0, 1)
                    logger.warning(f"Request failed (attempt {i+1}/{retries}). Retrying in {sleep_time:.2f}s...")
                    time.sleep(sleep_time)
        return wrapper
    return decorator

class MacroFetcher:
    def __init__(self, fred_api_key: str):
        self.fred = Fred(api_key=fred_api_key)
        self.db_path = str(DB_PATH)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._init_macro_table()

    def _init_macro_table(self):
        """Initialize the macro_data table with a verified schema."""
        with self.conn:
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS macro_data (
                date TEXT PRIMARY KEY,
                eq_mom_1m REAL, eq_mom_3m REAL, vix REAL,
                us10y REAL, us2y REAL, us3m REAL,
                us10y3m REAL, us10y2y REAL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""")
            logger.info("Verified 'macro_data' table schema.")

    @rate_limited(max_per_second=0.2)
    @retry_with_backoff()
    def _fetch_fred_data(self, series_id: str, as_of_date: str) -> Optional[float]:
        """Safely fetches a single data point from FRED."""
        data = self.fred.get_series(series_id, as_of_date, as_of_date)
        if data.empty or pd.isna(data.iloc[0]):
            logger.warning(f"No data for FRED series '{series_id}' on {as_of_date}.")
            return None
        return data.iloc[0]

    @rate_limited(max_per_second=2)
    def _fetch_kospi_momentum(self, as_of_date: str) -> Tuple[Optional[float], Optional[float]]:
        """Fetch KOSPI momentum data using yfinance, ensuring native float types are returned."""
        try:
            end_dt = pd.to_datetime(as_of_date) + pd.Timedelta(days=1)
            start_dt = end_dt - pd.Timedelta(days=180)
            
            kospi = yf.download(
                '^KS11',
                start=start_dt.strftime('%Y-%m-%d'),
                end=end_dt.strftime('%Y-%m-%d'),
                progress=False, auto_adjust=True
            )
            
            if kospi.empty or 'Close' not in kospi.columns:
                logger.warning("yfinance returned no KOSPI data or is missing the 'Close' column.")
                return None, None
            
            kospi_close = kospi['Close'].dropna()
            
            mom_1m = None
            mom_3m = None

            if len(kospi_close) >= 22:
                latest = kospi_close.iloc[-1].item()
                past = kospi_close.iloc[-22].item()
                mom_1m = (latest / past - 1) * 100 if past != 0 else 0.0

            if len(kospi_close) >= 64:
                latest = kospi_close.iloc[-1].item()
                past = kospi_close.iloc[-64].item()
                mom_3m = (latest / past - 1) * 100 if past != 0 else 0.0

            return (mom_1m, mom_3m)

        except Exception as e:
            logger.error(f"Could not fetch KOSPI momentum: {e}")
            return None, None

    @rate_limited(max_per_second=2)
    def _fetch_vix(self, as_of_date: str) -> Optional[float]:
        """Fetch VIX close price from Yahoo Finance."""
        try:
            end_dt = pd.to_datetime(as_of_date) + pd.Timedelta(days=1)
            vix_data = yf.Ticker('^VIX').history(start=as_of_date, end=end_dt.strftime('%Y-%m-%d'))
            if vix_data.empty:
                raise ValueError("yfinance returned no VIX data")
            return vix_data['Close'].iloc[0].item()
        except Exception as e:
            logger.error(f"Could not fetch VIX data: {e}")
            return None

    def fetch_and_store(self, as_of_date: str):
        """Fetch all macroeconomic data for a given date and store it in the database."""
        logger.info(f"Processing macro data for {as_of_date}")
        
        if self.conn.execute("SELECT 1 FROM macro_data WHERE date = ?", (as_of_date,)).fetchone():
            logger.info(f"Data for {as_of_date} already exists. Skipping.")
            return

        nyse = mcal.get_calendar('NYSE')
        if not nyse.valid_days(start_date=as_of_date, end_date=as_of_date).size:
            logger.info(f"Skipping {as_of_date} as it's a US market holiday or weekend.")
            return

        eq_mom_1m, eq_mom_3m = self._fetch_kospi_momentum(as_of_date)
        vix = self._fetch_vix(as_of_date)
        us10y = self._fetch_fred_data('DGS10', as_of_date)
        us2y = self._fetch_fred_data('DGS2', as_of_date)
        us3m = self._fetch_fred_data('DGS3MO', as_of_date)

        us10y3m = (us10y - us3m) if us10y is not None and us3m is not None else None
        us10y2y = (us10y - us2y) if us10y is not None and us2y is not None else None

        data = (as_of_date, eq_mom_1m, eq_mom_3m, vix, us10y, us2y, us3m, us10y3m, us10y2y)

        sql = "INSERT OR REPLACE INTO macro_data (date, eq_mom_1m, eq_mom_3m, vix, us10y, us2y, us3m, us10y3m, us10y2y) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
        with self.conn:
            self.conn.execute(sql, data)
        logger.info(f"Successfully stored macro data for {as_of_date}.")

    def __del__(self):
        if self.conn:
            self.conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch macroeconomic data over a date range.")
    parser.add_argument("-s", "--start-date", required=True, help="Start date in YYYY-MM-DD format")
    parser.add_argument("-e", "--end-date", required=True, help="End date in YYYY-MM-DD format")
    args = parser.parse_args()

    fetcher = MacroFetcher(FRED_API_KEY)
    for dt in pd.date_range(start=args.start_date, end=args.end_date):
        fetcher.fetch_and_store(dt.strftime('%Y-%m-%d'))