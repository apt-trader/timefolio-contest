# financials_fetcher.py
# Fetch, store, and compute financial-statement–based factors via OpenDARTReader (DART OpenAPI)

import os
import sqlite3
import logging
import time
from datetime import datetime, timedelta
import pandas as pd
from OpenDartReader.dart import OpenDartReader
from dotenv import load_dotenv
import argparse

logger = logging.getLogger(__name__)

# --- Rate Limiter for OpenDARTReader API ---
from collections import deque

class RateLimiter:
    """
    Simple token-bucket style rate limiter to prevent exceeding OpenDARTReader limits.
    Allows up to max_per_minute calls per 60-second window and max_per_day per 24-hour window.
    """
    def __init__(self, max_per_minute: int = 900, max_per_day: int = 20000):
        self.max_per_minute = max_per_minute
        self.max_per_day = max_per_day
        self.calls_minute = deque()
        self.calls_day = deque()

    def wait(self):
        now = time.time()
        # Remove entries older than 60 seconds
        while self.calls_minute and now - self.calls_minute[0] > 60:
            self.calls_minute.popleft()
        # Remove entries older than 24 hours
        while self.calls_day and now - self.calls_day[0] > 86400:
            self.calls_day.popleft()
        # If at minute limit, sleep until a slot frees
        if len(self.calls_minute) >= self.max_per_minute:
            sleep_time = 60 - (now - self.calls_minute[0]) + 0.1
            time.sleep(sleep_time)
        # If at daily limit, raise error
        if len(self.calls_day) >= self.max_per_day:
            raise RuntimeError("Daily OpenDARTReader API call limit reached")
        # Record this call
        self.calls_minute.append(now)
        self.calls_day.append(now)

class FinancialsFetcher:
    """
    A. Uses DART OpenAPI (via OpenDartReader) to pull corporate filings, XBRL statements, and financials
    B. Persists into SQLite [financials](cci:1://file:///Users/kmj/Desktop/Python/timefolio-2025-main/financial_fetcher.py:96:4-111:35) table
    C. Computes key financial factors for TimeFolio workflow
    """

    def __init__(self, db_path: str = "krx_data.db"):
        """
        Initialize FinancialsFetcher with database path.
        DART API key is loaded from environment variables.
        """
        # Load environment variables from .env file
        load_dotenv()
        
        # Get API key from environment
        self.api_key = os.getenv('DART_API_KEY')
        if not self.api_key:
            raise ValueError("DART_API_KEY not found in environment variables")
            
        self.dart = OpenDartReader(self.api_key)
        self.db_path = db_path
        self._init_financials_table()
        self.rate_limiter = RateLimiter()

    def _init_financials_table(self):
        """
        Initialize SQLite schema for raw financial data
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
            CREATE TABLE IF NOT EXISTS financials (
                code TEXT,
                report_date DATE,
                report_type TEXT,
                revenue REAL,
                net_income REAL,
                total_assets REAL,
                total_equity REAL,
                operating_cf REAL,
                capex REAL,
                dividend REAL,
                shares_outstanding INTEGER,
                PRIMARY KEY(code, report_date, report_type)
            )""")
            conn.commit()

    def find_corp_code(self, identifier: str) -> str:
        """
        Find corporation code by company name or stock code
        Args:
            identifier: Company name or stock code (e.g., '삼성전자' or '005930')
        Returns:
            str: Corporation code
        """
        self.rate_limiter.wait()
        try:
            return self.dart.find_corp_code(identifier)
        except Exception as e:
            print(f"Error finding corp code for {identifier}: {str(e)}")
            raise

    def fetch_financials(self, corp_code: str, year: int, report_type: str = '11013') -> pd.DataFrame:
        """
        Fetch financial statements for a given corporation and year
        """
        try:
            self.rate_limiter.wait()
        except RuntimeError:
            logger.warning("일일 호출 한도 초과")
            # Optional: send external notification here
            now = datetime.now()
            tomorrow = datetime(now.year, now.month, now.day) + timedelta(days=1)
            seconds = (tomorrow - now).total_seconds() + 1
            time.sleep(seconds)
            return None
        try:
            fs = self.dart.finstate_all(corp_code, year, report_type)
            if fs is not None and not fs.empty:
                self._store_financials(fs, corp_code, year, report_type)
            return fs
        except Exception as e:
            print(f"Error fetching financials for {corp_code}: {str(e)}")
            return None

    def _store_financials(self, df: pd.DataFrame, corp_code: str, year: int, report_type: str):
        """
        Store financial data in SQLite database
        """
        try:
            # Add metadata
            df['code'] = corp_code
            df['report_date'] = f"{year}-12-31"  # Default to year-end
            df['report_type'] = report_type
            
            # Store in database
            with sqlite3.connect(self.db_path) as conn:
                # Convert DataFrame to records
                records = df.to_dict('records')
                
                # Prepare columns and placeholders
                cols = list(records[0].keys())
                placeholders = ','.join(['?'] * len(cols))
                cols_str = ','.join(cols)
                
                # Prepare update clause for ON CONFLICT
                updates = [f"{col}=excluded.{col}" for col in cols if col not in ('code', 'report_date', 'report_type')]
                update_clause = ', '.join(updates)
                
                # Insert or update
                sql = f"""
                INSERT INTO financials ({cols_str})
                VALUES ({placeholders})
                ON CONFLICT(code, report_date, report_type) 
                DO UPDATE SET {update_clause}
                """
                
                # Execute for each record
                cur = conn.cursor()
                for record in records:
                    cur.execute(sql, [record.get(col) for col in cols])
                conn.commit()
                
        except Exception as e:
            print(f"Error storing financials: {str(e)}")
            raise

    def load_financials(self, codes: list, as_of_date: str) -> pd.DataFrame:
        """
        Load the most recent financials up to as_of_date for each code
        """
        try:
            con = sqlite3.connect(self.db_path)
            placeholders = ','.join(['?'] * len(codes))
            sql = f"""
            SELECT f.* FROM financials f
            WHERE f.code IN ({placeholders})
              AND f.report_date = (
                SELECT MAX(report_date) FROM financials
                WHERE code=f.code AND report_date <= ?)
            """
            df = pd.read_sql(sql, con, params=codes + [as_of_date], 
                           parse_dates=['report_date'])
            return df.set_index('code')
        except Exception as e:
            print(f"Error loading financials: {str(e)}")
            raise
        finally:
            con.close()

    def compute_financial_factors(self, price_df: pd.DataFrame, as_of_date: str) -> pd.DataFrame:
        """
        Compute financial factors for given stocks
        """
        try:
            # Load financial data
            fin = self.load_financials(price_df.index.tolist(), as_of_date)
            if fin.empty:
                raise ValueError("No financial data found for the given codes")
                
            # Join with price data
            df = price_df[['close']].join(fin, how='inner')
            
            # Compute value factors
            df['pb'] = (fin['total_equity'] / fin['shares_outstanding']) / df['close']
            df['ep'] = (fin['net_income'] / fin['shares_outstanding']) / df['close']
            df['dividend_yield'] = fin['dividend'] / df['close']
            df['cf_yield'] = (fin['operating_cf'] / fin['shares_outstanding']) / df['close']
            df['market_cap'] = df['close'] * fin['shares_outstanding']
            
            # Compute quality factors
            df['roe'] = fin['net_income'] / fin['total_equity']
            df['roa'] = fin['net_income'] / fin['total_assets']
            df['leverage'] = fin['total_assets'] / fin['total_equity']
            
            return df
            
        except Exception as e:
            print(f"Error computing financial factors: {str(e)}")
            raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch financial statements for a given company over a date range."
    )
    parser.add_argument(
        "--identifier", "-i",
        required=True,
        help="Company name or stock code (e.g., '삼성전자' or '005930')"
    )
    parser.add_argument(
        "--start-date", "-s",
        required=True,
        help="Start date in YYYYMMDD format"
    )
    parser.add_argument(
        "--end-date", "-e",
        required=True,
        help="End date in YYYYMMDD format"
    )
    args = parser.parse_args()

    # Convert dates to years
    start_year = int(datetime.strptime(args.start_date, "%Y%m%d").year)
    end_year   = int(datetime.strptime(args.end_date,   "%Y%m%d").year)

    # Initialize fetcher
    fetcher = FinancialsFetcher(db_path="krx_data.db")
    # Find corporation code
    corp_code = fetcher.find_corp_code(args.identifier)
    # Loop through years and fetch
    for year in range(start_year, end_year + 1):
        fetcher.fetch_financials(corp_code, year)