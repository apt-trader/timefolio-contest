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
import yaml
from pathlib import Path
from typing import Optional
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Load sector universe file from config
config_path = Path(__file__).parent / "config.yaml"
with open(config_path) as f:
    cfg = yaml.safe_load(f)
universe_file = cfg['data_settings']['stock_universe_file']

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

    def __init__(self, db_path: str = str(Path(__file__).parent / "krx_data.db")):
        """
        Initialize FinancialsFetcher with database path.
        DART API key is loaded from environment variables.
        """
        # Load environment variables from .env file
        loaded = load_dotenv()
        if not loaded:
            logger.warning(".env file not found or failed to load")
        
        # Get API key from environment
        self.api_key = os.getenv('DART_API_KEY')
        if not self.api_key:
            raise ValueError("DART_API_KEY not found in environment variables")
            
        self.dart = OpenDartReader(self.api_key)
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._init_financials_table()
        # Cache existing DB columns to avoid repeated PRAGMA calls
        self.existing_columns = self._get_existing_columns()
        self.rate_limiter = RateLimiter()

    def _init_financials_table(self):
        """
        Initialize SQLite schema for raw financial data with all possible columns from DART API
        """
        # Create new table with all possible columns
        self.conn.execute("""
            -- Basic info
            rcept_no TEXT,
            reprt_code TEXT,
            bsns_year TEXT,
            corp_code TEXT,
            
            -- Statement info
            sj_div TEXT,
            sj_nm TEXT,
            fs_div TEXT,
            fs_nm TEXT,
            fs_sn TEXT,
            
            -- Account info
            account_id TEXT,
            account_nm TEXT,
            account_detail TEXT,
            
            -- Current term (thstrm)
            thstrm_nm TEXT,
            thstrm_amount REAL,
            thstrm_add_amount REAL,
            thstrm_q_nm TEXT,
            thstrm_q_amount REAL,
            thstrm_q_add_amount REAL,
            
            -- Previous term (frmtrm)
            frmtrm_nm TEXT,
            frmtrm_amount REAL,
            frmtrm_add_amount REAL,
            frmtrm_q_nm TEXT,
            frmtrm_q_amount REAL,
            frmtrm_q_add_amount REAL,
            
            -- Term before previous (bfefrmtrm)
            bfefrmtrm_nm TEXT,
            bfefrmtrm_amount REAL,
            bfefrmtrm_q_nm TEXT,
            bfefrmtrm_q_amount REAL,
            
            -- Additional fields
            ord INTEGER,
            currency TEXT,
            
            -- English translations
            sj_nm_eng TEXT,
            account_nm_eng TEXT,
            account_detail_eng TEXT,
            thstrm_nm_eng TEXT,
            frmtrm_nm_eng TEXT,
            bfefrmtrm_nm_eng TEXT,
            fs_nm_eng TEXT,
            
            -- Japanese translations
            sj_nm_jp TEXT,
            account_nm_jp TEXT,
            account_detail_jp TEXT,
            thstrm_nm_jp TEXT,
            frmtrm_nm_jp TEXT,
            bfefrmtrm_nm_jp TEXT,
            fs_nm_jp TEXT,
            
            -- Chinese translations
            sj_nm_zh TEXT,
            account_nm_zh TEXT,
            account_detail_zh TEXT,
            thstrm_nm_zh TEXT,
            frmtrm_nm_zh TEXT,
            bfefrmtrm_nm_zh TEXT,
            fs_nm_zh TEXT,
            
            -- Metadata
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            
            -- Primary key
            PRIMARY KEY (rcept_no, account_id, fs_div, fs_sn)
        )""")
    def _get_existing_columns(self) -> list:
        """
        Return a list of existing column names in the financials table.
        """
        cursor = self.conn.execute("PRAGMA table_info(financials)")
        return [row[1] for row in cursor.fetchall()]
        
        # Create indexes for better query performance
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_financials_corp_code ON financials(corp_code)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_financials_bsns_year ON financials(bsns_year)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_financials_account_id ON financials(account_id)")
        
        # Create a trigger to update the updated_at timestamp
        self.conn.execute("""
        CREATE TRIGGER IF NOT EXISTS update_financials_timestamp
        AFTER UPDATE ON financials
        FOR EACH ROW
        BEGIN
            UPDATE financials SET updated_at = CURRENT_TIMESTAMP 
            WHERE rcept_no = NEW.rcept_no 
            AND account_id = NEW.account_id 
            AND fs_div = NEW.fs_div 
            AND fs_sn = NEW.fs_sn;
        END;
        """)
        
        self.conn.commit()

    def find_corp_code(self, identifier: str) -> Optional[str]:
        """
        Find corporation code by company name or stock code
        Args:
            identifier: Company name or stock code (e.g., '삼성전자' or '005930')
        Returns:
            str: Corporation code
        """
        logger.debug(f"Looking up corp code for identifier: {identifier}")
        # First check if it's a stock code
        if identifier.isdigit() and len(identifier) in (6, 7, 8):
            try:
                # Search by stock code
                logger.debug(f"Searching by stock code: {identifier}")
                corp_info = self.dart.company(identifier)
                logger.debug(f"Company info from DART: {corp_info}")
                if corp_info and 'corp_code' in corp_info:
                    logger.debug(f"Found corp code: {corp_info['corp_code']}")
                    return corp_info['corp_code']
                else:
                    logger.warning(f"No corp code found for stock code: {identifier}")
            except Exception as e:
                logger.error(f"Error finding corp code for {identifier}: {e}", exc_info=True)
        
        # If not found or not a stock code, try searching by company name
        try:
            logger.debug(f"Searching by company name: {identifier}")
            companies = self.dart.company_by_name(identifier)
            # Normalize return type to DataFrame
            if isinstance(companies, list):
                companies = pd.DataFrame(companies)
            logger.debug(f"Companies found: {companies}")
            if not companies.empty and 'corp_code' in companies.columns:
                corp_code = companies.iloc[0]['corp_code']
                logger.debug(f"Using corp code: {corp_code}")
                return corp_code
            else:
                logger.warning(f"No companies found for name: {identifier}")
        except Exception as e:
            logger.error(f"Error finding company by name {identifier}: {e}", exc_info=True)
            
        logger.error(f"Could not find corporation code for identifier: {identifier}")
        return None

    def fetch_financials(self, corp_code: str, year: int, report_type: str = '11013') -> Optional[pd.DataFrame]:
        """
        Fetch financial statements for a given corporation and year
        
        Args:
            corp_code: Corporation code
            year: Year to fetch data for
            report_type: Report type code (default: '11013' for 1분기보고서). Valid codes:
                '11013' (1분기보고서), '11012' (반기보고서), '11014' (3분기보고서), '11011' (사업보고서)
            
        Returns:
            DataFrame with financial data if successful, None otherwise
        """
        logger.info(f"Starting to fetch financials for corp_code={corp_code}, year={year}, report_type={report_type}")
        
        # Check rate limits before making the API call
        self.rate_limiter.wait()
        
        # Check if we've already fetched this data
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT 1 FROM financials 
            WHERE corp_code = ? AND bsns_year = ? AND reprt_code = ?
            LIMIT 1
        """, (corp_code, str(year), report_type))
        if cursor.fetchone() is not None:
            logger.debug(f"Financial data for {corp_code} {year} (report_type: {report_type}) already exists in database")
            return None
            
        # Check if we've exceeded daily rate limits
        now = datetime.now()
        tomorrow = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        if len(self.rate_limiter.calls_day) >= self.rate_limiter.max_per_day:
            logger.warning("Daily rate limit reached. Waiting until tomorrow...")
            logger.info(f"Next available in: {tomorrow - now}")
            seconds = (tomorrow - now).total_seconds() + 1
            time.sleep(seconds)
            return None

        logger.debug(f"Fetching financial statements for {corp_code} {year}...")
        try:
            # Try consolidated financials (CFS)
            logger.info(f"Making API call to DART for corp_code={corp_code}, year={year}, report_type={report_type}, fs_div=CFS")
            fs_all = self.dart.finstate_all(corp_code, year, report_type, fs_div='CFS')
            
            # Log the response
            logger.info(f"API Response Type: {type(fs_all)}")
            if isinstance(fs_all, pd.DataFrame):
                logger.info(f"DataFrame shape: {fs_all.shape}")
                if not fs_all.empty:
                    logger.info(f"Columns: {fs_all.columns.tolist()}")
                    logger.info(f"First few rows:\n{fs_all.head(2).to_string()}")
            else:
                logger.info(f"Response content: {fs_all}")
                
            # If DataFrame and it's empty, fallback to separate financials (OFS)
            if isinstance(fs_all, pd.DataFrame) and fs_all.empty:
                logger.info(f"No consolidated FS (CFS) data for {corp_code} {year}, trying separate FS (OFS)")
                fs_all = self.dart.finstate_all(corp_code, year, report_type, fs_div='OFS')
                
                # Log OFS response
                if isinstance(fs_all, pd.DataFrame):
                    logger.info(f"OFS DataFrame shape: {fs_all.shape}")
                    if not fs_all.empty:
                        logger.info(f"OFS Columns: {fs_all.columns.tolist()}")
                        logger.info(f"OFS First few rows:\n{fs_all.head(2).to_string()}")
                
            # Store the data if we have any
            if isinstance(fs_all, pd.DataFrame) and not fs_all.empty:
                success = self._store_financials(fs_all, corp_code, year, report_type)
                if success:
                    logger.info(f"Successfully stored financials for {corp_code} {year}")
                return fs_all
                
        except Exception as e:
            logger.error(f"Error processing {corp_code} {year}: {str(e)}", exc_info=True)
            
        logger.warning(f"No financial data found for {corp_code} {year} (report_type: {report_type})")
        return None

    def _store_financials(self, df: pd.DataFrame, corp_code: str, year: int, report_type: str) -> bool:
        """
        Store financial data in SQLite database with proper error handling and logging
        
        Args:
            df: DataFrame containing financial data
            corp_code: Corporation code
            year: Year of the report
            report_type: Type of the report
            
        Returns:
            bool: True if storage was successful, False otherwise
        """
        if df is None or df.empty:
            logger.warning("No data to store")
            return False
            
        try:
            # Add metadata
            df['corp_code'] = corp_code
            df['bsns_year'] = str(year)
            df['reprt_code'] = report_type
            
            # Ensure all required columns exist in the DataFrame
            required_columns = ['rcept_no', 'account_id', 'fs_div', 'fs_sn']
            for col in required_columns:
                if col not in df.columns:
                    logger.debug(f"Adding missing required column: {col}")
                    df[col] = None
            
            # Convert DataFrame to records
            records = df.to_dict('records')
            if not records:
                logger.warning("No records to store after conversion")
                return False
            
            # Get all unique columns from records
            all_columns = set()
            for record in records:
                all_columns.update(record.keys())
            
            # Ensure all columns exist in the table
            existing_columns = self.existing_columns
            for col in all_columns:
                if col not in existing_columns:
                    logger.debug(f"Adding new column to database: {col}")
                    try:
                        self.conn.execute(f"ALTER TABLE financials ADD COLUMN {col} TEXT")
                        self.conn.commit()
                        # Update cache
                        self.existing_columns.append(col)
                    except sqlite3.OperationalError as e:
                        logger.warning(f"Could not add column {col}: {str(e)}")
                        # Continue even if column addition fails
            
            # Prepare for batch insert
            columns = list(all_columns)
            if not columns:
                logger.error("No columns to insert")
                return False
                
            placeholders = ','.join(['?'] * len(columns))
            columns_str = ','.join([f'"{col}"' for col in columns])  # Quote column names
            
            # Prepare the INSERT OR REPLACE statement
            update_columns = [f'"{col}"=excluded."{col}"' for col in columns 
                            if col not in ('rcept_no', 'account_id', 'fs_div', 'fs_sn')]
            
            if not update_columns:
                logger.error("No columns to update in ON CONFLICT clause")
                return False
            
            sql = f"""
            INSERT INTO financials ({columns_str})
            VALUES ({placeholders})
            ON CONFLICT(rcept_no, account_id, fs_div, fs_sn) 
            DO UPDATE SET {','.join(update_columns)}
            """
            
            # Prepare values ensuring all records have all columns
            values = []
            for record in records:
                row = []
                for col in columns:
                    # Convert all values to string and handle None/NaN
                    val = record.get(col)
                    if pd.isna(val) or val is None or val == '':
                        row.append(None)
                    else:
                        row.append(str(val))
                values.append(tuple(row))
            
            if not values:
                logger.error("No valid values to insert")
                return False
            
            # Execute in chunks to avoid too many SQL variables
            chunk_size = 100
            total_inserted = 0
            
            for i in range(0, len(values), chunk_size):
                chunk = values[i:i + chunk_size]
                try:
                    self.conn.executemany(sql, chunk)
                    self.conn.commit()
                    total_inserted += len(chunk)
                except sqlite3.Error as e:
                    self.conn.rollback()
                    logger.error(f"Database error while inserting chunk {i//chunk_size + 1}: {str(e)}")
                    # Try to continue with next chunk
            
            if total_inserted > 0:
                logger.info(f"Successfully stored {total_inserted} records for {corp_code} {year}")
                return True
            else:
                logger.error(f"Failed to store any records for {corp_code} {year}")
                return False
            
        except Exception as e:
            logger.exception(f"Unexpected error storing financials for {corp_code} {year}")
            try:
                self.conn.rollback()
            except:
                pass  # Ignore errors during rollback
            return False

    def load_financials(self, codes: list, as_of_date: str) -> pd.DataFrame:
        """
        Load the most recent financials up to as_of_date for each code
        """
        try:
            placeholders = ','.join(['?'] * len(codes))
            sql = f"""
            SELECT f.* FROM financials f
            WHERE f.code IN ({placeholders})
              AND f.report_date = (
                SELECT MAX(report_date) FROM financials
                WHERE code=f.code AND report_date <= ?)
            """
            df = pd.read_sql(sql, self.conn, params=codes + [as_of_date], 
                           parse_dates=['report_date'])
            return df.set_index('code')
        except Exception as e:
            logger.exception("Error loading financials")
            raise

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
            logger.exception("Error computing financial factors")
            raise


    def __del__(self):
        try:
            self.conn.close()
        except Exception:
            pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch financial statements for a given company or entire universe."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-i", "--identifier",
        help="Single company name or stock code (e.g., '삼성전자' or '005930')"
    )
    group.add_argument(
        "-a", "--all",
        action="store_true",
        help="Fetch for all tickers listed in the sector universe CSV"
    )
    parser.add_argument(
        "-s", "--start-date",
        required=True,
        type=lambda s: datetime.strptime(s, "%Y%m%d"),
        help="Start date (YYYYMMDD)"
    )
    parser.add_argument(
        "-e", "--end-date",
        required=True,
        type=lambda s: datetime.strptime(s, "%Y%m%d"),
        help="End date (YYYYMMDD)"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO)"
    )
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    start_year = args.start_date.year
    end_year   = args.end_date.year
    fetcher    = FinancialsFetcher(db_path="krx_data.db")

    if args.all:
        import pandas as pd
        # Read the CSV with the correct column name '종목코드' (stock code)
        df = pd.read_csv(universe_file)
        # Remove 'A' prefix and ensure 6-digit format with leading zeros
        tickers = df['종목코드'].astype(str).str.lstrip('A').str.zfill(6).tolist()
        logger.info(f"Loaded {len(tickers)} tickers from {universe_file}")
        logger.info(f"Sample tickers: {tickers[:5]}...")  # Log first 5 tickers for verification
    else:
        tickers = [args.identifier]
        if not tickers[0].isdigit() or len(tickers[0]) != 6:
            raise ValueError("Ticker must be a 6-digit number")

    missing = []

    for ticker in tqdm(tickers, desc="Processing tickers"):
        logger.info(f"Processing ticker: {ticker}")
        try:
            corp_code = fetcher.find_corp_code(ticker)
            if not corp_code:
                logger.warning(f"No corporation code found for ticker: {ticker}")
                continue

            for year in range(start_year, end_year + 1):
                report_types = [
                    '11013',  # 1분기보고서 (Q1)
                    '11012',  # 반기보고서 (Half-year)
                    '11014',  # 3분기보고서 (Q3)
                    '11011',  # 사업보고서 (Annual)
                ]
                data_found = False
                for report_code in report_types:
                    logger.debug(f"Fetching report {report_code} for {ticker} year {year}")
                    result = fetcher.fetch_financials(corp_code, year, report_type=report_code)
                    if isinstance(result, pd.DataFrame) and not result.empty:
                        data_found = True
                if not data_found:
                    missing.append(f"{ticker}-{year}")

        except Exception as e:
            logger.error(f"Error processing ticker {ticker}: {str(e)}")
            continue

    if missing:
        logger.info(f"No financial data found for the following ticker-years: {', '.join(missing)}")