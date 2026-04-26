#fetchers/krx_fetcher.py
import time
import sqlite3
import logging
import random
import argparse
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Union, Optional
from io import BytesIO
import pandas as pd
import numpy as np
import requests
from requests.adapters import HTTPAdapter, Retry
from tqdm import tqdm

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, will use system env vars only

# Selenium imports for KakaoTalk login
try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.options import Options
    from selenium.common.exceptions import TimeoutException, NoSuchElementException
    SELENIUM_AVAILABLE = True
except ImportError as e:
    SELENIUM_AVAILABLE = False
    logging.warning(f"Selenium not fully available: {e}. Install with: pip install selenium")

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
KRX_LOGIN_URL = "https://data.krx.co.kr/contents/MDC/COMS/client/MDCCOMS001.cmd"
GEN_OTP_URL = "https://data.krx.co.kr/comm/fileDn/GenerateOTP/generate.cmd"
DOWNLOAD_URL = "https://data.krx.co.kr/comm/fileDn/download_csv/download.cmd"
DEFAULT_HEADERS = {
    "Referer": "https://data.krx.co.kr/contents/MDC/MDI/mdiLoader",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

class KRXDataFetcher:
    """
    A class to fetch and manage KRX market data with KakaoTalk OAuth login support.
    """
    def __init__(self, cache_dir: str = None, db_path: str = str(Path(__file__).parent.parent / "db" / "krx_data.db"), 
                 throttle: float = 0.5, max_retries: int = 3, 
                 kakao_email: str = None, kakao_password: str = None,
                 cookies_file: str = None, headless: bool = True):
        
        if cache_dir is None:
            self.cache_dir = Path(__file__).parent.parent / 'cache' / 'krx_cache'
        else:
            self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.throttle = throttle
        self.max_retries = max_retries
        self.last_request = 0
        
        # Authentication credentials
        self.kakao_email = kakao_email or os.getenv('KAKAO_EMAIL')
        self.kakao_password = kakao_password or os.getenv('KAKAO_PASSWORD')
        self.headless = headless
        
        # Cookies file for session persistence
        if cookies_file is None:
            self.cookies_file = self.cache_dir / 'krx_cookies.pkl'
        else:
            self.cookies_file = Path(cookies_file)
        
        self.session = self._create_session()
        self._authenticated = False
        
        # Try to load saved cookies first
        if self._load_cookies():
            logger.info("Loaded saved cookies, checking if still valid...")
            if not self._verify_authentication():
                logger.info("Saved cookies expired, will login with KakaoTalk")
                self._login_with_kakao()
        elif self.kakao_email and self.kakao_password:
            self._login_with_kakao()
        else:
            logger.warning("No credentials provided. Set KAKAO_EMAIL and KAKAO_PASSWORD environment variables.")
        
        self._init_database()
    
    def _create_session(self) -> requests.Session:
        session = requests.Session()
        retry_strategy = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        session.headers.update(DEFAULT_HEADERS)
        return session
    
    def _save_cookies(self):
        """Save session cookies to file for reuse."""
        try:
            with open(self.cookies_file, 'wb') as f:
                pickle.dump(self.session.cookies, f)
            logger.info(f"Saved cookies to {self.cookies_file}")
        except Exception as e:
            logger.error(f"Failed to save cookies: {e}")
    
    def _load_cookies(self) -> bool:
        """Load session cookies from file."""
        try:
            if self.cookies_file.exists():
                with open(self.cookies_file, 'rb') as f:
                    self.session.cookies.update(pickle.load(f))
                logger.info(f"Loaded cookies from {self.cookies_file}")
                return True
        except Exception as e:
            logger.error(f"Failed to load cookies: {e}")
        return False
    
    def _verify_authentication(self) -> bool:
        """Verify if current session is authenticated for both website and API."""
        try:
            # Check main website access
            response = self.session.get(
                "https://data.krx.co.kr/contents/MDC/MAIN/main/index.cmd", 
                timeout=10,
                allow_redirects=False
            )
            if response.status_code != 200 or 'login' in response.url.lower():
                logger.info("Main website session not authenticated")
                return False
            
            # IMPORTANT: Test API access with a RECENT date (yesterday or 2 days ago)
            # Using old dates might pass even with expired tokens due to caching
            logger.info("Verifying API access with test OTP request...")
            from datetime import datetime, timedelta
            test_date = (datetime.now() - timedelta(days=2)).strftime('%Y%m%d')
            
            test_otp = self._get_otp_without_retry("dbms/MDC/STAT/standard/MDCSTAT01501", 
                                                   mktId="STK", trdDd=test_date, share="1", money="1")
            
            if not test_otp or test_otp.strip().startswith('<html'):
                logger.info(f"API access not authenticated - OTP generation failed for {test_date}")
                return False
            
            self._authenticated = True
            logger.info("Session is authenticated for both website and API")
            return True
        except Exception as e:
            logger.error(f"Authentication verification failed: {e}")
            return False
    
    def _get_otp_without_retry(self, bld: str, **kwargs) -> Optional[str]:
        """Get OTP without retry logic - for verification only."""
        try:
            payload = {
                "bld": bld,
                "csvxls_isNo": "false",
                "name": "fileDown",
                "url": bld
            }
            payload.update(kwargs)
            response = self.session.post(GEN_OTP_URL, data=payload, timeout=10)
            
            # Check for 403 or HTML error pages
            if response.status_code == 403:
                logger.debug("OTP request returned 403 - API access expired")
                return None
            if response.status_code == 200 and not response.text.strip().startswith('<html'):
                return response.text.strip()
            return None
        except Exception as e:
            logger.debug(f"OTP verification request failed: {e}")
            return None
    
    def _login_with_kakao(self) -> bool:
        """Login using KakaoTalk OAuth via Selenium."""
        if not SELENIUM_AVAILABLE:
            logger.error("Selenium is required for KakaoTalk login. Install with: pip install selenium")
            return False
        
        if not self.kakao_email or not self.kakao_password:
            logger.error("KakaoTalk credentials not provided")
            return False
        
        driver = None
        try:
            logger.info("Starting KakaoTalk login via Selenium...")
            
            # Setup Chrome options
            chrome_options = Options()
            if self.headless:
                chrome_options.add_argument('--headless')
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument(f'user-agent={DEFAULT_HEADERS["User-Agent"]}')
            
            # Initialize driver with automatic version management
            try:
                from selenium.webdriver.chrome.service import Service
                from webdriver_manager.chrome import ChromeDriverManager
                driver = webdriver.Chrome(
                    service=Service(ChromeDriverManager().install()),
                    options=chrome_options
                )
            except ImportError:
                # Fallback to system ChromeDriver if webdriver-manager not installed
                logger.warning("webdriver-manager not installed, using system ChromeDriver. Install with: pip install webdriver-manager")
                driver = webdriver.Chrome(options=chrome_options)
            driver.set_page_load_timeout(30)
            
            # Navigate to KRX login page
            logger.info("Navigating to KRX login page...")
            driver.get(KRX_LOGIN_URL)
            time.sleep(2)
            
            # Click on KakaoTalk login button
            logger.info("Looking for KakaoTalk login button...")
            try:
                # Use the correct selector for KRX's Kakao login button
                kakao_button = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, "a.ms-kakao.jsSnsMember"))
                )
                kakao_button.click()
                logger.info("Clicked KakaoTalk login button")
                time.sleep(3)
            except TimeoutException:
                logger.error("Could not find KakaoTalk login button.")
                logger.info("Current page URL: " + driver.current_url)
                if not self.headless:
                    logger.info("Please manually click the Kakao login button...")
                    input("Press Enter after manually logging in...")
                else:
                    return False
            
            # Switch to KakaoTalk login window if popup
            if len(driver.window_handles) > 1:
                driver.switch_to.window(driver.window_handles[-1])
            
            # Wait for KakaoTalk login page
            logger.info("Waiting for KakaoTalk login page...")
            time.sleep(3)
            
            current_url = driver.current_url
            
            # Check if we're on Kakao login page
            if 'accounts.kakao.com' not in current_url:
                logger.info("Not on Kakao login page. Checking if already authenticated...")
                # Might already be logged in, try to extract cookies
            else:
                # We're on Kakao login page, try to enter credentials
                logger.info("On Kakao login page, attempting to enter credentials...")
                
                # Enter KakaoTalk email
                try:
                    email_input = WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='email'], input[name='loginId'], input[id='loginId']"))
                    )
                    email_input.clear()
                    email_input.send_keys(self.kakao_email)
                    logger.info("Entered email")
                except TimeoutException:
                    logger.warning("Could not find email input field - might already be logged in")
                    # Continue to cookie extraction
                
                # Enter KakaoTalk password
                try:
                    password_input = driver.find_element(By.CSS_SELECTOR, "input[type='password'], input[name='password'], input[id='password']")
                    password_input.clear()
                    password_input.send_keys(self.kakao_password)
                    logger.info("Entered password")
                    
                    # Click login button
                    login_button = driver.find_element(By.CSS_SELECTOR, "button[type='submit'], button.btn_g, .submit")
                    login_button.click()
                    logger.info("Clicked login button")
                except NoSuchElementException:
                    logger.warning("Could not find password/login button - might already be logged in")
            
            # Wait for redirect back to KRX or check if already logged in
            logger.info("Waiting for authentication to complete...")
            max_wait = 30
            for i in range(max_wait):
                time.sleep(1)
                current_url = driver.current_url
                
                # Check if we're back on KRX domain (login successful)
                if 'data.krx.co.kr' in current_url and 'login' not in current_url.lower():
                    logger.info("Successfully authenticated! Back on KRX domain.")
                    break
                    
                # Check if still on Kakao login page
                if 'accounts.kakao.com' in current_url:
                    continue
                    
            else:
                logger.warning("Timed out waiting for redirect. Checking authentication status...")
            
            # Try to navigate to KRX main page to verify login
            driver.get("https://data.krx.co.kr/contents/MDC/MAIN/main/index.cmd")
            time.sleep(3)
            
            # Extract cookies from Selenium and add to requests session
            logger.info("Extracting cookies from browser...")
            selenium_cookies = driver.get_cookies()
            for cookie in selenium_cookies:
                self.session.cookies.set(cookie['name'], cookie['value'], domain=cookie.get('domain'))
            
            # Also try to visit the data page to ensure we have all necessary cookies
            logger.info("Visiting data page to initialize session...")
            driver.get("https://data.krx.co.kr/contents/MDC/MDI/mdiLoader/index.cmd?menuId=MDC0201")
            time.sleep(3)
            
            # Re-extract cookies after visiting data page
            selenium_cookies = driver.get_cookies()
            for cookie in selenium_cookies:
                self.session.cookies.set(cookie['name'], cookie['value'], domain=cookie.get('domain'))
            
            logger.info("Successfully logged in with KakaoTalk")
            self._authenticated = True
            
            # Save cookies for future use
            self._save_cookies()
            
            return True
            
        except Exception as e:
            logger.error(f"KakaoTalk login failed: {e}", exc_info=True)
            return False
        
        finally:
            if driver:
                driver.quit()
    
    def _ensure_authenticated(self) -> bool:
        """Check if authenticated and attempt re-login if needed."""
        if not self._authenticated:
            if self.kakao_email and self.kakao_password:
                return self._login_with_kakao()
            else:
                logger.error("Not authenticated and no credentials available")
                return False
        return True
    
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
            
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS listing_info (
                code TEXT PRIMARY KEY, name TEXT, market TEXT, listing_date DATE,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_listing_info_date ON listing_info (listing_date)")
    
    def _get_otp(self, bld: str, **kwargs) -> Optional[str]:
        """Generic OTP generator for different KRX endpoints."""
        if not self._ensure_authenticated():
            logger.error("Cannot get OTP - not authenticated")
            return None
        
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
                
                if response.status_code == 401 or 'login' in response.url.lower():
                    logger.warning("Session expired, attempting to re-login...")
                    self._authenticated = False
                    if self._ensure_authenticated():
                        continue
                    else:
                        return None
                
                response.raise_for_status()
                otp = response.text.strip()
                if not otp: 
                    raise ValueError("Empty OTP received")
                return otp
                
            except (requests.RequestException, ValueError) as e:
                if attempt == self.max_retries:
                    logger.error(f"Failed to get OTP after {self.max_retries+1} attempts: {e}")
                    return None
                logger.warning(f"Attempt {attempt+1} to get OTP failed, retrying... Error: {e}")
                time.sleep(attempt + 1)

    def _download_data_raw(self, otp: str) -> Optional[bytes]:
        """Helper function to download the raw bytes of the CSV file with retries."""
        if not otp: 
            return None
        
        if not self._ensure_authenticated():
            logger.error("Cannot download data - not authenticated")
            return None
            
        for attempt in range(self.max_retries + 1):
            try:
                self._throttle()
                logger.debug(f"Downloading raw data with OTP: {otp} (attempt {attempt+1})")
                response = self.session.post(DOWNLOAD_URL, data={"code": otp}, timeout=60)
                
                if response.status_code == 401 or 'login' in response.url.lower():
                    logger.warning("Session expired during download, attempting to re-login...")
                    self._authenticated = False
                    if self._ensure_authenticated():
                        continue
                    else:
                        return None
                
                response.raise_for_status()
                content = response.content

                if content.strip().lower().startswith(b'<html>') or content.strip().lower().startswith(b'<!doctype html'):
                    raise ValueError("Received HTML error page instead of CSV data.")

                if not content:
                    logger.warning("Received empty file, likely a non-trading day.")
                    return None
                return content

            except (requests.RequestException, ValueError) as e:
                if attempt == self.max_retries:
                    logger.error(f"Failed to download raw data after {self.max_retries+1} attempts: {e}")
                    return None
                logger.warning(f"Attempt {attempt+1} to download raw data failed, retrying... Error: {e}")
                time.sleep(2 ** attempt)

    def fetch_daily_data(self, date, markets: List[str] = None) -> Dict[str, pd.DataFrame]:
        """Fetches, processes, and cleans daily data, using a file cache."""
        if markets is None: 
            markets = ["STK", "KSQ"]
        
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
                        self.cache_dir.mkdir(exist_ok=True, parents=True)
                        with open(cache_file, 'wb') as f:
                            f.write(raw_bytes)
                    else:
                        logger.warning(f"No data downloaded for {mkt} on {date_str}.")
                        df = pd.DataFrame()
                
                if df.empty:
                    results[mkt] = df
                    continue

                column_map = {
                    '종목코드': 'code', '종목명': 'name', '시장구분': 'market_name',
                    '시가': 'open', '고가': 'high', '저가': 'low', '종가': 'close',
                    '거래량': 'volume', '거래대금': 'value', '상장주식수': 'listed_shares'
                }
                df.rename(columns=column_map, inplace=True)
                df['date'] = date_dt
                df['market'] = mkt
                df['code'] = df['code'].str.zfill(6)

                numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'value', 'listed_shares']
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = df[col].astype(str).str.replace(',', '', regex=False)
                        df[col] = pd.to_numeric(df[col], errors='coerce')

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
        if end_date is None: 
            end_date = datetime.now()
        if isinstance(start_date, str): 
            start_date = datetime.strptime(start_date, '%Y%m%d')
        if isinstance(end_date, str): 
            end_date = datetime.strptime(end_date, '%Y%m%d')

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
        return pd.DataFrame()

    def update_database(self, df: pd.DataFrame) -> int:
        """Updates the daily_prices table in the database."""
        if df.empty:
            return 0

        rows_to_process = len(df)
        try:
            update_query = """
            INSERT OR REPLACE INTO daily_prices (code, date, open, high, low, close, volume, value, market_cap, market, name)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
            
            df_for_db = df[['code', 'date', 'open', 'high', 'low', 'close', 'volume', 'value', 'market_cap', 'market', 'name']].copy()
            df_for_db.replace({np.nan: None}, inplace=True)
            update_data = [tuple(x) for x in df_for_db.to_numpy()]

            with self.conn:
                self.conn.executemany(update_query, update_data)
            
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
            otp = self._get_otp("dbms/MDC/STAT/standard/MDCSTAT01901", mktId='ALL', share='1')
            raw_bytes = self._download_data_raw(otp)
            if not raw_bytes:
                logger.error("Received no data from listing info endpoint.")
                return None

            df = pd.read_csv(BytesIO(raw_bytes), encoding='cp949', dtype={'단축코드': str})
            
            column_map = {
                '단축코드': 'code',
                '한글 종목약명': 'name',
                '시장구분': 'market',
                '상장일': 'listing_date'
            }
            df.rename(columns=column_map, inplace=True)
            
            df['code'] = df['code'].str.zfill(6)
            df['listing_date'] = pd.to_datetime(df['listing_date'], errors='coerce')
            
            final_cols = ['code', 'name', 'market', 'listing_date']
            df = df.reindex(columns=final_cols).dropna(subset=['listing_date'])
            
            logger.info(f"Successfully fetched and processed {len(df)} listings.")
            return df
        except Exception as e:
            logger.error(f"Failed to fetch listing info: {e}", exc_info=True)
            return None

    def update_listing_info_in_db(self, df: pd.DataFrame) -> int:
        """Updates the listing_info table in the database."""
        if df.empty: 
            return 0
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
        if self.conn: 
            self.conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fetch KRX market data with KakaoTalk authentication')
    parser.add_argument('-s', '--start-date', help='Start date (YYYYMMDD)')
    parser.add_argument('-e', '--end-date', help='End date (YYYYMMDD)')
    parser.add_argument('--market', default='ALL', choices=['STK', 'KSQ', 'ALL'], help='Market to fetch')
    parser.add_argument('--update-db', action='store_true', help='Update the database with fetched data')
    parser.add_argument('--force-refetch', action='store_true', help='Force re-fetch of existing data')
    parser.add_argument('--fetch-listings', action='store_true', help='Fetch and update listing information')
    parser.add_argument('--kakao-email', help='KakaoTalk email (or set KAKAO_EMAIL env var)')
    parser.add_argument('--kakao-password', help='KakaoTalk password (or set KAKAO_PASSWORD env var)')
    parser.add_argument('--no-headless', action='store_true', help='Show browser window during login (useful for debugging)')
    parser.add_argument('--cookies-file', help='Path to save/load session cookies')
    args = parser.parse_args()

    fetcher = KRXDataFetcher(
        kakao_email=args.kakao_email,
        kakao_password=args.kakao_password,
        headless=not args.no_headless,
        cookies_file=args.cookies_file
    )

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
            print(df[['code', 'name', 'date', 'open', 'close', 'volume', 'value', 'market_cap']].head())
    
    if not args.fetch_listings and not (args.start_date and args.end_date):
        parser.print_help()