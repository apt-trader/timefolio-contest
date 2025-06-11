#!/usr/bin/env python3
"""
KIS Data Manager for TimeFolio project
Handles API requests, caching, and data processing
"""
import os
import time
import random
import pandas as pd
import numpy as np
import datetime as dt
import logging
import requests
import json
import pathlib
from pathlib import Path
from typing import List, Dict, Optional, Union
from kis_auth import get_token
from typing import Tuple, List, Dict, Optional, Union
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import time
import sqlite3
import os
from pathlib import Path
import logging
from dateutil.relativedelta import relativedelta

def generate_date_chunks(start_date: Union[str, datetime], 
                       end_date: Union[str, datetime, None] = None,
                       chunk_days: int = 29) -> List[Tuple[datetime, datetime]]:
    """
    Generate date chunks for fetching data in fixed-size windows
    
    Args:
        start_date: Start date (YYYYMMDD or datetime)
        end_date: End date (YYYYMMDD or datetime, defaults to today)
        chunk_days: Number of days per chunk (default 29 to stay under 30-day limit)
        
    Returns:
        List of (start, end) datetime tuples
    """
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, '%Y%m%d')
    if end_date is None:
        end_date = datetime.now()
    elif isinstance(end_date, str):
        end_date = datetime.strptime(end_date, '%Y%m%d')
        
    chunks = []
    current = start_date
    
    while current <= end_date:
        chunk_end = min(current + timedelta(days=chunk_days), end_date)
        chunks.append((current, chunk_end))
        current = chunk_end + timedelta(days=1)
    
    return chunks


def _extract_rows(js: dict, default_key: str = "output") -> list[dict]:
    """
    Return the first non-empty list among
    ['output2', 'output1', 'output'].
    Raises ValueError if nothing is found.
    """
    for k in ("output2", "output1", default_key):
        rows = js.get(k)
        if isinstance(rows, list) and rows:
            return rows
        if isinstance(rows, dict):          # older price call (output1 dict)
            return [rows]
    raise ValueError("KIS API: response payload has no recognised output key")

BASE = "https://openapi.koreainvestment.com:9443"
HEAD = {
    "Content-Type":"application/json",
    "appKey":    os.environ["KIS_APP_KEY"],
    "appSecret": os.environ["KIS_APP_SECRET"],
    "authorization": f"Bearer {get_token()}",
    "tr_id": "FHKST01010100"          
}

def inquire_price(code:str)->int:
    """6-자리 종목코드 → 현재가(int) 반환"""
    params = {"fid_cond_mrkt_div_code":"J", "fid_input_iscd":code}
    res = requests.get(
        f"{BASE}/uapi/domestic-stock/v1/quotations/inquire-price",
        headers=HEAD, params=params
    ).json()
    rows = _extract_rows(res)
    return int(rows[0]["stck_prpr"])

def fetch_index_data(
    index_code: str,
    start: str,
    end: str | None = None,
    period: str = "D",
) -> pd.DataFrame:
    """
    Retrieve domestic index (KOSPI, KOSDAQ, etc.) daily/weekly/monthly/yearly OHLCV data
    Parameters
    ----------
    index_code : str   # 0001=KOSPI, 1001=KOSDAQ, 2001=KOSPI200 ...
    start      : str   # 'YYYYMMDD'
    end        : str   # 'YYYYMMDD' (None → today)
    period     : str   # 'D','W','M','Y'
    """
    if end is None:
        end = dt.date.today().strftime("%Y%m%d")

    url = (
        "/uapi/domestic-stock/v1/quotations/"
        "inquire-daily-indexchartprice"
    )  # TR: FHKUP03500100
    params = {
        "FID_COND_MRKT_DIV_CODE": "U",      # Index/sector distinction
        "FID_INPUT_ISCD": index_code,
        "FID_INPUT_DATE_1": start,
        "FID_INPUT_DATE_2": end,
        "FID_PERIOD_DIV_CODE": period,
    }
    headers = {
        "Content-Type": "application/json",
        "appKey": os.environ["KIS_APP_KEY"],
        "appSecret": os.environ["KIS_APP_SECRET"],
        "authorization": f"Bearer {get_token()}",
        "tr_id": "FHKUP03500100"
    }
    r = requests.get(BASE + url, headers=headers, params=params, timeout=5)
    r.raise_for_status()
    
    # Use the _extract_rows helper to handle both API formats
    try:
        response_data = r.json()
        if "output1" in response_data and "chart" in response_data["output1"]:
            js = response_data["output1"]["chart"]
        else:
            # For the new API format
            rows = _extract_rows(response_data)
            js = rows  # Assuming the structure matches what we need
    except Exception as e:
        logger.error(f"Error parsing index data response: {e}")
        raise ValueError(f"Failed to parse index data: {e}")
    df = (
        pd.DataFrame(js)
        .rename(
            columns={
                "stck_bsop_date": "date",
                "bstp_nmix_prpr": "close",
                "bstp_nmix_oprc": "open",
                "bstp_nmix_hgpr": "high",
                "bstp_nmix_lwpr": "low",
                "acml_vol": "volume",
            }
        )
        .assign(date=lambda d: pd.to_datetime(d["date"]))
        .set_index("date")
        .astype(float)
        .sort_index()
    )
    return df


def get_stock_listing_info(start_date: str, end_date: str | None = None) -> pd.DataFrame:
    """
    Get stock listing/delisting information
    
    Parameters
    ----------
    start_date : str   # 'YYYYMMDD'
    end_date   : str   # 'YYYYMMDD' (None → today)
    
    Returns
    -------
    DataFrame with stock listing information
    """
    if end_date is None:
        end_date = dt.date.today().strftime("%Y%m%d")
        
    url = "/uapi/domestic-stock/v1/ksdinfo/list-info"   # TR: FHPUP02120000
    params = {
        "SHT_CD": "",             # Empty → All
        "F_DT": start_date,       # YYYYMMDD
        "T_DT": end_date,         # YYYYMMDD
        "CTS": "",                # No paging
    }
    
    headers = {
        "Content-Type": "application/json",
        "appKey": os.environ["KIS_APP_KEY"],
        "appSecret": os.environ["KIS_APP_SECRET"],
        "authorization": f"Bearer {get_token()}",
        "tr_id": "FHPUP02120000"
    }
    
    r = requests.get(BASE + url, headers=headers, params=params, timeout=5)
    r.raise_for_status()
    
    try:
        # Use the _extract_rows helper to handle both API formats
        rows = _extract_rows(r.json())
        return pd.DataFrame(rows)
    except Exception as e:
        logger.error(f"Error parsing stock listing info: {e}")
        return pd.DataFrame()

# Create cache directories
CACHE = pathlib.Path.home() / "KRX_cache"
CACHE.mkdir(exist_ok=True)
today = dt.date.today().strftime("%Y%m%d")

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(ch)

# Constants
ENDPOINT = "[https://openapi.koreainvestment.com](https://openapi.koreainvestment.com):9443"
CACHE_DIR = os.path.expanduser("~/KRX_cache")
Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)

# Global API call tracker to respect rate limits
request_tracker = {
    'recent_calls': [],  # List of timestamps for recent API calls
    'daily_count': 0,    # Count of API calls today
    'daily_reset': 0     # Timestamp when daily count was last reset
}

def reset_daily_counter():
    """Reset daily API call counter if it's a new day"""
    now = time.time()
    day_start = now - (now % 86400)  # Start of current day
    
    if request_tracker['daily_reset'] < day_start:
        logger.info(f"Resetting API call counter (previous: {request_tracker['daily_count']})")
        request_tracker['daily_count'] = 0
        request_tracker['daily_reset'] = day_start

def track_api_call():
    """Track an API call for rate limiting purposes"""
    now = time.time()
    
    # Update daily count
    reset_daily_counter()
    request_tracker['daily_count'] += 1
    
    # Track recent calls for per-second rate limiting
    request_tracker['recent_calls'].append(now)
    
    # Only keep the last 30 calls (should be sufficient for 20/sec limit)
    if len(request_tracker['recent_calls']) > 30:
        request_tracker['recent_calls'] = request_tracker['recent_calls'][-30:]
    
    # Check if approaching daily limit (300k)
    if request_tracker['daily_count'] > 250000:
        logger.warning(f"Approaching daily API call limit: {request_tracker['daily_count']}/300000")
    
    # Return seconds since most recent call
    if len(request_tracker['recent_calls']) > 1:
        return now - request_tracker['recent_calls'][-2]
    return 10.0  # Arbitrary large value if first call

def adaptive_sleep():
    """
    Adaptive sleep with jitter to avoid pattern detection
    and respect rate limits
    """
    # Base delay - start conservative
    base_delay = 0.2
    
    # Check how many calls we've made in the last second
    now = time.time()
    recent_window = 1.0  # 1 second window
    calls_in_window = sum(1 for t in request_tracker['recent_calls'] 
                          if now - t < recent_window)
    
    # If we're approaching the rate limit, increase delay
    if calls_in_window > 15:  # Getting close to 20/sec
        base_delay = 0.5
    elif calls_in_window > 10:
        base_delay = 0.3
        
    # Add jitter to avoid predictable patterns
    jitter = random.uniform(0.1, 0.4)
    
    # Occasionally add much larger delay (1% chance)
    if random.random() < 0.01:
        logger.debug("Adding extended delay to avoid pattern detection")
        extended_delay = random.uniform(1.5, 3.0)
        time.sleep(extended_delay)
        return
    
    # Time to sleep
    sleep_time = base_delay + jitter
    
    # Don't sleep if last call was long ago
    time_since_last = track_api_call()
    if time_since_last > sleep_time:
        return
        
    time.sleep(max(0.05, sleep_time - time_since_last))

def should_throttle():
    """Determine if requests should be throttled based on time of day"""
    hour = dt.datetime.now().hour
    # More throttling during market hours (9:00-15:30 KST)
    if 9 <= hour < 16:
        return True
    return False

def api_request_with_retry(url: str, 
                          headers: Dict, 
                          params: Dict = None, 
                          max_retries: int = 5,
                          is_post: bool = False,
                          json_data: Dict = None) -> Dict:
    """
    Make API request with exponential backoff retry
    
    Args:
        url: API endpoint URL
        headers: Request headers
        params: Query parameters for GET requests
        max_retries: Maximum retry attempts
        is_post: Whether this is a POST request
        json_data: JSON data for POST requests
        
    Returns:
        API response data as dictionary
    """
    headers = headers.copy()  # Don't modify the original
    
    # Add user agent rotation if needed
    user_agents = [
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 11_1_0) AppleWebKit/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_6) AppleWebKit/605.1.15",
    ]
    headers["User-Agent"] = random.choice(user_agents)
    
    # Apply throttling if during market hours
    if should_throttle():
        time.sleep(random.uniform(0.5, 1.2))  # Longer delay during market hours
        
    for attempt in range(max_retries):
        try:
            # Ensure URL is properly formatted without markdown style square brackets
            clean_url = url
            if '[' in clean_url:
                # Remove markdown formatting if present
                clean_url = clean_url.replace('[', '').replace(']', '')
                logger.warning(f"Fixed malformed URL: {url} -> {clean_url}")
            
            if is_post:
                response = requests.post(clean_url, headers=headers, json=json_data, timeout=10)
            else:
                response = requests.get(clean_url, headers=headers, params=params, timeout=10)
                
            data = response.json()
            
            # Check for rate limiting or error responses
            if data.get('rt_cd') == '0202':  # Rate limit exceeded
                wait_time = 2 ** attempt + random.uniform(1, 3)  # Exponential backoff
                logger.warning(f"Rate limit hit, backing off for {wait_time:.2f}s")
                time.sleep(wait_time)
                continue
                
            elif data.get('rt_cd') not in ['0', '0000', None]:  # Other API error
                error_msg = data.get('msg_cd', 'Unknown error')
                logger.warning(f"API error {data.get('rt_cd')}: {error_msg}")
                
                if attempt < max_retries - 1:  # Don't sleep on last attempt
                    wait_time = 1 + random.uniform(0, 2)
                    time.sleep(wait_time)
                    continue
            
            return data
            
        except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
            if attempt < max_retries - 1:  # Don't sleep on last attempt
                wait_time = 2 ** attempt + random.uniform(0, 1)
                logger.error(f"Request error: {e}. Retrying in {wait_time:.2f}s")
                time.sleep(wait_time)
            else:
                logger.error(f"Final request attempt failed: {e}")
            
    # If we get here, all retries failed
    raise Exception(f"Failed to make API request after {max_retries} attempts")

def fetch_daily(code: str, start: str = "20230101", start_date: str = None) -> pd.DataFrame:
    """
    Fetch daily price data for a stock with pagination support
    
    Args:
        code: Stock code
        start: Start date in YYYYMMDD format (primary parameter)
        start_date: Alternative parameter name for backward compatibility
        
    Returns:
        DataFrame with OHLCV data
    """
    # Ensure backward compatibility with both parameter names
    if start_date is not None and start == "20230101":
        start = start_date
        
    all_rows = []
    current_end = dt.datetime.now().strftime("%Y%m%d")
    current_start = start
    max_requests = 20  # Safety limit to prevent infinite loops
    request_count = 0
    
    while request_count < max_requests:
        request_count += 1
        adaptive_sleep()  # Rate limiting
        
        # Prepare API request
        url = f"{BASE}/uapi/domestic-stock/v1/quotations/inquire-daily-price"
        token = get_token()
        headers = {
            "Content-Type": "application/json", 
            "authorization": f"Bearer {token}",
            "appkey": os.environ.get('KIS_APP_KEY'),
            "appsecret": os.environ.get('KIS_APP_SECRET'),
            "tr_id": "FHKST01010400"
        }
        
        payload = {
            "fid_cond_mrkt_div_code": "J",  # KOSPI and KOSDAQ
            "fid_input_iscd": code,
            "fid_input_date_1": current_start,
            "fid_input_date_2": current_end,
            "fid_period_div_code": "D",
            "fid_org_adj_prc": "1"
        }
        
        try:
            # Make API request
            response = api_request_with_retry(url, headers, params=payload)
            
            # Extract data from response
            try:
                api_rows = _extract_rows(response)
                if not api_rows:
                    logger.info(f"No more data available for {code}")
                    break
                    
                # Process the extracted rows
                batch_rows = []
                for item in api_rows:
                    # Stop if we've gone past our start date
                    if item['stck_bsop_date'] < start:
                        break
                        
                    try:
                        row = [
                            item['stck_bsop_date'],  # Date
                            float(item['stck_oprc']),  # Open
                            float(item['stck_hgpr']),  # High
                            float(item['stck_lwpr']),  # Low
                            float(item['stck_clpr']),  # Close
                            int(item['acml_vol'])      # Volume
                        ]
                        batch_rows.append(row)
                    except (KeyError, ValueError) as e:
                        logger.warning(f"Error parsing data row for {code}: {e}")
                        continue
                
                if not batch_rows:
                    logger.info(f"No new data in current batch for {code}")
                    break
                    
                all_rows.extend(batch_rows)
                
                # If we got fewer than 100 rows, we've reached the earliest available data
                if len(api_rows) < 100:
                    logger.info(f"Reached earliest available data for {code}")
                    break
                    
                # Update dates for next request (go backwards in time)
                last_date = pd.to_datetime(batch_rows[-1][0])
                current_end = (last_date - dt.timedelta(days=1)).strftime("%Y%m%d")
                
                # If we've gone past our start date, we're done
                if last_date < pd.to_datetime(start):
                    break
                    
            except ValueError as e:
                logger.error(f"{e} for {code}: {response}")
                break
                
        except Exception as e:
            logger.error(f"Error in fetch_daily for {code}: {e}")
            break
    
    # Create final DataFrame if we have data
    if not all_rows:
        logger.warning(f"No data returned for {code}")
        return pd.DataFrame(columns=['date', 'open', 'high', 'low', 'close', 'volume'])
    
    # Create DataFrame with proper columns and data types
    df = pd.DataFrame(
        all_rows,
        columns=['date', 'open', 'high', 'low', 'close', 'volume']
    )
    
    # Convert date to datetime and sort
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # Log the data range we fetched
    logger.info(f"Fetched {len(df)} rows of data for {code} from {df['date'].min()} to {df['date'].max()}")
    
    return df

def fetch_chunked_daily(
    code: str, 
    start_date: Union[str, datetime], 
    end_date: Union[str, datetime, None] = None,
    use_cache: bool = True,
    min_days: int = 120,
    force_refresh: bool = False
) -> pd.DataFrame:
    """
    Fetch daily price data with 30-day chunking and caching
    
    Args:
        code: Stock code (6 digits)
        start_date: Start date (YYYYMMDD or datetime)
        end_date: End date (YYYYMMDD or datetime, defaults to today)
        use_cache: Whether to use cached data
        min_days: Minimum number of trading days to return
        force_refresh: Force refresh even if cache exists
        
    Returns:
        DataFrame with OHLCV data
    """
    # Convert string dates to datetime objects
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, '%Y%m%d')
    if end_date is None:
        end_date = datetime.now()
    elif isinstance(end_date, str):
        end_date = datetime.strptime(end_date, '%Y%m%d')
    
    # Initialize cache
    cache_file = CACHE / f"{code}.parquet"
    cached_data = None
    
    # Try to load from cache if available and requested
    if use_cache and cache_file.exists() and not force_refresh:
        try:
            cached_data = pd.read_parquet(cache_file)
            if 'date' in cached_data.columns:
                cached_data['date'] = pd.to_datetime(cached_data['date'])
                # Check if we have enough data in cache
                if len(cached_data) >= min_days:
                    last_cached_date = cached_data['date'].max()
                    if last_cached_date >= end_date - timedelta(days=1):
                        return cached_data[
                            (cached_data['date'] >= start_date) & 
                            (cached_data['date'] <= end_date)
                        ]
        except Exception as e:
            logger.warning(f"Error reading cache for {code}: {e}")
            cached_data = None
    
    # Generate date chunks for fetching
    chunks = generate_date_chunks(start_date, end_date)
    all_data = []
    
    for chunk_start, chunk_end in chunks:
        try:
            # Format dates for API
            start_str = chunk_start.strftime('%Y%m%d')
            end_str = chunk_end.strftime('%Y%m%d')
            
            # Fetch data for this chunk
            chunk_data = fetch_daily(code, start_str, end_str)
            if not chunk_data.empty:
                all_data.append(chunk_data)
                
            # Respect rate limits
            adaptive_sleep()
            
        except Exception as e:
            logger.error(f"Error fetching data for {code} ({start_str}-{end_str}): {e}")
            continue
    
    if not all_data:
        if cached_data is not None:
            logger.warning("Using cached data due to fetch errors")
            return cached_data
        return pd.DataFrame()
    
    # Combine all chunks
    combined = pd.concat(all_data)
    if 'date' in combined.columns:
        combined = combined.drop_duplicates('date').sort_values('date')
    else:
        logger.error("No 'date' column in combined data")
        return pd.DataFrame()
    
    # Update cache
    try:
        if cached_data is not None and not cached_data.empty and 'date' in cached_data.columns:
            # Remove any overlapping data
            min_date = combined['date'].min()
            cached_data = cached_data[cached_data['date'] < min_date]
            combined = pd.concat([cached_data, combined])
            
        combined.to_parquet(cache_file)
    except Exception as e:
        logger.error(f"Error updating cache for {code}: {e}")
    
    return combined[
        (combined['date'] >= start_date) & 
        (combined['date'] <= end_date)
    ]

def get_stock_data(
    code: str, 
    start_date: str, 
    end_date: Union[str, None] = None,
    force_refresh: bool = False,
    min_days: int = 120
) -> pd.DataFrame:
    """
    Get stock data with intelligent caching
    
    Args:
        code: Stock code (6 digits)
        start_date: Start date in YYYYMMDD format
        end_date: End date in YYYYMMDD format (default: today)
        force_refresh: Force refresh even if cache exists
        min_days: Minimum number of trading days to return
        
    Returns:
        DataFrame with OHLCV data
    """
    # Convert dates to datetime for comparison
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date) if end_date else pd.Timestamp.now()
    
    # Initialize cache
    cache_file = CACHE / f"{code}.parquet"
    
    # If forcing refresh or no cache, fetch fresh data
    if force_refresh or not cache_file.exists():
        logger.info(f"Fetching fresh data for {code} from {start_date}")
        df = fetch_chunked_daily(
            code=code,
            start_date=start_date,
            end_date=end_date,
            use_cache=False,
            min_days=min_days,
            force_refresh=True
        )
    else:
        # Try to load from cache first
        try:
            df = pd.read_parquet(cache_file)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                
                # Check if we need to update the cache
                last_date = df['date'].max()
                if last_date < end_dt - pd.Timedelta(days=1):  # If cache is not up to date
                    logger.info(f"Updating cache for {code} from {last_date}")
                    new_data = fetch_chunked_daily(
                        code=code,
                        start_date=last_date.strftime('%Y%m%d'),
                        end_date=end_dt.strftime('%Y%m%d'),
                        use_cache=False,
                        force_refresh=True
                    )
                    if not new_data.empty and 'date' in new_data.columns:
                        # Remove any overlapping data
                        df = df[df['date'] < new_data['date'].min()]
                        df = pd.concat([df, new_data])
                        df.to_parquet(cache_file)
        except Exception as e:
            logger.error(f"Error reading/updating cache for {code}: {e}")
            df = pd.DataFrame()
    
    # Filter for the requested date range
    if not df.empty and 'date' in df.columns:
        df = df[
            (df['date'] >= start_dt) & 
            (df['date'] <= end_dt)
        ].sort_values('date')
    
    return df
    """
    Get stock data with intelligent caching
    
    Args:
        code: Stock code
        start_date: Start date in YYYYMMDD format
        force_refresh: Whether to force refresh data from API
        
    Returns:
        DataFrame with OHLCV data
    """
    cache_file = Path(CACHE_DIR) / f"{code}.parquet"
    
    # Check if we need fresh data
    refresh_needed = force_refresh
    if not refresh_needed and cache_file.exists():
        # Check if cache is recent enough
        last_modified = dt.datetime.fromtimestamp(os.path.getmtime(cache_file))
        cache_age_days = (dt.datetime.now() - last_modified).days
        
        # Read cached data to check last date
        cached_data = pd.read_parquet(cache_file)
        if not cached_data.empty:
            last_date = cached_data.index.max()
            today = dt.datetime.now().date()
            days_behind = (today - last_date.date()).days
            
            # Refresh if cache is old or missing recent data (but not on weekends)
            weekday = dt.datetime.now().weekday()
            is_weekend = weekday >= 5  # Saturday or Sunday
            
            if (cache_age_days > 7 or (days_behind > 3 and not is_weekend)):
                refresh_needed = True
                logger.info(f"Cache for {code} is {days_behind} days old, refreshing")
    else:
        refresh_needed = True
        if not cache_file.exists():
            logger.info(f"No cache found for {code}, creating new cache")
    
    # Fetch new data if needed
    if refresh_needed:
        try:
            df = fetch_chunked_daily(
                code,
                start_date,
                end_date=end_date,
                use_cache=False
            )
            
            # If we have existing data, merge only the new data
            if os.path.exists(cache_file) and not force_refresh:
                existing_data = pd.read_parquet(cache_file)
                # Find newest data to add
                last_existing_date = existing_data.index.max()
                new_data = df[df.index > last_existing_date]
                
                # Only append if we have new data
                if not new_data.empty:
                    logger.info(f"Adding {len(new_data)} new rows to cache for {code}")
                    df = pd.concat([existing_data, new_data])
                else:
                    logger.info(f"No new data for {code}, using cached data")
                    df = existing_data
            
            # Save to cache if we have data
            if not df.empty:
                df.to_parquet(cache_file)
                logger.info(f"Updated cache for {code} with {len(df)} rows")
            return df
            
        except Exception as e:
            logger.error(f"Error refreshing data for {code}: {e}")
            # Fall back to cached data if available
            if os.path.exists(cache_file):
                logger.warning(f"Using cached data for {code} due to fetch error")
                return pd.read_parquet(cache_file)
            # No cache and fetch failed
            logger.error(f"No data available for {code}")
            return pd.DataFrame()
    else:
        # Use cached data
        return pd.read_parquet(cache_file)

def fetch_multiple_daily(codes: List[str], start_date: str = "20230101", max_per_batch: int = 10) -> Dict[str, pd.DataFrame]:
    """
    Fetch data for multiple stock codes with intelligent batching
    
    Args:
        codes: List of stock codes
        start_date: Start date in YYYYMMDD format
        max_per_batch: Maximum number of stocks to process in a batch
        
    Returns:
        Dictionary of DataFrames with OHLCV data keyed by stock code
    """
    results = {}
    
    # Randomize order to avoid predictable patterns
    codes_copy = codes.copy()
    random.shuffle(codes_copy)
    
    # Process in smaller batches
    for i in range(0, len(codes_copy), max_per_batch):
        batch = codes_copy[i:i+max_per_batch]
        logger.info(f"Processing batch {i//max_per_batch + 1}/{(len(codes_copy)-1)//max_per_batch + 1} with {len(batch)} codes")
        
        for code in batch:
            try:
                results[code] = get_stock_data(code, start_date)
                # Shorter sleep within a batch
                time.sleep(random.uniform(0.1, 0.25))
            except Exception as e:
                logger.error(f"Error processing {code}: {e}")
                
        # Longer sleep between batches
        batch_delay = random.uniform(2.0, 5.0)
        logger.debug(f"Batch complete, waiting {batch_delay:.1f}s before next batch")
        time.sleep(batch_delay)
    
    return results

def create_price_df(data_dict: Dict[str, pd.DataFrame], column: str = 'close') -> pd.DataFrame:
    """
    Convert dictionary of OHLCV DataFrames to a price DataFrame
    
    Args:
        data_dict: Dictionary of DataFrames keyed by stock code
        column: Column to extract (default: close)
        
    Returns:
        DataFrame with prices by stock code
    """
    # Initialize empty DataFrame
    price_data = pd.DataFrame()
    
    # Extract specified column from each DataFrame
    for code, df in data_dict.items():
        if column in df.columns and not df.empty:
            price_data[code] = df[column]
    
    return price_data

def load_price_data_kis(tickers: List[str], start_date: str) -> pd.DataFrame:
    """
    Load price data from KIS cache for TimeFolio integration
    
    Args:
        tickers: List of stock tickers
        start_date: Start date in YYYYMMDD format
        
    Returns:
        DataFrame with close prices by ticker
    """
    price_data = pd.DataFrame()
    
    logger.info(f"Loading price data for {len(tickers)} tickers")
    for ticker in tickers:
        try:
            # Get data from KIS cache
            df = get_stock_data(ticker, start_date)
            
            # Add to price DataFrame
            if 'close' in df.columns and not df.empty:
                price_data[ticker] = df['close']
            else:
                logger.warning(f"Missing close price for {ticker}")
                
        except Exception as e:
            logger.error(f"Failed to get data for {ticker}: {e}")
            
    return price_data

def check_cache_status():
    """
    Check status of cached data
    
    Returns:
        Dictionary with cache statistics
    """
    cache_dir = Path(CACHE_DIR)
    files = list(cache_dir.glob('*.parquet'))
    
    if not files:
        return {"status": "empty", "message": "No cached files found"}
    
    stats = {
        "total_files": len(files),
        "total_size_mb": sum(f.stat().st_size for f in files) / (1024 * 1024),
        "oldest_file": min(files, key=lambda f: f.stat().st_mtime).name,
        "newest_file": max(files, key=lambda f: f.stat().st_mtime).name,
        "oldest_time": dt.datetime.fromtimestamp(min(f.stat().st_mtime for f in files)),
        "newest_time": dt.datetime.fromtimestamp(max(f.stat().st_mtime for f in files))
    }
    
    # Sample a few files to check date ranges
    sample_size = min(5, len(files))
    samples = random.sample(files, sample_size)
    sample_stats = []
    
    for f in samples:
        try:
            df = pd.read_parquet(f)
            if not df.empty:
                sample_stats.append({
                    "file": f.name,
                    "rows": len(df),
                    "earliest": df.index.min().strftime("%Y-%m-%d"),
                    "latest": df.index.max().strftime("%Y-%m-%d")
                })
        except Exception as e:
            logger.warning(f"Error reading sample file {f.name}: {e}")
    
    stats["samples"] = sample_stats
    return stats

__all__ = [
    "inquire_price",
    "fetch_index_data",
    "get_stock_listing_info",
    "fetch_daily",
    "get_stock_data",
    "fetch_multiple_daily",
    "create_price_df",
    "load_price_data_kis",
    "check_cache_status"
]

if __name__ == "__main__":
    # Setup console logging when run directly
    logging.basicConfig(level=logging.INFO)
    
    # Simple CLI interface
    import argparse
    parser = argparse.ArgumentParser(description="KIS Data Manager")
    parser.add_argument("--check-cache", action="store_true", help="Check cache status")
    parser.add_argument("--sample", type=str, help="Fetch sample data for a specific code")
    parser.add_argument("--start-date", type=str, default="20230101", help="Start date (YYYYMMDD)")
    
    args = parser.parse_args()
    
    if args.check_cache:
        stats = check_cache_status()
        print("Cache Status:")
        for k, v in stats.items():
            if k != "samples":
                print(f"  {k}: {v}")
        if "samples" in stats:
            print("\nSample Files:")
            for s in stats["samples"]:
                print(f"  {s['file']}: {s['rows']} rows, {s['earliest']} to {s['latest']}")
    
    if args.sample:
        try:
            code = args.sample
            print(f"Fetching sample data for {code} from {args.start_date}")
            df = get_stock_data(code, args.start_date)
            print(f"Retrieved {len(df)} rows of data")
            if not df.empty:
                print(f"Date range: {df.index.min().date()} to {df.index.max().date()}")
                print("\nSample data:")
                print(df.head())
        except Exception as e:
            print(f"Error: {e}")