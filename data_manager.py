#!/usr/bin/env python3
"""
Data Manager for TimeFolio project
Handles data loading and processing for KRX data
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Union, Any, Tuple
from datetime import datetime, timedelta
import sqlite3
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Path to the SQLite DB where KRX OHLCV + factors are stored
KRX_DB_PATH = Path(__file__).parent.parent / "krx_data.db"

def load_sector_codes(path: str) -> List[str]:
    """
    Load stock codes from a CSV file defining your sector universe.
    
    Args:
        path: Path to the CSV file containing sector universe
        
    Returns:
        List of stock codes as strings
    """
    try:
        df = pd.read_csv(path)
        if 'code' in df.columns:
            return df['code'].astype(str).str.zfill(6).tolist()
        elif '종목코드' in df.columns:
            return df['종목코드'].astype(str).str.strip().str.zfill(6).tolist()
        else:
            logger.error(f"CSV file must contain 'code' or '종목코드' column: {path}")
            return []
    except Exception as e:
        logger.error(f"Error loading sector codes from {path}: {e}")
        return []

def get_latest_close(codes: list[str], as_of_date: str) -> pd.DataFrame:
    """
    Fetch the closing prices for each code on the given date from the KRX DB.
    
    Args:
        codes: List of stock codes to fetch
        as_of_date: Date in YYYY-MM-DD format
        
    Returns:
        DataFrame with 'close' prices, indexed by stock code
    """
    if not codes:
        return pd.DataFrame()
        
    try:
        with sqlite3.connect(KRX_DB_PATH) as conn:
            placeholders = ','.join(['?'] * len(codes))
            query = f"""
                SELECT code, close 
                FROM daily_prices 
                WHERE code IN ({placeholders})
                AND date = (
                    SELECT MAX(date) 
                    FROM daily_prices 
                    WHERE date <= ? AND code IN ({placeholders})
                )
            """
            params = codes + [as_of_date] + codes
            df = pd.read_sql_query(query, conn, params=params)
            return df.set_index('code') if not df.empty else pd.DataFrame()
    except Exception as e:
        logger.error(f"Error fetching latest close prices: {e}")
        return pd.DataFrame()

def create_price_df(data_dict: Dict[str, pd.DataFrame], column: str = 'close') -> pd.DataFrame:
    """
    Convert dictionary of OHLCV DataFrames to a price DataFrame
    
    Args:
        data_dict: Dictionary of DataFrames keyed by stock code
        column: Column to extract (default: close)
        
    Returns:
        DataFrame with prices by stock code
    """
    if not data_dict:
        return pd.DataFrame()
        
    try:
        # Extract the specified column from each DataFrame
        price_series = []
        for code, df in data_dict.items():
            if not df.empty and column in df.columns:
                price_series.append(df[column].rename(code))
        
        return pd.concat(price_series, axis=1) if price_series else pd.DataFrame()
    except Exception as e:
        logger.error(f"Error creating price DataFrame: {e}")
        return pd.DataFrame()
    
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



def load_sector_codes(path: str) -> list[str]:
    """
    Load stock codes from a CSV file defining your sector universe.
    
    Args:
        path: Path to the CSV file containing sector universe
        
    Returns:
        List of stock codes as strings
    """
    df = pd.read_csv(path)
    # Assumes the CSV has a column named 'code'
    return df['code'].astype(str).tolist()


def get_latest_close(codes: list[str], as_of_date: str) -> pd.DataFrame:
    """
    Fetch the closing prices for each code on the given date from the KRX DB.
    
    Args:
        codes: List of stock codes to fetch
        as_of_date: Date in YYYY-MM-DD format
        
    Returns:
        DataFrame with 'close' prices, indexed by stock code
    """
    conn = sqlite3.connect(str(KRX_DB_PATH))
    placeholders = ",".join(["?"] * len(codes))
    sql = f"""
        SELECT code, close
          FROM daily_prices
         WHERE date = ?
           AND code IN ({placeholders})
    """
    params = [as_of_date] + codes
    df = pd.read_sql(sql, conn, params=params)
    conn.close()
    return df.set_index('code')[['close']]


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

__all__ = [
    "load_sector_codes",
    "get_latest_close",
    "create_price_df",
    "KRX_DB_PATH"
]

if __name__ == "__main__":
    # Setup console logging when run directly
    logging.basicConfig(level=logging.INFO)
    
    # Simple CLI interface
    import argparse
    parser = argparse.ArgumentParser(description="KRX Data Manager")
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