#!/usr/bin/env python3
"""
KIS Data Batch Update Script for TimeFolio
Handles weekly batch updates of stock data with optimized API handling
"""
import os
import time
import random
import pandas as pd
import numpy as np
import logging
import argparse
import datetime as dt
import json
import pathlib
from pathlib import Path
import traceback
from typing import List, Dict, Optional
from data_manager import inquire_price
from universe import get_universe


CODES = get_universe("STK")           # 자동 필터 후 약 2 300 종목
CACHE = pathlib.Path.home()/"KRX_cache"; CACHE.mkdir(exist_ok=True)
today = dt.date.today().isoformat()

rows, fails = [], []
for i, code in enumerate(CODES, 1):
    try:
        rows.append({"code":code, "date":today,
                     "close": inquire_price(code)})
    except Exception as e:
        fails.append((code, str(e)))

    # 쿼터 보호
    if i % 18 == 0:  time.sleep(1.2)
    else:            time.sleep(random.uniform(0.05,0.08))

# Save results (try parquet first, fall back to CSV if needed)
try:
    # Try to save as parquet
    data_df = pd.DataFrame(rows)
    try:
        data_df.to_parquet(CACHE/f"{today}.parquet")
        print(f"Saved price data in parquet format to {CACHE/f'{today}.parquet'}")
    except ImportError:
        # Fall back to CSV if parquet libraries are missing
        csv_path = CACHE/f"{today}.csv"
        data_df.to_csv(csv_path, index=False)
        print(f"Parquet libraries not found. Saved data in CSV format to {csv_path}")
        print("To use parquet format, install: pip install pyarrow")

    # Report on data collection
    if fails:
        print(f"Failed to fetch {len(fails)} stocks: {fails[:5]}...")
    else:
        print(f"✓ Successfully fetched prices for {len(rows)} stocks")
    
    # Save failures to help diagnose issues
    if fails:
        pd.DataFrame(fails, columns=["code","error"]).to_csv(
            CACHE/f"fail_{today}.csv", index=False)
        
    print(f"Data update completed. Total stocks processed: {len(rows)}")
except Exception as e:
    print(f"Error saving data: {e}")
    # Last resort emergency save
    pd.DataFrame(rows).to_csv(Path.home()/f"stock_data_emergency_{today}.csv", index=False)

print(f"✓ {len(rows)} 종목 저장, 실패 {len(fails)}")

# Import local modules
from data_manager import get_stock_data, fetch_multiple_daily, check_cache_status
from kis_auth import get_token, check_token_status

# Setup logging
log_format = "%(asctime)s [%(levelname)s] %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    handlers=[
        logging.FileHandler("/tmp/kis_batch.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("kis_batch")

# Configuration
CACHE_DIR = os.path.expanduser("~/KRX_cache")
Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)

# Backup directory
BACKUP_DIR = os.path.expanduser("~/KRX_backup")
Path(BACKUP_DIR).mkdir(parents=True, exist_ok=True)

def load_universe() -> List[str]:
    """
    Load the investment universe for TimeFolio
    
    Returns:
        List of stock codes from the universe file
    """
    # Fixed path to universe file
    universe_file = Path(os.path.expanduser("~/Desktop/Python/timefolio-2025/sector_universe.csv"))
    
    if not universe_file.exists():
        # Clear error if file doesn't exist
        raise FileNotFoundError(
            f"Universe file not found: {universe_file}\n"
            f"Please create the file with your stock universe\n"
            f"Each row should contain a stock code (e.g., '005930' for Samsung)"
        )
    
    try:
        # Check if file contains HTML error page (from KRX maintenance)
        with open(universe_file, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip().lower()
            if '<html' in first_line or '<!doctype' in first_line or '<meta' in first_line:
                # Create a backup of the corrupted file for reference
                import shutil
                backup_path = str(universe_file) + ".html_backup"
                shutil.copy2(universe_file, backup_path)
                
                # Clear error message with actionable steps
                error_msg = (
                    f"⚠️ CRITICAL: Universe file contains HTML content instead of CSV data!\n"
                    f"This is likely due to KRX service maintenance.\n"
                    f"\n"
                    f"1. The corrupted file has been backed up to: {backup_path}\n"
                    f"2. Please create a new CSV file with your stock universe\n"
                    f"3. Save it as: {universe_file}\n"
                    f"\n"
                    f"Format example:\n"
                    f"ticker,name\n"
                    f"005930,Samsung Electronics\n"
                    f"000660,SK Hynix\n"
                    f"...(more stocks)\n"
                )
                logger.error(error_msg)
                print("\n" + error_msg + "\n")
                raise ValueError("Universe file contains HTML error page, not CSV data")
        
        # Try parsing as CSV
        try:
            df = pd.read_csv(universe_file)
        except Exception as e1:
            # If standard parsing fails, try with custom delimiter
            try:
                logger.info(f"Standard CSV parsing failed: {e1}. Trying with custom delimiter.")
                df = pd.read_csv(universe_file, sep=', ', engine='python')
            except Exception as e2:
                # Clear error with helpful instructions
                error_msg = (
                    f"Failed to parse universe file: {universe_file}\n"
                    f"Please ensure the file is a properly formatted CSV:\n"
                    f"1. It should use comma (,) as the delimiter\n"
                    f"2. It should have a header row\n"
                    f"3. One of the columns should contain stock codes (e.g., '005930')\n"
                    f"\n"
                    f"Original errors:\n"
                    f"- Standard CSV: {e1}\n"
                    f"- Custom parsing: {e2}"
                )
                logger.error(error_msg)
                print("\n" + error_msg + "\n")
                raise ValueError("Cannot parse universe file as CSV")
        
        # Find the column containing stock codes
        ticker_col = None
        # Try exact match first
        for col in ['ticker', 'code', '종목코드', 'stock_code', 'stockcode']:
            if col in df.columns:
                ticker_col = col
                break
                
        # If no exact match, try partial match
        if ticker_col is None:
            for col in df.columns:
                col_lower = col.lower()
                if any(term in col_lower for term in ['ticker', 'code', '종목', 'stock']):
                    ticker_col = col
                    logger.info(f"Using column '{col}' for stock codes based on name matching")
                    break
        
        if ticker_col is None:
            available_cols = ', '.join(df.columns)
            error_msg = (
                f"Could not identify which column contains stock codes.\n"
                f"Available columns: {available_cols}\n"
                f"Please rename one column to 'ticker' or 'code' to indicate stock codes."
            )
            logger.error(error_msg)
            print("\n" + error_msg + "\n")
            raise ValueError("Stock code column not found")
        
        # Process and clean stock codes
        codes = []
        for code in df[ticker_col].astype(str):
            # Remove any 'A' prefix (common in KRX data) and keep only digits
            if code.startswith('A'):
                code = code[1:]
            code = ''.join(filter(str.isdigit, code))
            # Ensure 6-digit format with leading zeros
            if code and len(code) <= 6:
                codes.append(code.zfill(6))
        
        if not codes:
            error_msg = (
                f"No valid stock codes found in the universe file.\n"
                f"Please ensure the '{ticker_col}' column contains valid codes\n"
                f"(e.g., '005930' for Samsung Electronics)."
            )
            logger.error(error_msg)
            print("\n" + error_msg + "\n")
            raise ValueError("No valid stock codes in universe file")
        
        logger.info(f"Successfully loaded {len(codes)} stocks from universe file")
        return codes
        
    except Exception as e:
        # If this is not one of our custom errors with clear messages, provide more context
        if not isinstance(e, (ValueError, FileNotFoundError)):
            error_msg = (
                f"Unexpected error loading universe: {str(e)}\n"
                f"This is likely due to an issue with the universe file.\n"
                f"Please ensure {universe_file} is a valid CSV with stock codes."
            )
            logger.error(error_msg)
            print("\n" + error_msg + "\n")
        # Re-raise the exception to be handled by the caller
        raise

def update_data(force_refresh: bool = False, 
               universe: Optional[List[str]] = None, 
               start_date: str = "20200101",
               max_per_batch: int = 5) -> Dict:
    """
    Update all stock data with intelligent batching
    
    Args:
        force_refresh: Whether to force refresh data from API
        universe: List of stock codes (if None, load from universe file)
        start_date: Start date in YYYYMMDD format
        max_per_batch: Maximum stocks per batch
        
    Returns:
        Statistics about the update process
    """
    # Check token status
    token_status = check_token_status()
    if token_status["status"] == "expired":
        logger.info("Token expired, requesting new token")
        get_token(force_refresh=True)
    elif token_status["status"] == "missing":
        logger.info("Token missing, requesting new token")
        get_token(force_refresh=True)
    
    # Load universe if not provided
    if universe is None:
        universe = load_universe()
        
    logger.info(f"Starting data update for {len(universe)} stocks")
    
    # Randomize order to avoid predictable patterns
    universe_copy = universe.copy()
    random.shuffle(universe_copy)
    
    success_count = 0
    error_count = 0
    start_time = time.time()
    
    # Calculate dynamic batch size based on time of day
    hour = dt.datetime.now().hour
    if 9 <= hour < 16:  # Market hours - use smaller batches
        max_per_batch = max(2, max_per_batch - 2)
        logger.info(f"Market hours detected, reducing batch size to {max_per_batch}")
    
    # Process in smaller batches
    batch_count = (len(universe_copy) - 1) // max_per_batch + 1
    for i in range(0, len(universe_copy), max_per_batch):
        batch = universe_copy[i:i+max_per_batch]
        batch_no = i // max_per_batch + 1
        
        # Add some randomness to the batch size for less predictable patterns
        if random.random() < 0.3 and len(batch) > 2:  # 30% chance
            skip = random.randint(1, len(batch) // 2)
            batch = batch[:-skip]
            logger.info(f"Random reduction: batch size reduced to {len(batch)}")
            
        logger.info(f"Processing batch {batch_no}/{batch_count} with {len(batch)} stocks")
        
        try:
            for code in batch:
                try:
                    logger.info(f"Updating {code}")
                    df = get_stock_data(code, start_date, force_refresh=force_refresh)
                    
                    # Log results
                    if not df.empty:
                        date_range = f"{df.index.min().date()} to {df.index.max().date()}"
                        logger.info(f"Got {len(df)} data points for {code} ({date_range})")
                        success_count += 1
                    else:
                        logger.warning(f"Empty dataframe for {code}")
                        error_count += 1
                    
                    # Add varied delays between requests
                    delay = random.uniform(0.1, 0.5)
                    # Occasionally longer delay
                    if random.random() < 0.15:  # 15% chance
                        delay = random.uniform(0.7, 1.5)
                    time.sleep(delay)
                    
                except Exception as e:
                    logger.error(f"Error updating {code}: {e}")
                    error_count += 1
                    time.sleep(random.uniform(1, 3))  # Longer delay after error
            
            # Circuit breaker: if too many consecutive errors, take longer break
            if error_count > 5 and success_count == 0:
                logger.warning("Multiple consecutive errors detected, taking longer break")
                time.sleep(random.uniform(30, 60))
            
            # Calculate time for extra randomness in batch spacing
            if batch_no < batch_count:
                # Base delay between batches
                batch_delay = random.uniform(2.0, 5.0)
                
                # Add occasional longer pauses to avoid detection patterns
                if random.random() < 0.1:  # 10% chance
                    extended_delay = random.uniform(8, 15)
                    logger.info(f"Taking extended break ({extended_delay:.1f}s) to randomize pattern")
                    batch_delay = extended_delay
                    
                # Add extra delay if we had errors
                if error_count > success_count in batch:
                    logger.info("Adding extra delay due to errors")
                    batch_delay += random.uniform(3, 7)
                    
                logger.info(f"Batch {batch_no} complete, waiting {batch_delay:.1f}s")
                time.sleep(batch_delay)
                
        except Exception as e:
            logger.error(f"Batch error: {e}")
            logger.error(traceback.format_exc())
    
    # Create a backup
    backup_result = create_backup()
    
    # Compile statistics
    elapsed = time.time() - start_time
    statistics = {
        "success_count": success_count,
        "error_count": error_count,
        "total_stocks": len(universe),
        "elapsed_time": elapsed,
        "avg_time_per_stock": elapsed / max(1, success_count),
        "timestamp": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "backup_file": backup_result.get("backup_file", None)
    }
    
    # Save statistics to JSON
    stats_file = Path(CACHE_DIR) / "update_stats.json"
    try:
        if stats_file.exists():
            # Load existing stats
            with open(stats_file, 'r') as f:
                existing_stats = json.load(f)
                
            # Append new stats
            if "history" not in existing_stats:
                existing_stats["history"] = []
            existing_stats["history"].append(statistics)
            existing_stats["last_update"] = statistics
            
            # Keep last 30 updates
            existing_stats["history"] = existing_stats["history"][-30:]
            
            with open(stats_file, 'w') as f:
                json.dump(existing_stats, f, indent=2)
        else:
            # Create new stats file
            with open(stats_file, 'w') as f:
                json.dump({
                    "last_update": statistics,
                    "history": [statistics]
                }, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving statistics: {e}")
    
    logger.info(f"Data update complete. Success: {success_count}, Errors: {error_count}")
    logger.info(f"Total time: {elapsed:.2f} seconds, Avg: {elapsed/max(1, len(universe)):.2f}s per stock")
    
    return statistics

def create_backup() -> Dict:
    """
    Create a backup of the cache directory
    
    Returns:
        Dictionary with backup results
    """
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = f"{BACKUP_DIR}/krx_{timestamp}.zip"
    
    try:
        # Use os.system for simplicity
        cmd = f"zip -r {backup_file} {CACHE_DIR}/*.parquet"
        logger.info(f"Running backup command: {cmd}")
        ret = os.system(cmd)
        
        if ret != 0:
            logger.error(f"Backup command failed with return code {ret}")
            return {"status": "error", "error": f"Command returned {ret}"}
            
        logger.info(f"Created backup: {backup_file}")
        
        # Clean up old backups (keep last 10)
        backups = sorted(Path(BACKUP_DIR).glob("krx_*.zip"))
        if len(backups) > 10:
            for old_backup in backups[:-10]:
                old_backup.unlink()
                logger.info(f"Removed old backup: {old_backup}")
                
        return {
            "status": "success", 
            "backup_file": backup_file,
            "size_mb": Path(backup_file).stat().st_size / (1024 * 1024)
        }
    except Exception as e:
        logger.error(f"Failed to create backup: {e}")
        return {"status": "error", "error": str(e)}

def validate_universe_data(universe: Optional[List[str]] = None, min_days: int = 100) -> Dict:
    """
    Validate data for universe stocks
    
    Args:
        universe: List of stock codes
        min_days: Minimum number of days required
        
    Returns:
        Validation results
    """
    if universe is None:
        universe = load_universe()
        
    logger.info(f"Validating data for {len(universe)} stocks")
    
    results = {
        "total": len(universe),
        "valid": 0,
        "missing": 0,
        "incomplete": 0,
        "issues": []
    }
    
    for code in universe:
        cache_file = Path(CACHE_DIR) / f"{code}.parquet"
        if not cache_file.exists():
            results["missing"] += 1
            results["issues"].append({
                "code": code,
                "issue": "missing_file"
            })
            continue
            
        try:
            df = pd.read_parquet(cache_file)
            if df.empty:
                results["incomplete"] += 1
                results["issues"].append({
                    "code": code,
                    "issue": "empty_data"
                })
            elif len(df) < min_days:
                results["incomplete"] += 1
                results["issues"].append({
                    "code": code,
                    "issue": "insufficient_data",
                    "days": len(df)
                })
            else:
                results["valid"] += 1
        except Exception as e:
            results["issues"].append({
                "code": code,
                "issue": "read_error",
                "error": str(e)
            })
            
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KIS Data Batch Update")
    parser.add_argument("--force", action="store_true", help="Force refresh all data")
    parser.add_argument("--validate", action="store_true", help="Validate universe data")
    parser.add_argument("--check-cache", action="store_true", help="Check cache status")
    parser.add_argument("--start-date", type=str, default="20200101", help="Start date (YYYYMMDD)")
    parser.add_argument("--batch-size", type=int, default=5, help="Maximum stocks per batch")
    parser.add_argument("--stocks", type=str, help="Comma-separated list of stock codes to process (e.g. '005930,000660')")
    args = parser.parse_args()
    
    try:
        # Handle manual stock list if provided
        custom_universe = None
        if args.stocks:
            custom_universe = [code.strip() for code in args.stocks.split(',')]
            logger.info(f"Using custom stock list with {len(custom_universe)} stocks")
        
        # Check cache if requested
        if args.check_cache:
            stats = check_cache_status()
            print(json.dumps(stats, indent=2, default=str))
            
        # Validate universe if requested
        if args.validate:
            try:
                results = validate_universe_data(universe=custom_universe)
                print(json.dumps(results, indent=2))
            except Exception as e:
                logger.error(f"Universe validation failed: {e}")
                print(f"\nUniverse validation failed: {e}\n")
                print("You can still run data updates using a custom stock list:")
                print("python batch_update.py --stocks 005930,000660,035420")
                
        # Run data update if requested or if no other action specified
        if args.force or not (args.validate or args.check_cache):
            try:
                # Try to run the update process
                update_data(
                    force_refresh=args.force,
                    universe=custom_universe,
                    start_date=args.start_date,
                    max_per_batch=args.batch_size
                )
            except Exception as e:
                if isinstance(e, ValueError) and "HTML" in str(e):
                    # Special handling for HTML error in universe file
                    print("\n⚠️ KRX API ISSUE DETECTED\n")
                    print("The KRX API appears to be returning HTML error pages.")
                    print("This may indicate that the service is under maintenance.")
                    print("\nYou have two options to proceed:\n")
                    print("1. Wait until the KRX service is back online")
                    print("2. Manually specify stocks to update:\n")
                    print("   python batch_update.py --stocks 005930,000660,035420\n")
                else:
                    # Handle other errors
                    logger.error(f"Batch update failed: {e}")
                    logger.error(traceback.format_exc())
                    print(f"\nError: {e}\n")
                    
                # Provide fallback for manual stock processing
                if not custom_universe:
                    print("Fallback: You can manually specify stocks to update:")
                    print("python batch_update.py --stocks 005930,000660,035420,051910,035720")
                exit(1)
    except Exception as e:
        logger.error(f"Batch update failed: {e}")
        logger.error(traceback.format_exc())
        print(f"\nError: {e}\n")
        exit(1)
