#!/usr/bin/env python3
"""
KRX Data Backfill Script

This script backfills historical KRX market data for a specified date range.
It handles weekends, holidays, and rate limiting automatically.
"""
import sys
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from fetchers.krx_fetcher import KRXDataFetcher

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('backfill_krx_data.log')
    ]
)
logger = logging.getLogger('backfill_krx')

def get_krx_trading_days(start_date: str, end_date: str) -> list:
    """
    Get list of trading days between start_date and end_date.
    
    Args:
        start_date: Start date in 'YYYYMMDD' format
        end_date: End date in 'YYYYMMDD' format
        
    Returns:
        List of trading days in 'YYYYMMDD' format
    """
    # Convert string dates to datetime objects
    start = datetime.strptime(start_date, '%Y%m%d')
    end = datetime.strptime(end_date, '%Y%m%d')
    
    # Generate all dates in the range
    date_range = pd.date_range(start, end, freq='B')  # Business days only
    
    # Convert to list of strings in 'YYYYMMDD' format
    return [d.strftime('%Y%m%d') for d in date_range]

def backfill_data(start_date: str, end_date: str, db_path: str = 'krx_data.db',
                 cache_dir: str = 'krx_cache', min_delay: float = 2.0, 
                 max_delay: float = 5.0) -> None:
    """
    Backfill KRX data for the specified date range.
    
    Args:
        start_date: Start date in 'YYYYMMDD' format
        end_date: End date in 'YYYYMMDD' format
        db_path: Path to SQLite database
        cache_dir: Directory for cached files
        throttle: Seconds to wait between requests
    """
    # Initialize fetcher
    fetcher = KRXDataFetcher(
        db_path=db_path,
        cache_dir=cache_dir,
        throttle=min_delay  # Use min_delay as the base throttle time
    )
    
    # Get trading days in the range
    trading_days = get_krx_trading_days(start_date, end_date)
    logger.info(f"Found {len(trading_days)} trading days between {start_date} and {end_date}")
    
    # Get already processed dates from database
    try:
        existing_dates = set(
            d.strftime('%Y%m%d') 
            for d in fetcher.get_available_dates()
        )
    except Exception as e:
        logger.warning(f"Could not fetch existing dates: {e}")
        existing_dates = set()
    
    # Filter out already processed dates
    dates_to_process = [d for d in trading_days if d not in existing_dates]
    
    if not dates_to_process:
        logger.info("All dates already processed")
        return
    
    logger.info(f"Processing {len(dates_to_process)} new trading days")
    
    # Process each date with progress bar
    success_count = 0
    error_count = 0
    
    for date_str in tqdm(dates_to_process, desc="Processing trading days"):
        try:
            # Fetch data for both markets
            results = fetcher.fetch_daily_data(date_str)
            
            if not results:
                logger.warning(f"No data returned for {date_str}")
                error_count += 1
                continue
            
            # Update database for each market
            for market, df in results.items():
                if df.empty:
                    logger.warning(f"Empty DataFrame for {market} on {date_str}")
                    continue
                    
                try:
                    count = fetcher.update_database(df)
                    logger.debug(f"Updated {count} rows for {market} on {date_str}")
                except Exception as e:
                    logger.error(f"Error updating database for {market} on {date_str}: {e}")
                    error_count += 1
                    continue
            
            success_count += 1
            
            # Add randomized delay between 2-5 seconds to be nice to the server
            delay = 2 + (time.time() % 3)  # Random delay between 2-5 seconds
            time.sleep(delay)
            
        except Exception as e:
            logger.error(f"Error processing {date_str}: {e}")
            error_count += 1
            # Continue with next date even if one fails
            continue
    
    logger.info(f"Backfill complete. Success: {success_count}, Failed: {error_count}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Backfill KRX market data')
    parser.add_argument('--start-date', default='20220101',
                       help='Start date in YYYYMMDD format (default: 20220101)')
    parser.add_argument('--end-date', 
                       default=datetime.now().strftime('%Y%m%d'),
                       help='End date in YYYYMMDD format (default: today)')
    parser.add_argument('--db-path', default='krx_data.db',
                       help='Path to SQLite database file')
    parser.add_argument('--cache-dir', default='krx_cache',
                       help='Directory for cached files')
    parser.add_argument('--min-delay', type=float, default=2.0,
                       help='Minimum seconds to wait between requests (default: 2.0)')
    parser.add_argument('--max-delay', type=float, default=5.0,
                       help='Maximum seconds to wait between requests (default: 5.0)')
    
    args = parser.parse_args()
    
    # Validate dates
    try:
        start = datetime.strptime(args.start_date, '%Y%m%d')
        end = datetime.strptime(args.end_date, '%Y%m%d')
        
        if start > end:
            logger.error("Start date must be before end date")
            sys.exit(1)
            
        logger.info(f"Starting backfill from {args.start_date} to {args.end_date}")
        if args.min_delay >= args.max_delay:
            logger.error("min-delay must be less than max-delay")
            sys.exit(1)
            
        logger.info(f"Using random delay between {args.min_delay:.1f} and {args.max_delay:.1f} seconds between requests")
        
        backfill_data(
            start_date=args.start_date,
            end_date=args.end_date,
            db_path=args.db_path,
            cache_dir=args.cache_dir,
            min_delay=args.min_delay,
            max_delay=args.max_delay
        )
        
    except ValueError as e:
        logger.error(f"Invalid date format. Please use YYYYMMDD format. Error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        sys.exit(1)
