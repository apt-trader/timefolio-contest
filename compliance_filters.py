#!/usr/bin/env python3
"""
Compliance filter system for TimeFolio portfolio.
Automates checks for:
- Liquidity requirements (average daily volume)
- IPO age restrictions (minimum days since listing)
- Caution/Warning status (from exchange)
- Other regulatory restrictions

Maintains and updates a forbidden.csv file of excluded tickers.
"""
import os
import logging
import os
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

# Import get_token from data_manager
from data_manager import get_token
import random

# Data source
try:
    # Try to use our new KIS API implementation
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # Add current directory to path
    from data_manager import get_stock_data, load_price_data_kis
    USE_KIS_API = True
except ImportError:
    # Fall back to FinanceDataReader if KIS API is not available
    import FinanceDataReader as fdr
    USE_KIS_API = False
    logging.warning("KIS API not available, falling back to FinanceDataReader")
import requests
from bs4 import BeautifulSoup
import re
import csv

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(ch)


class ComplianceFilter:
    """Automated compliance filter for Korean equities."""
    
    def __init__(self, 
                 min_avg_daily_volume: int = 3_000_000_000,  # 30억원 (KRW)
                 min_ipo_days: int = 90,                    # ~3 months
                 output_file: str = 'forbidden.csv',
                 cache_dir: str = 'cache',
                 use_cached: bool = True,
                 cache_expiry_days: int = 1):
        """
        Initialize compliance filter.
        
        Args:
            min_avg_daily_volume: Minimum average daily trading volume in KRW (default: 3 billion KRW)
            min_ipo_days: Minimum days since IPO
            output_file: Output file for forbidden tickers
            cache_dir: Directory for cached data
            use_cached: Whether to use cached data if available
            cache_expiry_days: Days before cached data expires
        """
        self.min_avg_daily_volume = min_avg_daily_volume
        self.min_ipo_days = min_ipo_days
        self.output_file = output_file
        
        # Setup cache
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.use_cached = use_cached
        self.cache_expiry_days = cache_expiry_days
    
    def _get_cached_path(self, name: str) -> Path:
        """Get path for cached data file."""
        return self.cache_dir / f"{name}.pkl"
    
    def _is_cache_valid(self, path: Path) -> bool:
        """Check if cache is still valid based on expiry days."""
        if not path.exists():
            return False
            
        # Check file modification time
        mtime = datetime.fromtimestamp(path.stat().st_mtime)
        age = datetime.now() - mtime
        
        return age.days < self.cache_expiry_days
    
    def _load_cached_data(self, name: str) -> Optional[pd.DataFrame]:
        """Load data from cache if valid."""
        if not self.use_cached:
            return None
            
        path = self._get_cached_path(name)
        if self._is_cache_valid(path):
            try:
                return pd.read_pickle(path)
            except Exception as e:
                logger.warning(f"Error loading cached {name}: {e}")
                
        return None
    
    def _save_to_cache(self, name: str, data: pd.DataFrame):
        """Save data to cache."""
        path = self._get_cached_path(name)
        try:
            data.to_pickle(path)
            logger.info(f"Saved {name} to cache")
        except Exception as e:
            logger.warning(f"Error saving {name} to cache: {e}")
    
    def get_liquidity_filter(self, 
                            universe: List[str],
                            lookback_days: int = 20) -> Set[str]:
        """
        Get tickers that fail liquidity requirements based on 5-day average trading value in KRW.
        
        Args:
            universe: List of tickers to check
            lookback_days: Number of trading days to check for data retrieval
        
        Returns:
            Set of tickers that fail liquidity requirements
        """
        # Try to load from cache
        cached_data = self._load_cached_data('liquidity_filter')
        if cached_data is not None:
            logger.info(f"Using cached liquidity filter data")
            return set(cached_data['ticker'].tolist())
        
        # Calculate date range - get more days than needed to ensure we have enough trading days
        end_date = datetime.now().strftime("%Y-%m-%d")
        # Get more days than needed to account for weekends, holidays
        start_date = (datetime.now() - timedelta(days=lookback_days * 2)).strftime("%Y-%m-%d")
        
        failed_tickers = set()
        price_data = {}
        volume_data = {}
        
        # Fetch price and volume data
        for ticker in universe:
            try:
                # Try KIS API first, fall back to FDR if needed
                if USE_KIS_API:
                    try:
                        # Convert date format from YYYY-MM-DD to YYYYMMDD for KIS API
                        kis_start = start_date.replace('-', '')
                        df = get_stock_data(ticker, kis_start)
                        
                        if df.empty:
                            logger.warning(f"No data found for {ticker} from KIS API, marking as illiquid")
                            failed_tickers.add(ticker)
                            continue
                            
                        # KIS API returns columns as lowercase
                        price_data[ticker] = df['close']
                        volume_data[ticker] = df['volume']
                    except Exception as kis_e:
                        logger.warning(f"Error fetching data for {ticker} from KIS API: {kis_e}, trying FDR")
                        try:
                            # Fall back to Finance Datareader
                            df = fdr.DataReader(ticker, start_date, end_date)
                            if df.empty:
                                logger.warning(f"No data found for {ticker}, marking as illiquid")
                                failed_tickers.add(ticker)
                                continue
                                
                            price_data[ticker] = df['Close']
                            volume_data[ticker] = df['Volume']
                        except Exception as fdr_e:
                            logger.warning(f"Error fetching data for {ticker} from FDR: {fdr_e}")
                            failed_tickers.add(ticker)
                else:
                    # Use KRX data through Finance Datareader
                    df = fdr.DataReader(ticker, start_date, end_date)
                    if df.empty:
                        logger.warning(f"No data found for {ticker}, marking as illiquid")
                        failed_tickers.add(ticker)
                        continue
                        
                    price_data[ticker] = df['Close']
                    volume_data[ticker] = df['Volume']
            except Exception as e:
                logger.warning(f"Error fetching data for {ticker}: {e}")
                failed_tickers.add(ticker)
        
        # Convert to DataFrames
        prices = pd.DataFrame(price_data)
        volumes = pd.DataFrame(volume_data)
        
        # Calculate daily trading value in KRW (price × volume)
        daily_value_krw = prices * volumes
        
        # Calculate 5-day average trading value
        # This aligns with the TimeFolio contest rules for liquidity
        avg_value_5d = daily_value_krw.tail(5).mean().dropna()
        
        logger.info(f"Calculated 5-day avg trading value for {len(avg_value_5d)} tickers")
        
        # Find tickers that fail liquidity requirement
        illiquid_tickers = avg_value_5d[avg_value_5d < self.min_avg_daily_volume].index.tolist()
        logger.info(f"Found {len(illiquid_tickers)} illiquid tickers below {self.min_avg_daily_volume:,} KRW threshold")
        failed_tickers.update(illiquid_tickers)
        
        # Save to cache
        cache_df = pd.DataFrame({'ticker': list(failed_tickers)})
        self._save_to_cache('liquidity_filter', cache_df)
        
        return failed_tickers
    
    def get_ipo_age_filter(self, universe: List[str]) -> Set[str]:
        """
        Get tickers that fail IPO age requirements.
        
        Args:
            universe: List of tickers to check
        
        Returns:
            Set of tickers that fail IPO age requirements
        """
        # Try to load from cache
        cached_data = self._load_cached_data('ipo_filter')
        if cached_data is not None:
            logger.info(f"Using cached IPO filter data")
            return set(cached_data['ticker'].tolist())
        
        # Get stock listing info
        failed_tickers = set()
        
        # Use KRX website or FDR to get listing dates
        try:
            listings = fdr.StockListing('KRX')
            listings['Code'] = listings['Code'].str.zfill(6)
            
            # Keep only tickers in our universe
            universe_set = set(universe)
            listings = listings[listings['Code'].isin(universe_set)]
            
            # Convert listing dates to datetime
            if 'ListingDate' in listings.columns:
                listings['ListingDate'] = pd.to_datetime(listings['ListingDate'], errors='coerce')
            else:
                # Try alternative column names
                date_columns = [col for col in listings.columns if '날짜' in col or 'date' in col.lower()]
                if date_columns:
                    listings['ListingDate'] = pd.to_datetime(listings[date_columns[0]], errors='coerce')
                else:
                    logger.warning("Could not find listing date column")
                    # Skip this check if we can't get listing dates
                    return set()
            
            # Calculate days since IPO
            today = datetime.now().date()
            listings['DaysSinceIPO'] = (today - listings['ListingDate'].dt.date).dt.days
            
            # Find tickers that are too new
            too_new = listings[listings['DaysSinceIPO'] < self.min_ipo_days]['Code'].tolist()
            failed_tickers.update(too_new)
            
        except Exception as e:
            logger.error(f"Error fetching listing data: {e}")
        
        # Save to cache
        cache_df = pd.DataFrame({'ticker': list(failed_tickers)})
        self._save_to_cache('ipo_filter', cache_df)
        
        return failed_tickers
    
    def get_caution_status_filter(self, universe: List[str]) -> Set[str]:
        """
        Get tickers with caution/warning status using KRX API.
        
        Args:
            universe: List of tickers to check
        
        Returns:
            Set of tickers with caution/warning status
        """
        # Try to load from cache
        cached_data = self._load_cached_data('caution_filter')
        if cached_data is not None:
            logger.info("Using cached caution status filter data")
            return set(cached_data['ticker'].tolist())
        
        failed_tickers = set()
        
        try:
            # Use FinanceDataReader to get KRX stock listing info
            import FinanceDataReader as fdr
            
            # Get all KRX stocks with their status
            krx_list = fdr.StockListing('KRX')
            
            # Ensure we have the necessary columns
            if 'Code' not in krx_list.columns or 'Market' not in krx_list.columns:
                logger.warning("KRX listing doesn't have required columns. Cannot check caution status.")
                return set()
            
            # Standardize ticker format
            krx_list['Code'] = krx_list['Code'].astype(str).str.zfill(6)
            
            # Filter for our universe
            universe_set = set(universe)
            krx_filtered = krx_list[krx_list['Code'].isin(universe_set)]
            
            # Check for stocks with caution/warning status
            # These columns might indicate caution/warning status
            caution_columns = [
                'Market', 'Dept', 'Open', 'High', 'Low', 'Close', 'Volume',
                'Amount', 'Marcap', 'Stocks', 'MarketId', 'Rank', 'ListingDate'
            ]
            
            # Check each stock
            for _, row in krx_filtered.iterrows():
                ticker = row['Code']
                
                # Check for trading halt (시장구분)
                if 'Market' in row and pd.notna(row['Market']):
                    market_status = str(row['Market']).strip()
                    if market_status in ['거래정지', '관리종목', '투자경고', '투자위험', '거래정지(관리)']:
                        failed_tickers.add(ticker)
                        logger.info(f"Added {ticker} to caution list - status: {market_status}")
                        continue
                
                # Check for other warning signs in other columns
                for col in caution_columns:
                    if col in row and pd.notna(row[col]):
                        value = str(row[col]).strip().lower()
                        warning_terms = ['정지', '경고', '위험', '관리', '유상증자', '무상증자', '감리', '거래정지']
                        if any(term in value for term in warning_terms):
                            failed_tickers.add(ticker)
                            logger.info(f"Added {ticker} to caution list - {col}: {value}")
                            break
            
            logger.info(f"Found {len(failed_tickers)} tickers with caution/status issues")
            
        except Exception as e:
            logger.error(f"Error in get_caution_status_filter using KRX: {str(e)}")
            # If we can't get status, don't fail the whole process
            return set()
            
        # Save to cache
        cache_df = pd.DataFrame({'ticker': list(failed_tickers)})
        self._save_to_cache('caution_filter', cache_df)
        
        return failed_tickers
    
    def apply_all_filters(self, universe: List[str]) -> Dict[str, Set[str]]:
        """
        Apply all compliance filters to universe.
        
        Args:
            universe: List of tickers to check
        
        Returns:
            Dictionary mapping filter names to sets of filtered tickers
        """
        logger.info(f"Applying compliance filters to {len(universe)} tickers")
        
        results = {
            'liquidity': self.get_liquidity_filter(universe),
            'ipo_age': self.get_ipo_age_filter(universe),
            'caution_status': self.get_caution_status_filter(universe)
        }
        
        # Additional custom filters can be added here
        
        # Collect all failed tickers
        all_filtered = set()
        for filter_name, filtered_tickers in results.items():
            logger.info(f"{filter_name} filter removed {len(filtered_tickers)} tickers")
            all_filtered.update(filtered_tickers)
        
        logger.info(f"All filters removed {len(all_filtered)} tickers")
        
        # Update the results dict with a summary
        results['all'] = all_filtered
        
        return results
    
    def save_forbidden_tickers(self, filtered_tickers: Set[str], reason_dict: Dict[str, Set[str]]):
        """
        Save forbidden tickers to CSV with reasons.
        
        Args:
            filtered_tickers: Set of all filtered tickers
            reason_dict: Dictionary mapping filter names to sets of filtered tickers
        """
        # Prepare data for CSV
        rows = []
        for ticker in filtered_tickers:
            reasons = []
            for filter_name, ticker_set in reason_dict.items():
                if filter_name != 'all' and ticker in ticker_set:
                    reasons.append(filter_name)
            
            rows.append({
                'ticker': ticker,
                'reason': ', '.join(reasons),
                'date_added': datetime.now().strftime("%Y-%m-%d")
            })
        
        # Save to CSV
        df = pd.DataFrame(rows)
        df.to_csv(self.output_file, index=False)
        logger.info(f"Saved {len(filtered_tickers)} forbidden tickers to {self.output_file}")
    
    def run_and_save(self, universe: List[str]):
        """
        Run all filters and save results.
        
        Args:
            universe: List of tickers to check
        """
        filter_results = self.apply_all_filters(universe)


def filter_universe(
    universe: List[str],
    min_avg_daily_volume: int = 3_000_000_000,  # 3 billion KRW minimum 5-day avg volume
    min_ipo_days: int = 90,                     # 3 months minimum IPO age
    output_file: str = 'forbidden.csv',
    cache_dir: str = 'cache',
    use_cached: bool = True,
    cache_expiry_days: int = 1
) -> List[str]:
    """
    Convenience function to filter universe and return clean list.
    
    Args:
        universe: List of tickers to check
        min_avg_daily_volume: Minimum average daily trading volume in KRW
        min_ipo_days: Minimum days since IPO
        output_file: Output file for forbidden tickers
        cache_dir: Directory for cached data
        use_cached: Whether to use cached data if available
        cache_expiry_days: Days before cached data expires
    
    Returns:
        List of compliant tickers
    """
    # Create filter
    cf = ComplianceFilter(
        min_avg_daily_volume=min_avg_daily_volume,
        min_ipo_days=min_ipo_days,
        output_file=output_file,
        cache_dir=cache_dir,
        use_cached=use_cached,
        cache_expiry_days=cache_expiry_days
    )
    
    # Apply filters
    filter_results = cf.apply_all_filters(universe)
    
    # Save forbidden tickers
    cf.save_forbidden_tickers(filter_results['all'], filter_results)
    
    # Return clean universe
    return [ticker for ticker in universe if ticker not in filter_results['all']]


def load_forbidden_tickers(file_path: str = 'forbidden.csv') -> Set[str]:
    """
    Load forbidden tickers from CSV.
    
    Args:
        file_path: Path to forbidden tickers CSV
    
    Returns:
        Set of forbidden tickers
    """
    if not os.path.exists(file_path):
        logger.warning(f"Forbidden tickers file {file_path} not found")
        return set()
        
    try:
        df = pd.read_csv(file_path)
        return set(df['ticker'].astype(str).str.zfill(6).tolist())
    except Exception as e:
        logger.error(f"Error loading forbidden tickers: {e}")
        return set()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Apply compliance filters to stock universe")
    parser.add_argument("--universe-file", required=True, help="CSV file with stock universe")
    parser.add_argument("--ticker-column", default="ticker", help="Column name for tickers")
    parser.add_argument("--min-volume", type=int, default=300_000_000, help="Minimum avg daily volume in KRW")
    parser.add_argument("--min-ipo-days", type=int, default=90, help="Minimum days since IPO")
    parser.add_argument("--output", default="forbidden.csv", help="Output file for forbidden tickers")
    parser.add_argument("--refresh-cache", action="store_true", help="Ignore cached data")
    args = parser.parse_args()
    
    # Load universe
    universe_df = pd.read_csv(args.universe_file)
    universe = universe_df[args.ticker_column].astype(str).str.strip().str.zfill(6).tolist()
    
    logger.info(f"Loaded {len(universe)} tickers from {args.universe_file}")
    
    # Create and run filter
    cf = ComplianceFilter(
        min_avg_daily_volume=args.min_volume,
        min_ipo_days=args.min_ipo_days,
        output_file=args.output,
        use_cached=not args.refresh_cache
    )
    
    cf.run_and_save(universe)
    
    # Report results
    forbidden = load_forbidden_tickers(args.output)
    logger.info(f"Filtered out {len(forbidden)} tickers ({len(forbidden)/len(universe):.1%} of universe)")
    logger.info(f"Clean universe contains {len(universe) - len(forbidden)} tickers")
