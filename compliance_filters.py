# compliance_filters.py
import logging
import pandas as pd
from datetime import datetime
import os
from pathlib import Path
from typing import Dict, List, Set

# Configure logging for this module
logger = logging.getLogger(__name__)

class ComplianceFilter:
    """
    Applies compliance filters based on pre-fetched market data.
    This class does NOT fetch data itself; it processes DataFrames provided to it.
    """
    def __init__(self,
                 min_avg_daily_value: int = 3_000_000_000,  # 3 billion KRW
                 min_ipo_days: int = 90,
                 output_file: str = 'forbidden.csv'):
        """
        Initializes the compliance filter with contest rules.

        Args:
            min_avg_daily_value: Minimum 5-day average trading value in KRW.
            min_ipo_days: Minimum number of days a stock must be listed to be eligible.
            output_file: Path to save the list of filtered tickers.
        """
        self.min_avg_daily_value = min_avg_daily_value
        self.min_ipo_days = min_ipo_days
        self.output_file = Path(output_file)
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        logger.info("ComplianceFilter initialized.")

    def get_liquidity_filter(self, prices: pd.DataFrame, volumes: pd.DataFrame) -> Set[str]:
        """
        Filters tickers that fail the liquidity requirement.

        Args:
            prices (pd.DataFrame): DataFrame of closing prices.
            volumes (pd.DataFrame): DataFrame of trading volumes.

        Returns:
            A set of ticker codes that are considered illiquid.
        """
        logger.debug("Applying liquidity filter...")
        if prices.empty or volumes.empty:
            logger.warning("Price or volume data is empty. Skipping liquidity filter.")
            return set()

        daily_value = prices * volumes
        # Use a 5-day rolling average to match contest rules
        avg_5day_value = daily_value.rolling(window=5, min_periods=1).mean().iloc[-1]

        illiquid_tickers = avg_5day_value[avg_5day_value < self.min_avg_daily_value]
        logger.info(f"Liquidity filter identified {len(illiquid_tickers)} tickers below the "
                    f"{self.min_avg_daily_value:,.0f} KRW threshold.")
        return set(illiquid_tickers.index)

    def get_ipo_age_filter(self, listings_info: pd.DataFrame) -> Set[str]:
        """
        Filters tickers that fail the minimum IPO age requirement.

        Args:
            listings_info (pd.DataFrame): DataFrame containing stock listing information,
                                          including a 'ListingDate' column.

        Returns:
            A set of ticker codes that are too new to be traded.
        """
        logger.debug("Applying IPO age filter...")
        if 'ListingDate' not in listings_info.columns:
            logger.warning("'ListingDate' column not found. Skipping IPO age filter.")
            return set()

        # Ensure ListingDate is in datetime format
        listings_info['ListingDate'] = pd.to_datetime(listings_info['ListingDate'], errors='coerce')
        
        # Calculate days since IPO
        days_since_listing = (datetime.now() - listings_info['ListingDate']).dt.days
        
        # Contest rule: tradeable from the 6th business day. A 90-day filter is much stricter
        # and safer. We use the configured min_ipo_days.
        too_new_tickers = listings_info[days_since_listing < self.min_ipo_days]
        logger.info(f"IPO age filter identified {len(too_new_tickers)} tickers listed "
                    f"less than {self.min_ipo_days} days ago.")
        return set(too_new_tickers.index)

    def get_caution_status_filter(self, listings_info: pd.DataFrame) -> Set[str]:
        """
        Filters tickers with a 'caution', 'warning', 'risk', or 'halted' status.

        Args:
            listings_info (pd.DataFrame): DataFrame containing stock listing information.
                                          Expected to have a 'Market' or status-like column.

        Returns:
            A set of ticker codes with cautionary designations.
        """
        logger.debug("Applying caution status filter...")
        caution_statuses = {'관리종목', '거래정지', '투자주의', '투자경고', '투자위험'}
        
        # The 'Market' column in fdr.StockListing often contains this status info
        if 'Market' not in listings_info.columns:
            logger.warning("'Market' column not found. Skipping caution status filter.")
            return set()

        caution_tickers = listings_info[listings_info['Market'].isin(caution_statuses)]
        logger.info(f"Caution status filter identified {len(caution_tickers)} tickers.")
        return set(caution_tickers.index)

    def apply_all_filters(self,
                          prices: pd.DataFrame,
                          volumes: pd.DataFrame,
                          listings_info: pd.DataFrame) -> Dict[str, Set[str]]:
        """
        Applies all compliance filters and returns a dictionary of filtered sets.

        Args:
            prices (pd.DataFrame): Closing price data.
            volumes (pd.DataFrame): Volume data.
            listings_info (pd.DataFrame): Stock exchange listing information.

        Returns:
            A dictionary where keys are filter names and values are sets of
            tickers excluded by that filter. Includes an 'all' key for the union.
        """
        logger.info("Applying all compliance filters...")
        results = {
            'liquidity': self.get_liquidity_filter(prices, volumes),
            'ipo_age': self.get_ipo_age_filter(listings_info),
            'caution_status': self.get_caution_status_filter(listings_info)
        }
        
        # Combine all filtered tickers into a single set
        all_filtered = set().union(*results.values())
        results['all'] = all_filtered
        
        logger.info(f"Total unique tickers filtered out: {len(all_filtered)}")
        return results

    def save_forbidden_list(self, filter_results: Dict[str, Set[str]]):
        """
        Saves the consolidated list of forbidden tickers to a CSV file.

        Args:
            filter_results (Dict[str, Set[str]]): The output from apply_all_filters.
        """
        rows = []
        for ticker in sorted(list(filter_results.get('all', set()))):
            reasons = [name for name, f_set in filter_results.items() if name != 'all' and ticker in f_set]
            rows.append({
                'ticker': ticker,
                'reason': ', '.join(reasons),
                'date_added': datetime.now().strftime("%Y-%m-%d")
            })
        
        if not rows:
            logger.info("No tickers were filtered. 'forbidden.csv' will be empty.")
            # Create an empty file with headers
            pd.DataFrame(columns=['ticker', 'reason', 'date_added']).to_csv(self.output_file, index=False)
            return

        df = pd.DataFrame(rows)
        df.to_csv(self.output_file, index=False)
        logger.info(f"Saved {len(df)} forbidden tickers to {self.output_file}")


def load_forbidden_tickers(file_path: str = 'forbidden.csv') -> Set[str]:
    """
    Loads the set of forbidden tickers from the specified CSV file.

    Args:
        file_path (str): The path to the forbidden tickers CSV file.

    Returns:
        A set of ticker codes that should be excluded from trading.
    """
    if not os.path.exists(file_path):
        return set()
    try:
        df = pd.read_csv(file_path, dtype={'ticker': str})
        return set(df['ticker'].str.zfill(6).tolist())
    except Exception as e:
        logger.error(f"Error loading forbidden tickers from {file_path}: {e}")
        return set()