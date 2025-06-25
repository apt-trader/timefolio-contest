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
                 output_file: str = 'forbidden.csv',
                 skip_rules: bool = False):
        """
        Initializes the compliance filter with contest rules.

        Args:
            min_avg_daily_value: Minimum 5-day average trading value in KRW.
            min_ipo_days: Minimum number of days a stock must be listed to be eligible.
            output_file: Path to save the list of filtered tickers.
            skip_rules: If True, skip all compliance rules (liquidity, IPO age, etc.)
        """
        self.min_avg_daily_value = min_avg_daily_value
        self.min_ipo_days = min_ipo_days
        self.output_file = Path(output_file)
        self.skip_rules = skip_rules
            
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        
        if self.skip_rules:
            logger.info("ComplianceFilter initialized with ALL RULES DISABLED (tuning mode).")
        else:
            logger.info("ComplianceFilter initialized with all rules ENABLED (contest mode).")

    def get_liquidity_filter(self, prices: pd.DataFrame, volumes: pd.DataFrame) -> Set[str]:
        """
        Filters tickers that fail the liquidity requirement.

        Args:
            prices (pd.DataFrame): DataFrame of closing prices.
            volumes (pd.DataFrame): DataFrame of trading volumes.

        Returns:
            A set of ticker codes that are considered illiquid.
        """
        if self.skip_rules:
            logger.debug("Skipping liquidity filter (tuning mode).")
            return set()
            
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

    def get_ipo_age_filter(self, listings_info: pd.DataFrame, end_date: datetime) -> Set[str]:
        """
        Filters tickers that fail the minimum IPO age requirement as of a specific date.

        Args:
            listings_info (pd.DataFrame): DataFrame with 'code' and 'listing_date'.
            end_date (datetime): The date to calculate the IPO age against.

        Returns:
            A set of ticker codes that are too new to be traded.
        """
        if self.skip_rules:
            logger.debug("Skipping IPO age filter (tuning mode).")
            return set()

        logger.debug(f"Applying IPO age filter for date {end_date.strftime('%Y-%m-%d')}...")
        if listings_info.empty or 'listing_date' not in listings_info.columns:
            logger.warning("No listing date information available. Skipping IPO age filter.")
            return set()

        # Ensure DataFrame is indexed by ticker code for correct filtering
        if 'code' in listings_info.columns:
            listings_info = listings_info.set_index('code')

        # Calculate days since listing relative to the end_date
        days_since_listing = (end_date - pd.to_datetime(listings_info['listing_date'])).dt.days

        # Filter stocks that haven't met the minimum listing period
        new_tickers = listings_info[days_since_listing < self.min_ipo_days].index
        logger.info(f"IPO age filter identified {len(new_tickers)} tickers listed for "
                    f"less than {self.min_ipo_days} days as of {end_date.strftime('%Y-%m-%d')}.")
        return set(new_tickers)

    def get_caution_status_filter(self, listings_info: pd.DataFrame) -> Set[str]:
        """
        Filters tickers with a 'caution', 'warning', 'risk', or 'halted' status.

        Args:
            listings_info (pd.DataFrame): DataFrame containing stock listing information.
                                          Expected to have a 'Market' or status-like column.

        Returns:
            A set of ticker codes with cautionary designations.
        """
        if self.skip_rules:
            logger.debug("Skipping caution status filter (tuning mode).")
            return set()
            
        logger.debug("Applying caution/warning status filter...")
        if listings_info.empty:
            logger.warning("No listing information available. Skipping caution status filter.")
            return set()

        # Check for caution/warning indicators in the data
        # This is a placeholder - adjust based on your actual data structure
        caution_indicators = ['caution', 'warning', 'risk', 'halted', '관리', '경고', '위험']
        
        # Initialize set to store tickers with cautionary status
        caution_tickers = set()
        
        # Check each column that might contain status information
        for col in listings_info.columns:
            if listings_info[col].dtype == 'object':  # Only check string columns
                for indicator in caution_indicators:
                    # Find rows where the column contains any caution indicator
                    mask = listings_info[col].str.contains(indicator, case=False, na=False)
                    if mask.any():
                        matched = set(listings_info[mask].index)
                        caution_tickers.update(matched)
                        logger.debug(f"Found {len(matched)} tickers with '{indicator}' in column '{col}'")
        
        logger.info(f"Caution status filter identified {len(caution_tickers)} tickers with caution/warning indicators.")
        return caution_tickers

    def apply_all_filters(self,
                          prices: pd.DataFrame,
                          volumes: pd.DataFrame,
                          listings_info: pd.DataFrame,
                          end_date: datetime) -> Dict[str, Set[str]]:
        """
        Applies all compliance filters and returns a dictionary of filtered sets.

        Args:
            prices (pd.DataFrame): Closing price data.
            volumes (pd.DataFrame): Volume data.
            listings_info (pd.DataFrame): Stock exchange listing information.
            end_date (datetime): The reference date for time-sensitive filters like IPO age.

        Returns:
            A dictionary where keys are filter names and values are sets of
            tickers excluded by that filter. Includes an 'all' key for the union.
        """
        logger.info(f"Applying all compliance filters for date {end_date.strftime('%Y-%m-%d')}...")
        results = {
            'liquidity': self.get_liquidity_filter(prices, volumes),
            'ipo_age': self.get_ipo_age_filter(listings_info, end_date),
            'caution_status': self.get_caution_status_filter(listings_info)
        }
        
        # Combine all filtered tickers into a single set
        all_filtered = set().union(*list(results.values()))
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
            reasons = [name for name, f_set in list(filter_results.items()) if name != 'all' and ticker in f_set]
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