# utils/sector_parser.py
import logging
import pandas as pd
from datetime import datetime
from typing import Dict, Optional
import re

logger = logging.getLogger(__name__)

def parse_sector_limits(
    file_path: str,
    as_of_date: Optional[datetime] = None
) -> Dict[str, float]:
    """
    Parses the market_sectors.csv file to find the most recent sector weights
    and calculates the corresponding portfolio limits based on Timefolio rules.

    The rule is: limit = max(2 * market_weight, 10%)

    Args:
        file_path (str): The path to the market_sectors.csv file.
        as_of_date (datetime, optional): The date to find the latest limits for.
                                          Defaults to the current date.

    Returns:
        A dictionary mapping sector codes (e.g., 'En', 'Ma') to their
        calculated maximum weight limit (e.g., 0.20 for 20%).
    """
    if as_of_date is None:
        as_of_date = datetime.now()

    logger.info(f"Parsing sector limits from '{file_path}' for date {as_of_date.strftime('%Y-%m-%d')}.")

    try:
        with open(file_path, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        logger.error(f"Sector limits file not found at: {file_path}. Returning default limits.")
        return {"default": 0.10}

    # Find all date sections in the file, e.g., [250404]
    date_sections = re.findall(r'\[(\d{6})\]', content)
    
    # Convert them to datetime objects
    valid_dates = {}
    for date_str in date_sections:
        try:
            # DART dates are YYMMDD, so we need to add the century
            dt = datetime.strptime(f"20{date_str}", '%Y%m%d')
            valid_dates[dt] = date_str
        except ValueError:
            continue

    # Find the most recent date that is not after our as_of_date
    relevant_dates = [d for d in valid_dates if d <= as_of_date]
    if not relevant_dates:
        logger.warning("No valid sector date found before or on the target date. Using default limits.")
        return {"default": 0.10}

    latest_date = max(relevant_dates)
    latest_date_str = valid_dates[latest_date]
    logger.info(f"Using sector weights from section: [{latest_date_str}]")

    # Extract the text block for the latest date
    pattern = re.compile(rf'\[{latest_date_str}\](.*?)(?=\[\d{{6}}\]|\Z)', re.DOTALL)
    match = pattern.search(content)

    if not match:
        logger.error("Could not extract data for the latest date section. Using default limits.")
        return {"default": 0.10}

    sector_block = match.group(1)
    
    # Parse the sector weights from the block
    limits = {}
    for line in sector_block.strip().split('\n'):
        if line.strip().startswith('#') or not line.strip():
            continue
        parts = line.split()
        if len(parts) == 2:
            sector_code, market_weight_pct = parts
            try:
                market_weight = float(market_weight_pct) / 100.0
                
                # Apply the Timefolio rule: max(2x market weight, 10%)
                # This also handles the "if weight < 5%, can go to 10%" case implicitly.
                limit = max(2 * market_weight, 0.10)
                limits[sector_code] = limit
            except ValueError:
                continue

    logger.info(f"Successfully parsed limits for {len(limits)} sectors.")
    return limits