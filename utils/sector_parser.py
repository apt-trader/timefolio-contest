# utils/sector_parser.py
import logging
import pandas as pd
from datetime import datetime
from typing import Dict

logger = logging.getLogger(__name__)

def parse_sector_limits_for_date(
    file_path: str,
    target_date: datetime
) -> Dict[str, float]:
    """
    Parses the market_sectors.csv file, which has a custom format with date headers,
    to find the most recent sector weights for a specific date. It uses the sector weights
    directly as portfolio limits (no calculation applied).

    Args:
        file_path (str): The path to the market_sectors.csv file.
        target_date (datetime): The date to find the latest limits for.

    Returns:
        A dictionary mapping sector codes to their direct weight limits from the CSV.
    """
    default_limits = {
        'En': 0.10, 'Ma': 0.10, 'In': 0.10, 'He': 0.10,
        'Fi': 0.10, 'CD': 0.10, 'CS': 0.10, 'IT': 0.10,
        'Co': 0.10, 'Ut': 0.10, 'Re': 0.10
    }

    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        logger.error(f"Sector limits file not found at: {file_path}. Using default limits.")
        return default_limits

    data = []
    current_date = None
    logger.debug(f"Starting to parse file: {file_path}")
    for i, line in enumerate(lines):
        line = line.strip()
        logger.debug(f"Processing line {i+1}: '{line}'")

        if not line or line.startswith('#'):
            logger.debug("Line is empty or a comment, skipping.")
            continue

        if line.startswith('[') and line.endswith(']'):
            logger.debug("Found potential date header.")
            try:
                date_str = line.strip('[]')
                current_date = pd.to_datetime(date_str, format='%y%m%d')
                logger.debug(f"Parsed date: {current_date}")
            except (ValueError, TypeError):
                logger.warning(f"Could not parse date header: {line}. Resetting current_date.")
                current_date = None
            continue

        if current_date and ',' in line:
            logger.debug("Found potential sector line.")
            try:
                sector, weight_str = line.split(',')
                sector = sector.strip()
                weight = float(weight_str.strip())
                data.append({
                    'date': current_date,
                    'sector': sector,
                    'weight': weight
                })
                logger.debug(f"Appended data: {{'date': {current_date}, 'sector': '{sector}', 'weight': {weight}}})")
            except (ValueError, IndexError):
                logger.warning(f"Could not parse sector line: {line}. Skipping.")
        else:
            logger.debug("Line is not a date header or a valid sector line, skipping.")


    if not data:
        logger.error(f"No valid data parsed from '{file_path}'. Using default limits.")
        return default_limits

    df = pd.DataFrame(data)
    logger.info(f"Parsing sector limits from '{file_path}' for date {target_date.strftime('%Y-%m-%d')}.")

    valid_dates = df[df['date'] <= target_date]['date']
    if valid_dates.empty:
        logger.warning("No valid sector date found on or before target. Using default limits.")
        return default_limits

    most_recent_date = valid_dates.max()
    latest_weights = df[df['date'] == most_recent_date].set_index('sector')['weight']

    # Use sector weights directly as limits (convert from percentage to decimal)
    sector_limits = {sector: weight / 100.0 for sector, weight in latest_weights.items()}

    for sector in default_limits:
        if sector not in sector_limits:
            sector_limits[sector] = 0.10

    logger.info(f"Successfully parsed limits for {len(sector_limits)} sectors.")
    return sector_limits