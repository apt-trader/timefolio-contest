import os
import logging
from dotenv import load_dotenv
from tqdm import tqdm

from config import Config
from data_manager import DataManager
from fetchers.financial_fetcher import FinancialsFetcher

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("CachePopulator")

def populate_corp_code_cache():
    """
    Fetches and caches corporation codes for all tickers in the universe
    to speed up subsequent data loading operations.
    """
    load_dotenv()
    api_key = os.getenv('DART_API_KEY')
    if not api_key:
        logger.error("DART_API_KEY not found in .env file. Please create it.")
        return

    logger.info("Initializing components to populate corporation code cache...")

    # Load configuration using the Config class
    try:
        cfg = Config()
        logger.info("Configuration object created successfully.")
    except Exception as e:
        logger.error(f"Error creating configuration object: {e}. Aborting.")
        return
    
    # Initialize DataManager to get the full list of tickers
    dm = DataManager(cfg)
    listing_info_df = dm.get_listing_info()
    
    if listing_info_df.empty:
        logger.error("Could not retrieve listing info. Aborting cache population.")
        return

    tickers = listing_info_df['code'].unique().tolist()
    if not tickers:
        logger.error("No tickers found in listing info. Aborting.")
        return
        
    # Initialize FinancialFetcher, which contains the caching logic
    fetcher = FinancialsFetcher(api_key=api_key)
    
    logger.info(f"Found {len(tickers)} unique tickers. Starting cache population.")
    logger.info("This may take several minutes due to API rate limits, but only needs to be run once.")
    
    # Iterate through all tickers and call find_corp_code
    # This will fetch from API if not in cache, and save the result.
    for ticker in tqdm(tickers, desc="Populating Corp Code Cache"):
        fetcher.find_corp_code(ticker)
        
    logger.info("Corporation code cache has been successfully populated.")
    logger.info("You can now run the tuner, and it will start up much faster.")

if __name__ == "__main__":
    populate_corp_code_cache()
