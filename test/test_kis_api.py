#!/usr/bin/env python3
"""
Test script to verify KIS API connectivity
"""
import os
import sys
import logging
from data_manager import get_stock_data, inquire_price, check_cache_status

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def test_kis_api():
    """Test KIS API connectivity with a sample stock"""
    try:
        # Test 1: Get current price of Samsung Electronics (005930)
        logger.info("Testing KIS API with Samsung Electronics (005930)...")
        price = inquire_price("005930")
        logger.info(f"Current price of 005930: {price:,} KRW")
        
        # Test 2: Get historical data for the past 5 days
        logger.info("\nTesting historical data fetch...")
        import datetime as dt
        end_date = dt.datetime.now().strftime("%Y%m%d")
        start_date = (dt.datetime.now() - dt.timedelta(days=5)).strftime("%Y%m%d")
        
        data = get_stock_data("005930", start_date=start_date)
        logger.info(f"\nRetrieved {len(data)} days of data:")
        print(data[['open', 'high', 'low', 'close', 'volume']].tail())
        
        # Test 3: Check cache status
        logger.info("\nCache status:")
        cache_status = check_cache_status()
        for k, v in cache_status.items():
            logger.info(f"{k}: {v}")
            
        return True
        
    except Exception as e:
        logger.error(f"KIS API test failed: {str(e)}", exc_info=True)
        return False

if __name__ == "__main__":
    print("Testing KIS API connectivity...\n" + "="*50)
    success = test_kis_api()
    if success:
        print("\n✅ KIS API test completed successfully!")
    else:
        print("\n❌ KIS API test failed. Please check the error messages above.")
