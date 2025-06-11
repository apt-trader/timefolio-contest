#!/usr/bin/env python3
"""
Ticker Validation and Cleaning Script for TimeFolio

This script validates all tickers in the sector_universe.csv file by:
1. Checking if they can be looked up in the KIS API
2. Verifying they have recent price data
3. Generating a clean universe file with only valid tickers
"""
import os
import sys
import time
import pandas as pd
import logging
from datetime import datetime, timedelta
from pathlib import Path
from tqdm import tqdm
from data_manager import get_stock_data
from kis_auth import get_token

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('ticker_validation.log')
    ]
)
logger = logging.getLogger(__name__)

# Ensure required environment variables are set
required_env_vars = ['KIS_APP_KEY', 'KIS_APP_SECRET']
for var in required_env_vars:
    if var not in os.environ:
        logger.error(f"Required environment variable {var} is not set")
        sys.exit(1)

class TickerValidator:
    def __init__(self, universe_file='sector_universe.csv', min_data_points=5):
        self.universe_file = universe_file
        self.min_data_points = min_data_points
        self.results = []
        self.valid_tickers = []
        self.invalid_tickers = []
        self.validation_report = {
            'total_tickers': 0,
            'valid': 0,
            'invalid': 0,
            'reasons': {}
        }
        # Map Korean column names to English
        self.column_map = {
            '종목코드': 'ticker',
            '종목명': 'name',
            '섹터코드': 'sector_code',
            '섹터명': 'sector_name'
        }
    
    def validate_ticker(self, ticker, sector_info):
        """
        Validate a single ticker and return results
        
        Args:
            ticker: Ticker symbol to validate
            sector_info: Dictionary containing sector information
            
        Returns:
            Dictionary with validation results
        """
        # Clean the ticker (remove any 'A' prefix and ensure it's a string)
        clean_ticker = str(ticker).lstrip('A')
        
        result = {
            'ticker': clean_ticker,
            'original_ticker': ticker,
            'sector_code': sector_info.get('sector_code', ''),
            'sector_name': sector_info.get('sector_name', ''),
            'status': 'valid',
            'reason': '',
            'last_price': None,
            'last_trade_date': None,
            'data_points': 0,
            'api_response': None
        }
        
        try:
            # Add delay to avoid rate limiting
            time.sleep(0.5)  # 500ms delay between requests
            
            logger.info(f"Validating ticker: {ticker} (cleaned: {clean_ticker})")
            
            # First try with the cleaned ticker
            test_ticker = clean_ticker
            
            # Get recent data (last 30 days)
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')
            
            try:
                df = get_stock_data(test_ticker, start_date=start_date, force_refresh=True)
            except Exception as e:
                # If that fails, try with the original ticker
                if test_ticker != ticker:
                    logger.warning(f"Failed with cleaned ticker {test_ticker}, trying original {ticker}")
                    test_ticker = ticker
                    df = get_stock_data(test_ticker, start_date=start_date, force_refresh=True)
                else:
                    raise
            
            # Log the first few rows for debugging
            if df is not None and not df.empty:
                logger.debug(f"Data for {test_ticker}:\n{df.head()}")
            
            if df is None or df.empty:
                raise ValueError("No data returned from API")
                
            if len(df) < self.min_data_points:
                raise ValueError(f"Insufficient data points: {len(df)} (min: {self.min_data_points})")
                
            # Get the latest price
            latest_price = df['close'].iloc[-1]
            if pd.isna(latest_price) or latest_price <= 0:
                raise ValueError(f"Invalid price data: {latest_price}")
            
            # If we get here, the ticker is valid
            result.update({
                'ticker': test_ticker,  # Use the ticker that worked
                'last_price': float(latest_price),
                'last_trade_date': df.index.max().strftime('%Y-%m-%d'),
                'data_points': len(df),
                'first_trade_date': df.index.min().strftime('%Y-%m-%d') if not df.empty else None
            })
            
            self.valid_tickers.append(test_ticker)
            logger.info(f"✅ Validated {test_ticker}: {len(df)} data points, price: {latest_price:,.0f}")
            
        except Exception as e:
            error_msg = str(e)
            result.update({
                'status': 'invalid',
                'reason': error_msg,
                'ticker_used': test_ticker if 'test_ticker' in locals() else 'unknown'
            })
            
            # Track reasons for invalidity
            reason = error_msg.split(':')[0].strip()
            if len(reason) > 50:  # Truncate long error messages
                reason = reason[:47] + "..."
                
            self.validation_report['reasons'][reason] = self.validation_report['reasons'].get(reason, 0) + 1
            
            logger.warning(f"❌ Invalid ticker {ticker} (tried: {test_ticker if 'test_ticker' in locals() else ticker}): {error_msg}")
            
        self.results.append(result)
        return result
    
    def validate_universe(self, sample_size=None):
        """
        Validate tickers in the universe file
        
        Args:
            sample_size: Number of tickers to validate (None or 0 for all)
        """
        try:
            # Load universe file with proper encoding for Korean
            try:
                df = pd.read_csv(self.universe_file, dtype={'종목코드': str}, encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(self.universe_file, dtype={'종목코드': str}, encoding='cp949')
            
            # Rename columns to English for easier processing
            df = df.rename(columns=self.column_map)
            
            # Ensure required columns exist
            if 'ticker' not in df.columns:
                raise ValueError("Could not find ticker column in the universe file")
            
            # Sample tickers if requested
            if sample_size and 0 < sample_size < len(df):
                df = df.sample(n=sample_size, random_state=42)
                logger.info(f"Sampled {sample_size} tickers for validation")
            elif sample_size == 0 or sample_size is None:
                logger.info("Processing all tickers in the universe")
            
            self.validation_report['total_tickers'] = len(df)
            
            # Convert to list of dicts for easier processing
            tickers_to_validate = df.to_dict('records')
            
            logger.info(f"\nValidating {len(tickers_to_validate)} tickers...")
            
            # Process each ticker with progress bar
            for item in tqdm(tickers_to_validate, desc="Validating tickers"):
                self.validate_ticker(item['ticker'], item)
                
                # Add small delay to avoid rate limiting
                time.sleep(0.5)
            
            # Generate and print report
            report = self.generate_report()
            print("\n" + "="*80)
            print("VALIDATION REPORT")
            print("="*80)
            print(report)
            
            # Save results
            self.save_clean_universe()
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating universe: {str(e)}", exc_info=True)
            return False
    
    def generate_report(self):
        """Generate a detailed validation report"""
        report = [
            "\n=== TICKER VALIDATION REPORT ===",
            f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Universe file: {self.universe_file}",
            "\n--- Summary ---",
            f"Total tickers: {self.validation_report['total_tickers']}",
            f"Valid tickers: {self.validation_report['valid']}",
            f"Invalid tickers: {self.validation_report['invalid']}",
            "\n--- Invalid Ticker Reasons ---"
        ]
        
        for reason, count in self.validation_report['reasons'].items():
            report.append(f"- {reason}: {count}")
            
        # Add list of invalid tickers
        if self.invalid_tickers:
            report.extend([
                "\n--- Invalid Tickers ---",
                ", ".join(self.invalid_tickers)
            ])
            
        return "\n".join(report)
    
    def save_clean_universe(self, output_file=None):
        """
        Save a clean version of the universe with only valid tickers
        
        Args:
            output_file: Path to save the output file
            
        Returns:
            Path to the saved file if successful, None otherwise
        """
        if not output_file:
            base_name = os.path.splitext(os.path.basename(self.universe_file))[0]
            output_dir = 'validated_universes'
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"{base_name}_validated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        
        valid_records = [r for r in self.results if r['status'] == 'valid']
        
        if not valid_records:
            logger.warning("No valid tickers to save!")
            return None
            
        try:
            # Create DataFrame with original Korean column names
            valid_df = pd.DataFrame(valid_records)
            
            # Map column names back to Korean for output
            reverse_map = {v: k for k, v in self.column_map.items()}
            output_columns = {
                'ticker': reverse_map.get('ticker', 'ticker'),
                'sector_code': reverse_map.get('sector_code', 'sector_code'),
                'sector_name': reverse_map.get('sector_name', 'sector_name')
            }
            
            # Select and rename columns
            valid_df = valid_df[['ticker', 'sector_code', 'sector_name']].rename(columns=output_columns)
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            
            # Save with UTF-8 encoding for Korean characters
            valid_df.to_csv(output_file, index=False, encoding='utf-8-sig')
            
            logger.info(f"Saved {len(valid_df)} valid tickers to {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Error saving clean universe: {str(e)}", exc_info=True)
            return None

def parse_arguments():
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate tickers in the universe file')
    parser.add_argument('--sample', type=int, default=10, 
                       help='Number of tickers to sample (0 for all)')
    parser.add_argument('--full', action='store_true',
                       help='Run validation on all tickers (overrides --sample)')
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Initialize validator
    validator = TickerValidator()
    
    # Determine sample size
    sample_size = 0 if args.full else args.sample
    
    # Validate universe
    if validator.validate_universe(sample_size=sample_size):
        # Save clean universe
        output_file = validator.save_clean_universe()
        if output_file:
            print(f"\nSaved clean universe to: {output_file}")
        else:
            print("\nNo valid tickers to save!")
    
    print("\nValidation complete!")

if __name__ == "__main__":
    main()
