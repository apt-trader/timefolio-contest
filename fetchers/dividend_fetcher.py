# fetchers/dividend_fetcher.py
"""
DS002 - 배당정보 (Dividend Info) Fetcher
API: alotMatter (배당에 관한 사항)

Fetches dividend data from DART OpenAPI for Value signal enhancement.
"""
import os
import sys
import time
import random
import argparse
import logging
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import OpenDartReader
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

logs_dir = Path(__file__).parent.parent / 'logs'
logs_dir.mkdir(exist_ok=True, parents=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(logs_dir / 'dividend_fetcher.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DividendFetcher:
    """Fetches dividend data from DART API (DS002 - alotMatter)."""
    
    def __init__(self, api_key: Optional[str], db_path: str):
        self.db_path = db_path
        self.dart = None
        
        if api_key:
            try:
                self.dart = OpenDartReader(api_key)
            except ValueError as e:
                logger.error(f"Failed to initialize OpenDartReader: {e}")
        else:
            logger.warning("DART API key not provided.")
        
        self._init_db_structure()
        self.ticker_to_corp_code_map = self._load_company_mapping()
        logger.info(f"DividendFetcher initialized for DB at {db_path}.")
    
    def _create_connection(self):
        return sqlite3.connect(self.db_path)
    
    def _init_db_structure(self):
        """Initialize dividends table if not exists."""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS dividends (
            corp_code TEXT NOT NULL,
            ticker TEXT NOT NULL,
            fiscal_year INTEGER NOT NULL,
            report_code TEXT NOT NULL,
            se TEXT,                          -- 구분 (주식종류)
            stock_knd TEXT,                   -- 주식종류
            thstrm REAL,                      -- 당기 배당금
            frmtrm REAL,                      -- 전기 배당금
            lwfr REAL,                        -- 전전기 배당금
            thstrm_rate REAL,                 -- 당기 배당률
            frmtrm_rate REAL,                 -- 전기 배당률
            lwfr_rate REAL,                   -- 전전기 배당률
            dividend_per_share REAL,          -- 주당 배당금
            dividend_yield REAL,              -- 배당수익률
            total_dividend REAL,              -- 배당금 총액
            payout_ratio REAL,                -- 배당성향
            fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (corp_code, fiscal_year, report_code, se)
        );
        """
        try:
            with self._create_connection() as conn:
                conn.execute(create_table_sql)
                conn.commit()
            logger.info("Dividends table initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize dividends table: {e}")
            raise
    
    def _load_company_mapping(self) -> Dict[str, str]:
        """Load ticker to corp_code mapping from database."""
        logger.info("Loading company ticker-to-corp_code mapping...")
        try:
            with self._create_connection() as conn:
                df = pd.read_sql_query(
                    "SELECT ticker, corp_code FROM company_mapping WHERE ticker IS NOT NULL", 
                    conn
                )
                df['ticker'] = df['ticker'].astype(str).str.zfill(6)
                mapping = df.set_index('ticker')['corp_code'].to_dict()
                logger.info(f"Loaded {len(mapping)} company mappings.")
                return mapping
        except Exception as e:
            logger.error(f"Failed to load company mapping: {e}")
            return {}
    
    def find_corp_code(self, ticker: str) -> Optional[str]:
        """Find corp_code for a ticker."""
        return self.ticker_to_corp_code_map.get(ticker)
    
    def fetch_dividend_for_ticker(self, ticker: str, year: int, report_code: str) -> bool:
        """
        Fetch dividend data for a single ticker.
        
        Args:
            ticker: Stock ticker (6-digit)
            year: Fiscal year
            report_code: Report type ('11011'=Annual, '11012'=Q2, '11013'=Q1, '11014'=Q3)
        
        Returns:
            True if successful, False otherwise
        """
        corp_code = self.find_corp_code(ticker)
        if not corp_code:
            logger.debug(f"No corp_code for {ticker}")
            return False
        
        if not self.dart:
            logger.error("DART API not configured")
            return False
        
        try:
            # Call DART API for dividend info (alotMatter)
            df = self.dart.report(corp_code, '배당', year, report_code)
            
            if df is None or df.empty:
                logger.debug(f"No dividend data for {ticker} ({year} {report_code})")
                return False
            
            # Process and store
            df['corp_code'] = corp_code
            df['ticker'] = str(ticker).zfill(6)
            df['fiscal_year'] = year
            df['report_code'] = report_code
            
            self._store_dividend_data(df)
            return True
            
        except Exception as e:
            logger.debug(f"Error fetching dividend for {ticker}: {e}")
            return False
    
    def _store_dividend_data(self, df: pd.DataFrame):
        """Store dividend data to database."""
        if df.empty:
            return
        
        # Map Korean column names to English
        column_mapping = {
            '구분': 'se',
            '주식의 종류': 'stock_knd',
            '당기': 'thstrm',
            '전기': 'frmtrm',
            '전전기': 'lwfr',
        }
        
        df = df.rename(columns=column_mapping)
        
        # Extract key metrics
        processed_rows = []
        
        for _, row in df.iterrows():
            se = row.get('se', '')
            
            # Look for dividend per share row
            if '주당' in str(se) and '배당금' in str(se):
                processed_rows.append({
                    'corp_code': row['corp_code'],
                    'ticker': row['ticker'],
                    'fiscal_year': row['fiscal_year'],
                    'report_code': row['report_code'],
                    'se': se,
                    'stock_knd': row.get('stock_knd', ''),
                    'dividend_per_share': self._parse_numeric(row.get('thstrm')),
                    'thstrm': self._parse_numeric(row.get('thstrm')),
                    'frmtrm': self._parse_numeric(row.get('frmtrm')),
                    'lwfr': self._parse_numeric(row.get('lwfr')),
                })
            
            # Look for dividend yield row
            if '배당수익률' in str(se):
                for pr in processed_rows:
                    if pr['ticker'] == row['ticker'] and pr['fiscal_year'] == row['fiscal_year']:
                        pr['dividend_yield'] = self._parse_numeric(row.get('thstrm'))
            
            # Look for payout ratio row
            if '배당성향' in str(se):
                for pr in processed_rows:
                    if pr['ticker'] == row['ticker'] and pr['fiscal_year'] == row['fiscal_year']:
                        pr['payout_ratio'] = self._parse_numeric(row.get('thstrm'))
        
        if not processed_rows:
            # Store raw data if no specific rows found
            for _, row in df.iterrows():
                processed_rows.append({
                    'corp_code': row['corp_code'],
                    'ticker': row['ticker'],
                    'fiscal_year': row['fiscal_year'],
                    'report_code': row['report_code'],
                    'se': row.get('se', ''),
                    'stock_knd': row.get('stock_knd', ''),
                    'thstrm': self._parse_numeric(row.get('thstrm')),
                    'frmtrm': self._parse_numeric(row.get('frmtrm')),
                    'lwfr': self._parse_numeric(row.get('lwfr')),
                })
        
        if processed_rows:
            result_df = pd.DataFrame(processed_rows)
            try:
                with self._create_connection() as conn:
                    result_df.to_sql('dividends_temp', conn, if_exists='replace', index=False)
                    
                    # Get common columns
                    main_cols = {col[1] for col in conn.execute("PRAGMA table_info(dividends)").fetchall()}
                    temp_cols = {col[1] for col in conn.execute("PRAGMA table_info(dividends_temp)").fetchall()}
                    common_cols = list(main_cols.intersection(temp_cols))
                    cols_str = ', '.join(f'"{c}"' for c in common_cols)
                    
                    conn.execute(f"INSERT OR REPLACE INTO dividends ({cols_str}) SELECT {cols_str} FROM dividends_temp")
                    conn.execute("DROP TABLE dividends_temp")
                    conn.commit()
                    
                logger.info(f"Stored {len(processed_rows)} dividend records")
            except Exception as e:
                logger.error(f"Error storing dividend data: {e}")
    
    def _parse_numeric(self, value) -> Optional[float]:
        """Parse numeric value from various formats."""
        if pd.isna(value) or value is None:
            return None
        try:
            if isinstance(value, (int, float)):
                return float(value)
            s = str(value).replace(',', '').replace('%', '').strip()
            if s == '-' or s == '':
                return None
            return float(s)
        except:
            return None
    
    def fetch_all_for_year(self, year: int, report_code: str):
        """
        Fetch dividend data for all tickers for a specific year and report type.
        
        Args:
            year: Fiscal year
            report_code: Report type ('11011'=Annual, '11012'=Q2, '11013'=Q1, '11014'=Q3)
        """
        if not self.dart:
            logger.error("DART API not configured")
            return
        
        report_names = {'11013': 'Q1', '11012': 'Q2', '11014': 'Q3', '11011': 'Annual'}
        report_name = report_names.get(report_code, report_code)
        
        tickers = list(self.ticker_to_corp_code_map.keys())
        logger.info(f"Fetching {report_name} {year} dividend data for {len(tickers)} tickers...")
        
        success_count = 0
        error_count = 0
        
        for ticker in tqdm(tickers, desc=f"Fetching Dividends {report_name} {year}"):
            try:
                if self.fetch_dividend_for_ticker(ticker, year, report_code):
                    success_count += 1
                else:
                    error_count += 1
            except Exception as e:
                logger.debug(f"Error fetching {ticker}: {e}")
                error_count += 1
            time.sleep(random.uniform(0.3, 0.6))
        
        logger.info(f"Completed: {success_count} success, {error_count} no data/errors")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fetch dividend data from DART API (DS002)')
    parser.add_argument('--year', type=int, required=True, help='Fiscal year (e.g., 2024)')
    parser.add_argument('--report-type', type=str, default='11011',
                        choices=['11011', '11012', '11013', '11014'],
                        help='Report type: 11011=Annual (default), 11012=Q2, 11013=Q1, 11014=Q3')
    parser.add_argument('--db-path', type=str, default='db/krx_data.db', help='Path to database')
    
    args = parser.parse_args()
    
    api_key = os.getenv('DART_API_KEY')
    if not api_key:
        logger.error("DART_API_KEY environment variable not set")
        sys.exit(1)
    
    fetcher = DividendFetcher(api_key=api_key, db_path=args.db_path)
    fetcher.fetch_all_for_year(year=args.year, report_code=args.report_type)
