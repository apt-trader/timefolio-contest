# fetchers/treasury_stock_fetcher.py
"""
DS002 - 자기주식 (Treasury Stock) Fetcher
API: tesstkAcqsDspsSttus (자기주식 취득/처분 현황)

Fetches treasury stock (buyback) data from DART OpenAPI.
Buybacks signal management confidence in company value.
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
        logging.FileHandler(logs_dir / 'treasury_stock_fetcher.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TreasuryStockFetcher:
    """Fetches treasury stock (buyback) data from DART API (DS002)."""
    
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
        logger.info(f"TreasuryStockFetcher initialized for DB at {db_path}.")
    
    def _create_connection(self):
        return sqlite3.connect(self.db_path)
    
    def _init_db_structure(self):
        """Initialize treasury_stock table if not exists."""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS treasury_stock (
            corp_code TEXT NOT NULL,
            ticker TEXT NOT NULL,
            fiscal_year INTEGER NOT NULL,
            report_code TEXT NOT NULL,
            rcept_no TEXT,                    -- 접수번호
            acqs_mth1 TEXT,                   -- 취득방법 (직접취득/신탁계약)
            acqs_mth2 TEXT,                   -- 취득방법 상세
            acqs_mth3 TEXT,                   -- 취득방법 상세2
            stock_knd TEXT,                   -- 주식종류
            bsis_qy REAL,                     -- 기초수량
            change_qy_acqs REAL,              -- 변동수량(취득)
            change_qy_dsps REAL,              -- 변동수량(처분)
            change_qy_incnr REAL,             -- 변동수량(소각)
            trmend_qy REAL,                   -- 기말수량
            rm TEXT,                          -- 비고
            fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (corp_code, fiscal_year, report_code, acqs_mth1, stock_knd)
        );
        """
        try:
            with self._create_connection() as conn:
                conn.execute(create_table_sql)
                conn.commit()
            logger.info("Treasury stock table initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize treasury_stock table: {e}")
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
    
    def fetch_treasury_stock_for_ticker(self, ticker: str, year: int, report_code: str) -> bool:
        """
        Fetch treasury stock data for a single ticker.
        
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
            # Call DART API for treasury stock info
            df = self.dart.report(corp_code, '자기주식', year, report_code)
            
            if df is None or df.empty:
                logger.debug(f"No treasury stock data for {ticker} ({year} {report_code})")
                return False
            
            # Process and store
            df['corp_code'] = corp_code
            df['ticker'] = str(ticker).zfill(6)
            df['fiscal_year'] = year
            df['report_code'] = report_code
            
            self._store_treasury_stock_data(df)
            return True
            
        except Exception as e:
            logger.debug(f"Error fetching treasury stock for {ticker}: {e}")
            return False
    
    def _store_treasury_stock_data(self, df: pd.DataFrame):
        """Store treasury stock data to database."""
        if df.empty:
            return
        
        # Map Korean column names to English
        column_mapping = {
            '접수번호': 'rcept_no',
            '취득방법': 'acqs_mth1',
            '취득방법 대분류': 'acqs_mth1',
            '취득방법 중분류': 'acqs_mth2',
            '취득방법 소분류': 'acqs_mth3',
            '주식의 종류': 'stock_knd',
            '주식종류': 'stock_knd',
            '기초수량': 'bsis_qy',
            '변동수량 취득': 'change_qy_acqs',
            '변동수량 처분': 'change_qy_dsps',
            '변동수량 소각': 'change_qy_incnr',
            '기말수량': 'trmend_qy',
            '비고': 'rm',
        }
        
        df = df.rename(columns=column_mapping)
        
        # Convert numeric columns
        numeric_cols = ['bsis_qy', 'change_qy_acqs', 'change_qy_dsps', 'change_qy_incnr', 'trmend_qy']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].apply(self._parse_numeric)
        
        # Ensure required columns exist
        for col in ['acqs_mth1', 'stock_knd']:
            if col not in df.columns:
                df[col] = ''
        
        try:
            with self._create_connection() as conn:
                df.to_sql('treasury_stock_temp', conn, if_exists='replace', index=False)
                
                # Get common columns
                main_cols = {col[1] for col in conn.execute("PRAGMA table_info(treasury_stock)").fetchall()}
                temp_cols = {col[1] for col in conn.execute("PRAGMA table_info(treasury_stock_temp)").fetchall()}
                common_cols = list(main_cols.intersection(temp_cols))
                cols_str = ', '.join(f'"{c}"' for c in common_cols)
                
                conn.execute(f"INSERT OR REPLACE INTO treasury_stock ({cols_str}) SELECT {cols_str} FROM treasury_stock_temp")
                conn.execute("DROP TABLE treasury_stock_temp")
                conn.commit()
                
            logger.info(f"Stored {len(df)} treasury stock records for ticker {df['ticker'].iloc[0]}")
        except Exception as e:
            logger.error(f"Error storing treasury stock data: {e}")
    
    def _parse_numeric(self, value) -> Optional[float]:
        """Parse numeric value from various formats."""
        if pd.isna(value) or value is None:
            return None
        try:
            if isinstance(value, (int, float)):
                return float(value)
            s = str(value).replace(',', '').replace('주', '').strip()
            if s == '-' or s == '':
                return None
            return float(s)
        except:
            return None
    
    def fetch_all_for_year(self, year: int, report_code: str):
        """
        Fetch treasury stock data for all tickers for a specific year and report type.
        
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
        logger.info(f"Fetching {report_name} {year} treasury stock data for {len(tickers)} tickers...")
        
        success_count = 0
        error_count = 0
        
        for ticker in tqdm(tickers, desc=f"Fetching Treasury Stock {report_name} {year}"):
            try:
                if self.fetch_treasury_stock_for_ticker(ticker, year, report_code):
                    success_count += 1
                else:
                    error_count += 1
            except Exception as e:
                logger.debug(f"Error fetching {ticker}: {e}")
                error_count += 1
            time.sleep(random.uniform(0.3, 0.6))
        
        logger.info(f"Completed: {success_count} success, {error_count} no data/errors")
    
    def get_buyback_summary(self) -> pd.DataFrame:
        """
        Get summary of treasury stock activity for signal construction.
        
        Returns:
            DataFrame with ticker, total_acquired, total_disposed, net_buyback
        """
        query = """
        SELECT 
            ticker,
            fiscal_year,
            SUM(COALESCE(change_qy_acqs, 0)) as total_acquired,
            SUM(COALESCE(change_qy_dsps, 0)) as total_disposed,
            SUM(COALESCE(change_qy_acqs, 0)) - SUM(COALESCE(change_qy_dsps, 0)) as net_buyback,
            MAX(trmend_qy) as ending_treasury_shares
        FROM treasury_stock
        GROUP BY ticker, fiscal_year
        ORDER BY ticker, fiscal_year DESC
        """
        try:
            with self._create_connection() as conn:
                return pd.read_sql_query(query, conn)
        except Exception as e:
            logger.error(f"Error getting buyback summary: {e}")
            return pd.DataFrame()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fetch treasury stock data from DART API (DS002)')
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
    
    fetcher = TreasuryStockFetcher(api_key=api_key, db_path=args.db_path)
    fetcher.fetch_all_for_year(year=args.year, report_code=args.report_type)
