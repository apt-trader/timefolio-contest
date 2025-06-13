import sqlite3
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple

import pandas as pd
import yfinance as yf
from fredapi import Fred

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent / "krx_data.db"

class MacroFetcher:
    def __init__(self, fred_api_key: str):
        self.fred = Fred(api_key=fred_api_key)
        self.db_path = str(DB_PATH)
        self._init_macro_table()

    def _init_macro_table(self):
        """Initialize the macro_data table if it doesn't exist"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
            CREATE TABLE IF NOT EXISTS macro_data (
                date DATE PRIMARY KEY,
                slope_10y_2y REAL,
                credit_spread REAL,
                equity_mom_1m REAL,
                equity_mom_3m REAL,
                bond_mom_1m REAL,
                bond_mom_3m REAL,
                eq_vs_bond_mom REAL,
                vix REAL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)
            conn.commit()

    def _fetch_fred_data(self, series_id: str, as_of_date: str) -> float:
        """Helper method to fetch data from FRED API with error handling"""
        try:
            logger.info(f"Fetching {series_id} data from FRED...")
            series = self.fred.get_series(series_id, as_of_date, as_of_date)
            if series.empty:
                raise ValueError(f"No data returned for {series_id}")
            return float(series.iloc[0])
        except Exception as e:
            logger.error(f"Error fetching {series_id}: {str(e)}")
            raise

    def _fetch_kospi_momentum(self, as_of_date: str) -> Tuple[float, float]:
        """Fetch KOSPI momentum data"""
        try:
            logger.info("Fetching KOSPI data...")
            kospi = pd.read_sql(
                """
                SELECT date, close FROM daily_prices
                WHERE code='KOSPI' AND date <= ?
                ORDER BY date DESC LIMIT 121
                """, 
                sqlite3.connect(self.db_path), 
                params=[as_of_date]
            )
            
            if kospi.empty:
                raise ValueError("No KOSPI data found in the database")
                
            eq_mom_1m = kospi['close'].pct_change(21).iloc[-1]
            eq_mom_3m = kospi['close'].pct_change(63).iloc[-1]
            
            return float(eq_mom_1m), float(eq_mom_3m)
            
        except Exception as e:
            logger.error(f"Error fetching KOSPI data: {str(e)}")
            raise

    def _fetch_vix(self, as_of_date: str) -> float:
        """Fetch VIX data from Yahoo Finance"""
        try:
            logger.info("Fetching VIX data from Yahoo Finance...")
            end_date = (pd.to_datetime(as_of_date) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
            vix_data = yf.Ticker('^VIX').history(start=as_of_date, end=end_date)
            
            if vix_data.empty:
                raise ValueError("No VIX data available for the specified date")
                
            return float(vix_data['Close'].iloc[0])
            
        except Exception as e:
            logger.error(f"Error fetching VIX data: {str(e)}")
            raise

    def fetch_and_store(self, as_of_date: str) -> bool:
        """
        Fetch macroeconomic data and store it in the database
        
        Args:
            as_of_date: Date in 'YYYY-MM-DD' format
            
        Returns:
            bool: True if operation was successful, False otherwise
        """
        try:
            logger.info(f"Starting macro data update for {as_of_date}")
            
            # Validate date format
            try:
                as_of_date_dt = pd.to_datetime(as_of_date).date()
                logger.debug(f"Validated date: {as_of_date_dt}")
            except Exception as e:
                raise ValueError(f"Invalid date format: {as_of_date}. Expected 'YYYY-MM-DD'")
            
            # 1) Fetch FRED data
            logger.info("Fetching FRED data...")
            dgs10 = self._fetch_fred_data('DGS10', as_of_date)
            dgs2 = self._fetch_fred_data('DGS2', as_of_date)
            baa = self._fetch_fred_data('BAA', as_of_date)
            
            # 2) Fetch KOSPI momentum
            eq_mom_1m, eq_mom_3m = self._fetch_kospi_momentum(as_of_date)
            
            # 3) Bond momentum (placeholder - implement as needed)
            bond_mom_1m = None  
            bond_mom_3m = None
            
            # 4) Fetch VIX
            vix = self._fetch_vix(as_of_date)
            
            # 5) Calculate derived metrics
            slope = dgs10 - dgs2
            credit_spread = baa - dgs10
            eq_vs_bond_mom = eq_mom_1m - (bond_mom_1m or 0)
            
            # 6) Store in database
            logger.info("Storing data in database...")
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO macro_data (
                        date, slope_10y_2y, credit_spread, 
                        equity_mom_1m, equity_mom_3m, 
                        bond_mom_1m, bond_mom_3m,
                        eq_vs_bond_mom, vix
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, 
                    (
                        as_of_date_dt, slope, credit_spread,
                        eq_mom_1m, eq_mom_3m,
                        bond_mom_1m, bond_mom_3m,
                        eq_vs_bond_mom, vix
                    )
                )
                conn.commit()
            
            logger.info(f"Successfully updated macro data for {as_of_date}")
            return True
            
        except sqlite3.Error as e:
            logger.error(f"Database error: {str(e)}")
            raise
            
        except Exception as e:
            logger.error(f"Failed to update macro data: {str(e)}", exc_info=True)
            raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch macroeconomic data over a date range."
    )
    parser.add_argument(
        "--fred-api-key", "-k",
        required=True,
        help="FRED API key"
    )
    parser.add_argument(
        "--start-date", "-s",
        required=True,
        help="Start date in YYYY-MM-DD format"
    )
    parser.add_argument(
        "--end-date", "-e",
        required=True,
        help="End date in YYYY-MM-DD format"
    )
    args = parser.parse_args()

    fetcher = MacroFetcher(fred_api_key=args.fred_api_key)
    # Generate daily dates between start and end
    for dt in pd.date_range(start=args.start_date, end=args.end_date, freq='D'):
        date_str = dt.strftime('%Y-%m-%d')
        try:
            fetcher.fetch_and_store(date_str)
        except Exception as e:
            logger.error(f"Error on {date_str}: {e}")