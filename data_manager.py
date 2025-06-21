# data_manager.py
import logging
import pandas as pd
from datetime import datetime
import sqlite3
from typing import Dict, List, Tuple
import os
from config import Config
from fetchers.krx_fetcher import KRXFetcher
from fetchers.financial_fetcher import FinancialsFetcher
from fetchers.macro_fetcher import MacroFetcher
from compliance_filters import ComplianceFilter

logger = logging.getLogger(__name__)

class DataManager:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.db_path = cfg.db_path
        self._krx = KRXFetcher(db_path=self.db_path)
        self._fin = FinancialsFetcher(api_key=os.getenv('DART_API_KEY'), db_path=self.db_path)
        self._macro = MacroFetcher(fred_api_key=os.getenv('FRED_API_KEY'))
        self.tickers, self.sector_map = self._initialize_universe()
        logger.info(f"DataManager initialized with {len(self.tickers)} compliant tickers.")

    def _get_universe_at_date(self, date_str: str) -> pd.DataFrame:
        """Gets all tickers that were actively traded on a specific date from our own database."""
        with sqlite3.connect(self.db_path) as conn:
            query = "SELECT DISTINCT code, name FROM daily_prices WHERE date = ?"
            df = pd.read_sql_query(query, conn, params=(date_str,))
        return df.set_index('code')

    def _run_compliance_filters(self) -> Tuple[List[str], Dict[str, str]]:
        end_date = datetime.strptime(self.cfg.end_date, '%Y-%m-%d')
        logger.info(f"Building and filtering universe for date: {end_date.strftime('%Y-%m-%d')}")
        base_universe_df = self._get_universe_at_date(end_date.strftime('%Y-%m-%d'))
        full_sector_map_df = pd.read_csv(self.cfg.stock_universe_file, dtype={'종목코드':str})
        full_sector_map_df['종목코드'] = full_sector_map_df['종목코드'].str.lstrip('A').str.zfill(6)
        
        listings_info = base_universe_df.merge(
            full_sector_map_df[['종목코드', '섹터코드']], left_index=True, right_on='종목코드', how='left'
        ).set_index('종목코드')

        initial_tickers = listings_info.index.tolist()
        initial_sector_map = listings_info['섹터코드'].to_dict()

        start_dt_str = (end_date - pd.DateOffset(days=60)).strftime('%Y%m%d')
        end_dt_str_krx = end_date.strftime('%Y%m%d')
        prices = pd.DataFrame({t: self._krx.get_stock_data(t, start_dt_str, end_dt_str_krx)['close'] for t in initial_tickers}).ffill()
        volumes = pd.DataFrame({t: self._krx.get_stock_data(t, start_dt_str, end_dt_str_krx)['volume'] for t in initial_tickers}).ffill()
        
        simulated_listings = pd.DataFrame(index=initial_tickers)
        simulated_listings['ListingDate'] = pd.to_datetime(self.cfg.start_date)

        comp_filter = ComplianceFilter(
            min_avg_daily_value=self.cfg.risk_management.get('min_avg_daily_value', 3e9),
            min_ipo_days=self.cfg.risk_management.get('min_ipo_days', 90)
        )
        filter_results = comp_filter.apply_all_filters(prices, volumes, simulated_listings)
        comp_filter.save_forbidden_list(filter_results)
        
        forbidden_tickers = filter_results['all']
        compliant_tickers = [t for t in initial_tickers if t not in forbidden_tickers]
        compliant_sector_map = {t: s for t, s in initial_sector_map.items() if t in compliant_tickers}
        
        return compliant_tickers, compliant_sector_map

    def get_market_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        logger.info(f"Fetching market data for {len(self.tickers)} tickers...")
        start, end = self.cfg.start_date.replace('-', ''), self.cfg.end_date.replace('-', '')
        prices = pd.DataFrame({t: self._krx.get_stock_data(t, start, end)['close'] for t in self.tickers}).ffill()
        volumes = pd.DataFrame({t: self._krx.get_stock_data(t, start, end)['volume'] for t in self.tickers}).ffill()
        market_caps = pd.DataFrame({t: self._krx.get_stock_data(t, start, end)['market_cap'] for t in self.tickers}).ffill()
        returns = prices.pct_change().dropna(how='all', axis=0)
        
        common_idx = returns.index
        return prices.loc[common_idx], volumes.loc[common_idx], returns, market_caps.loc[common_idx]

    def get_fundamental_data(self) -> pd.DataFrame:
        return self._fin.get_latest_fundamentals(self.tickers, self.cfg.end_date)

    def get_historical_fundamental_data(self) -> Dict[str, pd.DataFrame]:
        return self._fin.get_historical_fundamentals(self.tickers, self.cfg.end_date, years=5)

    def get_macro_data(self) -> pd.DataFrame:
        self._macro.fetch_and_store(self.cfg.end_date)
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql("SELECT * FROM macro_data WHERE date <= ? ORDER BY date DESC LIMIT 1", conn, params=(self.cfg.end_date,))
        
    def get_sector_mappings(self) -> Dict[str, str]:
        return self.sector_map