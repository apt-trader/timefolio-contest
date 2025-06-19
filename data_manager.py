# data_manager.py
import logging
import pandas as pd
from datetime import datetime
import FinanceDataReader as fdr
import sqlite3

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
        self._fin = FinancialsFetcher(
            api_key=cfg.fetcher_settings.get('dart_api_key'), 
            db_path=self.db_path
        )
        self._macro = MacroFetcher(fred_api_key=cfg.fetcher_settings.get('fred_api_key'))
        self.tickers, self.sector_map = self._initialize_universe()
        logger.info(f"DataManager initialized with {len(self.tickers)} compliant tickers.")
    
    def _initialize_universe(self) -> tuple[list[str], dict[str, str]]:
        # ... (This logic is correct and remains the same) ...
        # It correctly uses ComplianceFilter to produce a clean universe
        return self._run_compliance_filters()

    def _run_compliance_filters(self) -> tuple[list[str], dict[str, str]]:
        raw_universe_df = pd.read_csv(self.cfg.stock_universe_file, dtype=str)
        raw_universe_df['종목코드'] = raw_universe_df['종목코드'].str.strip().str.lstrip('A').str.zfill(6)
        raw_universe_df = raw_universe_df.set_index('종목코드')
        initial_tickers = raw_universe_df.index.tolist()
        initial_sector_map = raw_universe_df['섹터코드'].to_dict()
        logger.info(f"Loaded {len(initial_tickers)} tickers from universe file.")

        logger.info("Fetching data required for compliance filters...")
        listings_info = fdr.StockListing('KRX-ALL').set_index('Code')
        
        start_date_str = (datetime.strptime(self.cfg.end_date, '%Y-%m-%d') - pd.DateOffset(days=60)).strftime('%Y-%m-%d')
        prices_df = self.get_all_prices_for_tickers(initial_tickers, start_date_str, self.cfg.end_date)
        volumes_df = self.get_all_volumes_for_tickers(initial_tickers, start_date_str, self.cfg.end_date)
        
        comp_filter = ComplianceFilter(
            min_avg_daily_value=self.cfg.risk_management.get('min_avg_daily_value', 3_000_000_000),
            min_ipo_days=self.cfg.risk_management.get('min_ipo_days', 90)
        )
        filter_results = comp_filter.apply_all_filters(prices_df, volumes_df, listings_info)
        comp_filter.save_forbidden_list(filter_results)
        
        forbidden_tickers = filter_results['all']
        compliant_tickers = [t for t in initial_tickers if t not in forbidden_tickers]
        compliant_sector_map = {t: s for t, s in initial_sector_map.items() if t in compliant_tickers}
        
        logger.info(f"Compliance filtering complete. Final universe size: {len(compliant_tickers)}.")
        return compliant_tickers, compliant_sector_map

    def get_market_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        logger.info(f"Fetching market data for {len(self.tickers)} tickers...")
        prices_df = self.get_all_prices_for_tickers(self.tickers, self.cfg.start_date, self.cfg.end_date)
        volumes_df = self.get_all_volumes_for_tickers(self.tickers, self.cfg.start_date, self.cfg.end_date)
        market_caps_df = self.get_all_market_caps_for_tickers(self.tickers, self.cfg.start_date, self.cfg.end_date)
        
        returns_df = prices_df.pct_change().dropna(how='all')
        market_index = self._krx.get_stock_data('KOSPI', self.cfg.start_date.replace('-', ''), self.cfg.end_date.replace('-', ''))['close']
        
        common_index = returns_df.index
        return prices_df.loc[common_index], volumes_df.loc[common_index], returns_df.loc[common_index], market_index.loc[common_index], market_caps_df.iloc[-1]
    
    # Helper methods for fetching data for a list of tickers
    def get_all_prices_for_tickers(self, tickers, start, end):
        return pd.DataFrame({t: self._krx.get_stock_data(t, start.replace('-', ''), end.replace('-', ''))['close'] for t in tickers}).ffill().bfill()
    
    def get_all_volumes_for_tickers(self, tickers, start, end):
        return pd.DataFrame({t: self._krx.get_stock_data(t, start.replace('-', ''), end.replace('-', ''))['volume'] for t in tickers}).ffill().bfill()
        
    def get_all_market_caps_for_tickers(self, tickers, start, end):
        return pd.DataFrame({t: self._krx.get_stock_data(t, start.replace('-', ''), end.replace('-', ''))['market_cap'] for t in tickers}).ffill().bfill()

    def get_fundamental_data(self) -> pd.DataFrame:
        """Provides the latest fundamental data for the compliant universe."""
        logger.info(f"Fetching latest fundamental data for {len(self.tickers)} tickers.")
        return self._fin.get_latest_fundamentals(self.tickers, self.cfg.end_date)

    def get_macro_data(self) -> pd.DataFrame:
        """Provides the latest macroeconomic data."""
        logger.info("Fetching latest macroeconomic data.")
        self._macro.fetch_and_store(self.cfg.end_date)
        with sqlite3.connect(self.db_path) as conn:
            query = "SELECT * FROM macro_data WHERE date <= ? ORDER BY date DESC LIMIT 1"
            df = pd.read_sql(query, conn, params=[self.cfg.end_date])
        return df
        
    def get_sector_mappings(self) -> dict[str, str]:
        return self.sector_map