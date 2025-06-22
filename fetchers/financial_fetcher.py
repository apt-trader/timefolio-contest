# fetchers/financial_fetcher.py
import os
import sys
import sqlite3
import logging
import time
from datetime import datetime
import pandas as pd
from OpenDartReader.dart import OpenDartReader
from dotenv import load_dotenv
import argparse
import yaml
from pathlib import Path
from typing import Optional, List, Dict
from tqdm import tqdm
from collections import deque

load_dotenv()
logger = logging.getLogger(__name__)

try:
    config_path = Path(__file__).parent.parent / "config/config.yaml"
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    universe_file = Path(__file__).parent.parent / cfg['data_settings']['stock_universe_file']
except FileNotFoundError:
    logger.error(f"Config file not found at {config_path}.")
    universe_file = None

class RateLimiter:
    def __init__(self, max_per_minute: int = 900, max_per_day: int = 9500):
        self.max_per_minute, self.max_per_day = max_per_minute, max_per_day
        self.calls_minute, self.calls_day = deque(), deque()
    def wait(self):
        now = time.time()
        while self.calls_minute and self.calls_minute[0] < now - 60: self.calls_minute.popleft()
        while self.calls_day and self.calls_day[0] < now - 86400: self.calls_day.popleft()
        if len(self.calls_day) >= self.max_per_day: raise RuntimeError(f"Daily API limit ({self.max_per_day}) reached.")
        if len(self.calls_minute) >= self.max_per_minute:
            time.sleep(60.1 - (now - self.calls_minute[0]))
        self.calls_minute.append(time.time()); self.calls_day.append(time.time())

class FinancialsFetcher:
    def __init__(self, api_key: str, db_path: str = str(Path(__file__).parent.parent / "db" / "krx_data.db")):
        if not api_key: raise ValueError("DART API key is required.")
        self.api_key = api_key
        self.cache_dir = Path(__file__).parent / 'docs_cache'
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        os.environ['OPENDART_CACHE_PATH'] = str(self.cache_dir)
        self.dart = OpenDartReader(self.api_key)
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.corp_code_cache: Dict[str, Optional[str]] = {}
        self._init_financials_table()
        self.existing_columns = self._get_existing_columns()
        self.rate_limiter = RateLimiter()

    def _init_financials_table(self):
        try:
            with self.conn:
                self.conn.execute("""
                    CREATE TABLE IF NOT EXISTS financials (
                        rcept_no TEXT, reprt_code TEXT, bsns_year TEXT, corp_code TEXT, sj_div TEXT, sj_nm TEXT, fs_div TEXT, fs_nm TEXT, fs_sn TEXT,
                        account_id TEXT, account_nm TEXT, account_detail TEXT, thstrm_nm TEXT, thstrm_amount REAL, thstrm_add_amount REAL, thstrm_q_nm TEXT, 
                        thstrm_q_amount REAL, frmtrm_nm TEXT, frmtrm_amount REAL, frmtrm_add_amount REAL, frmtrm_q_nm TEXT, frmtrm_q_amount REAL,
                        bfefrmtrm_nm TEXT, bfefrmtrm_amount REAL, ord INTEGER, currency TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, PRIMARY KEY (rcept_no, account_id, fs_div)
                    )""")
                self.conn.execute("CREATE INDEX IF NOT EXISTS idx_financials_corp_code_year ON financials(corp_code, bsns_year)")
        except sqlite3.Error as e: logger.critical(f"FATAL: DB schema init failed. Error: {e}"); raise

    def _get_existing_columns(self) -> list:
        return [row[1] for row in self.conn.execute("PRAGMA table_info(financials)").fetchall()]

    def find_corp_code(self, identifier: str) -> Optional[str]:
        if identifier in self.corp_code_cache: return self.corp_code_cache[identifier]
        try:
            self.rate_limiter.wait(); corp_code = self.dart.find_corp_code(identifier)
            self.corp_code_cache[identifier] = corp_code; return corp_code
        except Exception: self.corp_code_cache[identifier] = None; return None

    def fetch_financials(self, corp_code: str, year: int, report_type: str = '11011'):
        if self.conn.execute("SELECT 1 FROM financials WHERE corp_code = ? AND bsns_year = ? AND reprt_code = ? LIMIT 1", (corp_code, str(year), report_type)).fetchone():
            return
        try:
            self.rate_limiter.wait(); df = self.dart.finstate_all(corp_code, year, report_type, fs_div='CFS')
            if not isinstance(df, pd.DataFrame) or df.empty:
                self.rate_limiter.wait(); df = self.dart.finstate_all(corp_code, year, report_type, fs_div='OFS')
            if isinstance(df, pd.DataFrame) and not df.empty: self._store_financials(df)
        except Exception as e: logger.error(f"Error processing {corp_code} {year}: {e}", exc_info=True)

    def _store_financials(self, df: pd.DataFrame):
        if df.empty: return
        valid_cols = [c for c in df.columns if c in self.existing_columns]; df_to_store = df[valid_cols].copy()
        for col in ['thstrm_amount', 'frmtrm_amount', 'bfefrmtrm_amount']:
            if col in df_to_store: df_to_store[col] = pd.to_numeric(df_to_store[col], errors='coerce')
        pk_cols = ['rcept_no', 'account_id', 'fs_div']; update_cols = [f'"{c}" = excluded."{c}"' for c in df_to_store.columns if c not in pk_cols]
        sql = f"INSERT INTO financials ({','.join(f'\"{c}\"' for c in df_to_store.columns)}) VALUES ({','.join(['?']*len(df_to_store.columns))}) ON CONFLICT({','.join(pk_cols)}) DO UPDATE SET {','.join(update_cols)}"
        with self.conn: self.conn.executemany(sql, df_to_store.where(pd.notna(df_to_store), None).to_records(index=False).tolist())

    def get_latest_fundamentals(self, tickers: List[str], as_of_date: str) -> pd.DataFrame:
        logger.info(f"Loading latest fundamentals for {len(tickers)} tickers as of {as_of_date}.")
        account_map = {'Assets': 'ifrs-full_Assets', 'Liabilities': 'ifrs-full_Liabilities', 'Equity': 'ifrs-full_Equity', 'NetIncome': 'ifrs-full_ProfitLoss', 'GrossProfit': 'ifrs-full_GrossProfit', 'Sales': 'ifrs-full_Revenue', 'OperatingCF': 'ifrs-full_CashFlowsFromUsedInOperatingActivities'}
        corp_codes_map = {ticker: self.find_corp_code(ticker) for ticker in tickers}
        valid_corp_codes = [c for c in corp_codes_map.values() if c is not None]
        if not valid_corp_codes: return pd.DataFrame()
        ticker_lookup = {v: k for k, v in corp_codes_map.items()}
        sql = f"WITH LatestReports AS (SELECT corp_code, MAX(rcept_no) AS rcept_no FROM financials WHERE corp_code IN ({','.join(['?'] * len(valid_corp_codes))}) AND bsns_year < ? AND reprt_code = '11011' GROUP BY corp_code) SELECT f.corp_code, f.account_id, f.thstrm_amount FROM financials f JOIN LatestReports lr ON f.corp_code = lr.corp_code AND f.rcept_no = lr.rcept_no WHERE f.account_id IN ({','.join(['?'] * len(account_map))})"
        params = valid_corp_codes + [as_of_date[:4]] + list(account_map.values())
        with self.conn: raw_df = pd.read_sql_query(sql, self.conn, params=params)
        if raw_df.empty: return pd.DataFrame()
        pivot_df = raw_df.pivot_table(index='corp_code', columns='account_id', values='thstrm_amount').rename(columns={v: k for k, v in account_map.items()})
        pivot_df.index = pivot_df.index.map(ticker_lookup)
        return pivot_df

    def get_historical_fundamentals(self, tickers: List[str], as_of_date: str, years: int = 5) -> Dict[str, pd.DataFrame]:
        logger.info(f"Loading {years}-year historical fundamentals for {len(tickers)} tickers as of {as_of_date}.")
        account_map = {'Assets': 'ifrs-full_Assets', 'Equity': 'ifrs-full_Equity', 'NetIncome': 'ifrs-full_ProfitLoss'}
        end_year, start_year = int(as_of_date[:4]), int(as_of_date[:4]) - years
        corp_codes_map = {ticker: self.find_corp_code(ticker) for ticker in tickers}
        valid_corp_codes = [c for c in corp_codes_map.values() if c is not None]
        if not valid_corp_codes: return {}
        ticker_lookup = {v: k for k, v in corp_codes_map.items()}
        sql = f"SELECT corp_code, bsns_year, account_id, thstrm_amount FROM financials WHERE corp_code IN ({','.join(['?'] * len(valid_corp_codes))}) AND bsns_year BETWEEN ? AND ? AND reprt_code = '11011' AND account_id IN ({','.join(['?'] * len(account_map))})"
        params = valid_corp_codes + [str(start_year), str(end_year)] + list(account_map.values())
        with self.conn: raw_df = pd.read_sql_query(sql, self.conn, params=params)
        if raw_df.empty: return {}
        results = {}
        for corp_code, group in raw_df.groupby('corp_code'):
            if ticker := ticker_lookup.get(corp_code):
                pivot = group.pivot_table(index='bsns_year', columns='account_id', values='thstrm_amount').rename(columns={v: k for k, v in account_map.items()})
                results[ticker] = pivot.sort_index()
        return results

    def __del__(self):
        try: self.conn.close()
        except: pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch DART financial statements.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-i", "--identifier", help="Single stock code")
    group.add_argument("-a", "--all", action="store_true", help="Fetch for all tickers in universe file.")
    parser.add_argument("-s", "--start-year", required=True, type=int, help="Start year (YYYY)")
    parser.add_argument("-e", "--end-year", required=True, type=int, help="End year (YYYY)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    if not (api_key := os.getenv('DART_API_KEY')): logger.critical("DART_API_KEY not found."); sys.exit(1)
    fetcher = FinancialsFetcher(api_key=api_key)
    if args.all:
        if not (universe_file and universe_file.exists()): logger.critical("Universe file not found."); sys.exit(1)
        tickers = pd.read_csv(universe_file)['종목코드'].astype(str).str.lstrip('A').str.zfill(6).tolist()
    else: tickers = [args.identifier]
    for year in range(args.start_year, args.end_year + 1):
        for ticker in tqdm(tickers, desc=f"Processing Year {year}", unit="ticker"):
            if corp_code := fetcher.find_corp_code(ticker): fetcher.fetch_financials(corp_code, year)