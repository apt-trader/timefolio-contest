import sqlite3
import pandas as pd
from pathlib import Path

class DBChecker:
    """Analyzes and reports on the contents of the krx_data.db database."""

    def __init__(self, db_path: str):
        """Initializes the DBChecker with the path to the database."""
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found at {self.db_path}")
        self.conn = sqlite3.connect(self.db_path)

    def _execute_query(self, query: str, params=()) -> pd.DataFrame:
        """Executes a SQL query and returns the result as a pandas DataFrame."""
        return pd.read_sql_query(query, self.conn, params=params)

    def check_krx_data(self):
        """Analyzes tables populated by the conceptual krx_fetcher."""
        print("--- KRX & Financial Data Analysis Status ---")

        # Analyze daily_prices
        print("\n1. Table: daily_prices")
        try:
            count = self._execute_query("SELECT COUNT(*) FROM daily_prices;").iloc[0, 0]
            unique_tickers = self._execute_query("SELECT COUNT(DISTINCT code) FROM daily_prices;").iloc[0, 0]
            min_date, max_date = self._execute_query("SELECT MIN(date), MAX(date) FROM daily_prices;").iloc[0]
            print(f"   - Description: Contains daily stock prices (OHLCV) and market cap.")
            print(f"   - Total Records: {count:,}")
            print(f"   - Unique Tickers: {unique_tickers:,}")
            print(f"   - Date Range: {min_date} to {max_date}")
        except Exception as e:
            print(f"   - Could not analyze table: {e}")

        # Analyze listing_info
        print("\n2. Table: listing_info")
        try:
            count = self._execute_query("SELECT COUNT(*) FROM listing_info;").iloc[0, 0]
            print(f"   - Description: Contains company listing information like IPO dates.")
            print(f"   - Total Records: {count:,}")
        except Exception as e:
            print(f"   - Could not analyze table: {e}")

        # Analyze financials
        print("\n3. Table: financials")
        try:
            count = self._execute_query("SELECT COUNT(*) FROM financials;").iloc[0, 0]
            unique_corps = self._execute_query("SELECT COUNT(DISTINCT corp_code) FROM financials;").iloc[0, 0]
            
            # Get date range from report_date column (converting from Unix timestamp)
            date_range = self._execute_query("""
                SELECT datetime(MIN(CAST(report_date AS INTEGER)), 'unixepoch'), 
                       datetime(MAX(CAST(report_date AS INTEGER)), 'unixepoch')
                FROM financials 
                WHERE report_date != '' AND report_date IS NOT NULL;
            """).iloc[0]
            min_date, max_date = date_range.iloc[0], date_range.iloc[1]
            
            print(f"   - Description: Contains corporate financial statement data.")
            print(f"   - Total Records: {count:,}")
            print(f"   - Unique Companies: {unique_corps:,}")
            if min_date and max_date:
                print(f"   - Date Range: {min_date} to {max_date}")
            
            # Show distribution by year (converting from Unix timestamp)
            year_dist = self._execute_query("""
                SELECT strftime('%Y', datetime(CAST(report_date AS INTEGER), 'unixepoch')) as year, 
                       COUNT(DISTINCT ticker) as companies,
                       COUNT(*) as records
                FROM financials 
                WHERE report_date != '' AND report_date IS NOT NULL
                GROUP BY year 
                HAVING year IS NOT NULL
                ORDER BY year;
            """)
            if not year_dist.empty:
                print("   - Yearly Distribution:")
                for _, row in year_dist.iterrows():
                    print(f"      {row['year']}: {row['companies']} companies, {row['records']:,} records")
                    
        except Exception as e:
            print(f"   - Could not analyze table: {e}")

        # Analyze company_mapping
        print("\n4. Table: company_mapping")
        try:
            count = self._execute_query("SELECT COUNT(*) FROM company_mapping;").iloc[0, 0]
            print(f"   - Description: Maps corporate codes to stock tickers.")
            print(f"   - Total Records: {count:,}")
        except Exception as e:
            print(f"   - Could not analyze table: {e}")

    def check_macro_data(self):
        """Analyzes tables populated by the conceptual macro_fetcher."""
        print("\n--- Macro Data Analysis Status ---")

        # Analyze macro_data
        print("\n1. Table: macro_data")
        try:
            count = self._execute_query("SELECT COUNT(*) FROM macro_data;").iloc[0, 0]
            min_date, max_date = self._execute_query("SELECT MIN(date), MAX(date) FROM macro_data;").iloc[0]
            print(f"   - Description: Contains macroeconomic indicator data.")
            print(f"   - Total Records: {count:,}")
            print(f"   - Date Range: {min_date} to {max_date}")
        except Exception as e:
            print(f"   - Could not analyze table: {e}")

    def run_checks(self):
        """Runs all database checks and prints a summary report."""
        print("=" * 50)
        print("Starting Database Content Analysis")
        print("=" * 50)
        self.check_krx_data()
        self.check_macro_data()
        print("=" * 50)
        print("Analysis Complete")
        print("=" * 50)

    def close(self):
        """Closes the database connection."""
        self.conn.close()

if __name__ == "__main__":
    db_path = 'db/krx_data.db'
    try:
        checker = DBChecker(db_path)
        checker.run_checks()
        checker.close()
    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")