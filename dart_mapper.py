# dart_mapper.py
import os
import sys
import time
import sqlite3
import requests
import pandas as pd
import xml.etree.ElementTree as ET
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

class DartMapper:
    def __init__(self, api_key=None, db_path='db/krx_data.db'):
        self.api_key = api_key or os.getenv('DART_API_KEY')
        self.db_path = db_path
        self.base_url = "https://opendart.fss.or.kr/api"
        
    def get_listed_companies(self):
        """Get listed companies from the database"""
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql("""
                SELECT code as ticker, name as corp_name, market 
                FROM listing_info
                WHERE market IN ('KOSPI', 'KOSDAQ')
            """, conn)
        return df

    def download_dart_corp_codes(self):
        """Download and parse the complete list of corporations from DART"""
        import zipfile
        import io
        import tempfile
        
        url = f"{self.base_url}/corpCode.xml"
        params = {'crtfc_key': self.api_key}
        
        try:
            print("Downloading DART corporation codes...")
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            # Save the ZIP file to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_zip:
                tmp_zip.write(response.content)
                zip_path = tmp_zip.name
            
            print("Extracting ZIP file...")
            # Extract the XML file from the ZIP
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Get the first XML file in the ZIP
                xml_files = [f for f in zip_ref.namelist() if f.lower().endswith('.xml')]
                if not xml_files:
                    raise ValueError("No XML file found in the ZIP archive")
                
                # Read the XML content
                with zip_ref.open(xml_files[0]) as xml_file:
                    xml_content = xml_file.read()
            
            # Parse XML content
            print("Parsing XML data...")
            root = ET.fromstring(xml_content)
            companies = []
            
            # Extract company information
            for item in root.findall('.//list'):
                corp_code = item.find('corp_code')
                corp_name = item.find('corp_name')
                stock_code = item.find('stock_code')
                
                if (corp_code is not None and corp_code.text and 
                    corp_name is not None and corp_name.text and
                    stock_code is not None and stock_code.text.strip()):
                    
                    companies.append({
                        'corp_code': corp_code.text,
                        'corp_name': corp_name.text,
                        'stock_code': stock_code.text.zfill(6)  # Ensure 6 digits
                    })
            
            print(f"Found {len(companies)} companies with stock codes")
            return pd.DataFrame(companies)
            
        except Exception as e:
            print(f"Error processing DART corp codes: {e}")
            import traceback
            traceback.print_exc()
            
            if 'response' in locals():
                print(f"Response status: {response.status_code}")
                print(f"Response headers: {response.headers}")
                
            return None
        finally:
            # Clean up the temporary file if it exists
            if 'zip_path' in locals() and os.path.exists(zip_path):
                try:
                    os.unlink(zip_path)
                except Exception as e:
                    print(f"Warning: Could not delete temporary file {zip_path}: {e}")

    def update_mappings(self):
        """Update company mappings in the database"""
        # Get listed companies from our database
        krx_companies = self.get_listed_companies()
        print(f"Found {len(krx_companies)} listed companies in KRX")
        
        # Get all companies from DART
        dart_companies = self.download_dart_corp_codes()
        if dart_companies is None or dart_companies.empty:
            print("Failed to download DART company list")
            return
        
        # Debug: Print column names before merge
        print("\nKRX companies columns:", krx_companies.columns.tolist())
        print("DART companies columns:", dart_companies.columns.tolist())
        
        # Standardize column names for merging
        krx_companies = krx_companies.rename(columns={
            'corp_name': 'company_name'
        })
        
        # Debug: Print sample data
        print("\nSample KRX companies:")
        print(krx_companies.head(2).to_string())
        
        print("\nSample DART companies:")
        print(dart_companies.head(2).to_string())
        
        # Merge the datasets on stock code (ticker)
        merged = pd.merge(
            krx_companies, 
            dart_companies,
            left_on='ticker',
            right_on='stock_code',
            how='inner'
        )
        
        # Debug: Print merged columns
        print("\nMerged columns:", merged.columns.tolist())
        
        # Debug: Print the first row of merged data
        print("\nFirst row of merged data:")
        print(merged.iloc[0])
        
        # Define the exact column names we have in the merged DataFrame
        available_columns = merged.columns.tolist()
        print("\nAvailable columns in merged data:", available_columns)
        
        # Create a new DataFrame with exactly the columns we need
        result = pd.DataFrame()
        
        # Map the columns we have to the columns we need
        if 'corp_code' in available_columns:
            result['corp_code'] = merged['corp_code']
        else:
            print("Error: Required column 'corp_code' not found")
            return
            
        if 'ticker' in available_columns:
            result['ticker'] = merged['ticker']
        else:
            print("Error: Required column 'ticker' not found")
            return
            
        # Use company_name if available, otherwise use corp_name
        if 'company_name' in available_columns:
            result['company_name'] = merged['company_name']
        elif 'corp_name' in available_columns:
            result['company_name'] = merged['corp_name']
        else:
            print("Error: Neither 'company_name' nor 'corp_name' found")
            return
            
        if 'market' in available_columns:
            result['market'] = merged['market']
        else:
            print("Error: Required column 'market' not found")
            return
        
        # Ensure we have all required columns
        required_columns = ['corp_code', 'ticker', 'company_name', 'market']
        missing_columns = [col for col in required_columns if col not in result.columns]
        if missing_columns:
            print(f"\nError: Missing required columns: {missing_columns}")
            return
            
        # Use the new DataFrame
        merged = result[required_columns].copy()
        
        # Debug: Print the first row of the final DataFrame
        print("\nFirst row of final data:")
        print(merged.iloc[0])
        
        print(f"Found {len(merged)} matching companies between KRX and DART")
        
        if not merged.empty:
            # Insert into database
            with sqlite3.connect(self.db_path) as conn:
                # Clear existing mappings
                conn.execute("DELETE FROM company_mapping")
                
                # Insert new mappings
                merged['market'] = merged['market'].astype(str)  # Ensure market is string
                
                # Debug: Print the columns we're about to insert
                print("\nColumns being inserted:", merged[['corp_code', 'ticker', 'company_name', 'market']].columns.tolist())
                
                # Use the correct column name (company_name instead of corp_name)
                merged[['corp_code', 'ticker', 'company_name', 'market']].to_sql(
                    'company_mapping', 
                    conn, 
                    if_exists='append', 
                    index=False
                )
                print(f"Inserted {len(merged)} company mappings into database")
                
                # Verify the count
                count = pd.read_sql("SELECT COUNT(*) as count FROM company_mapping", conn).iloc[0]['count']
                print(f"Total company mappings in database: {count}")
        else:
            print("No matching companies found between KRX and DART")

if __name__ == "__main__":
    try:
        mapper = DartMapper()
        mapper.update_mappings()
        print("Company mapping update completed successfully")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)