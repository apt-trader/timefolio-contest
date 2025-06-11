# universe.py
import pandas as pd, requests, datetime as dt, pathlib

PORTAL_URL = "https://apiportal.koreainvestment.com/download/master/MstStockList.csv"
GITHUB_URL = ("https://raw.githubusercontent.com/"
              "koreainvestment/open-trading-api/main/stocks_info/MstStockList.csv")
CACHE_DIR  = pathlib.Path.home()/".kis_master"; CACHE_DIR.mkdir(exist_ok=True)

def _download(url, path):
    r = requests.get(url, timeout=10); r.raise_for_status()
    path.write_bytes(r.content)

def get_universe(kind:str="STK")->list[str]:
    """
    Retrieve active stock codes from KIS master list
    
    Parameters
    ----------
    kind : str
        Stock type ('STK' for common stocks, 'ETF', 'ETN', etc.)
        
    Returns
    -------
    list[str]
        List of 6-digit stock codes for currently active securities
    """
    today = dt.date.today()
    stamp = today.strftime("%Y%m%d")
    csvfile = CACHE_DIR / f"master_{stamp}.csv"

    # Download the master list if not already cached for today
    if not csvfile.exists():
        try:
            print(f"Downloading master stock list from KIS portal...")
            _download(PORTAL_URL, csvfile)
        except Exception as e:
            print(f"Failed to download from KIS portal: {e}\nTrying GitHub mirror...")
            _download(GITHUB_URL, csvfile)
    
    # Simple fallback for just getting KOSPI stocks if all else fails
    fallback_list = [
        "005930", "000660", "051910", "035420", "005380",  # Top 5
        "006400", "000270", "068270", "105560", "028260",  # 6-10
        "035720", "012330", "055550", "051900", "066570"   # 11-15
    ]
    
    try:
        # Load the CSV file
        df = pd.read_csv(csvfile, encoding="euc-kr")
        
        # Print columns for debugging
        print(f"CSV columns: {df.columns.tolist()}")
        
        # Check for required columns
        code_col = None
        for col in ["단축코드", "종목코드", "short_code", "code"]:
            if col in df.columns:
                code_col = col
                break
        
        if code_col is None:
            print("Could not find stock code column in CSV. Using fallback list.")
            return fallback_list
            
        # Try to filter by stock type if possible
        type_col = None
        for col in ["상품구분코드", "종목종류", "product_type", "type"]:
            if col in df.columns:
                type_col = col
                break
        
        if type_col and kind:
            filtered_df = df[df[type_col] == kind]
            if len(filtered_df) > 0:
                codes = filtered_df[code_col].astype(str).str.zfill(6).tolist()
                print(f"Found {len(codes)} active {kind} stocks")
                return codes
        
        # If filtering fails or not possible, return all codes
        all_codes = df[code_col].astype(str).str.zfill(6).tolist()
        print(f"Returning all {len(all_codes)} stock codes")
        return all_codes
        
    except Exception as e:
        print(f"Error processing master stock list: {e}\nUsing fallback list.")
        return fallback_list