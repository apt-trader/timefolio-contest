import sys
from fredapi import Fred

try:
    print("Initializing FRED client...")
    fred = Fred(api_key='c5a3033b4d0dfbb0014d152e22218ad9')
    print("FRED client initialized successfully")
    
    print("\nFetching NASDAQ100 data...")
    data = fred.get_series('NASDAQ100')
    
    if data is not None and not data.empty:
        print("\nSuccessfully retrieved data:")
        print(f"Data points: {len(data)}")
        print(f"Date range: {data.index[0].date()} to {data.index[-1].date()}")
        print("\nFirst 5 rows:")
        print(data.head())
    else:
        print("\nNo data was returned from FRED API")
        
except ImportError as e:
    print(f"\nError importing required packages: {e}")
    print("Please make sure you have installed all required packages (fredapi, pandas)")
    
except Exception as e:
    print(f"\nAn error occurred: {str(e)}", file=sys.stderr)
    print("\nTroubleshooting steps:")
    print("1. Check your internet connection")
    print("2. Verify your FRED API key is valid")
    print("3. Try running the script again")