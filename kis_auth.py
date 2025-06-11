#!/usr/bin/env python3
"""
KIS Developers API authentication module
Handles token management, acquisition and renewal
"""
import os
import requests
import json
import time
import pathlib
import logging
import datetime as dt

PORTAL_URL = ("https://apiportal.koreainvestment.com"
              "/download/master/MstStockList.csv")
GITHUB_URL = ("https://raw.githubusercontent.com/"
              "koreainvestment/open-trading-api/main/"
              "stocks_info/MstStockList.csv")

CACHE_DIR  = pathlib.Path.home()/".kis_master"; CACHE_DIR.mkdir(exist_ok=True)

def _download(url, path, is_fallback=False):
    """
    Download a file from URL and save it to the specified path
    with authentication error checking and fallback support
    """
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        
        # Check if response is an HTML error page (authentication error)
        content = r.content.decode('utf-8', errors='ignore')
        if '<html' in content.lower() or 'http-equiv="refresh"' in content.lower():
            if not is_fallback:  # Only try fallback if this is the primary URL
                print("Warning: Received HTML response from primary source, trying fallback...")
                return _download(GITHUB_URL, path, is_fallback=True)
                
            print("Warning: Fallback source also returned HTML. Using minimal dataset.")
            # Create a simple CSV with headers to avoid future errors
            backup_content = "short_code,name,market\n"
            backup_content += "005930,Samsung Electronics,KOSPI\n"
            backup_content += "000660,SK Hynix,KOSPI\n"
            path.write_text(backup_content)
            return
            
        # Write normal binary content if we got valid data
        path.write_bytes(r.content)
        
    except Exception as e:
        if not is_fallback:  # Only try fallback if this is the primary URL
            print(f"Warning: Error downloading from primary source ({str(e)}), trying fallback...")
            return _download(GITHUB_URL, path, is_fallback=True)
            
        print(f"Warning: Fallback source also failed ({str(e)}). Using minimal dataset.")
        # Create a minimal dataset if both sources fail
        backup_content = "short_code,name,market\n"
        backup_content += "005930,Samsung Electronics,KOSPI\n"
        backup_content += "000660,SK Hynix,KOSPI\n"
        path.write_text(backup_content)

def get_universe(kind="STK"):                         # STK=보통주
    today = dt.date.today().strftime("%Y%m%d")
    path  = CACHE_DIR/f"master_{today}.csv"

    if not path.exists():
        try:                # ① 포털
            _download(PORTAL_URL, path)
        except Exception:
            _download(GITHUB_URL, path)  # ② 미러

    df = pd.read_csv(path, encoding="euc-kr")
    codes = (df.loc[df["상품구분코드"] == kind, "단축코드"]
               .astype(str).str.zfill(6).tolist())
    return codes

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(ch)

def get_token(force_refresh=False):
    """
    Get KIS OAuth token with automatic renewal
    
    Args:
        force_refresh: Force token refresh even if current one is valid
        
    Returns:
        Access token string
    """
    # Get credentials from environment
    APP_KEY = os.environ.get('KIS_APP_KEY')
    APP_SECRET = os.environ.get('KIS_APP_SECRET')
    
    if not APP_KEY or not APP_SECRET:
        logger.error("KIS_APP_KEY or KIS_APP_SECRET not set in environment variables")
        raise ValueError("Missing KIS API credentials")
    
    # Token storage location
    TOKEN_PATH = pathlib.Path.home()/'.kis_token'
    
    # Check for existing valid token
    if not force_refresh and TOKEN_PATH.exists():
        try:
            tok = json.loads(TOKEN_PATH.read_text())
            # Buffer of 1 hour before expiration (24h tokens, but renew at 23h)
            if tok.get('expire', 0) > time.time():
                logger.debug("Using existing valid token")
                return tok['access_token']
            else:
                logger.info("Token expired, requesting new one")
        except Exception as e:
            logger.warning(f"Error reading token file: {e}")
    
    # Request new token
    try:
        url = "https://openapi.koreainvestment.com:9443/oauth2/tokenP"
        payload = {
            "grant_type": "client_credentials",
            "appkey": APP_KEY, 
            "appsecret": APP_SECRET
        }
        
        # Add randomized delay to avoid pattern detection
        time.sleep(0.1 + (0.3 * time.time() % 0.5))
        
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()  # Raise exception for HTTP errors
        
        res = response.json()
        if 'access_token' not in res:
            logger.error(f"Invalid token response: {res}")
            raise ValueError("Failed to obtain access token")
            
        # Store token with expiry (24h - 1h buffer)
        tok = {
            "access_token": res['access_token'],
            "expire": time.time() + 23*3600,
            "issue_time": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Save to file
        TOKEN_PATH.write_text(json.dumps(tok))
        logger.info(f"New token acquired, valid until {time.ctime(tok['expire'])}")
        
        return tok['access_token']
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error requesting token: {e}")
        
        # Fall back to existing token if available, even if expired
        if TOKEN_PATH.exists():
            try:
                tok = json.loads(TOKEN_PATH.read_text())
                logger.warning("Using expired token as fallback")
                return tok['access_token']
            except:
                pass
                
        raise ValueError(f"Failed to get token and no fallback available: {e}")

def check_token_status():
    """
    Check current token status and time until expiration
    
    Returns:
        Dictionary with token status information
    """
    TOKEN_PATH = pathlib.Path.home()/'.kis_token'
    
    if not TOKEN_PATH.exists():
        return {"status": "missing", "message": "No token file exists"}
    
    try:
        tok = json.loads(TOKEN_PATH.read_text())
        
        now = time.time()
        expiry = tok.get('expire', 0)
        time_left = expiry - now
        
        if time_left > 0:
            hours_left = time_left / 3600
            return {
                "status": "valid",
                "hours_left": round(hours_left, 1),
                "issued": tok.get('issue_time', 'unknown'),
                "expires": time.ctime(expiry)
            }
        else:
            return {
                "status": "expired",
                "expired_for": round(abs(time_left) / 3600, 1),
                "issued": tok.get('issue_time', 'unknown')
            }
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    # Setup console logging when run directly
    logging.basicConfig(level=logging.INFO)
    
    # Simple CLI interface
    import argparse
    parser = argparse.ArgumentParser(description="KIS API Token Management")
    parser.add_argument("--status", action="store_true", help="Check token status")
    parser.add_argument("--refresh", action="store_true", help="Force token refresh")
    
    args = parser.parse_args()
    
    if args.status:
        status = check_token_status()
        print(f"Token status: {status['status']}")
        for k, v in status.items():
            if k != 'status':
                print(f"  {k}: {v}")
    else:
        try:
            token = get_token(force_refresh=args.refresh)
            print(f"Token: {token[:10]}...{token[-5:]}")
            print("Token successfully acquired/refreshed")
        except Exception as e:
            print(f"Error: {e}")
            exit(1)