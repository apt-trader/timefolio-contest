import os
import json
import requests
import base64
import hashlib
import hmac
import datetime
from dotenv import load_dotenv

class KISMarketData:
    def __init__(self):
        load_dotenv()
        
        # Load KIS API credentials from environment variables
        self.APP_KEY = os.getenv('KIS_APP_KEY')
        self.APP_SECRET = os.getenv('KIS_APP_SECRET')
        
        if not all([self.APP_KEY, self.APP_SECRET]):
            raise ValueError("Missing required KIS API credentials in environment variables")
            
        # Base URLs
        self.base_url = "https://openapi.koreainvestment.com:9443"  # Real server
        # self.base_url = "https://openapivts.koreainvestment.com:29443"  # Virtual server for testing
        
        # Generate access token
        self.access_token = self._get_access_token()
        
    def _get_access_token(self):
        """Generate access token using API key and secret"""
        url = f"{self.base_url}/oauth2/tokenP"
        headers = {
            "content-type": "application/json"
        }
        body = {
            "grant_type": "client_credentials",
            "appkey": self.APP_KEY,
            "appsecret": self.APP_SECRET
        }
        
        res = requests.post(url, headers=headers, data=json.dumps(body))
        if res.status_code == 200:
            return res.json()['access_token']
        else:
            raise Exception(f"Failed to get access token: {res.text}")
    
    def get_headers(self):
        """Generate headers with authentication"""
        return {
            "Content-Type": "application/json",
            "authorization": f"Bearer {self.access_token}",
            "appKey": self.APP_KEY,
            "appSecret": self.APP_SECRET,
            "custtype": "P"  # Personal
        }
    
    # Example: Get stock price
    def get_stock_price(self, symbol: str, timeframe: str = "D", count: int = 100):
        """Get historical stock price data"""
        url = f"{self.base_url}/uapi/domestic-stock/v1/quotations/inquire-daily-price"
        
        headers = self.get_headers()
        headers["tr_id"] = "FHKST03010100"  # Historical price inquiry
        
        params = {
            "fid_cond_mrkt_div_code": "J",  # J: 주식
            "fid_input_iscd": symbol,       # 종목코드
            "fid_org_adj_prc": "0",         # 0: 수정주가, 1: 원주가
            "fid_period_div_code": timeframe  # D: 일, W: 주, M: 월, Y: 년
        }
        
        res = requests.get(url, headers=headers, params=params)
        return res.json()
    
    # Example: Get current price
    def get_current_price(self, symbol: str):
        """Get current stock price"""
        url = f"{self.base_url}/uapi/domestic-stock/v1/quotations/inquire-price"
        
        headers = self.get_headers()
        headers["tr_id"] = "FHKST01010100"  # Current price inquiry
        
        params = {
            "fid_cond_mrkt_div_code": "J",  # J: 주식
            "fid_input_iscd": symbol        # 종목코드
        }
        
        res = requests.get(url, headers=headers, params=params)
        return res.json()