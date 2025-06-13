from OpenDartReader.dart import OpenDartReader
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Get API key from environment
api_key = os.getenv('DART_API_KEY')
if not api_key:
    raise ValueError("DART_API_KEY not found in environment variables")

dart = OpenDartReader(api_key)

try:
    corp_code = dart.find_corp_code('005930')  # 삼성전자 고유번호
    print("Found corp code:", corp_code)
    df = dart.list(corp_code, start='2023-01-01')  # 공시 리스트 조회
    print(df.head())
except Exception as e:
    print("Error occurred:", str(e))