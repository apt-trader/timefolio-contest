import os
from dotenv import load_dotenv

print("Loading environment variables...")
loaded = load_dotenv()
print(f"Environment loaded: {loaded}")
print(f"DART_API_KEY exists: {'DART_API_KEY' in os.environ}")
print(f"FRED_API_KEY exists: {'FRED_API_KEY' in os.environ}")

# If DART_API_KEY exists, print first few characters (for security, not the full key)
if 'DART_API_KEY' in os.environ:
    key = os.environ['DART_API_KEY']
    print(f"DART_API_KEY starts with: {key[:4]}...{key[-4:] if len(key) > 8 else ''}")
else:
    print("DART_API_KEY not found in environment variables")
