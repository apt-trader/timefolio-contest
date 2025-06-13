import pandas as pd
import os

def ensure_six_digits(ticker):
    """Convert ticker to string and pad with leading zeros to ensure 6 digits."""
    return str(ticker).strip().zfill(6)

def process_file(file_path):
    """Process a single CSV file to standardize ticker format."""
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Check if 'ticker' column exists
        if 'ticker' not in df.columns:
            print(f"Warning: 'ticker' column not found in {file_path}")
            return False
            
        # Store original count for reporting
        original_count = len(df)
        
        # Convert tickers to 6-digit format
        df['ticker'] = df['ticker'].apply(ensure_six_digits)
        
        # Remove any potential duplicates that might have been created
        df = df.drop_duplicates(subset=['ticker'])
        
        # Save back to the same file
        df.to_csv(file_path, index=False)
        
        # Print summary
        print(f"Processed {file_path}:")
        print(f"  - Original count: {original_count}")
        print(f"  - Final count: {len(df)}")
        print(f"  - Removed {original_count - len(df)} duplicate tickers")
        print("  - Sample tickers:", ", ".join(df['ticker'].head(5).tolist() + ['...']))
        
        return True
        
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return False

def main():
    # Files to process
    files_to_process = [
        'forbidden.csv',
        'forbidden_backup.csv'
    ]
    
    # Process each file
    success_count = 0
    for file_path in files_to_process:
        if os.path.exists(file_path):
            if process_file(file_path):
                success_count += 1
        else:
            print(f"File not found: {file_path}")
    
    print(f"\nSuccessfully processed {success_count} out of {len(files_to_process)} files.")

if __name__ == "__main__":
    main()
