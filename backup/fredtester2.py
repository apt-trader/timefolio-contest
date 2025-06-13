from fredapi import Fred
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

# Configure plot style
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (14, 7)
plt.rcParams['font.size'] = 12

def fetch_nasdaq100_data():
    try:
        # Initialize FRED API client
        api_key = 'c5a3033b4d0dfbb0014d152e22218ad9'
        fred = Fred(api_key=api_key)
        
        print("Fetching NASDAQ-100 data from FRED...")
        
        # Fetch NASDAQ-100 data (the series code for NASDAQ-100 in FRED is 'NASDAQ100')
        nasdaq_data = fred.get_series('NASDAQ100')
        
        if nasdaq_data.empty:
            raise ValueError("No data returned from FRED API")
            
        # Convert to DataFrame and clean
        df = pd.DataFrame(nasdaq_data, columns=['NASDAQ100'])
        df = df.dropna()
        
        print(f"\nSuccessfully retrieved {len(df)} data points")
        print(f"Date range: {df.index[0].date()} to {df.index[-1].date()}")
        
        return df
        
    except Exception as e:
        print(f"\nError fetching data: {str(e)}")
        return None

def plot_nasdaq100_data(df):
    if df is None or df.empty:
        print("No data available to plot")
        return
    
    plt.figure(figsize=(14, 7))
    
    # Plot the data
    plt.plot(df.index, df['NASDAQ100'], 
             linewidth=2, 
             color='#2ecc71',
             label='NASDAQ-100 Index')
    
    # Customize the plot
    plt.title('NASDAQ-100 Index Performance\n', fontsize=16, fontweight='bold')
    plt.suptitle(f"Last Updated: {datetime.now().strftime('%Y-%m-%d')}", 
                 y=0.91, fontsize=10, color='gray')
    
    # Format y-axis with comma separators
    ax = plt.gca()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
    
    # Format x-axis to show years
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    # Add grid and legend
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Add value annotations for first and last points
    first_val = df.iloc[0]['NASDAQ100']
    last_val = df.iloc[-1]['NASDAQ100']
    
    plt.annotate(f"{first_val:,.0f}", 
                 xy=(df.index[0], first_val),
                 xytext=(10, 10), 
                 textcoords='offset points',
                 bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
    
    plt.annotate(f"{last_val:,.0f}", 
                 xy=(df.index[-1], last_val),
                 xytext=(-60, 10), 
                 textcoords='offset points',
                 bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
    
    # Add some margin
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('nasdaq100_performance.png', dpi=300, bbox_inches='tight')
    print("\nChart saved as 'nasdaq100_performance.png'")
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    # Fetch the data
    nasdaq_data = fetch_nasdaq100_data()
    
    # If data was fetched successfully, plot it
    if nasdaq_data is not None:
        plot_nasdaq100_data(nasdaq_data)
        
        # Show some basic statistics
        print("\nBasic Statistics:")
        print(nasdaq_data.describe())