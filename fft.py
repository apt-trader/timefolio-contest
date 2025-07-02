# fft.py
# An advanced FFT analysis tool designed to work with the project's krx_data.db

import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import detrend
import logging
import argparse
from pathlib import Path

# --- Configuration ---
# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Database Path (relative to the project root)
DB_PATH = Path(__file__).parent / "db" / "krx_data.db"

# Analysis Parameters
N_DOMINANT_CYCLES = 10  # Number of top cycles to use for reconstruction

def load_data_from_db(ticker: str, db_path: Path, start_date: str, end_date: str) -> pd.Series:
    """
    Loads time series data for a given ticker directly from the SQLite database.
    It can handle both stock codes and macro data series names.
    """
    logging.info(f"Loading data for '{ticker}' from database: {db_path}")
    
    if not db_path.exists():
        logging.error(f"Database file not found at {db_path}")
        return pd.Series(dtype=float)

    try:
        with sqlite3.connect(db_path) as conn:
            # Check if the ticker exists in the daily_prices table (for stocks/KOSPI)
            query_prices = """
                SELECT date, close FROM daily_prices 
                WHERE code = ? AND date BETWEEN ? AND ? ORDER BY date
            """
            df = pd.read_sql_query(query_prices, conn, params=(ticker, start_date, end_date),
                                   parse_dates=['date'], index_col='date')
            
            if not df.empty:
                logging.info(f"Found '{ticker}' in daily_prices table.")
                return df['close']

            # If not found, check if it's a column in the macro_data table
            query_macro = f"""
                SELECT date, "{ticker}" FROM macro_data
                WHERE date BETWEEN ? AND ? ORDER BY date
            """
            df_macro = pd.read_sql_query(query_macro, conn, params=(start_date, end_date),
                                         parse_dates=['date'], index_col='date')
            
            if not df_macro.empty and ticker in df_macro.columns:
                logging.info(f"Found '{ticker}' in macro_data table.")
                return df_macro[ticker].dropna()

            # If not found anywhere
            logging.error(f"Ticker or series name '{ticker}' not found in the database.")
            return pd.Series(dtype=float)

    except Exception as e:
        logging.error(f"Failed to load data from database: {e}")
        return pd.Series(dtype=float)


def plot_fft_results(
    original_data: pd.Series,
    detrended_data: np.ndarray,
    periods: np.ndarray,
    magnitude: np.ndarray,
    dominant_periods: pd.Series,
    reconstructed_signal: np.ndarray,
    ticker: str,
    start_date: str,
    end_date: str
):
    """Generates a comprehensive plot of the FFT analysis results."""
    # This function is unchanged from the previous version. It's a pure plotting utility.
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axs = plt.subplots(3, 1, figsize=(15, 18))
    fig.suptitle(f'Advanced Fourier Transform Analysis for {ticker}', fontsize=20, weight='bold')
    axs[0].plot(original_data.index, original_data, label='Original Series', color='deepskyblue', alpha=0.9)
    trend = original_data.values - detrended_data
    axs[0].plot(original_data.index, trend, label='Underlying Trend (Linear Fit)', color='red', linestyle='--', alpha=0.7)
    axs[0].set_title('Step 1: Original Time Series and Detrending', fontsize=14); axs[0].set_ylabel('Value'); axs[0].legend(); axs[0].grid(True)
    positive_freq_mask = periods > 0
    axs[1].plot(periods[positive_freq_mask], magnitude[positive_freq_mask], label='FFT Spectrum', color='darkviolet')
    axs[1].set_title('Step 2: FFT Power Spectrum', fontsize=14); axs[1].set_xlabel('Period (Days)'); axs[1].set_ylabel('Magnitude (Normalized)'); axs[1].set_xscale('log'); axs[1].grid(True, which="both", ls="--", linewidth=0.5)
    for period in dominant_periods.values:
        axs[1].axvline(x=period, color='green', linestyle='--', alpha=0.6)
    logging.info(f"Top {len(dominant_periods)} Dominant Periods (in days):\n{dominant_periods.to_string()}")
    axs[2].plot(original_data.index, original_data, label='Original Series', color='deepskyblue', alpha=0.8)
    axs[2].plot(original_data.index, reconstructed_signal, label=f'Reconstructed Signal (Top {N_DOMINANT_CYCLES} Cycles)', color='darkorange', linewidth=2)
    axs[2].set_title(f'Step 3: Signal Reconstruction using Dominant Cycles', fontsize=14); axs[2].set_ylabel('Value'); axs[2].legend(); axs[2].grid(True)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    # Save the plot as PNG file
    output_filename = f"fft_analysis_{ticker}_{start_date}_to_{end_date}.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    logging.info(f"FFT analysis plot saved as: {output_filename}")
    
    plt.show()


def run_fft_analysis(ticker: str, start: str, end: str):
    """
    Main function to perform the end-to-end FFT analysis on data from krx_data.db.
    """
    # --- THIS IS THE MAIN CHANGE: Data Loading ---
    original_series = load_data_from_db(ticker, DB_PATH, start, end)
    
    if original_series.empty:
        logging.error(f"Cannot perform analysis for '{ticker}' as no data was loaded.")
        return

    N = len(original_series)
    logging.info(f"Successfully loaded {N} data points for analysis.")
    
    # The rest of the FFT analysis logic is identical to the previous script
    detrended_series = detrend(original_series, type='linear')
    window = np.hanning(N)
    windowed_series = detrended_series * window
    n_padded = 2**int(np.ceil(np.log2(N)))
    padded_series = np.pad(windowed_series, (0, n_padded - N), 'constant')
    fft_result = np.fft.fft(padded_series)
    frequencies = np.fft.fftfreq(n_padded, d=1)
    
    # We take the one-sided spectrum and normalize
    magnitude = np.abs(fft_result)[:n_padded // 2] * 2 / N
    frequencies = frequencies[:n_padded // 2]
    
    with np.errstate(divide='ignore'):
        periods = 1 / frequencies
        
    valid_mask = (periods > 1) & (periods < N / 2)
    spectrum = pd.Series(magnitude[valid_mask], index=periods[valid_mask])
    dominant_periods = spectrum.nlargest(N_DOMINANT_CYCLES)
    
    # Reconstruct the signal
    fft_filtered = fft_result.copy()
    min_dominant_freq = 1 / dominant_periods.index.max()
    full_frequencies = np.fft.fftfreq(n_padded, d=1)
    fft_filtered[np.abs(full_frequencies) < min_dominant_freq] = 0
    
    inverse_fft_result = np.fft.ifft(fft_filtered)
    reconstructed_detrended = np.real(inverse_fft_result[:N])
    original_trend = original_series.values - detrended_series
    reconstructed_signal = reconstructed_detrended + original_trend
    
    plot_fft_results(original_series, detrended_series, periods, magnitude, dominant_periods, reconstructed_signal, ticker, start, end)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Perform advanced FFT analysis on time series data from the project database.")
    parser.add_argument("ticker", help="The stock code (e.g., '005930') or macro series name (e.g., 'vix', 'us10y2y') to analyze.")
    parser.add_argument("-s", "--start-date", default='2020-01-01', help="Start date for the analysis (YYYY-MM-DD).")
    parser.add_argument("-e", "--end-date", default='2024-06-01', help="End date for the analysis (YYYY-MM-DD).")
    args = parser.parse_args()
    
    run_fft_analysis(args.ticker, args.start_date, args.end_date)