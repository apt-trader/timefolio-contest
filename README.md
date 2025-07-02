# TimeFolio Portfolio System

A modular, professional-grade system for building, backtesting, and optimizing compliant Korean equity portfolios. It features a complete, end-to-end quantitative workflow, from data ingestion and multi-factor alpha modeling to hyperparameter tuning and risk analysis.

## Table of Contents

1. [Key Features](#key-features)
2. [System Architecture](#system-architecture)
3. [The Quantitative Workflow](#the-quantitative-workflow)
4. [Prerequisites](#prerequisites)
5. [Installation](#installation)
6. [Configuration](#configuration)
7. [CLI Usage](#cli-usage)
8. [Technical Framework](#technical-framework)
9. [Outputs](#outputs)
10. [License & Acknowledgements](#license--acknowledgements)

---

## Key Features

- **Robust Data Fetchers**: Modules for KRX (prices), DART (fundamentals), and FRED/yfinance (macro data), with built-in rate limiting and error handling.
- **Advanced Factor Engine**: Computes a suite of advanced alpha factors (Multi-dimensional Momentum, Robust Value, Deeper Quality, Investment).
- **Machine Learning Alpha Model**: Employs a **Principal Component Regression (PCR)** model to intelligently learn from historical data and combine all factors into a single, powerful predictive signal.
- **`CVXPY`-based Optimizer**: Implements a robust and fast two-step quadratic programming (QP) heuristic to construct efficient portfolios.
- **Scientific Backtesting**: Simulates historical strategy performance with key metrics like Sharpe Ratio and Maximum Drawdown.
- **Automated Tuning**: Uses `Optuna` to discover the optimal strategy hyperparameters automatically.

---

## System Architecture

The system is organized into distinct, decoupled modules.

```
/timefolio-2025
├── main.py                     # Main pipeline for a single-period run.

├── tuner.py                    # Hyperparameter optimization using Optuna.
├── config.py                   # Centralized configuration handler.
├── data_manager.py             # Data access layer, orchestrates fetchers.
├── factor_engine.py            # Computes multi-factor alpha signals.
├── optimizer.py                # Solves for the optimal portfolio.
├── risk_monitor.py             # Post-trade risk analysis and reporting.
|
├── fetchers/                   # Modules for fetching external data.
│   ├── __init__.py
│   ├── krx_fetcher.py          # Market data (OHLCV, Market Cap)
│   ├── financial_fetcher.py    # Fundamental data (DART)
│   └── macro_fetcher.py        # Macroeconomic data (FRED, yfinance)
|
├── config/
│   └── config.yaml             # Central configuration file.
|
└── krx_data.db                 # Central SQLite Database
```

---

## The Quantitative Workflow

The project follows a professional quantitative research and production lifecycle:

1. **Data Population:** Use the standalone `fetchers` to populate a local SQLite database with market, fundamental, and macro data. This is done once or periodically to keep the local data store fresh.
2. **Strategy Research & Tuning:** Use `tuner.py` to run dozens or hundreds of backtests, automatically finding the optimal parameters (e.g., factor windows, risk aversion) that maximize historical performance.
3. **Validation:** Update `config.yaml` with the best parameters found by the tuner for live trading deployment.
4. **Production Run:** Execute `main.py` to generate the final portfolio for the upcoming period using the validated, optimal configuration.

---

## Prerequisites

- Python 3.9+ & SQLite 3
- API keys for DART and FRED.

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-org/timefolio-2025.git
cd timefolio-2025

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Create a .env file for your API keys (in the project root)
cp .env.example .env
# -> Now, edit the .env file with your DART_API_KEY and FRED_API_KEY

# 4. Install all dependencies from the corrected requirements file
pip install -r requirements.txt
```

---

## Configuration

All system parameters are managed in `config/config.yaml`. This centralized approach allows for easy tuning and experimentation.

```yaml
# config/config.yaml
data_settings:
  # Paths are relative to the project root
  stock_universe_file: "sector_universe.csv"
  market_sectors_file: "market_sectors.csv"
  db_path: "db/krx_data.db"
```

---

## CLI Usage

### **Step 1: Populate Your Database (Run once, then periodically)**

Run these scripts from the project root to download data into `krx_data.db`. It's recommended to fetch data for one year at a time to respect API limits.

```bash
# Note: The -m flag is crucial for running scripts inside a package.

# 1. Fetch market data from KRX
# ex: For the year 2022
python -m fetchers.krx_fetcher -s 20240101 -e 20241231 --update-db

# 2. Fetch macroeconomic data
# ex: For the year 2022
python -m fetchers.macro_fetcher -s 2024-01-01 -e 2024-12-31

# 3. Fetch annual financial statements from DART for all universe stocks
# ex: For the year 2022's reports
python -m fetchers.financial_fetcher --all -s 2024 -e 2024
```

### **Step 2: Find Optimal Parameters (Research Phase)**

**Strategic Focus**: Train on 2023-2024 period (current market regime) while excluding the anomalous COVID era (2020-2022).

**Objectives (in order of priority)**:
1. **Sharpe Ratio Optimization**: Target 1.0+ for contest competitiveness
2. **Risk-Adjusted Returns**: Achieve 25-35% annualized returns with controlled volatility
3. **Drawdown Management**: Keep maximum drawdown under 15% for capital preservation
4. **Factor Balance**: Learn from realistic market conditions with proper Value/Momentum/Quality rotations
5. **Parameter Robustness**: Avoid overfitting to unrepeatable market anomalies

**Time Estimation**: 2-4 hours with parallelization, 7-10 hours single-threaded (300 trials).

```bash
# Full hyperparameter tuning on strategic period
python tuner.py --start-date 2023-01-01 --end-date 2024-12-27 --n-trials 300 \
 --study-name "final-model-tuning-v1" --n-jobs 4
```

```bash
# Test optimized parameters on full strategic period
python test_improvement.py
```

After the run, copy the "Best Parameters" from the output into your `config.yaml`.

### **Step 3: Market Cycle Analysis (Research Tool)**

Use the advanced FFT analysis tool to understand cyclical patterns in market data, validate strategy parameters, and gain deeper insights into individual stock behaviors.

**Key Use Cases:**
- **Stock Research**: Deep-dive into individual stock cyclical patterns
- **Parameter Validation**: Confirm that optimized parameters align with market rhythms

```bash
# Analyze Samsung Electronics cyclical patterns
python fft.py 005930 -s 2023-01-01 -e 2024-06-01
```

**Analyzing the FFT Results**:

The generated PNG contains three key analysis panels:

**Panel 1: Original vs Detrended Data**
- **Blue line**: Original price series
- **Orange line**: Detrended series (linear trend removed)
- **Purpose**: Shows cyclical patterns without trend interference

**Panel 2: FFT Spectrum (Key Analysis)**
- **X-axis**: Period (in days) - identifies cycle lengths
- **Y-axis**: Magnitude (strength of each cycle)
- **Green dashed lines**: Dominant cycles (e.g., 30-day, 102-day)
- **Purpose**: Reveals the strongest cyclical patterns in the data

**Panel 3: Signal Reconstruction**
- **Blue line**: Original price series
- **Orange line**: Reconstructed using only dominant cycles
- **Purpose**: Validates that identified cycles capture real market patterns

**Strategic Validation**: Compare identified cycles (e.g., 30-day dominant cycle) with your optimized momentum window (e.g., 48 days) to confirm your strategy captures genuine market rhythms rather than noise.


### **Step 4: Generate Final Portfolio (Production Run)**

Execute the main pipeline to generate the portfolio for the upcoming period.

```bash
python main.py --output-dir output/
```

---

## Technical Framework

- **Alpha Model**: A machine-learning-driven multi-factor model.
  - **Factor Library**: The factor model is built on a comprehensive set of individual factors, grouped into five core families. This granular approach allows for more precise risk and return attribution.
    - **Value**: Book-to-Price (B/P), Earnings-to-Price (E/P), Sales-to-Price (S/P), and Cash-Flow-to-Price (CF/P).
    - **Quality**: Return-on-Equity (ROE), Financial Leverage, and ROE Stability.
    - **Profitability**: Gross Profitability (GPA), Operating Margin, and Net Margin.
    - **Momentum**: 12-Month Momentum, 6-Month Acceleration, and Volatility-Scaled Momentum.
    - **Investment**: Total Asset Growth and CAPEX Growth.
  - **Signal Generation**: All factors are individually winsorized and standardized to ensure robustness. The final alpha score is generated using a Principal Component Regression (PCR) model, which creates a diversified signal from these inputs. This approach reduces noise and captures the most significant drivers of alpha. The final signal is dynamically tilted based on a macro regime indicator (the 10y-2y yield spread).
- **Portfolio Construction**: A two-step Mean-Variance Optimization (MVO) process.
  - **Step 1 (Pre-selection)**: A candidate universe of high-potential assets is selected based on expected returns to reduce the problem size and control the number of positions.
  - **Step 2 (Optimization)**: A continuous Quadratic Programming (QP) solver (OSQP) finds the optimal weights for the selected assets.
  - `Objective: max  μ'w - λ·w'Σw`
- **Constraints & Heuristics**:
  - **Full Investment**: `Σw = 1` (enforced in solver).
  - **Weight Limits**: `w_i ≤ max_weight` (e.g., 0.15, enforced in solver).
  - **Sector Exposure**: Dynamic deviation limits relative to a market benchmark (enforced in solver).
  - **Turnover**: Limits on portfolio rebalancing to control transaction costs.
- **Risk Management**: Post-optimization analysis via `RiskMonitor` checks HHI for weight and return concentration, MDD, and weekly turnover to ensure compliance.

---

## Outputs

All outputs are saved to the directory specified by `--output-dir` (default: `output/`), unless otherwise noted.

### From `main.py` (Production Run)
- **Final Portfolio**: `final_portfolio.csv`
- **Risk Report**: `risk_report_{timestamp}.txt`


- **Performance Summary**: `backtest_summary.txt`
- **Equity Curve Plot**: `backtest_equity_curve.png`
- **Daily Returns Series**: `backtest_returns.csv`

### From `tuner.py` (Tuning Run)
- **Tuning Database**: `tuning_results.db` (saved in the project root)

---

## License & Acknowledgements

- **License**: MIT
- **Libraries**: Optuna, CVXPY, Pandas, NumPy, Scikit-learn, XGBoost, PyYAML, yfinance, fredapi, pandas-market-calendars, OpenDartReader-unofficial
- **Data Sources**: Korea Exchange (KRX), DART, FRED Economic Data
