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
- **Advanced Factor Engine**: Computes a sophisticated alpha signal from a composite of **Momentum**, **Value**, **Quality**, and **Macro-Regime** factors.
- **`CVXPY`-based Optimizer**: Enforces all constraints (weight, cardinality, sector, small-cap) in a single, powerful mixed-integer solver.
- **Scientific Backtesting**: Simulates historical strategy performance with key metrics like Sharpe Ratio and Maximum Drawdown.
- **Automated Tuning**: Uses `Optuna` to discover the optimal strategy hyperparameters automatically.

---

## System Architecture

The system is organized into distinct, decoupled modules.

/timefolio-2025
├── main.py                     # Main pipeline for a single-period run.
├── backtester.py               # Simulates strategy performance over time.
├── tuner.py                    # Hyperparameter optimization using Optuna.
├── config.py                   # Centralized configuration handler.
├── data_manager.py             # Data access layer, orchestrates fetchers.
├── factor_engine.py            # Computes multi-factor alpha signals.
├── optimizer.py                # Solves for the optimal portfolio.
├── risk_monitor.py             # Post-trade risk analysis and reporting.
|
├── fetchers/                   # Modules for fetching external data.
│   ├── **init**.py
│   ├── krx_fetcher.py          # Market data (OHLCV, Market Cap)
│   ├── financial_fetcher.py    # Fundamental data (DART)
│   └── macro_fetcher.py        # Macroeconomic data (FRED, yfinance)
|
├── config/
│   └── config.yaml             # Central configuration file.
|
└── krx_data.db                 # Central SQLite Database

---

## The Quantitative Workflow

The project follows a professional quantitative research and production lifecycle:

1. **Data Population:** Use the standalone `fetchers` to populate a local SQLite database with market, fundamental, and macro data. This is done once or periodically to keep the local data store fresh.
2. **Strategy Research & Tuning:** Use `tuner.py` to run dozens or hundreds of backtests, automatically finding the optimal parameters (e.g., factor windows, risk aversion) that maximize historical performance.
3. **Validation:** Update `config.yaml` with the best parameters found by the tuner. Then, use `backtester.py` to run a single, full backtest to generate a detailed performance report and equity curve for the final, tuned strategy.
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
  db_path: "krx_data.db"
```

---

## CLI Usage

### **Step 1: Populate Your Database (Run once, then periodically)**

Run these scripts from the project root to download data into `krx_data.db`. It's recommended to fetch data for one year at a time to respect API limits.

```bash
# Note: The -m flag is crucial for running scripts inside a package.

# 1. Fetch market data from KRX
# ex: For the year 2022
python -m fetchers.krx_fetcher -s 20220101 -e 20221231 --update-db

# 2. Fetch macroeconomic data
# ex: For the year 2022
python -m fetchers.macro_fetcher -s 2022-01-01 -e 2022-12-31

# 3. Fetch annual financial statements from DART for all universe stocks
# ex: For the year 2022's reports
python -m fetchers.financial_fetcher --all -s 2022 -e 2022
```

### **Step 2: Find Optimal Parameters (Research Phase)**

This may take several hours. Results are saved in `tuning_results.db`.

```bash
python tuner.py --n-trials 100 --study-name "tuning-v1"
```

After the run, copy the "Best Parameters" from the output into your `config.yaml`.

### **Step 3: Validate Strategy (Verification Phase)**

Run a full backtest using your tuned parameters to confirm performance.

```bash
python backtester.py --start 2022-01-01 --end 2023-12-31
```

### **Step 4: Generate Final Portfolio (Production Run)**

Execute the main pipeline to generate the portfolio for the upcoming period.

```bash
python main.py --output-dir output/
```

---

## Technical Framework

- **Alpha Model**: A multi-factor model combining **Momentum**, **Value**, **Quality**, **Profitability**, and **Macro Regime** signals. Factors are intelligently weighted based on the macro environment.
- **Portfolio Construction**: Mean-Variance Optimization with L2 regularization and mixed-integer constraints.
  - `Objective: max  μ'w - λ·w'Σw - η·||w||₂²`
- **Constraints Enforced in Solver**:
  - Full Investment: `Σw = 1`
  - Cardinality: `Σz ≤ 12` (max 12 positions)
  - Weight Limits: `0.01 ≤ w_i ≤ 0.15`
  - Sector Exposure: Dynamic limits based on `market_sectors.csv`.
  - Small-Cap Limit: `Σw_small_cap ≤ 0.40`.
- **Risk Management**: Post-optimization analysis via `RiskMonitor` checks HHI, MDD, and weekly turnover to ensure compliance.

---

## Outputs

- **Final Portfolio**: `output/final_portfolio.csv`
- **Risk Report**: `output/risk_report_{date}.txt`
- **Backtest Equity Curve**: `output/backtest_equity_curve.png`
- **Forbidden Tickers List**: `forbidden.csv`
- **Tuning Database**: `tuning.db` (stores results of all tuning trials)

---

## License & Acknowledgements

- **License**: MIT
- **Libraries**: Optuna, CVXPY, Pandas, NumPy, Scikit-learn, XGBoost, PyYAML, yfinance, fredapi, pandas-market-calendars, OpenDartReader-unofficial
- **Data Sources**: Korea Exchange (KRX), DART, FRED Economic Data
