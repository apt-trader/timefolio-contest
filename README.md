# TimeFolio Portfolio System

A modular, professional-grade system for building, backtesting, and optimizing compliant Korean equity portfolios. It features a complete, end-to-end quantitative workflow, from data ingestion and multi-factor alpha modeling to hyperparameter tuning and risk analysis.
---
## Table of Contents

1.  [Key Features](#key-features)
2.  [System Architecture](#system-architecture)
3.  [The Quantitative Workflow](#the-quantitative-workflow)
4.  [Prerequisites](#prerequisites)
5.  [Installation](#installation)
6.  [Configuration](#configuration)
7.  [CLI Usage](#cli-usage)
8.  [Outputs](#outputs)
9.  [License & Acknowledgements](#license--acknowledgements)
---
## Key Features

- **Robust Data Fetchers**: Parallel-safe modules for KRX (prices), DART (fundamentals), and FRED (macro data).
- **Advanced Factor Engine**: Computes Momentum, Value, Quality, and Macro-regime factors.
- **`CVXPY`-based Optimizer**: Enforces all constraints (weight, cardinality, sector, etc.) in a single, powerful solver.
- **Scientific Backtesting**: Simulates historical performance with key metrics (Sharpe, MDD).
- **Automated Tuning**: Uses `Optuna` to find the best strategy parameters automatically.
---
## System Architecture

The system is organized into distinct, decoupled modules. The `main.py` script orchestrates the portfolio generation pipeline, while the `backtester.py` and `tuner.py` scripts provide a powerful research and optimization framework.

```
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
│ ├── init.py
│ ├── krx_fetcher.py            # Market data (OHLCV, Market Cap)
│ ├── financial_fetcher.py      # Fundamental data (DART)
│ └── macro_fetcher.py          # Macroeconomic data (FRED,yfinance)
|
├── config/
│ └── config.yaml               # Central configuration file.
|
└── krx_data.db                 # Central SQLite Database
```
---
## The Quantitative Workflow

The project follows a professional quantitative research and production lifecycle:

1.  **Data Population:** Use the standalone `fetchers` (or `backfill_krx_data.py`) to populate a local SQLite database with market, fundamental, and macro data. This is done once or periodically to keep the local data store fresh.
2.  **Strategy Research & Tuning:** Use `tuner.py` to run dozens or hundreds of backtests, automatically finding the optimal parameters (e.g., factor windows, risk aversion) that maximize historical performance. This is the core research step.
3.  **Validation:** Update `config.yaml` with the best parameters found by the tuner. Then, use `backtester.py` to run a single, full backtest to generate a detailed performance report and equity curve for the final, tuned strategy.
4.  **Production Run:** Execute `main.py` to generate the final portfolio for the upcoming period using the validated, optimal configuration.
---
## Prerequisites
-   Python 3.9+ & SQLite 3
-   API keys in a `.env` file: `DART_API_KEY`, `FRED_API_KEY`.
-   All packages from `requirements.txt`.
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
# Edit the .env file with your DART_API_KEY and FRED_API_KEY

# 4. Install all dependencies
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

### Step 1: Data Population (Run once, then periodically)
Run these scripts to download the latest data into your local `krx_data.db`. Use the -m flag to run modules as scripts from the project root. Recommended to fetch data for 1 year at a time.

```bash
# Fetch historical market data from KRX
python -m fetchers.krx_fetcher -s 20220101 -e 20221231 --update-db

# Fetch annual financial statements from DART for all universe stocks
python -m fetchers.financial_fetcher --all -s 20220101 -e 20221231

# Fetch historical macroeconomic data from FRED and yfinance
python -m fetchers.macro_fetcher -s 2022-01-01 -e 2022-12-31
```

### Step 2: Find Optimal Parameters (Research Phase)
Use the tuner to discover the best parameters for your strategy. This is an intensive process that may take several hours Results are saved in tuning_results.db.

```bash
python tuner.py --n-trials 100 --study-name "tuning-v1"
```
After the run, copy the "Best Parameters" from the output and update your `config.yaml`.

### Step 3: Validate Strategy (Verification Phase)
Run a full backtest using your newly tuned parameters to confirm performance and generate an equity curve.

```bash
python backtester.py --start 2022-01-01 --end 2025-06-30
```

### Step 4: Generate Final Portfolio (Production Run)
Execute the main pipeline to generate the portfolio for the upcoming period.

```bash
python main.py --output-dir output/
```
---
## Technical Framework

-   **Alpha Model**: A multi-factor model combining **Momentum**, **Value**, **Quality**, **Profitability**, **Low Beta**, and **Macro Regime** signals. Factors are intelligently weighted based on the macro environment.
-   **Portfolio Construction**: Mean-Variance Optimization with L2 regularization and mixed-integer constraints.
    -   `Objective: max  μ'w - λ·w'Σw - η·||w||₂²`
-   **Constraints Enforced in Solver**:
    -   Full Investment: `Σw = 1`
    -   Cardinality: `Σz ≤ 12` (max 12 positions)
    -   Weight Limits: `0.01 ≤ w_i ≤ 0.15`
    -   Sector Exposure: Dynamic limits based on `market_sectors.csv`.
    -   Small-Cap Limit: `Σw_small_cap ≤ 0.40`.
-   **Risk Management**: Post-optimization analysis via `RiskMonitor` checks HHI for weight and return concentration, MDD, and weekly turnover to ensure full contest compliance.
---
## Outputs

-   **Final Portfolio**: `output/final_portfolio.csv`
-   **Risk Report**: `output/risk_report_{date}.txt`
-   **Backtest Equity Curve**: `output/backtest_equity_curve.png`
-   **Forbidden Tickers List**: `forbidden.csv`
-   **Tuning Database**: `tuning.db` (stores results of all tuning trials)

---
## License & Acknowledgements
-   **License**: MIT
-   **Libraries**: Optuna, CVXPY, Pandas, NumPy, Scikit-learn, XGBoost, PyYAML, yfinance, fredapi, pandas-market-calendars
-   **Data Sources**: Korea Exchange (KRX), DART, FRED Economic Data