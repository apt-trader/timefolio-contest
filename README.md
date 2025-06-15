# TimeFolio Portfolio System

A modular, A+-grade system for building, backtesting, and optimizing compliant Korean equity portfolios. It features a complete, end-to-end quantitative workflow, from data ingestion and multi-factor alpha modeling to hyperparameter tuning and risk analysis.

## Table of Contents

1.  [System Architecture](#system-architecture)
2.  [The Quantitative Workflow](#the-quantitative-workflow)
3.  [Prerequisites](#prerequisites)
4.  [Installation](#installation)
5.  [Configuration](#configuration)
6.  [Core Components](#core-components)
7.  [CLI Usage](#cli-usage)
8.  [Technical Framework](#technical-framework)
9.  [Outputs](#outputs)
10. [License & Acknowledgements](#license--acknowledgements)

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
│   ├── krx_fetcher.py          # Market data (OHLCV, Market Cap)
│   ├── financial_fetcher.py    # Fundamental data (DART)
│   └── macro_fetcher.py        # Macroeconomic data (FRED)
|
├── utils/                      # Common utility functions.
│   ├── portfolio_metrics.py
│   └── sector_parser.py
|
└── config/
    └── config.yaml             # Central configuration file.
```

## The Quantitative Workflow

The project follows a professional quantitative research and production lifecycle:

1.  **Data Population:** Use the standalone `fetchers` (or `backfill_krx_data.py`) to populate a local SQLite database with market, fundamental, and macro data. This is done once or periodically to keep the local data store fresh.
2.  **Strategy Research & Tuning:** Use `tuner.py` to run dozens or hundreds of backtests, automatically finding the optimal parameters (e.g., factor windows, risk aversion) that maximize historical performance. This is the core research step.
3.  **Validation:** Update `config.yaml` with the best parameters found by the tuner. Then, use `backtester.py` to run a single, full backtest to generate a detailed performance report and equity curve for the final, tuned strategy.
4.  **Production Run:** Execute `main.py` to generate the final portfolio for the upcoming period using the validated, optimal configuration.

## Prerequisites
-   Python 3.8+ & SQLite 3
-   API keys in a `.env` file: `DART_API_KEY`, `FRED_API_KEY`.
-   All packages from `requirements.txt`.

## Installation

```bash
# 1. Clone the repository and navigate into it
git clone https://github.com/your-org/timefolio-2025.git
cd timefolio-2025

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install all dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Edit the .env file with your personal API keys
```
---
## Configuration

All system parameters are managed in `config/config.yaml`. This centralized approach allows for easy tuning and experimentation.

```yaml
# config/config.yaml
data_settings:
  stock_universe_file: "data/sector_universe.csv"
  market_sectors_file: "data/market_sectors.csv"
  db_path: "krx_data.db"
  start_date: '2022-01-01'
  end_date: '2024-06-01'

optimization:
  max_positions: 12
  individual_limit: 0.15
  min_weight: 0.01
  risk_aversion: 1.25      # Tunable parameter
  l2_penalty: 0.2          # Tunable parameter
  
risk_management:
  min_avg_daily_value: 3000000000
  min_ipo_days: 90
  small_cap_threshold: 1000000000000 # 1 Trillion KRW
  max_small_cap_weight: 0.40

factor_engine:
  mom_windows: [20, 60, 120] # Tunable parameters
  vol_window: 25             # Tunable parameter
```
---
## Core Components

-   **`fetchers/`**: A suite of robust modules for pulling data from KRX, DART (financials), and FRED (macro), and persisting it to a local SQLite DB.
-   **`data_manager.py`**: The single source of truth for data. It runs compliance filters and provides the rest of the application with clean, aligned, and analysis-ready data from all sources.
-   **`factor_engine.py`**: A sophisticated alpha model that calculates and combines technical, fundamental (Value, Quality, Profitability), and macroeconomic factors into a unified return forecast.
-   **`optimizer.py`**: A powerful `CVXPY`-based optimization engine that enforces all contest rules (position count, weight limits, sector caps, small-cap limits) within a mixed-integer quadratic program.
-   **`backtester.py`**: A scientific tool for simulating strategy performance over historical periods, providing key metrics like Sharpe Ratio and Max Drawdown.
-   **`tuner.py`**: An automated `Optuna`-based script that runs hundreds of backtests to find the optimal set of strategy parameters.

---
## CLI Usage

### **Step 1: Data Population (As Needed)**
Run these scripts to download the latest data into your local `krx_data.db`.

```bash
# Fetch historical market data from KRX (e.g., for the last 2 years)
python fetchers/krx_fetcher.py --start-date 20220101 --end-date 20250630

# Fetch historical financial statements from DART for all universe stocks
python fetchers/financial_fetcher.py --all --start-year 2022 --end-year 2025

# Fetch historical macroeconomic data from FRED
python fetchers/macro_fetcher.py --start-date 2022-01-01 --end-date 2025-06-30
```

### **Step 2: Find Optimal Parameters (Research Phase)**
Use the tuner to discover the best parameters for your strategy. This is an intensive process that may take several hours.

```bash
python tuner.py --n-trials 100 --study-name "full-factor-tuning-v1"
```
After the run, copy the "Best Parameters" from the output and update your `config.yaml`.

### **Step 3: Validate Strategy (Verification Phase)**
Run a full backtest using your newly tuned parameters to confirm performance and generate an equity curve.

```bash
python backtester.py --config config/config.yaml --start 2022-01-01 --end 2025-06-30
```

### **Step 4: Generate Final Portfolio (Production Run)**
Execute the main pipeline to generate the portfolio for the upcoming period.

```bash
python main.py --config config/config.yaml --output-dir output/
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

## Outputs

-   **Final Portfolio**: `output/final_portfolio.csv`
-   **Risk Report**: `output/risk_report_{date}.txt`
-   **Backtest Equity Curve**: `output/backtest_equity_curve.png`
-   **Forbidden Tickers List**: `forbidden.csv`
-   **Tuning Database**: `tuning.db` (stores results of all tuning trials)

---
## License & Acknowledgements
-   **License**: MIT
-   **Libraries**: Optuna, CVXPY, Pandas, NumPy, Scikit-learn, XGBoost, PyYAML
-   **Data Sources**: Korea Exchange (KRX), DART, FRED Economic Data