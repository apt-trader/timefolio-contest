# TimeFolio Portfolio System

A modular system for building and managing Korean equity portfolios that comply with TimeFolio contest rules. It features a six-stage workflow from data ingestion to portfolio execution, with built-in risk management and compliance controls.

## Table of Contents

1. [System Architecture](#system-architecture)  
2. [Prerequisites](#prerequisites)  
3. [Installation](#installation)  
4. [Configuration](#configuration)  
5. [Workflow Components](#workflow-components)  
6. [CLI Usage Examples](#cli-usage-examples)  
7. [License & Acknowledgements](#license--acknowledgements)

## System Architecture

```
╔═══════════════════════════════════════════════════════╗
                  I. Data Ingestion                      
  ┌─────────────┐ ┌───────────────────┐ ┌───────────────┐ 
  │ KRX Fetcher │ │ Financial Fetcher │ │ Macro Fetcher │ 
  └──────┬──────┘ └─────────┬─────────┘ └───────┬───────┘ 
╚═══════════════════════════════════════════════════════╝
         │                  │                   │
         ▼                  ▼                   ▼
╔═══════════════════════════════════════════════════════╗
                  II. Data Manager                    
   - Centralized data access & caching               
   - Handles raw data storage (SQLite)               
   - Manages data validation & cleaning              
   - Implements request throttling and retry logic   
   - Maintains data consistency during market holidays
╚═══════════════════════════════════════════════════════╝
                           │
                           ▼
╔═══════════════════════════════════════════════════════╗
                  III. Factor Engine                   
  - Computes investment factors (Momentum, Mean Reversion)  
  - Handles factor calculations and standardization  
  - Manages factor persistence and normalization     
  - Implements volatility-adjusted calculations     
  - Generates composite factor scores               
╚═══════════════════════════════════════════════════════╝
                           │
                           ▼
╔═══════════════════════════════════════════════════════╗
                  IV. Portfolio Optimizer                 
  - Generates target portfolio allocations      
  - Implements optimization strategies (CVXPY)  
  - Manages risk constraints and position limits
  - Handles sector exposure and turnover constraints
  - Implements L2 regularization for diversification    
╚═══════════════════════════════════════════════════════╝
                           │
                           ▼
╔═══════════════════════════════════════════════════════╗
                  V. Risk Monitoring System              
  - Tracks portfolio risk metrics (MDD, HHI, etc.)    
  - Implements real-time alerts via Slack/Email        
  - Generates comprehensive risk reports               
  - Enforces compliance with trading rules             
  - Monitors position concentration and drawdowns      
╚═══════════════════════════════════════════════════════╝
                           │
                           ▼
╔═══════════════════════════════════════════════════════╗
                  VI. Portfolio Execution                 
  - Generates trade lists and order routing           
  - Handles implementation shortfall optimization     
  - Manages transaction cost modeling                 
  - Trades execution and position tracking            
  - Updates portfolio performance metrics             
╚═══════════════════════════════════════════════════════╝
```
The system follows a unidirectional data flow where each component processes data and passes it to the next stage, with feedback loops for performance analysis and optimization.

## Prerequisites

- Python 3.8+
- SQLite 3
- UNIX-style shell or Windows PowerShell
- Valid API keys:
  - **DART** (`DART_API_KEY`)
  - **FRED** (`FRED_API_KEY`)
- Required Python packages (see `requirements.txt`)

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-org/timefolio-2025.git
cd timefolio-2025

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

---

## Configuration

The system is configured via `config.yaml` in the root directory. Key sections include:

```yaml
data_settings:
  stock_universe_file: "validated_universes/sector_universe_latest.csv"
  market_sectors_file: "market_sectors.csv"
  start_date: '2022-01-01'
  end_date: '2025-09-30'
  cache_dir: 'cache'
  output_dir: 'out'

optimization:
  individual_limit: 0.15     # max 15% per stock
  risk_aversion: 1.0         # λ in μᵀw - λ·wᵀΣw - η·∥w∥₂²
  l2_penalty: 0.1            # η for concentration control
  max_positions: 12
  rebalance_step_days: 5

risk_management:
  mdd_threshold_6m: 0.15
  hhi_weight_threshold: 0.20
  # Additional risk parameters...

data_fetch:
  max_attempts: 3
  retry_delay: 5
```

---

## Workflow Components

### I. Data Ingestion

#### 1. Market Data (KRX)
- **Module**: `krx_fetcher.py`
- **Features**:
  - Fetches historical OHLCV data from KRX
  - Implements OTP-based authentication
  - Handles rate limiting with exponential backoff
  - Caches responses locally to minimize API calls
  - Computes technical indicators (SMA, RSI, etc.)
  - Stores data in `daily_prices` table

#### 2. Financial Data (DART)
- **Module**: `financial_fetcher.py`
- **Features**:
  - Interfaces with OpenDartReader API
  - Handles rate limiting and daily quotas
  - Parses XBRL financial statements
  - Computes fundamental ratios (P/E, P/B, etc.)
  - Updates `financials` table with standardized metrics

#### 3. Macroeconomic Data
- **Module**: `macro_fetcher.py`
- **Features**:
  - Fetches economic indicators from FRED/Yahoo
  - Tracks yield curves and credit spreads
  - Normalizes time series data
  - Updates `macro_data` table
  - Implements data quality checks

### II. Data Manager
- **Module**: `data_manager.py`
- **Key Functions**:
  - `load_sector_codes()`: Loads universe of stocks
  - `get_latest_close()`: Retrieves most recent prices
  - `create_price_df()`: Standardizes price data format
- **Features**:
  - Centralized data access layer
  - Handles data validation and cleaning
  - Implements caching for performance

### III. Factor Engine
- **Module**: `factor_engine.py`
- **Factor Types**:
  1. **Momentum** (20/60/120-day lookback)
  2. **Mean Reversion** (RSI + Bollinger Bands)
  3. **Liquidity** (5-day average volume)
- **Features**:
  - Volatility-adjusted factor calculations
  - Z-score normalization
  - Composite score generation

### IV. Portfolio Optimizer
- **Module**: `portfolio_optimizer.py`
- **Optimization Problem**:
  ```
  max_w μᵀw - λ·wᵀΣw - η·∥w∥₂²
  s.t. Σw = 1, w ≥ 0
  ```
- **Constraints**:
  - Individual position limits (≤ 15%)
  - Sector exposure limits
  - Turnover constraints
- **Implementation**:
  - Uses CVXPY for convex optimization
  - Implements L2 regularization for diversification
  - Handles cardinality constraints

### V. Risk Monitoring
- **Module**: `risk_monitor.py`
- **Metrics Tracked**:
  - Maximum Drawdown (6M/12M)
  - Portfolio Concentration (HHI)
  - Turnover
  - Tracking Error
- **Features**:
  - Real-time alerts via Slack/Email
  - Automated reporting
  - Threshold-based notifications

### VI. Portfolio Execution
- **Module**: `execution.py`
- **Features**:
  - Trade list generation
  - Implementation shortfall optimization
  - Transaction cost modeling
  - Broker integration

---

## CLI Usage

### 1. Update Market Data (Run at the start of each week)
   ```bash
   # Backfill any missing market data (only needed if there were holidays/errors)
   python backfill_krx_data.py --start-date $(date -v-7d "+%Y%m%d") --end-date $(date "+%Y%m%d")
   
   # Fetch fresh market data (past 30 days)
   python krx_fetcher.py -s $(date -v-30d "+%Y%m%d") -e $(date "+%Y%m%d")
   
   # Update financial statements (past year)
   python financial_fetcher.py -s $(date -v-1y "+%Y%m%d") -e $(date "+%Y%m%d")
   
   # Update macroeconomic data (past year)
   python macro_fetcher.py -s $(date -v-1y "+%Y%m%d") -e $(date "+%Y%m%d")
   ```

### 2. Generate Portfolio (After data updates)
   ```bash
   python competition_portfolio.py \
     --config config.yaml \
     --positions 12 \
     --out-dir output/
   ```

### 3. Monitor Risk (Run daily/continuously)
  ```bash
   python risk_monitor.py --config config/risk_config.yaml
   ```

### 4. Backtest Portfolio (Additional command)
  
  ```bash
  python backtest_portfolio.py --start-date 20240101 --end-date 20241231
  ```

### 5. Update Universe (Additional command)
  
  ```bash
  python update_universe.py --market KOSPI --min-cap 100000000000  # 100B KRW
  ```

**Notes**
- Use `nohup` or `tmux` for long-running processes
- Check `logs/` directory for execution logs
- Set up alerts for any failures in the workflow

### 6. Factor Engine (`factor_engine.py`)
Implements a multi-factor model for alpha generation.

**Key Features:**
- **Momentum Factors**
  - 20/60/120-day price momentum
  - Volatility-adjusted returns
  - Cross-sectional normalization

- **Mean Reversion**
  - RSI (14-day default)
  - Price deviation from moving averages
  - Short-term reversal signals

### 7. Risk Monitoring System (`risk_monitor.py`)
- **Risk Metrics**
  - Tracks 6M and 12M rolling maximum drawdown
  - Monitors HHI concentration (weight and return)
  - Enforces 5% minimum weekly turnover
  - Implements position-level MDD triggers

- **Alert System**
  - Configurable alert thresholds
  - Email notifications for critical issues
  - Slack integration for team notifications
  - Daily risk reports

### 8. Compliance Filters (`compliance_filters.py`)
- **Stock Screening**
  - 3 billion KRW minimum 5-day average volume
  - 90-day minimum trading history
  - KRX caution/warning status monitoring
  - Automatic maintenance of restricted stock list

---

## Technical Framework

### Portfolio Optimizer
```
Objective:  max  PCR-Sharpe(w)
             - λ₁·max(0, HHI(w) - τ₁)
             - λ₂·max(0, ReturnHHI(w) - τ₂)
Where:
  - PCR-Sharpe: Principal Component Risk-adjusted Sharpe ratio
  - HHI: Herfindahl-Hirschman Index for weight concentration
  - ReturnHHI: Herfindahl-Hirschman Index for return contributions
  - λ₁, λ₂: Penalty weights for concentration metrics (configured via lambda_hhi, lambda_return_hhi)
Constraints:
  ∑w = 1
  0 ≤ w_i ≤ 0.15
  sector_sum_s ≤ max(2 × market_weight_s, 0.10)
  ||w − w_prev||₁ ≥ 0.05   # Weekly turnover constraint enforced in-solver
  liquidity_i ≥ 3B KRW  (5-day ADTV)
```

### Risk Management
- **Position Limits**: 15% per stock (40% for Samsung Electronics)
- **Sector Caps**: 2× market weight per sector
- **Liquidity**: Minimum 3B KRW 5-day average volume
- **Turnover**: Minimum 5% weekly turnover
- **Concentration**: HHI monitoring for weights and returns
- **Drawdowns**: 6M and 12M rolling MDD tracking

### Alpha Generation
- **Momentum Factors**: 20/60/120-day price momentum
- **Mean Reversion**: RSI and price deviation signals

---

## Risk Management Protocols

### Position Monitoring
- **Drawdown Triggers**:
  - -5%: Review position thesis
  - -8%: Reduce position by 30-50%
  - -10%: Reduce to ≤5% weight
  - -15%: Full exit

### Portfolio Risk Controls
- **Concentration Limits**:
  - Max 15% per position (40% for Samsung Electronics)
  - Max 2× market weight or 10%
  - HHI weight threshold: 0.08
  - HHI return threshold: 0.30

### Automated Monitoring
- **Daily Checks**:
  - Position-level MDD
  - Sector exposures
  - Liquidity constraints
  - Turnover compliance

---

## Outputs
- Portfolio Weights: `output/current_portfolio.csv`
- Risk Report: `output/risk_report.md`
- Performance Metrics: `output/performance_metrics.json`
- Logs: `logs/optimization_*.log`

---

## License & Acknowledgements

- **License**: MIT
- **Data Sources**:
  - Korea Exchange (KRX)
  - DART (Data Analysis, Retrieval and Transfer System)
  - FRED Economic Data
- **Libraries**:
  - Pandas, NumPy, SciPy
  - CVXPY
  - scikit-learn
  - SQLAlchemy

For support or to report issues, please open an issue on our GitHub repository.