# TimeFolio Portfolio System

A production-grade system for building and stress-testing Korean equity portfolios that satisfy the TimeFolio contest rules (≤ 15% per stock, sector cap = max(2 × market-weight, 10%), weekly turnover ≥ 5%).

## System Architecture

```
╔═══════════════════════════════════════════════════════╗
                  I. Data Ingestion                      
  ┌─────────────┐ ┌───────────────────┐ ┌───────────────┐ 
  │ KRX Fetcher │ │ Financial Fetcher │ │ Macro Fetcher │ 
  └──────┬──────┘ └─────────┬─────────┘ └───────┬───────┘ 
╚═══════════════════════════════════════════════════════╝
         │                  │                    │
         ▼                  ▼                    ▼
╔═══════════════════════════════════════════════════════╗
                  II. Data Manager                    
   - Centralized data access & caching               
   - Handles raw data storage (SQLite)               
   - Manages data validation & cleaning              
╚═══════════════════════════════════════════════════════╝
                           │
                           ▼
╔═══════════════════════════════════════════════════════╗
                  III. Factor Engine                   
  - Computes investment factors                 
  - Handles factor calculations                 
  - Manages factor persistence                  
╚═══════════════════════════════════════════════════════╝
                           │
                           ▼
╔═══════════════════════════════════════════════════════╗
                  IV. Portfolio Optimizer                 
  - Generates target portfolio allocations      
  - Implements optimization strategies          
  - Manages risk constraints                    
╚═══════════════════════════════════════════════════════╝
                           │
                           ▼
╔═══════════════════════════════════════════════════════╗
                  V. Risk Monitoring System                    
  - Executes trades based on target portfolio   
  - Handles order routing                       
  - Manages position tracking                   
╚═══════════════════════════════════════════════════════╝
                           │
                           ▼
╔═══════════════════════════════════════════════════════╗
                  VI. Portfolio Management                 
  - Executes trades based on target portfolio   
  - Handles order routing                       
  - Manages position tracking                   
╚═══════════════════════════════════════════════════════╝
```

## Pre-Contest Checklist

> **Note**: For the July 2025 contest, ensure all pre-contest checks are completed before the competition starts.

### 1-2 Weeks Before Contest
- [ ] Update universe files with latest stock listings
- [ ] Clear `forbidden.csv` to reset liquidity restrictions
- [ ] Run full backtest with recent market data
- [ ] Verify sector weight limits and position constraints

### 1-2 Days Before Start
- [ ] Run `backfill_krx_data.py --start-date 20220701 --end-date 20250711`
- [ ] Verify data integrity in SQLite database
- [ ] Execute dry-run with `--dry-run` flag

### Contest Day
- [ ] Review generated portfolio in `output/` directory
- [ ] Check risk metrics and compliance reports
- [ ] Manually verify top positions against latest news

## KRX Data FetcherM

### Key Features
- Fetches daily OHLCV data for all KOSPI and KOSDAQ stocks using their OTP-based download system
- Handles rate limiting and automatic retries
- Stores data in an efficient SQLite database
- Supports incremental updates and backfilling
- Includes comprehensive error handling and logging

### Backfilling Data
To backfill historical data, use the provided script:

```bash
# Backfill with default settings (weekly update)
python backfill_krx_data.py --start-date 20220701 --end-date 20250704
python backfill_krx_data.py --start-date 20220707 --end-date 20250711

# With custom delay settings
python backfill_krx_data.py --min-delay 3 --max-delay 7
```

### Database Schema
Data is stored in an SQLite database with the following schema:

```sql
CREATE TABLE daily_prices (
    code TEXT,
    date DATE,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    volume INTEGER,
    value INTEGER,
    market TEXT,
    name TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (code, date)
)
```

## Setup Preparations

### 1. Environment Setup
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies
```bash
# Core requirements (includes KRX fetcher dependencies)
pip install -r requirements-combined.txt

# Verify installation
python -c "import pandas as pd; print(f'Success! pandas version: {pd.__version__}')"
```

### 3. Initial Data Setup
```bash
# Create necessary directories
mkdir -p data/processed validated_universes output

# Run initial data update
python batch_update.py --start-date $(date -v-1y +%Y%m%d)

# Verify data
ls -lh data/processed/*.csv
```

### 4. Configuration
1. Create a symlink to the latest universe file:
   ```bash
   # Create a symlink that always points to the latest universe file
   cd validated_universes
   ln -sf $(ls -t sector_universe_validated_* | head -1) sector_universe_latest.csv
   ```

2. Configure `config.yaml` with relative dates:
   ```yaml
   data_settings:
     stock_universe_file: "validated_universes/sector_universe_latest.csv"
     lookback_years: 3  # Automatically calculates start date from end_date
     # end_date: YYYY-MM-DD  # Optional: defaults to today if not specified
     
   optimization:
     # backtest_split_date: YYYY-MM-DD  # Optional: defaults to end_date if not specified
     max_positions: 15
     position_limit: 0.15  # 15% per position
     sector_limit_multiplier: 2.0  # 2x market weight
     min_turnover: 0.05
     lambda_hhi: 10.0             # Penalty for weight HHI above threshold
   ```

3. Update market_sector.csv with new weights

### 5. Verify Setup
```bash
# Dry run
python competition_portfolio.py --config config.yaml --dry-run

# Check logs
tail -n 20 logs/optimization_*.log
```

### 6. Regular Updates
```bash
# Update market data (run weekly)
python batch_update.py --start-date $(date -v-1w +%Y%m%d)

# Retrain models (monthly)
python -m ml_forecast.train

# Generate new portfolio
python competition_portfolio.py --config config.yaml --positions 15 --output-dir output
```

### 7. Configure API Access (optional)
```bash
# Set KIS API credentials
export KIS_APP_KEY='your_key_here'
export KIS_APP_SECRET='your_secret_here'

# Test authentication
python kis_auth.py --status
python kis_auth.py --refresh
```

## Commands

### Basic Run
```bash
# Update market data
python batch_update.py --start-date 2025MMDD

# Run portfolio optimization
python competition_portfolio.py --config config.yaml --positions 15 --output-dir output
```

## Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--config` | Path to YAML config file | `config.yaml` |
| `--positions` | Target number of positions | 12 |
| `--use-ml-forecast` | Enable ML return predictions | `False` |
| `--tune-params` | Enable hyperparameter tuning | `False` |
| `--n-tuning-calls` | Tuning iterations | 20 |
| `--output-dir` | Output directory | `output` |
| `--min-data-points` | Minimum data points required | 63 |
| `--data-interval` | Data frequency | `1d` |

## Key Components

### 1. Data Manager (`data_manager.py`)
- **Data Loading & Validation**
  - Fetches market data from KIS API and other sources
  - Validates data completeness and quality
  - Implements retry logic with exponential backoff
  - Handles KRX maintenance periods gracefully
  - Caches data locally for performance

### 2. Enhanced Data Loader (`data_loader.py`)
- **Risk-Free Rate Management**
  - Fetches 3-month KTB yields from KIS API
  - Implements 1-week TTL caching
  - Fallback to cached values during KRX maintenance
  - Configurable update frequency

- **Data Processing**
  - Handles multiple data frequencies (daily/weekly/monthly)
  - Implements data validation and cleaning
  - Supports both KOSPI and KOSDAQ markets
  - Maintains data consistency during market holidays

- **Universe Management**
  - Multi-encoding support for universe files
  - Ticker validation against market data
  - Sector and market cap validation

### 3. Orchestration (`competition_portfolio.py`)
- Loads data, runs factor engine (PCA+PCR) & portfolio optimizer
- Executes risk monitoring & exports results

### 4. Portfolio Optimizer (`portfolio_optimizer.py`)
- **Core solver** maximizing wᵀ·r̂ − λ_risk·wᵀΣw − λ_HHI·∑w_i²
- Enforces position (≤15%), sector caps, and in-solver turnover (≥5%) constraints
- Mixed-integer programming for exact position counts with convex fallback
- Logs detailed metrics from `portfolio_metrics.py`

### 5. Portfolio Metrics (`portfolio_metrics.py`)
- **Optimization Framework**
  - Implements PCR-Sharpe ratio optimization
  - Uses Ledoit-Wolf shrinkage for covariance estimation
  - Enforces 15% position limits (40% for Samsung Electronics)
  - Implements sector constraints (2× market weight cap)
  - Enforces 5% minimum weekly turnover
  - Supports mixed-integer programming for exact position counts

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

### 8. Backtesting Framework (`backtest.py`)
- **Walk-Forward Validation**
  - Implements rolling window backtesting
  - Supports daily and weekly rebalancing
  - Tracks key performance metrics
  - Generates detailed performance reports

### 9. Factor Engine (`factor_engine.py`)
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

### 10. Compliance Filters (`compliance_filters.py`)
- **Stock Screening**
  - 3 billion KRW minimum 5-day average volume
  - 90-day minimum trading history
  - KRX caution/warning status monitoring
  - Automatic maintenance of restricted stock list



## Data Management

### Data Pipeline
- **Data Sources**
  - KIS API for market data
  - Local cache for offline use
  - Fallback to FinanceDataReader when needed

### Performance Features
- **Caching**
  - Disk-based caching of price data
  - In-memory caching for frequent queries
  - Automatic cache invalidation
  - Efficient data structures for large datasets

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
- **ML Forecasts**: XGBoost-based return predictions

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
  - Max 40% per sector (2× market weight)
  - HHI weight threshold: 0.08
  - HHI return threshold: 0.30

### Automated Monitoring
- **Daily Checks**:
  - Position-level MDD
  - Sector exposures
  - Liquidity constraints
  - Turnover compliance

- **Alerts**:
  - Email for critical issues
  - Slack for team notifications
  - Daily risk reports

## Troubleshooting

### Data Issues
- **KRX Maintenance**: Check KRX website for scheduled maintenance
- **API Limits**: Verify KIS API key usage and limits
- **Cache**: Clear cache with `rm -rf data/cache/*` if needed

### Optimization Issues
- **Infeasible Solution**: Check constraint parameters in config
- **Solver Errors**: Try different solvers (ECOS, SCS, OSQP)
- **Numerical Issues**: Scale returns or adjust solver settings

### Performance Issues
- **Slow Execution**: Reduce universe size or use weekly data
- **Memory Usage**: Process data in smaller chunks
- **Model Drift**: Retrain models with recent data

## Outputs
- Portfolio Weights: `output/current_portfolio.csv`
- Risk Report: `output/risk_report.md`
- Performance Metrics: `output/performance_metrics.json`
- Logs: `logs/optimization_*.log`