# TimeFolio Portfolio System

A production-grade system for building and stress-testing Korean equity portfolios that satisfy the TimeFolio contest rules (≤ 15% per stock, sector cap = max(2 × market-weight, 10%), weekly turnover ≥ 5%).

## System Architecture

```
╔═══════════════════════════════════╦═══════════════════════════════════╗
   I. DATA HUB                      ⇣                                    
   • 3-M KTB fetch & cache         ⇠  Multi-source prices / fundamentals
   • Retry-throttle pipeline       ⇢  Disk-cache & validation           
╚═══════════════════════════════════╩═══════════════════════════════════╝
                     │
                     ▼
╔═══════════════════════════════════════════════════════════════════════╗
   II. FACTOR & ML LAB                                                   
   ── Momentum 20/60/120d ───┐                                           
   ── Mean-reversion (RSI,%B)│──►  PCA + Ledoit Σ  ──► Feature store    
   ── XGBoost / LSTM         ┘                                           
╚═══════════════════════════════════════════════════════════════════════╝
                     │
                     ▼
╔═══════════════════════════════════════════════════════════════════════╗
   III. PORTFOLIO ENGINE                                                  
   max PCR-Sharpe − λ₁·HHI − λ₂·ReturnHHI                              
   • Mixed-Integer solver (N=12)                                       
   • Position ≤15 %, Sector ≤max(2×mkt,10%)                           
   • L1/L2 sparsity, Turnover ≥5 %                                     
╚═══════════════════════════════════════════════════════════════════════╝
                     │
                     ▼
╔═══════════════════════════════════════════════════════════════════════╗
   IV. RISK & COMPLIANCE                                                 
   • Liquidity >3 bn KRW ADTV  • Micro-cap ≤40 %                        
   • Diversification-Ratio ≥2  • Rolling MDD guardrails                
   • Real-time breach alerts (Slack / email)                            
╚═══════════════════════════════════════════════════════════════════════╝
                     │
                     ▼
╔═══════════════════════════════════════════════════════════════════════╗
   V. OUTPUT & MONITOR                                                  
   • current_portfolio.csv  • trade blotter                            
   • HTML dashboards (risk, P&L, attribution)                          
   • Model artefacts + config hash for audit                           
╚═══════════════════════════════════════════════════════════════════════╝
```

## Pre-Contest Checklist

> **Note**: For the July 2025 contest, ensure all pre-contest checks are completed before the competition starts.

### 1-2 Weeks Before Contest
- [ ] Verify all API credentials are valid (KIS, etc.)
- [ ] Update universe files with latest stock listings
- [ ] Clear `forbidden.csv` to reset liquidity restrictions
- [ ] Run full backtest with recent market data
- [ ] Verify sector weight limits and position constraints

### 1-2 Days Before Start
- [ ] Run `batch_update.py` with `--start-date` to refresh all data
- [ ] Execute dry-run with `--dry-run` flag
- [ ] Verify output directory has proper write permissions
- [ ] Confirm backup system is working

### Contest Day
- [ ] Run optimization with `--use-ml-forecast` if enabled
- [ ] Review generated portfolio in `output/` directory
- [ ] Check risk metrics and compliance reports
- [ ] Manually verify top positions against latest news

## KRX Data Fetcher

A robust module for fetching historical market data from the Korea Exchange (KRX) using their OTP-based download system. This module is designed to be resilient against rate limiting and provides a clean interface for both one-time backfills and incremental updates.

### Key Features

- Fetches daily OHLCV data for all KOSPI and KOSDAQ stocks
- Handles rate limiting and automatic retries
- Stores data in an efficient SQLite database
- Supports incremental updates and backfilling
- Includes comprehensive error handling and logging

### Quick Start

```python
from krx_fetcher import KRXDataFetcher

# Initialize with default settings
fetcher = KRXDataFetcher()

# Fetch data for a specific date
results = fetcher.fetch_daily_data('20250530')

# Get historical data for a stock
df = fetcher.get_stock_data('005930', start_date='20220101', end_date='20221231')

# Backfill historical data (see backfill_krx_data.py for full options)
# python backfill_krx_data.py --start-date 20220101 --end-date 20250530
```

### Backfilling Data

To backfill historical data, use the provided script:

```bash
# Backfill with default settings (2-5 second delay between requests)
python backfill_krx_data.py --start-date 20220101 --end-date 20250530

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

### Error Handling

- Automatic retries for failed requests
- Rate limiting to avoid IP bans
- Detailed logging to `krx_fetcher.log`
- Graceful handling of weekends and holidays

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

# For ML features (optional)
pip install -r requirements-ml.txt

# Verify installation
python -c "import pandas as pd; print(f'Success! pandas version: {pd.__version__}')"
```

### 3. Configure API Access
```bash
# Set KIS API credentials
export KIS_APP_KEY='your_key_here'
export KIS_APP_SECRET='your_secret_here'

# Test authentication
python kis_auth.py --status
python kis_auth.py --refresh
```

### 4. Initial Data Setup
```bash
# Create necessary directories
mkdir -p data/processed validated_universes output

# Run initial data update
python batch_update.py --start-date $(date -v-1y +%Y%m%d)

# Verify data
ls -lh data/processed/*.csv
```

### 5. Configuration
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
   ```

3. Update market_sector.csv with new weights

### 6. Verify Setup
```bash
# Dry run
python competition_portfolio.py --config config.yaml --dry-run

# Check logs
tail -n 20 logs/optimization_*.log
```

### 7. Regular Updates
```bash
# Update market data (run weekly)
python batch_update.py --start-date $(date -v-1w +%Y%m%d)

# Retrain models (monthly)
python -m ml_forecast.train

# Generate new portfolio
python competition_portfolio.py --config config.yaml --positions 15 --output-dir output
```

## Commands

### Basic Run
```bash
# Update market data
python batch_update.py --start-date 2025MMDD

# Run portfolio optimization
python competition_portfolio.py --config config.yaml --positions 15 --output-dir output
```

### With ML Forecasting
```bash
python competition_portfolio.py --config config.yaml --positions 15 --use-ml-forecast --output-dir output
```

### With Parameter Tuning
```bash
python competition_portfolio.py --config config.yaml --positions 15 --use-ml-forecast --tune-params --n-tuning-calls 20 --output-dir output
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

### 3. Portfolio Optimizer (`competition_portfolio.py`)
- **Main Orchestrator**
  - Coordinates data loading, optimization, and risk management
  - Handles command-line arguments and configuration
  - Manages output generation and reporting

### 4. Portfolio Metrics (`portfolio_metrics.py`)
- **Optimization Framework**
  - Implements PCR-Sharpe ratio optimization
  - Uses Ledoit-Wolf shrinkage for covariance estimation
  - Enforces 15% position limits (40% for Samsung Electronics)
  - Implements sector constraints (2× market weight cap)
  - Enforces 5% minimum weekly turnover
  - Supports mixed-integer programming for exact position counts

### 4. ML Return Forecaster (`ml_forecast.py`)
- **XGBoost-based return prediction**
  - Implements gradient boosting for return forecasting
  - Feature importance analysis
  - Model versioning and persistence
  - Automated hyperparameter tuning

### 5. Risk Monitoring System (`risk_monitor.py`)
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

### 6. Backtesting Framework (`backtest.py`)
- **Walk-Forward Validation**
  - Implements rolling window backtesting
  - Supports daily and weekly rebalancing
  - Tracks key performance metrics
  - Generates detailed performance reports

### 7. Factor Engine (`factor_engine.py`)
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

### 8. Compliance Filters (`compliance_filters.py`)
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
Objective:  max  PCR-Sharpe(w) - λ₁·HHI(w) - λ₂·ReturnHHI(w)

Where:
- PCR-Sharpe: Principal Component Risk-adjusted Sharpe ratio
- HHI: Herfindahl-Hirschman Index for concentration risk
- ReturnHHI: Concentration of return contributions
- λ₁, λ₂: Penalty weights for concentration metrics

Constraints:
  ∑w = 1
  0 ≤ w_i ≤ 0.15   (individual position limit, 0.25 for Samsung Electronics)
  sector_sum_s ≤ 2 × market_weight_s  (sector caps)
  ||w − w_prev||₁ ≥ 0.05  (minimum 5% weekly turnover)
  liquidity_i ≥ 3B KRW (5-day average)
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