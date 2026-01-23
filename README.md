# TimeFolio Portfolio System

A **signal-based portfolio optimization system** for the Korean equity market (KOSPI/KOSDAQ). Implements Mean-Variance Optimization at the signal level rather than individual stocks, achieving higher signal-to-noise ratio and more robust portfolios.

## Table of Contents

1. [Key Features](#key-features)
2. [System Architecture](#system-architecture)
3. [Quick Start](#quick-start)
4. [Configuration](#configuration)
5. [CLI Usage](#cli-usage)
6. [Technical Framework](#technical-framework)
7. [Backtest Results](#backtest-results)
8. [Outputs](#outputs)

---

## Key Features

### Signal-Based MVO (Core Innovation)
- **5 Orthogonal Signals**: Value, Quality, Momentum, Low Volatility, Growth
- **Signal-Level Optimization**: MVO on 5 signals instead of 200+ stocks
- **Higher SNR**: Signal portfolios have ~10x higher signal-to-noise ratio
- **Robust Covariance**: Ledoit-Wolf shrinkage for stable estimation

### DS002 Data Enhancement
- **Dividend Data**: Dividend yield integrated into Value signal (alotMatter API)
- **Buyback Data**: Treasury stock activity integrated into Quality signal (tesstkAcqsDspsSttus API)
- **Management Confidence**: Net buyback signals management's belief in undervaluation

### Toraniko Factor Model Integration (NEW)
- **Barra-Style Risk Model**: WLS regression for factor return estimation
- **Factor Covariance**: Ledoit-Wolf shrinkage on factor returns
- **Risk Attribution**: Systematic vs idiosyncratic risk decomposition
- **Enhanced Signals**: Exponential momentum, proper cross-sectional standardization

### Transaction Cost Awareness
- **Korean Market Costs**: 0.1% commission + 0.23% securities transaction tax
- **Turnover Control**: Max 15% one-way turnover per rebalance
- **Small Trade Filtering**: Minimum 1% trade threshold

### Regime Detection
- **Market Regimes**: Bull, Bear, High Volatility, Low Volatility, Neutral
- **Adaptive Weights**: Signal weights adjust based on detected regime
- **Smooth Transitions**: Gradual regime shifts to avoid whipsaws

### Compliance & Risk
- **Forbidden Tickers**: Automatic filtering via `forbidden.csv`
- **Sector Limits**: Dynamic sector concentration constraints
- **Market Cap Filter**: Min 100B KRW (시가총액 1,000억원 이상)
- **Liquidity Filter**: Min 3B KRW 5-day avg trading value (5일 평균 거래대금 30억원 이상)

### Signal Quality Monitoring (IC Monitor)
- **Information Coefficient**: Tracks IC = cor(signal/σ, forward_return/σ) for each signal
- **Rolling IC History**: 26-week rolling average for trend detection
- **Alert System**: Warns when IC drops below threshold (default: 0.02)
- **Persistent Storage**: IC history saved to `output/ic_history.json`

---

## System Architecture

```
/timefolio-contest
├── main_signal.py              # ★ NEW ENTRY POINT
├── signals/                    # ★ NEW SIGNAL-BASED MODULES
│   ├── signal_constructor.py   # Builds 5 orthogonal signals
│   ├── signal_optimizer.py     # Signal-level MVO + FactorEnhancedOptimizer
│   ├── portfolio_mapper.py     # Maps signals to stock weights
│   ├── signal_pipeline.py      # End-to-end orchestration
│   ├── factor_model.py         # ★ Toraniko: Barra-style factor model
│   ├── risk_attribution.py     # ★ Toraniko: Risk decomposition
│   ├── toraniko_signals.py     # ★ Toraniko: Enhanced signal construction
│   ├── polars_adapter.py       # ★ Toraniko: Pandas↔Polars conversion
│   ├── transaction_costs.py    # Transaction cost model
│   ├── regime_signal_weights.py # Regime detection & adjustment
│   └── ic_monitor.py           # Information Coefficient monitoring
│
├── config.py                   # Configuration loader
├── data_manager.py             # Data access layer
├── compliance_filters.py       # Forbidden ticker filtering
│
├── fetchers/                   # Data fetching modules
│   ├── krx_fetcher.py          # Market data (OHLCV, Market Cap)
│   ├── financial_fetcher.py    # Fundamental data (DART)
│   ├── dividend_fetcher.py     # ★ DS002: Dividend data (alotMatter)
│   ├── treasury_stock_fetcher.py # ★ DS002: Buyback data (tesstkAcqsDspsSttus)
│   └── macro_fetcher.py        # Macroeconomic data
│
├── config/
│   └── config.yaml             # Central configuration
│
├── db/
│   └── krx_data.db             # SQLite database
│
├── forbidden.csv               # Excluded tickers
├── sector_universe.csv         # Stock universe
└── market_sectors.csv          # Sector definitions
```

### Legacy Files (Preserved but Unused)
```
legacy/                         # Old stock-level MVO system
├── main.py                     # Old entry point
├── factor_engine.py            # Old factor model
├── optimizer.py                # Old stock-level optimizer
├── ml_ensemble_alpha.py        # ML ensemble (disabled)
├── robust_regression_models.py # Robust regression (disabled)
└── ...                         # Other legacy modules
```

---

## Quick Start

### Prerequisites
- Python 3.9+
- SQLite 3
- API keys for DART and FRED (in `.env` file)

### Installation

```bash
# Clone and setup
git clone https://github.com/your-org/timefolio-contest.git
cd timefolio-contest

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Setup API keys
cp .env.example .env
# Edit .env with your DART_API_KEY and FRED_API_KEY
```

### Generate Portfolio (Production)

```bash
# Generate live portfolio for this week
python main_signal.py --mode live
```

### Run Backtest

```bash
# Backtest 2024 performance
python main_signal.py --mode backtest --start 2024-01-01 --end 2024-12-31
```

---

## Configuration

### Date Ranges in `config.yaml`

There are **two separate date ranges** with different purposes:

```yaml
data_settings:
  start_date: '2020-01-01'   # Data loading range (need 2+ years for warmup)
  end_date: '2025-12-31'     # Latest available data

training_settings:           # LEGACY - only used by old main.py
  start_date: '2025-01-01'   # Not used by main_signal.py
  end_date: '2026-01-01'
```

| Setting | Purpose | Used By |
|---------|---------|---------|
| `data_settings.start_date` | How far back to load data (momentum needs 252 days, warmup needs 52 weeks) | `main_signal.py` |
| `data_settings.end_date` | Latest data to load | `main_signal.py` |
| `training_settings.*` | Legacy model training period | Old `main.py` only |

**Rule**: Set `data_settings.start_date` to at least 2 years before your backtest start date.

### Signal Pipeline Settings

```yaml
signal_settings:
  # Core optimization
  max_positions: 15
  risk_aversion: 0.27
  
  # Turnover control (OPTIMIZED)
  max_turnover: 0.15              # 15% max one-way turnover
  min_trade_threshold: 0.01       # 1% minimum trade size
  
  # Transaction costs
  commission_rate: 0.001          # 0.1%
  tax_rate: 0.0023                # 0.23% (sells only)
  
  # Regime detection
  use_regime_adjustment: true
  regime_adjustment_strength: 0.3
```

---

## CLI Usage

### Step 1: Update Database

```bash
# Fetch latest market data
rm cache/krx_cache/krx_cookies.pkl
python fetchers/krx_fetcher.py -s 20260112 -e 20260118 --update-db --no-headless

# Fetch macroeconomic data
python -m fetchers.macro_fetcher -s 2026-01-12 -e 2026-01-18

# Fetch financial statements
python -m fetchers.financial_fetcher --year 2025 --report-type 11011  # Annual
python -m fetchers.financial_fetcher --year 2025 --report-type 11012  # Q2
python -m fetchers.financial_fetcher --year 2025 --report-type 11013  # Q1
python -m fetchers.financial_fetcher --year 2025 --report-type 11014  # Q3

# Fetch DS002 data (dividend and treasury stock)
python -m fetchers.dividend_fetcher --year 2024 --report-type 11011      # Dividend data
python -m fetchers.treasury_stock_fetcher --year 2024 --report-type 11011  # Buyback data
```

### Step 2: Update Compliance Files

```bash
# Edit forbidden.csv - add tickers with caution/warning status
# Edit market_sectors.csv - update sector limits if needed
```

### Step 3: Generate Portfolio

```bash
# Live mode - generates portfolio for current week
python main_signal.py --mode live

# Backtest mode - test historical performance
python main_signal.py --mode backtest --start 2024-01-01 --end 2024-06-30
```

---

## Technical Framework

### Signal Construction

| Signal | Components | Description |
|--------|------------|-------------|
| **Value** | B/P, E/P, S/P, C/P, **DivYield** | Composite value score (includes dividend yield) |
| **Quality** | ROE, Low Leverage, Earnings Quality, **Buyback** | Fundamental quality (includes treasury stock) |
| **Momentum** | 12-1 month return | Price momentum (skip recent month) |
| **Low Volatility** | Inverse realized volatility | Defensive signal |
| **Growth** | Revenue/Earnings growth (or ROE proxy) | Growth characteristics |

### Signal-Level MVO

```
Objective: max  μ'w - λ·w'Σw

Where:
  μ = Expected signal returns (shrinkage estimator)
  Σ = Signal covariance (Ledoit-Wolf shrinkage)
  λ = Risk aversion (default: 0.27)
  w = Signal weights (sum to 1, each ≥ 0)
```

### Portfolio Mapping

1. **Compute composite score**: `score_i = Σ(signal_weight_j × signal_score_ij)`
2. **Select top stocks**: Top 15 by composite score
3. **Apply constraints**: Max 15% per stock, sector limits
4. **Filter forbidden**: Remove tickers in `forbidden.csv`

### Regime Detection

| Regime | Momentum | LowVol | Value | Quality |
|--------|----------|--------|-------|---------|
| Bull | +30% | -30% | -20% | -10% |
| Bear | -40% | +40% | +20% | +30% |
| High Vol | -50% | +50% | 0% | +20% |
| Low Vol | +10% | -10% | 0% | 0% |

---

## Pipeline Flow

### 1. Signal Construction (`SignalConstructor`)
- Calculates 5 orthogonal signals from raw data
- Applies winsorization and robust standardization
- Outputs z-scores for each stock per signal

### 2. Signal Return Estimation (`SignalReturnEstimator`)
- Forms long-only portfolios for each signal (top quintile)
- Tracks historical signal portfolio returns
- Requires ~26 weeks warmup for reliable estimation

### 3. Signal Optimization (`SignalOptimizer`)
- Estimates signal expected returns (with shrinkage)
- Estimates signal covariance (Ledoit-Wolf)
- Performs MVO on 5x5 covariance matrix
- Outputs optimal signal weights

### 4. Portfolio Mapping (`PortfolioMapper`)
- Calculates composite score per stock
- Selects top stocks by composite score
- Applies cardinality constraint (max 15)
- Applies sector constraints

### 5. Turnover Management (`TurnoverManager`)
- Limits one-way turnover to 15%
- Smooths transitions between rebalances

---

## Backtest Results

### 2024 H1 Performance

| Phase | Total Return | Ann. Return | Volatility | Sharpe | Costs |
|-------|--------------|-------------|------------|--------|-------|
| Phase 1 (Signal MVO) | +5.16% | +11.02% | 9.82% | **1.12** | N/A |
| Phase 2 (+ Costs & Regime) | +5.07% | +10.84% | 9.95% | **1.09** | 149.5 bps |

### Comparison with Old System

| Aspect | Old System | New System |
|--------|-----------|------------|
| Optimization level | Stock (200+) | Signal (5) |
| Covariance matrix | 200x200 (unstable) | 5x5 (stable) |
| SNR | ~0.05 | ~0.5-1.0 |
| Model complexity | ML ensemble | Simple MVO |
| Interpretability | Low | High |
| Turnover | Uncontrolled | Managed |

---

## Outputs

### From `main_signal.py --mode live`
- `output/portfolio_YYYYMMDD.csv` - Final portfolio weights

### From `main_signal.py --mode backtest`
- Console output with performance metrics
- Log file in `logs/`

---

## License

MIT License

**Libraries**: CVXPY, Pandas, NumPy, Scikit-learn, PyYAML, yfinance, fredapi, OpenDartReader

**Data Sources**: Korea Exchange (KRX), DART, FRED
