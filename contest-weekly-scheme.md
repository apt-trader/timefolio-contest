# TimeFolio Weekly Operational Scheme

## Pre-Flight Checklist

- [ ] `config/config.yaml`: Ensure `data_settings.end_date` covers latest data
- [ ] `market_sectors.csv`: Update sector limits if needed
- [ ] `forbidden.csv`: Check KIND for caution/warning/risk tickers

---

## Step 1: Update Database

```bash
# Fetch latest market data (adjust dates as needed)
python fetchers/krx_fetcher.py -s 20250106 -e 20250111 --update-db

# Fetch macroeconomic data
python -m fetchers.macro_fetcher -s 2025-01-06 -e 2025-01-11
```

---

## Step 2: Generate Portfolio

```bash
# Run signal-based portfolio generation
python main_signal.py --mode live
```

---

## Step 3: Review & Submit

- [ ] Review `output/portfolio_YYYYMMDD.csv`
- [ ] Verify no forbidden tickers in output
- [ ] Check sector concentration
- [ ] Submit portfolio to contest platform
- [ ] Archive output for the week

---

## Data Collection Log

| Period | Status |
|--------|--------|
| 2025-01-06 to 2025-01-11 | [ ] Pending |

---

## Notes

### Signal-Based System (NEW)
- Entry point: `main_signal.py` (not `main.py`)
- Output: `output/portfolio_YYYYMMDD.csv`
- No need to modify `training_settings` - the new system uses CLI args

### Key Config Parameters
```yaml
signal_settings:
  max_positions: 15
  max_turnover: 0.15        # 15% max turnover per rebalance
  use_regime_adjustment: true
```

### Transaction Costs (Built-in)
- Commission: 0.1% per trade
- Tax: 0.23% on sells only
- Round-trip: ~0.43%