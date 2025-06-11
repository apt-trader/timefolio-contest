import itertools
import subprocess
import os
import shutil
import re
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Experiment parameter grid (focused on Enhanced tactical strategy with EliteAlpha)
TACTICAL_SOURCES = ['enhanced']
TACTICAL_ALLOCS  = [0.15, 0.20, 0.25, 0.30]
ELITE_FLAGS      = [True]

# Load sector limits for validation
def load_sector_limits(path='market_sectors.csv') -> dict:
    limits = {}
    in_section = False
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('[') and line.endswith(']'):
                in_section = not in_section
                continue
            if in_section:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        pct = float(parts[1]) / 100.0
                        limits[parts[0]] = min(2 * pct, 0.10)
                    except ValueError:
                        continue
    return limits


# Load sector map from universe file
def load_sector_map(path='sector_universe.csv') -> dict:
    # allow for stray spaces in header
    df = pd.read_csv(path, skipinitialspace=True)
    # normalize column names
    df.columns = df.columns.str.strip()
    # normalize ticker codes to 6 digits
    df['code'] = df['종목코드'].astype(str).str.zfill(6)
    # build map: code → sector
    return dict(zip(df['code'], df['섹터 코드']))



# Ensure results directory exists
os.makedirs('results', exist_ok=True)
results = []
sector_limits = load_sector_limits()
sector_map    = load_sector_map()

for source, alloc, elite in itertools.product(TACTICAL_SOURCES, TACTICAL_ALLOCS, ELITE_FLAGS):
    tag = f"{source}_{int(alloc*100)}_{'elite' if elite else 'noelite'}"
    output_dir = os.path.join('results', tag)
    os.makedirs(output_dir, exist_ok=True)

    # Build command
    cmd = [
        'python','competition_portfolio.py',
        '--config', 'config.yaml',
        '--positions','12',
        '--tactical-count','2',
        '--tactical-source', source,
        '--tactical-allocation', str(alloc),
        '--output-dir', output_dir
    ]
    if elite:
        cmd.append('--use-elite-alpha')

    logger.info(f"Running experiment: {tag}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    out = proc.stdout + proc.stderr
    status = 'ok' if proc.returncode == 0 else 'error'

    # Check output file
    port_path = os.path.join(output_dir, 'current_portfolio.csv')
    if not os.path.exists(port_path):
        status = 'missing_output'

    # Parse metrics
    sharpe = None
    m1 = re.search(r"Sharpe=([0-9.\-]+)", out)
    if m1:
        sharpe = float(m1.group(1))

    mc_cvar = None
    mc_var  = None
    m2 = re.search(r"MC CVaR stress mean=([0-9.\-]+)%.*, VaR=([0-9.\-]+)%", out)
    if m2:
        mc_cvar = float(m2.group(1)) / 100
        mc_var  = float(m2.group(2)) / 100

    # Validate constraints
    max_stock_w = None
    sector_violation = False
    if status == 'ok':
        dfw = pd.read_csv(port_path, index_col=0)
        max_stock_w = dfw['weight'].max()
        # check sector sums
        sect_sums = {}
        for code, w in dfw['weight'].items():
            sec = sector_map.get(code)
            if sec:
                sect_sums[sec] = sect_sums.get(sec, 0) + w
        for sec, total in sect_sums.items():
            if total > sector_limits.get(sec, 1):
                sector_violation = True
                break

    results.append({
        'tag': tag,
        'status': status,
        'tactical_source': source,
        'allocation': alloc,
        'use_elite_alpha': elite,
        'sharpe': sharpe,
        'MC_CVaR': mc_cvar,
        'MC_VaR': mc_var,
        'max_stock_weight': max_stock_w,
        'sector_violation': sector_violation
    })

# Save summary
df = pd.DataFrame(results)
df.to_csv('results/summary.csv', index=False)
logger.info('Experiment completed. Summary saved to results/summary.csv')

# Visualization of results
import matplotlib.pyplot as plt
import seaborn as sns

# Load summary
df_vis = pd.read_csv('results/summary.csv')

# 1. Sharpe vs Tactical Allocation
plt.figure(figsize=(8,5))
plt.plot(df_vis['allocation'], df_vis['sharpe'], marker='o')
plt.xlabel('Tactical Allocation')
plt.ylabel('Sharpe Ratio')
plt.title('Sharpe vs Tactical Allocation')
plt.grid(True)
plt.savefig('results/sharpe_vs_alloc.png')
plt.close()

# 2. MC CVaR Heatmap
# Pivot by allocation and elite flag
pivot = df_vis.pivot(index='allocation', columns='use_elite_alpha', values='MC_CVaR')
plt.figure(figsize=(6,4))
sns.heatmap(pivot, annot=True, fmt='.2f', cbar_kws={'label': 'MC CVaR'})
plt.title('MC CVaR Heatmap')
plt.savefig('results/mc_cvar_heatmap.png')
plt.close()
