#!/usr/bin/env python3
import os
import yaml
import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from retrying import retry
import numpy as np
import pandas as pd
import cvxpy as cp
import FinanceDataReader as fdr
from scipy.stats.mstats import winsorize

def find_latest_validation_file(directory: str = "validated_universes", 
                              pattern: str = "sector_universe_validated_\d{8}_\d{6}\.csv") -> Optional[str]:
    """
    Find the most recent validation file matching the pattern in the specified directory.
    
    Args:
        directory: Directory to search in
        pattern: Regex pattern to match filenames
        
    Returns:
        Path to the most recent matching file, or None if no files found
    """
    try:
        # Ensure directory exists
        dir_path = Path(directory)
        if not dir_path.exists():
            global_logger.warning(f"Validation directory not found: {directory}")
            return None
            
        # Find all matching files
        files = [f for f in dir_path.glob("*") if re.match(pattern, f.name)]
        
        if not files:
            global_logger.warning(f"No validation files found matching pattern: {pattern}")
            return None
            
        # Sort by modification time (newest first)
        latest_file = max(files, key=os.path.getmtime)
        global_logger.info(f"Using validation file: {latest_file}")
        return str(latest_file)
        
    except Exception as e:
        global_logger.error(f"Error finding latest validation file: {str(e)}")
        return None

 # ── 로깅 설정 ─────────────────────────────────────────────────────────────
global_logger = logging.getLogger("portfolio_optimizer")
global_logger.setLevel(logging.INFO)
if not global_logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    global_logger.addHandler(ch)

def alert(msg: str):
    url = os.getenv("SLACK_WEBHOOK")
    if url:
        try:
            import requests
            requests.post(url, json={"text": f"[portfolio_optimizer] {msg}"}, timeout=3)
        except Exception as e:
            global_logger.warning(f"Slack alert failed: {e}")

class Config:
    def __init__(self, path: str):
        with open(path) as f:
            root = yaml.safe_load(f)
        ds = root.get('data_settings', {})
        
        # Get the base universe file from config
        universe_file = ds.get('stock_universe_file', 'sector_universe.csv')
        
        # If it's a validation file pattern, try to find the latest one
        if 'sector_universe_validated_' in universe_file:
            latest_file = find_latest_validation_file()
            if latest_file:
                self.stock_universe_file = latest_file
            else:
                self.stock_universe_file = universe_file
                global_logger.warning(f"Using configured universe file: {universe_file}")
        else:
            self.stock_universe_file = universe_file
            
        self.market_sectors_file = ds.get('market_sectors_file', 'market_sectors.csv')
        self.cache_dir = ds.get('cache_dir', 'cache')
        self.output_dir = ds.get('output_dir', 'out')
        self.start_date = ds.get('start_date')
        self.end_date = ds.get('end_date')
        
        # 최적화 파라미터 
        opt = root.get('optimization', {})
        # self.cvar_alpha            = opt.get('cvar_alpha', 0.05)  # Removed CVaR parameter
        
        # individual max per‐stock weight from config.optimization.individual_limit
        self.individual_limit      = opt.get('individual_limit', 0.15)
        self.risk_free_rate        = opt.get('risk_free_rate', 0.02)

        # 전술 전략 파라미터 
        tac = root.get('tactical', {})
        self.min_tactical_alloc    = tac.get('min_tactical_alloc', 0.20)
        self.max_tactical_alloc    = tac.get('max_tactical_alloc', 0.20)
        self.use_elite_alpha       = tac.get('use_elite_alpha', False)
        self.elite_alpha_blend     = tac.get('elite_alpha_blend', 0.30)
        
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        global_logger.info(f"Config loaded from {path}")

class DataManager:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        df = pd.read_csv(cfg.stock_universe_file, dtype=str)
        df.columns = df.columns.str.strip()
        if 'ticker' in df.columns:
            raw = df['ticker']
        elif '종목코드' in df.columns:
            raw = df['종목코드']
        else:
            raise KeyError(f"No ticker column in {cfg.stock_universe_file}")
        codes = raw.astype(str).str.strip().str.lstrip('A').str.zfill(6)
        codes = codes[codes.str.match(r'^\d{6}$')]
        self.tickers = codes.tolist()
        global_logger.info(f"Loaded {len(self.tickers)} universe tickers")
        self.sector_limits = self._load_sector_limits()

    def _load_sector_limits(self) -> dict:
        limits = {}
        in_section = False
        with open(self.cfg.market_sectors_file, 'r') as f:
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
        global_logger.info(f"Loaded sector limits for {len(limits)} sectors")
        return limits

    @retry(stop_max_attempt_number=3, wait_fixed=5000)
    def fetch_stock_prices(self, ticker, start, end):
        clean = str(ticker).strip().lstrip('A').zfill(6)
        
        # First try KIS API if available
        try:
            from data_manager import get_stock_data
            df = get_stock_data(clean, start_date=start)
            if df is not None and not df.empty:
                # Filter for the requested date range
                mask = (df.index >= pd.to_datetime(start)) & (df.index <= pd.to_datetime(end))
                df = df.loc[mask]
                if not df.empty:
                    global_logger.info(f"Successfully fetched {clean} from KIS API")
                    return df["close"]
        except ImportError:
            global_logger.warning("KIS API not available, falling back to FinanceDataReader")
        except Exception as e:
            global_logger.warning(f"[{clean}] KIS API failed: {str(e)}")
        
        # Fallback to FinanceDataReader
        try:
            df = fdr.DataReader(clean, start, end)
            if df is not None and not df.empty:
                global_logger.info(f"Fetched {clean} from FinanceDataReader")
                return df["Close"]
        except Exception as e:
            global_logger.warning(f"[{clean}] FinanceDataReader failed: {e}")
            
        # Final fallback to Yahoo Finance
        try:
            from pandas_datareader import data as pdr
            df2 = pdr.DataReader(clean + ".KS", "yahoo", start, end)
            if df2 is not None and not df2.empty:
                global_logger.info(f"Fetched {clean} from Yahoo Finance")
                return df2["Close"]
        except Exception as e2:
            global_logger.warning(f"[{clean}] Yahoo Finance fallback failed: {e2}")
            
        global_logger.error(f"All data sources failed for {clean}")
        return None

    def get_returns(self, start: str, end: str) -> pd.DataFrame:
        prices = {}
        for ticker in self.tickers:
            try:
                df = fdr.DataReader(ticker, start, end)
                prices[ticker] = df['Close'].sort_index()
            except Exception as e:
                global_logger.warning(f"[{ticker}] price fetch failed: {e}")

        # Build price frame and fill missing values in both directions
        price_df = pd.DataFrame(prices).sort_index()
        price_df = price_df.ffill().bfill()

        # Compute returns, dropping *only* rows where all tickers are NaN
        rets = price_df.pct_change().dropna(how='all')

        # Winsorize outliers and return
        arr = winsorize(rets.values, limits=[0.01, 0.01])
        return pd.DataFrame(arr, index=rets.index, columns=rets.columns)

class PortfolioOptimizer:
    """
    Portfolio optimizer maximizing PCR-Sharpe ratio with HHI & Return-HHI penalties
    enforces position, sector, and turnover constraints in-solver.
    """
    def __init__(self, cfg: Config, data: DataManager, beta_vec: pd.Series):
        self.cfg = cfg
        self.data = data
        # pull in the sector limits loaded by DataManager
        self.sector_limits = data.sector_limits
        self.beta_vec = beta_vec

    def calculate_risk_contributions(self, w: np.ndarray, cov: np.ndarray) -> np.ndarray:
        """
        Calculate risk contributions for each asset.
        
        Args:
            w: Weight vector
            cov: Covariance matrix
            
        Returns:
            Vector of risk contributions
        """
        port_var = w @ cov @ w
        if port_var < 1e-10:  # Avoid division by zero
            return np.ones_like(w) / len(w)
            
        port_vol = np.sqrt(port_var)
        marginal_contrib = cov @ w / port_vol
        risk_contrib = w * marginal_contrib
        return risk_contrib / risk_contrib.sum()  # Normalize to sum to 1
        
    def dispersion_penalty(self, w: cp.Variable, cov: np.ndarray, target_contrib: np.ndarray) -> cp.Expression:
        """
        Create a dispersion penalty term to encourage risk parity.
        Uses a quadratic penalty on the deviation from target risk contribution.
        
        Args:
            w: Weight variable (CVXPY Variable object)
            cov: Covariance matrix (numpy array)
            target_contrib: Target risk contribution (numpy array)
            
        Returns:
            CVXPY expression for the dispersion penalty
        """
        # Calculate portfolio volatility
        port_vol = cp.quad_form(w, cov)
        
        # Calculate risk contributions using the proper CVXPY expressions
        # risk_contrib = (w * (cov @ w)) / port_vol
        # But we need to handle this carefully in CVXPY
        
        # Get the number of assets from the covariance matrix
        n = cov.shape[0]
        
        # Calculate risk contributions using element-wise multiplication and sum
        # This avoids using len() on the CVXPY Variable
        risk_contrib = cp.multiply(w, cov @ w) / port_vol
        
        # Normalize to sum to 1 (not strictly necessary but helps with numerical stability)
        risk_contrib = risk_contrib / cp.sum(risk_contrib)
        
        # Calculate penalty as sum of squared differences from target contributions
        penalty = 0
        for i in range(n):
            penalty += (risk_contrib[i] - target_contrib[i])**2
            
        return penalty

    def optimise(self, rets: pd.DataFrame, expected: pd.Series, 
            dispersion_gamma: float = 0.0, 
            contrib_target: Optional[np.ndarray] = None,
            n_positions: int = 12) -> pd.Series:
        """
        Optimize portfolio with mean-variance objective.
        Uses a two-step approach to get exactly n_positions.
        
        Args:
            rets: Returns DataFrame (daily returns)
            expected: Expected returns Series
            dispersion_gamma: Not used in this simplified version
            contrib_target: Not used in this simplified version
            n_positions: Exact number of positions to include in the portfolio
            
        Returns:
            Optimized portfolio weights
        """
        n = len(rets.columns)  # Number of assets
        
        # Calculate expected returns and covariance
        mu = expected.values if isinstance(expected, pd.Series) else expected
        
        # Add small positive value to diagonal for numerical stability
        cov = np.cov(rets.T) * 252  # Annualized covariance
        np.fill_diagonal(cov, np.diag(cov) + 1e-6)
        
        # Make sure covariance matrix is positive semi-definite
        cov = 0.5 * (cov + cov.T)  # Ensure symmetry
        
        # Define optimization parameters from config
        risk_aversion = 1.0  # Balanced risk aversion
        min_weight = 0.05    # 5% minimum weight per position
        max_weight = self.cfg.individual_limit  # Max weight from config (15%)
        
        # Create binary variables for position selection
        z = cp.Variable(n, boolean=True)  # Binary variable for position inclusion
        w = cp.Variable(n)                # Continuous variable for weights
        # Auxiliary variables for turnover constraint
        u = cp.Variable(n, nonneg=True)
        prev_w = self.data.get_previous_weights()  # series or array of w_prev
        
        # Portfolio return and risk
        portfolio_return = mu.T @ w
        portfolio_risk = cp.quad_form(w, cov)

        # --- HHI and Return-HHI penalties ---
        cfg = self.cfg
        expected_returns = expected
        # Compute HHI penalty (already present)
        lambda_hhi = cfg.get('lambda_hhi', 10.0)
        hhi_weight_threshold = cfg.get('hhi_weight_threshold', 0.08)
        hhi_penalty = lambda_hhi * cp.pos(cp.sum_squares(w) - hhi_weight_threshold)

        # Compute Return-HHI penalty
        lambda_return_hhi = cfg.get('lambda_return_hhi', 10.0)
        return_hhi_threshold = cfg.get('return_hhi_threshold', 0.30)
        # Approximate Return-HHI: (sum((w*μ)^2) / (sum(w*μ))^2)
        mu_vec = expected_returns.values
        numerator = cp.sum_squares(cp.multiply(w, mu_vec))
        denominator = cp.square(cp.sum(cp.multiply(w, mu_vec)) + 1e-10)
        ret_hhi_expr = numerator / denominator
        return_hhi_penalty = lambda_return_hhi * cp.pos(ret_hhi_expr - return_hhi_threshold)

        # New objective: maximize return minus risk, HHI and Return-HHI penalties
        objective = cp.Maximize(
            portfolio_return
            - 0.5 * portfolio_risk
            - hhi_penalty
            - return_hhi_penalty
        )
        
        # Constraints
        constraints = [
            cp.sum(w) == 1,                     # Fully invested
            w >= 0,                              # No short selling
            w <= z * max_weight,                 # If z=0, w=0; if z=1, w <= max_weight
            w >= z * min_weight,                 # If z=1, w >= min_weight
            cp.sum(z) == n_positions,            # Exactly n_positions
            cp.norm(w, 2) <= 0.4                # L2 norm constraint for diversification
        ]
        # Turnover constraint: enforce weekly turnover >= min_turnover
        min_turnover = self.cfg.min_turnover  # 0.05
        constraints += [
            u >= w - prev_w,
            u >= prev_w - w,
            cp.sum(u) >= min_turnover
        ]
        
        # Log optimization settings
        global_logger.info("\n=== OPTIMIZATION SETTINGS ===")
        global_logger.info(f"- Number of assets: {n}")
        global_logger.info(f"- Target positions: {n_positions}")
        global_logger.info(f"- Risk aversion: {risk_aversion}")
        global_logger.info(f"- Position limits: {min_weight*100:.1f}% min, {max_weight*100:.1f}% max")
        global_logger.info(f"- Using config file: {self.cfg.stock_universe_file}")
        global_logger.info(f"- Individual limit: {self.cfg.individual_limit*100:.1f}%")
        global_logger.info(f"- Return weight: {self.cfg.return_weight}")
        # global_logger.info(f"- CVaR alpha: {self.cfg.cvar_alpha}")  # Removed CVaR alpha logging
        
        # Create and solve problem
        prob = cp.Problem(objective, constraints)
        
        # Try different solvers in sequence
        solvers = [
            ('ECOS_BB', {'max_iters': 1000, 'verbose': False}),
            ('SCIP', {'verbose': False}),
            ('GUROBI', {'verbose': False}) if 'GUROBI' in cp.installed_solvers() else None,
            ('MOSEK', {'verbose': False}) if 'MOSEK' in cp.installed_solvers() else None
        ]
        
        solution_found = False
        for solver_name, solver_opts in [s for s in solvers if s is not None]:
            try:
                global_logger.info(f"Trying solver: {solver_name}")
                prob.solve(solver=solver_name, **solver_opts)
                if prob.status == cp.OPTIMAL or prob.status == 'optimal':
                    solution_found = True
                    break
                else:
                    global_logger.warning(f"Solver {solver_name} returned status: {prob.status}")
            except Exception as e:
                global_logger.warning(f"Solver {solver_name} failed: {str(e)[:200]}")
        
        if not solution_found:
            global_logger.error("All solvers failed, falling back to heuristic")
            return self._fallback_weights(rets, n_positions, min_weight, max_weight)
        
        # Process results
        if w.value is None or np.any(np.isnan(w.value)) or z.value is None:
            global_logger.error("Optimization failed, falling back to heuristic")
            return self._fallback_weights(rets, n_positions, min_weight, max_weight)
            
        # Get weights and round small values to zero
        weights = pd.Series(w.value, index=rets.columns)
        weights[weights < 1e-6] = 0
        weights = weights / weights.sum()  # Re-normalize
        
        # Log results
        global_logger.info("\n=== OPTIMIZATION RESULTS ===")
        global_logger.info(f"Number of positions: {(weights > 0).sum()}")
        global_logger.info(f"Min weight: {weights[weights > 0].min():.2%}")
        global_logger.info(f"Max weight: {weights.max():.2%}")
        
        return weights
    
    def _fallback_weights(self, rets: pd.DataFrame, n_positions: int, 
                         min_weight: float, max_weight: float) -> pd.Series:
        """Fallback method when optimization fails"""
        global_logger.warning("Using fallback portfolio with top %d positions by Sharpe ratio", n_positions)
        
        # Calculate Sharpe ratios (simple version)
        expected_returns = rets.mean()
        vols = rets.std()
        sharpe_ratios = expected_returns / (vols + 1e-6)  # Avoid division by zero
        
        # Select top n_positions by Sharpe ratio
        top_tickers = sharpe_ratios.nlargest(n_positions).index
        
        # Create weights respecting min/max constraints
        weights = pd.Series(0.0, index=rets.columns)  # Initialize with float type
        
        # Assign weights based on Sharpe ratios, respecting constraints
        selected_sharpe = sharpe_ratios[top_tickers]
        weights[top_tickers] = selected_sharpe / selected_sharpe.sum()
        
        # Apply min/max constraints
        weights = weights.clip(lower=min_weight, upper=max_weight)
        weights = weights / weights.sum()  # Renormalize
        
        # Ensure exactly n_positions
        if (weights > 0).sum() > n_positions:
            # If we have more than n_positions due to min_weight, keep only top n
            top_n = weights.nlargest(n_positions).index
            weights = weights[top_tickers].reindex(weights.index, fill_value=0.0)
        
        return weights.astype(float)
        # Log results
        global_logger.info("\n=== OPTIMIZATION RESULTS ===")
        try:
            portfolio_vol = np.sqrt(weights @ cov @ weights)
            global_logger.info(f"Expected return: {(mu @ weights):.2%}")
            global_logger.info(f"Portfolio volatility: {portfolio_vol:.2%}")
        except Exception as e:
            global_logger.warning(f"Could not calculate portfolio statistics: {e}")
        
        # Log all positions with weights > 0.1%
        positions = weights[weights > 0.001].sort_values(ascending=False)
        global_logger.info(f"\nPortfolio Positions ({len(positions)}):")
        for ticker, weight in positions.items():
            global_logger.info(f"{ticker}: {weight*100:.2f}%")
        
        return weights
