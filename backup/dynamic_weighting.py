#!/usr/bin/env python3
"""
Dynamic Weighting Module for Portfolio Optimization
"""
import pandas as pd
import numpy as np
import logging
from scipy import stats
import statsmodels.api as sm
import os
from datetime import datetime

# Setup logging
glogger = logging.getLogger("dynamic_weighting")
glogger.setLevel(logging.INFO)
if not glogger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    glogger.addHandler(ch)

class BetaRegimeTracker:
    """
    Stateful tracker for beta regime that maintains history and provides smoothing.
    """
    def __init__(self, smoothing_alpha=0.2, max_jump=0.3):
        self.prev_beta = None
        self.alpha = smoothing_alpha
        self.max_jump = max_jump
        self.beta_history = []
        self.smoothed_history = []
        self.timestamps = []

    def update(self, new_beta):
        self.beta_history.append(new_beta)
        self.timestamps.append(datetime.now())
        if self.prev_beta is None:
            self.prev_beta = new_beta
            self.smoothed_history.append(new_beta)
            glogger.info(f"Initializing beta tracker with beta={new_beta:.4f}")
            return new_beta
        smoothed = (1 - self.alpha) * self.prev_beta + self.alpha * new_beta
        if abs(new_beta - self.prev_beta) > self.max_jump:
            direction = 1 if new_beta > self.prev_beta else -1
            bounded = self.prev_beta + self.max_jump * direction
            smoothed = (1 - self.alpha) * self.prev_beta + self.alpha * bounded
            glogger.warning(f"Large beta jump ({new_beta - self.prev_beta:.4f}); bounded to {bounded:.4f}")
        self.prev_beta = smoothed
        self.smoothed_history.append(smoothed)
        glogger.info(f"Updated beta: raw={new_beta:.4f}, smoothed={smoothed:.4f}")
        return smoothed

    def save_state(self, path):
        try:
            import pickle
            state = {
                'prev_beta': self.prev_beta,
                'alpha': self.alpha,
                'max_jump': self.max_jump,
                'beta_history': self.beta_history,
                'smoothed_history': self.smoothed_history,
                'timestamps': [ts.isoformat() for ts in self.timestamps]
            }
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'wb') as f:
                pickle.dump(state, f)
            glogger.info(f"Beta regime tracker state saved to {path}")
            return True
        except Exception as e:
            glogger.error(f"Error saving state: {e}")
            return False

    @classmethod
    def load_state(cls, path):
        try:
            import pickle
            with open(path, 'rb') as f:
                st = pickle.load(f)
            tracker = cls(smoothing_alpha=st['alpha'], max_jump=st['max_jump'])
            tracker.prev_beta = st['prev_beta']
            tracker.beta_history = st['beta_history']
            tracker.smoothed_history = st['smoothed_history']
            tracker.timestamps = [datetime.fromisoformat(ts) for ts in st['timestamps']]
            glogger.info(f"Loaded beta tracker from {path}")
            return tracker
        except Exception as e:
            glogger.error(f"Error loading state: {e}")
            return cls()

    def visualize(self, save_path: str = None):
        if not self.timestamps or not self.smoothed_history:
            glogger.warning("No beta history to visualize")
            return
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(self.timestamps, self.beta_history, 'o', alpha=0.4, label='Raw Beta')
        plt.plot(self.timestamps, self.smoothed_history, '-', lw=2, label='Smoothed Beta')
        plt.xlabel('Time')
        plt.ylabel('Beta')
        plt.title('Beta Regime History')
        plt.legend()
        if save_path:
            plt.savefig(save_path)
            glogger.info(f"Saved beta history plot to {save_path}")
        else:
            plt.show()

# Global instance for backward compatibility
_beta_tracker = BetaRegimeTracker()

def calculate_dynamic_beta(stock_returns: pd.DataFrame, market_returns: pd.Series = None, window: int = 20) -> float:
    if len(stock_returns) < max(window, 30):
        glogger.warning("Insufficient data for beta calculation; default=0.5")
        return 0.5
    if market_returns is None:
        market_returns = stock_returns.mean(axis=1)
    mvol = market_returns.rolling(window).std().iloc[-1]
    sample_cols = stock_returns.columns
    if len(sample_cols) > 30:
        sample_cols = np.random.choice(sample_cols, 30, replace=False)
    idio_vols = []
    for col in sample_cols:
        try:
            X = sm.add_constant(market_returns)
            model = sm.OLS(stock_returns[col], X).fit()
            idio_vols.append(model.resid.std())
        except:
            continue
    avg_idio = np.mean(idio_vols) if idio_vols else mvol
    beta = mvol / (mvol + avg_idio) if (mvol + avg_idio) > 0 else 0.5
    beta = min(max(beta, 0.0), 1.0)
    glogger.info(f"Calculated dynamic beta={beta:.4f}")
    return beta

def get_dynamic_weights(return_df: pd.DataFrame, smoothing_factor: float = 0.2) -> dict:
    market = return_df.mean(axis=1)
    raw_beta = calculate_dynamic_beta(return_df, market)
    cache = os.path.join(os.getcwd(), 'cache', 'beta_tracker.pkl')
    tracker = BetaRegimeTracker.load_state(cache) if os.path.exists(cache) else BetaRegimeTracker(smoothing_factor)
    smoothed = tracker.update(raw_beta)
    tracker.save_state(cache)
    corr = return_df.corr()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    avg_corr = upper.stack().mean()
    return {
        'sharpe_treynor_weight': 1 - smoothed,
        'raw_beta': raw_beta,
        'smoothed_beta': smoothed,
        'market_volatility': market.std() * np.sqrt(252),
        'average_correlation': avg_corr
    }

def save_beta_state(path: str) -> bool:
    return _beta_tracker.save_state(path)

def visualize_beta_history(save_path: str = None):
    _beta_tracker.visualize(save_path)