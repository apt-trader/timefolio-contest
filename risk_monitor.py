# risk_monitor.py
import logging
import pandas as pd
import numpy as np
from typing import Dict, Optional, Set
from pathlib import Path

# Configure logging for this module
logger = logging.getLogger(__name__)

class RiskMonitor:
    """
    Calculates and reports on key portfolio risk and performance metrics.
    Designed to be used after an optimization run.
    """
    def __init__(self, settings: Dict):
        """
        Initializes the RiskMonitor with risk thresholds from the configuration.

        Args:
            settings (Dict): A dictionary of risk management settings.
        """
        self.settings = settings
        self.mdd_threshold_6m = settings.get('mdd_threshold_6m', 0.15)
        self.hhi_weight_threshold = settings.get('hhi_weight_threshold', 0.08)
        self.hhi_return_threshold = settings.get('hhi_return_threshold', 0.30)
        self.min_turnover_threshold = settings.get('turnover_min', 0.05)
        logger.info("RiskMonitor initialized.")

    def _calculate_hhi(self, series: pd.Series) -> float:
        """
        Calculates the Herfindahl-Hirschman Index (HHI) for a given series.
        Used for both weight and return contribution concentration.
        """
        if series.sum() == 0:
            return 0.0
        # For return contributions, some can be negative. We use absolute values.
        contributions = series.abs()
        normalized_contributions = contributions / contributions.sum()
        hhi = (normalized_contributions ** 2).sum()
        return float(hhi)

    def _calculate_turnover(self, prev_weights: pd.Series, curr_weights: pd.Series) -> float:
        """
        Calculates one-way portfolio turnover.
        """
        # Align series to handle adds/drops, filling missing with 0
        aligned_prev, aligned_curr = prev_weights.align(curr_weights, fill_value=0)
        turnover = (aligned_curr - aligned_prev).abs().sum() / 2.0
        return float(turnover)

    def _calculate_mdd(self, portfolio_returns: pd.Series) -> float:
        """
        Calculates the Maximum Drawdown (MDD) of the portfolio returns.
        """
        if portfolio_returns.empty:
            return 0.0
        cumulative_returns = (1 + portfolio_returns).cumprod()
        peak = cumulative_returns.expanding(min_periods=1).max()
        drawdown = (cumulative_returns / peak) - 1
        return float(drawdown.min())

    def run_analysis(self,
                     weights: pd.Series,
                     returns_df: pd.DataFrame,
                     prev_weights: Optional[pd.Series] = None) -> Dict:
        """
        Performs a full risk analysis and returns a dictionary of metrics.

        Args:
            weights (pd.Series): The current portfolio weights.
            returns_df (pd.DataFrame): DataFrame of historical daily returns for all assets.
            prev_weights (pd.Series, optional): The previous period's weights for turnover calculation.

        Returns:
            A dictionary containing all calculated risk metrics.
        """
        logger.info("Running full portfolio risk analysis...")
        
        # Ensure we only work with assets in the portfolio
        portfolio_tickers = weights[weights > 0].index
        analysis_rets = returns_df[portfolio_tickers]
        analysis_weights = weights[portfolio_tickers]

        # 1. Portfolio Daily Returns
        portfolio_returns = analysis_rets.mul(analysis_weights, axis=1).sum(axis=1)

        # 2. Maximum Drawdown (6-month)
        mdd_6m = self._calculate_mdd(portfolio_returns.tail(126)) # Approx. 6 months

        # 3. HHI on Portfolio Weights (Concentration)
        hhi_weight = self._calculate_hhi(analysis_weights)

        # 4. HHI on Return Contributions (Profit Concentration)
        # Calculate total return contribution of each stock over the period
        return_contributions = analysis_rets.mean().multiply(analysis_weights)
        hhi_return = self._calculate_hhi(return_contributions)
        
        # 5. Turnover
        turnover = 0.0
        if prev_weights is not None:
            turnover = self._calculate_turnover(prev_weights, weights)

        # Compile results
        results = {
            'mdd_6m': {
                'value': mdd_6m,
                'threshold': self.mdd_threshold_6m,
                'status': 'FAIL' if abs(mdd_6m) > self.mdd_threshold_6m else 'PASS'
            },
            'hhi_weight': {
                'value': hhi_weight,
                'threshold': self.hhi_weight_threshold,
                'status': 'FAIL' if hhi_weight > self.hhi_weight_threshold else 'PASS'
            },
            'hhi_return': {
                'value': hhi_return,
                'threshold': self.hhi_return_threshold,
                'status': 'FAIL' if hhi_return > self.hhi_return_threshold else 'PASS'
            },
            'turnover': {
                'value': turnover,
                'threshold': self.min_turnover_threshold,
                'status': 'FAIL' if turnover < self.min_turnover_threshold else 'PASS'
            }
        }
        logger.info("Risk analysis complete.")
        return results

    def generate_report_text(self, analysis_results: Dict) -> str:
        """
        Generates a human-readable text report from the analysis results.
        
        Args:
            analysis_results (Dict): The output from run_analysis.
            
        Returns:
            A formatted string containing the risk report.
        """
        report_lines = [
            "========================================",
            "         PORTFOLIO RISK REPORT          ",
            "========================================",
            f"Report generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        ]
        
        for metric, data in analysis_results.items():
            value = data['value']
            threshold = data['threshold']
            status = data['status']
            
            line = f"  - {metric.replace('_', ' ').title():<20}: {value:8.2%} "
            if 'mdd' in metric:
                line += f" (Threshold: {threshold:.2%}, Status: {status})"
            elif 'hhi' in metric:
                 line += f" (Threshold: {threshold:.4f}, Status: {status})"
            elif 'turnover' in metric:
                line += f" (Min Threshold: {threshold:.2%}, Status: {status})"
            
            report_lines.append(line)
        
        report_lines.append("\n" + "="*40)
        return "\n".join(report_lines)

    def save_report(self, report_text: str, output_dir: str):
        """
        Saves the generated report text to a file.
        """
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        file_path = path / f"risk_report_{pd.Timestamp.now().strftime('%Y%m%d')}.txt"
        
        with open(file_path, 'w') as f:
            f.write(report_text)
        
        logger.info(f"Risk report saved to {file_path}")