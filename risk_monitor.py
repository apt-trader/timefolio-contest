#!/usr/bin/env python3
"""
Risk monitoring system for TimeFolio portfolio.
Tracks rolling maximum drawdown, Herfindahl-Hirschman Index (concentration),
and turnover, triggering alerts when thresholds are exceeded.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import json
import yaml

# For alerts
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(ch)


class RiskMonitor:
    """Portfolio risk monitoring system."""
    
    def __init__(self, 
                 config_file: Optional[str] = None,
                 mdd_threshold_6m: float = 0.15,
                 mdd_threshold_12m: float = 0.25,
                 hhi_weight_threshold: float = 0.20,
                 hhi_return_threshold: float = 0.30,
                 turnover_min: float = 0.05,
                 turnover_max: float = 0.50,
                 tracking_error_threshold: float = 0.08,
                 risk_report_dir: Optional[str] = None,
                 slack_token: Optional[str] = None,
                 slack_channel: Optional[str] = None,
                 email_address: Optional[str] = None,
                 email_password: Optional[str] = None):
        """
        Initialize risk monitor with thresholds and notification settings.
        
        Args:
            config_file: Path to YAML configuration file
            mdd_threshold_6m: Maximum drawdown threshold for 6-month window
            mdd_threshold_12m: Maximum drawdown threshold for 12-month window
            hhi_weight_threshold: HHI threshold for portfolio weight concentration
            hhi_return_threshold: HHI threshold for return contribution concentration
            turnover_min: Minimum portfolio turnover threshold
            turnover_max: Maximum portfolio turnover threshold
            tracking_error_threshold: Tracking error threshold vs benchmark
            risk_report_dir: Directory to save risk reports
            slack_token: Slack API token for alerts
            slack_channel: Slack channel for alerts
            email_address: Email address for sending alerts
            email_password: Email password
        """
        # Initialize with default thresholds
        self.config = {
            'mdd': {
                'threshold_6m': mdd_threshold_6m,
                'threshold_12m': mdd_threshold_12m,
                'alert_level': 'WARNING'
            },
            'hhi': {
                'weight_threshold': hhi_weight_threshold,
                'return_threshold': hhi_return_threshold,
                'alert_level': 'WARNING'
            },
            'turnover': {
                'min_threshold': turnover_min,
                'max_threshold': turnover_max,
                'alert_level': 'WARNING'
            },
            'tracking_error': {
                'threshold': tracking_error_threshold,
                'alert_level': 'WARNING'
            },
            'notify': {
                'slack': True,
                'email': True,
                'log_file': True
            }
        }
        
        # Load configuration from YAML file if provided
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    yaml_config = yaml.safe_load(f)
                    # Update default config with loaded values
                    self._update_nested_dict(self.config, yaml_config)
                logger.info(f"Loaded risk alert configuration from {config_file}")
            except Exception as e:
                logger.error(f"Error loading config from {config_file}: {e}")
                
        # For backward compatibility, set these attributes directly
        self.mdd_threshold_6m = self.config['mdd']['threshold_6m']
        self.mdd_threshold_12m = self.config['mdd']['threshold_12m']
        self.hhi_weight_threshold = self.config['hhi']['weight_threshold']
        self.hhi_return_threshold = self.config['hhi']['return_threshold']
        self.turnover_min = self.config['turnover']['min_threshold']
        self.turnover_max = self.config['turnover']['max_threshold']
        self.risk_report_dir = Path(risk_report_dir)
        self.risk_report_dir.mkdir(parents=True, exist_ok=True)
        
        # Load alert configuration if provided
        self.alert_config = {}
        if slack_token:
            self.alert_config['slack'] = {
                'token': slack_token,
                'channel': slack_channel
            }
        if email_address and email_password:
            self.alert_config['email'] = {
                'address': email_address,
                'password': email_password
            }
    
    def rolling_mdd(self, 
                   returns: pd.Series, 
                   window_days: int = 126,
                   data_frequency: str = 'daily') -> pd.Series:
        """
        Calculate rolling maximum drawdown with support for different data frequencies.
        
        Args:
            returns: Portfolio returns (daily or weekly)
            window_days: Rolling window size in days
            data_frequency: Data frequency ('daily' or 'weekly')
        
        Returns:
            Series of rolling maximum drawdowns
        """
        # Adjust window size for weekly data
        if data_frequency.lower() == 'weekly':
            # Convert days to weeks (approximately)
            window_periods = max(1, round(window_days / 7))
            logger.info(f"Adjusting {window_days}-day window to {window_periods} weeks for weekly data")
        else:
            window_periods = window_days
        
        # Calculate cumulative returns
        cum_returns = (1 + returns).cumprod()
        
        # Initialize rolling MDD series
        rolling_mdd = pd.Series(index=returns.index, dtype=float)
        
        # Calculate rolling MDD
        for i in range(len(returns)):
            if i < window_periods:
                rolling_mdd.iloc[i] = np.nan  # Not enough data for full window
                continue
                
            window = cum_returns.iloc[i-window_periods:i+1]
            peak = window.cummax()
            drawdown = (window / peak - 1)
            rolling_mdd.iloc[i] = drawdown.min()
            
        return rolling_mdd
    
    def calculate_hhi(self, weights: pd.Series) -> float:
        """
        Calculate Herfindahl-Hirschman Index for portfolio concentration.
        
        Args:
            weights: Portfolio weights or contribution values
            
        Returns:
            HHI value between 0 and 1
        """
        # Handle absolute values by default (needed for return contributions)
        weights_abs = weights.abs()
        
        # Normalize to sum to 1
        weights_normalized = weights_abs / weights_abs.sum()
        
        # Square weights and sum
        return np.sum(weights_normalized ** 2)
        
    def calculate_return_contributions(self, returns_by_ticker: pd.DataFrame, weights: pd.Series) -> pd.Series:
        """
        Calculate each ticker's contribution to portfolio returns.
        
        Args:
            returns_by_ticker: DataFrame of returns by ticker
            weights: Portfolio weights at the beginning of the period
            
        Returns:
            Series of return contributions by ticker
        """
        # Match indices between weights and returns
        common_tickers = returns_by_ticker.columns.intersection(weights.index)
        
        # Filter weights and returns to common tickers
        filtered_weights = weights.loc[common_tickers]
        filtered_returns = returns_by_ticker.loc[:, common_tickers]
        
        # Normalize weights (in case they don't sum to 1)
        normalized_weights = filtered_weights / filtered_weights.sum()
        
        # Calculate return contribution for each ticker
        return_contrib = pd.Series(
            (normalized_weights * filtered_returns.mean()).values,
            index=common_tickers
        )
        
        return return_contrib
    
    def calculate_turnover(self, 
                          old_weights: pd.Series, 
                          new_weights: pd.Series) -> float:
        """
        Calculate portfolio turnover.
        
        Args:
            old_weights: Previous portfolio weights
            new_weights: New portfolio weights
        
        Returns:
            Turnover as a fraction (0-1)
        """
        # Align weights to same index
        old_aligned = old_weights.reindex(new_weights.index, fill_value=0)
        # Calculate turnover (sum of absolute differences divided by 2)
        return (new_weights - old_aligned).abs().sum() / 2
    
    def check_risk_metrics(self, 
                          returns: pd.Series,
                          weights: pd.Series,
                          previous_weights: Optional[pd.Series] = None,
                          returns_by_ticker: Optional[pd.DataFrame] = None,
                          benchmark_returns: Optional[pd.Series] = None,
                          data_frequency: str = 'auto') -> Dict:
        """
        Check all risk metrics for the portfolio.
        
        Args:
            returns: Portfolio returns
            weights: Current portfolio weights
            previous_weights: Previous portfolio weights (for turnover)
            returns_by_ticker: Individual asset returns (optional, for contribution HHI)
            benchmark_returns: Benchmark returns (optional, for tracking error)
            data_frequency: Data frequency ('daily', 'weekly', or 'auto')
            
        Returns:
            Dictionary with risk metric results
        """
        # Check for minimum data
        if len(returns) < 10:
            logger.warning("Insufficient return data for risk metrics. Need at least 10 data points.")
            return {}
        
        # Calculate HHI for weight concentration
        hhi_weight = self.calculate_hhi(weights)
        
        # Calculate HHI for return contribution if data available
        hhi_return = None
        if returns_by_ticker is not None and len(returns_by_ticker) > 0:
            # Calculate return contributions
            return_contribs = self.calculate_return_contributions(returns_by_ticker, weights)
            # Calculate HHI for return contributions
            hhi_return = self.calculate_hhi(return_contribs)
        
        # Calculate turnover if previous weights provided
        turnover = 0
        if previous_weights is not None:
            turnover = self.calculate_turnover(previous_weights, weights)
        
        # Determine the data frequency if not explicitly provided
        if data_frequency.lower() not in ['daily', 'weekly']:
            # Try to detect frequency from the returns index
            if isinstance(returns.index, pd.DatetimeIndex):
                avg_days = (returns.index[-1] - returns.index[0]).days / max(1, len(returns) - 1)
                data_frequency = 'weekly' if avg_days > 2.5 else 'daily'
                logger.info(f"Detected data frequency: {data_frequency} (avg days between data points: {avg_days:.1f})")
            else:
                # Default to daily if we can't determine
                data_frequency = 'daily'
                logger.info("Could not determine data frequency from index, assuming daily data")
        
        # Adjust min data points required based on frequency
        if data_frequency.lower() == 'weekly':
            min_6m_points = 26  # ~26 weeks = 6 months
            min_12m_points = 52  # 52 weeks = 12 months
        else:  # daily
            min_6m_points = 126  # ~126 trading days = 6 months
            min_12m_points = 252  # ~252 trading days = 12 months
            
        # Calculate tracking error if benchmark provided
        tracking_error = None
        if benchmark_returns is not None and len(benchmark_returns) >= min_6m_points:
            # Get common dates
            common_dates = returns.index.intersection(benchmark_returns.index)
            if len(common_dates) >= min_6m_points:
                # Calculate active returns
                active_returns = returns.loc[common_dates] - benchmark_returns.loc[common_dates]
                # Calculate tracking error (annualized std dev of active returns)
                ann_factor = 52 if data_frequency.lower() == 'weekly' else 252
                tracking_error = active_returns.std() * np.sqrt(ann_factor)
        
        # Check rolling maximum drawdown (6-month)
        mdd_6m = None
        if len(returns) >= min_6m_points:  # At least 6 months of data
            mdd_6m = self.rolling_mdd(returns, window_days=126, data_frequency=data_frequency).iloc[-1]
        
        # Calculate 12-month MDD if we have enough data
        mdd_12m = None
        if len(returns) >= min_12m_points:
            mdd_12m = self.rolling_mdd(returns, window_days=252, data_frequency=data_frequency).iloc[-1]
        
        # Structure risk results
        risk_results = {
            'timestamp': datetime.now(),
            'metrics': {}
        }
        
        # Add MDD metrics if available
        if mdd_6m is not None:
            risk_results['metrics']['mdd_6m'] = {
                'value': mdd_6m,
                'threshold': self.config['mdd']['threshold_6m'],
                'status': self.config['mdd']['alert_level'] if mdd_6m < -self.config['mdd']['threshold_6m'] else 'OK'
            }
        
        if mdd_12m is not None:
            risk_results['metrics']['mdd_12m'] = {
                'value': mdd_12m,
                'threshold': self.config['mdd']['threshold_12m'],
                'status': self.config['mdd']['alert_level'] if mdd_12m < -self.config['mdd']['threshold_12m'] else 'OK'
            }
            
        # Add HHI metrics
        risk_results['metrics']['hhi_weight'] = {
            'value': hhi_weight,
            'threshold': self.config['hhi']['weight_threshold'],
            'status': self.config['hhi']['alert_level'] if hhi_weight > self.config['hhi']['weight_threshold'] else 'OK'
        }
        
        # Add HHI return contribution if available
        if hhi_return is not None:
            risk_results['metrics']['hhi_return'] = {
                'value': hhi_return,
                'threshold': self.config['hhi']['return_threshold'],
                'status': self.config['hhi']['alert_level'] if hhi_return > self.config['hhi']['return_threshold'] else 'OK'
            }
        
        # Add turnover metrics
        risk_results['metrics']['turnover'] = {
            'value': turnover,
            'min_threshold': self.config['turnover']['min_threshold'],
            'max_threshold': self.config['turnover']['max_threshold'],
            'status': (
                self.config['turnover']['alert_level'] 
                if (turnover < self.config['turnover']['min_threshold'] or 
                    turnover > self.config['turnover']['max_threshold']) 
                else 'OK'
            )
        }
        
        # Add tracking error if available
        if tracking_error is not None:
            risk_results['metrics']['tracking_error'] = {
                'value': tracking_error,
                'threshold': self.config['tracking_error']['threshold'],
                'status': self.config['tracking_error']['alert_level'] if tracking_error > self.config['tracking_error']['threshold'] else 'OK'
            }
        
        return risk_results
    
    def generate_risk_report(self, 
                            returns: pd.Series,
                            weights: pd.Series,
                            previous_weights: Optional[pd.Series] = None,
                            additional_info: Optional[Dict] = None,
                            data_frequency: str = 'daily') -> str:
        """
        Generate risk report with visualizations.
        
        Args:
            returns: Historical daily returns
            weights: Current portfolio weights
            previous_weights: Previous portfolio weights
            additional_info: Additional information to include in report
            data_frequency: Data frequency ('daily' or 'weekly')
        
        Returns:
            Path to generated report
        """
        # Check risk levels
        risk_status = self.check_risk_metrics(returns, weights, previous_weights, data_frequency=data_frequency)
        
        # Create report timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d")
        report_path = self.risk_report_dir / f"risk_report_{timestamp}.txt"
        
        # Generate visualizations
        fig_path = self.risk_report_dir / f"risk_viz_{timestamp}.png"
        self._generate_risk_visualizations(returns, weights, risk_status, fig_path, data_frequency)
        
        # Write report
        with open(report_path, 'w') as f:
            f.write(f"TimeFolio Risk Report - {timestamp}\n")
            f.write("="*50 + "\n\n")
            
            # Write risk metrics
            f.write("RISK METRICS SUMMARY:\n")
            f.write("-"*50 + "\n")
            for metric, details in risk_status.items():
                f.write(f"{metric.upper()}: {details['value']:.4f} ")
                f.write(f"[Threshold: {details['threshold']:.4f}] ")
                f.write(f"Status: {details['status']}\n")
            f.write("\n")
            
            # Write portfolio metrics
            f.write("PORTFOLIO METRICS:\n")
            f.write("-"*50 + "\n")
            f.write(f"Number of positions: {len(weights)}\n")
            f.write(f"Top 3 holdings: {weights.sort_values(ascending=False).head(3).to_dict()}\n")
            f.write(f"Annualized volatility: {returns.std() * np.sqrt(252):.4f}\n")
            
            if previous_weights is not None:
                f.write(f"Turnover: {risk_status['turnover']['value']:.4f}\n")
                
            # Include additional info if provided
            if additional_info:
                f.write("\nADDITIONAL INFORMATION:\n")
                f.write("-"*50 + "\n")
                for key, value in additional_info.items():
                    f.write(f"{key}: {value}\n")
            
            f.write("\n")
            f.write("VISUALIZATIONS:\n")
            f.write(f"See {fig_path} for risk visualizations\n")
        
        logger.info(f"Risk report generated: {report_path}")
        
        # Check if any warnings and trigger alerts if needed
        self._check_and_send_alerts(risk_status, report_path)
        
        return str(report_path)
    
    def _generate_risk_visualizations(self, 
                                     returns: pd.Series,
                                     weights: pd.Series,
                                     risk_status: Dict,
                                     output_path: str,
                                     data_frequency: str):
        """
        Generate risk visualizations.
        
        Args:
            returns: Historical daily returns
            weights: Current portfolio weights
            risk_status: Risk metrics and status
            output_path: Path to save visualization
            data_frequency: Data frequency ('daily' or 'weekly')
        """
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Rolling MDD
        mdd_6m = self.rolling_mdd(returns, window_days=126, data_frequency=data_frequency)
        mdd_12m = self.rolling_mdd(returns, window_days=252, data_frequency=data_frequency)
        
        axs[0, 0].plot(mdd_6m, label='6-Month MDD', color='blue')
        axs[0, 0].plot(mdd_12m, label='12-Month MDD', color='darkblue')
        axs[0, 0].axhline(y=-self.mdd_threshold_6m, color='r', linestyle='--', label='6M Threshold')
        axs[0, 0].axhline(y=-self.mdd_threshold_12m, color='darkred', linestyle='--', label='12M Threshold')
        axs[0, 0].set_title('Rolling Maximum Drawdown')
        axs[0, 0].legend()
        axs[0, 0].grid(True)
        
        # Plot 2: Cumulative returns
        cum_returns = (1 + returns).cumprod()
        axs[0, 1].plot(cum_returns, color='green')
        axs[0, 1].set_title('Cumulative Returns')
        axs[0, 1].grid(True)
        
        # Plot 3: Weight distribution (treemap or barplot)
        top_weights = weights.sort_values(ascending=False).head(10)
        sns.barplot(x=top_weights.values, y=top_weights.index, ax=axs[1, 0])
        axs[1, 0].set_title('Top 10 Holdings')
        axs[1, 0].set_xlabel('Weight')
        axs[1, 0].grid(True)
        
        # Plot 4: Risk metrics
        metric_values = [abs(risk_status['mdd_6m']['value']), 
                        abs(risk_status['mdd_12m']['value']),
                        risk_status['hhi']['value'],
                        risk_status['turnover']['value']]
        metric_thresholds = [risk_status['mdd_6m']['threshold'], 
                            risk_status['mdd_12m']['threshold'],
                            risk_status['hhi']['threshold'],
                            risk_status['turnover']['threshold']]
        metric_names = ['MDD 6M', 'MDD 12M', 'HHI', 'Turnover']
        
        x = np.arange(len(metric_names))
        width = 0.35
        
        axs[1, 1].bar(x - width/2, metric_values, width, label='Current')
        axs[1, 1].bar(x + width/2, metric_thresholds, width, label='Threshold')
        axs[1, 1].set_xticks(x)
        axs[1, 1].set_xticklabels(metric_names)
        axs[1, 1].set_title('Risk Metrics vs. Thresholds')
        axs[1, 1].legend()
        axs[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
    
    def _log_alert_to_file(self, alert_type: str, message: str) -> None:
        """
        Log alert to a local file as fallback when notification services fail.
        
        Args:
            alert_type: Type of alert (slack/email)
            message: Alert message
        """
        # Ensure the alerts directory exists
        alerts_dir = self.risk_report_dir / 'alerts'
        alerts_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = alerts_dir / f"{alert_type}_alert_{timestamp}.txt"
        
        try:
            with open(filename, 'w') as f:
                f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Type: {alert_type}\n")
                f.write(f"Message:\n{message}\n")
            logger.info(f"Alert logged to file: {filename}")
        except Exception as e:
            logger.error(f"Error logging alert to file: {e}")
    
    def _check_and_send_alerts(self, 
                              risk_status: Dict,
                              report_path: str):
        """
        Check risk status and send alerts if thresholds are exceeded.
        
        Args:
            risk_status: Risk metrics and status
            report_path: Path to risk report
        """
        # Check if any metrics have WARNING status
        warnings = [metric for metric, details in risk_status.items() 
                   if details['status'] == 'WARNING']
        
        if not warnings:
            return
            
        # Construct alert message
        message = f"⚠️ RISK ALERT - TimeFolio Portfolio ⚠️\n\n"
        message += f"The following risk metrics have exceeded thresholds:\n"
        
        for metric in warnings:
            message += f"• {metric.upper()}: {risk_status[metric]['value']:.4f} "
            message += f"(Threshold: {risk_status[metric]['threshold']:.4f})\n"
            
        message += f"\nSee full report at: {report_path}"
        
        # Send alerts based on configuration
        self._send_alerts(message, warnings)
    
    def _send_alerts(self, message: str, warning_types: List[str]):
        """
        Send alerts through configured channels.
        
        Args:
            message: Alert message
            warning_types: List of warning types that triggered the alert
        """
        severity = "high" if "mdd_12m" in warning_types else "medium"
        
        # Email alert
        if "email" in self.alert_config:
            try:
                self._send_email_alert(message, severity)
                logger.info("Email alert sent")
            except Exception as e:
                logger.error(f"Failed to send email alert: {e}")
        
        # Slack alert
        if "slack" in self.alert_config:
            try:
                self.send_slack_alert(message)
                logger.info("Slack alert sent")
            except Exception as e:
                logger.error(f"Failed to send Slack alert: {e}")
    
    def _send_email_alert(self, message: str, severity: str):
        """
        Send email alert.
        
        Args:
            message: Alert message
            severity: Alert severity
        """
        if "email" not in self.alert_config:
            return
            
        config = self.alert_config["email"]
        
        # Create message
        msg = MIMEMultipart()
        msg['From'] = config.get('from', 'timefolio-alert@example.com')
        msg['To'] = ', '.join(config.get('recipients', []))
        msg['Subject'] = f"[{severity.upper()}] TimeFolio Risk Alert"
        
        # Attach message body
        msg.attach(MIMEText(message, 'plain'))
        
        # Send email
        server = smtplib.SMTP(config.get('smtp_server', 'smtp.gmail.com'), 
                             config.get('smtp_port', 587))
        server.starttls()
        server.login(config.get('username'), config.get('password'))
        server.send_message(msg)
        server.quit()
    
    def send_slack_alert(self, message: str, channel: Optional[str] = None) -> bool:
        """
        Send alert to Slack channel with graceful fallback for missing credentials.
        
        Args:
            message: Alert message
            channel: Slack channel (overrides config)
            
        Returns:
            bool: Success status
        """
        try:
            # First check if Slack client is available
            try:
                from slack_sdk import WebClient
                from slack_sdk.errors import SlackApiError
            except ImportError:
                logger.warning("Slack SDK not installed. Alert logged locally only.")
                # Log the message to file as fallback
                self._log_alert_to_file("slack", message)
                return False
                
            # Then check if configuration exists
            if 'slack' not in self.alert_config:
                logger.warning("Slack configuration not found. Alert logged locally only.")
                self._log_alert_to_file("slack", message)
                return False
                
            slack_config = self.alert_config.get('slack', {})
            token = slack_config.get('token')
            default_channel = slack_config.get('channel', '#alerts')
            
            if not token:
                logger.warning("Slack API token not found. Alert logged locally only.")
                self._log_alert_to_file("slack", message)
                return False
                
            channel = channel or default_channel
            
            # Send the message
            client = WebClient(token=token)
            response = client.chat_postMessage(
                channel=channel,
                text=message
            )
            logger.info(f"Slack alert sent to {channel}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending Slack alert: {e}")
            # Log the message to file as fallback
            self._log_alert_to_file("slack", message)
            return False


def rolling_mdd(returns: pd.Series, window_days: int = 126, data_frequency: str = 'daily') -> pd.Series:
    """
    Convenience function to calculate rolling maximum drawdown.
    
    Args:
        returns: Daily portfolio returns
        window_days: Rolling window size in days
        data_frequency: Data frequency ('daily' or 'weekly')
    
    Returns:
        Series of rolling maximum drawdowns
    """
    monitor = RiskMonitor()
    return monitor.rolling_mdd(returns, window_days)


def check_hhi(weights: pd.Series, threshold: float = 0.20) -> Dict:
    """
    Convenience function to check HHI concentration.
    
    Args:
        weights: Portfolio weights
        threshold: HHI threshold
    
    Returns:
        Dictionary with HHI value and status
    """
    monitor = RiskMonitor(hhi_threshold=threshold)
    hhi = monitor.calculate_hhi(weights)
    return {
        'value': hhi,
        'threshold': threshold,
        'status': 'WARNING' if hhi > threshold else 'OK'
    }


def generate_risk_report(returns: pd.Series,
                        weights: pd.Series,
                        previous_weights: Optional[pd.Series] = None,
                        mdd_threshold_6m: float = 0.15,
                        mdd_threshold_12m: float = 0.25,
                        hhi_threshold: float = 0.20,
                        turnover_min: float = 0.05,
                        risk_report_dir: str = 'risk_reports',
                        alert_config_file: Optional[str] = None,
                        additional_info: Optional[Dict] = None,
                        data_frequency: str = 'daily') -> str:
    """
    Convenience function to generate a full risk report.
    
    Args:
        returns: Historical daily returns
        weights: Current portfolio weights
        previous_weights: Previous portfolio weights
        mdd_threshold_6m: Maximum drawdown threshold for 6-month window
        mdd_threshold_12m: Maximum drawdown threshold for 12-month window
        hhi_threshold: HHI concentration threshold
        turnover_min: Minimum required turnover
        risk_report_dir: Directory to save risk reports
        alert_config_file: Path to alert configuration file
        additional_info: Additional information to include in report
    
    Returns:
        Path to generated report
    """
    monitor = RiskMonitor(
        mdd_threshold_6m=mdd_threshold_6m,
        mdd_threshold_12m=mdd_threshold_12m,
        hhi_weight_threshold=hhi_threshold,  # Changed from hhi_threshold
        hhi_return_threshold=hhi_threshold,  # Using same threshold for both weight and return HHI
        turnover_min=turnover_min,
        risk_report_dir=risk_report_dir,
        alert_config_file=alert_config_file
    )
    
    return monitor.generate_risk_report(
        returns=returns,
        weights=weights,
        previous_weights=previous_weights,
        additional_info=additional_info
    )

def calculate_benchmark_tracking(date_str=None):
    """
    Calculate tracking error and other benchmark metrics
    
    Args:
        date_str: Date string in ISO format (YYYY-MM-DD)
                 If None, use today's date
    
    Returns:
        Dictionary with benchmark metrics
    """
    import pathlib
    import pandas as pd
    from datetime import datetime
    
    CACHE = pathlib.Path.home() / "KRX_cache"
    today = date_str or datetime.now().strftime("%Y-%m-%d")
    
    try:
        # 1. Load portfolio data
        pf = pd.read_parquet(CACHE / f"{today}.parquet")
        pf = pf.pivot(index="date", columns="code", values="close")
        pf_ret = pf.pct_change().mean(axis=1)
        
        # 2. Load benchmark index
        kospi = pd.read_parquet(CACHE / f"KOSPI_{today}.parquet")
        kospi_ret = kospi["close"].pct_change()
        
        # 3. Calculate benchmark metrics
        tracking_err = (pf_ret - kospi_ret).std() * (252**0.5)
        info_ratio = ((pf_ret - kospi_ret).mean() * 252) / tracking_err
        beta = pf_ret.cov(kospi_ret) / kospi_ret.var()
        
        metrics = {
            "tracking_error": tracking_err,
            "information_ratio": info_ratio,
            "beta": beta
        }
        
        print(f"Tracking Error vs KOSPI: {tracking_err:.4%}")
        print(f"Information Ratio: {info_ratio:.4f}")
        print(f"Portfolio Beta: {beta:.4f}")
        
        return metrics
    except Exception as e:
        print(f"Error calculating benchmark metrics: {e}")
        return {}


if __name__ == "__main__":
    import argparse
    from cvar_optimizer_revamped import Config, DataManager
    
    parser = argparse.ArgumentParser(description="Generate portfolio risk report")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--weights", required=True, help="Path to portfolio weights CSV")
    parser.add_argument("--previous-weights", help="Path to previous portfolio weights CSV")
    parser.add_argument("--alert-config", help="Path to alert configuration file")
    parser.add_argument("--output-dir", default="risk_reports", help="Output directory for risk reports")
    args = parser.parse_args()
    
    # Load config and data manager
    config = Config(args.config)
    data_manager = DataManager(config)
    
    # Load weights
    weights = pd.read_csv(args.weights, index_col=0).iloc[:, 0]
    
    # Load previous weights if provided
    previous_weights = None
    if args.previous_weights:
        previous_weights = pd.read_csv(args.previous_weights, index_col=0).iloc[:, 0]
    
    # Get historical returns
    start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")
    returns = data_manager.get_returns(start_date, end_date)
    
    # Calculate portfolio returns
    portfolio_returns = returns[weights.index].multiply(weights).sum(axis=1)
    
    # Generate risk report
    report_path = generate_risk_report(
        returns=portfolio_returns,
        weights=weights,
        previous_weights=previous_weights,
        risk_report_dir=args.output_dir,
        alert_config_file=args.alert_config
    )
    
    print(f"Risk report generated: {report_path}")
