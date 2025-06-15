# config.py
import yaml
import logging
import os
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file at the start
load_dotenv()
logger = logging.getLogger(__name__)

class Config:
    """
    Handles loading configuration from a YAML file and environment variables.
    """
    def __init__(self, config_path: str = 'config/config.yaml'):
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            logger.error(f"Configuration file not found at: {self.config_path}")
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        try:
            with open(self.config_path, 'r') as f:
                self._raw_config: Dict[str, Any] = yaml.safe_load(f)
            logger.info(f"Successfully loaded configuration from {self.config_path}")
        except Exception as e:
            logger.error(f"Error parsing YAML file {self.config_path}: {e}")
            raise

        self._load_settings()

    def _load_settings(self):
        """
        Loads settings from YAML and enriches with secrets from .env file.
        """
        # --- Data & Path Settings ---
        data_settings = self._raw_config.get('data_settings', {})
        self.stock_universe_file: str = data_settings.get('stock_universe_file', 'data/sector_universe.csv')
        self.market_sectors_file: str = data_settings.get('market_sectors_file', 'data/market_sectors.csv')
        self.db_path: str = data_settings.get('db_path', 'krx_data.db')
        self.start_date: str = data_settings.get('start_date', '2022-01-01')
        self.end_date: str = data_settings.get('end_date', '2025-09-30')

        # --- Optimization Settings ---
        self.optimization_settings: Dict[str, Any] = self._raw_config.get('optimization', {})

        # --- Factor Engine Settings ---
        self.factor_settings: Dict[str, Any] = self._raw_config.get('factor_engine', {})

        # --- Risk Management Settings ---
        self.risk_management: Dict[str, Any] = self._raw_config.get('risk_management', {})
        
        # --- Data Fetcher Settings (Enriched with API Keys) ---
        self.fetcher_settings: Dict[str, Any] = self._raw_config.get('data_fetch', {})
        # Load API keys from environment and add them to the settings dict
        self.fetcher_settings['dart_api_key'] = os.getenv('DART_API_KEY')
        self.fetcher_settings['fred_api_key'] = os.getenv('FRED_API_KEY')
        
        # Validate that keys were loaded
        if not self.fetcher_settings['dart_api_key']:
            logger.warning("DART_API_KEY not found in .env file.")
        if not self.fetcher_settings['fred_api_key']:
            logger.warning("FRED_API_KEY not found in .env file.")

        logger.info("Configuration loaded and enriched with API keys from .env")