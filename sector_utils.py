import logging
import sqlite3
from pathlib import Path
from typing import Dict, Optional, Union
import pandas as pd
import os

logger = logging.getLogger(__name__)

class SectorLimits:
    """
    Single source of truth for sector weight limits.
    
    Calculates sector weight limits based on market cap weights from the database.
    Applies the rule: max(2x market weight, 10%)
    """
    
    def __init__(self, db_path: Union[str, Path], fallback_limits: Optional[Dict[str, float]] = None):
        """
        Initialize SectorLimits with database connection.
        
        Args:
            db_path: Path to SQLite database
            fallback_limits: Optional fallback limits if database is unavailable
        """
        self.db_path = str(db_path)
        self.fallback_limits = fallback_limits or {"default": 0.10}
        self.limits: Dict[str, float] = {}
        self._load_limits()
        
    def _load_limits(self) -> None:
        """Load sector weights from database and calculate limits"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Query to get market cap by sector
                query = """
                SELECT sector, SUM(market_cap) as sector_mcap
                FROM stocks
                WHERE sector IS NOT NULL
                GROUP BY sector
                """
                df = pd.read_sql(query, conn)
                
                if df.empty:
                    logger.warning("No sector data found in database, using fallback limits")
                    self.limits = self.fallback_limits.copy()
                    return
                
                # Calculate weights and apply constraints
                total_mcap = df['sector_mcap'].sum()
                if total_mcap > 0:
                    for _, row in df.iterrows():
                        weight = row['sector_mcap'] / total_mcap
                        # Apply rule: max(2x market weight, 10%)
                        self.limits[row['sector']] = max(2 * weight, 0.10)
                else:
                    logger.warning("Total market cap is zero, using fallback limits")
                    self.limits = self.fallback_limits.copy()
                    
                # Add default if not already set
                if "default" not in self.limits:
                    self.limits["default"] = 0.10
                    
        except Exception as e:
            logger.error(f"Failed to load sector limits from database: {e}")
            self.limits = self.fallback_limits.copy()
            
    def get_limit(self, sector: str) -> float:
        """
        Get weight limit for a sector
        
        Args:
            sector: Sector name
            
        Returns:
            float: Maximum allowed weight for the sector (between 0.10 and 1.0)
        """
        return self.limits.get(sector, self.limits.get("default", 0.10))
    
    def refresh(self) -> None:
        """Refresh sector limits from database"""
        self.limits.clear()
        self._load_limits()