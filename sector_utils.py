import logging
from pathlib import Path
from typing import Dict, Optional
import pandas as pd

logger = logging.getLogger(__name__)

class SectorLimits:
    """Single source of truth for sector weight limits"""
    
    def __init__(self, market_weights_file: Path):
        self.limits: Dict[str, float] = {}
        self._load_limits(market_weights_file)
        
    def _load_limits(self, file_path: Path) -> None:
        try:
            with open(file_path) as f:
                current_date = None
                for line in f:
                    line = line.strip()
                    if line.startswith('[') and line.endswith(']'):
                        current_date = line[1:-1]
                    elif line and not line.startswith('#'):
                        parts = line.split()
                        if len(parts) >= 2:
                            sector = parts[0]
                            weight = float(parts[1]) / 100.0
                            # Consistent rule: max(2x market weight, 10%)
                            self.limits[sector] = max(2 * weight, 0.10)
        except Exception as e:
            logger.error(f"Failed to load sector limits: {e}")
            self.limits = {"default": 0.10}
            
    def get_limit(self, sector: str) -> float:
        """Get limit for sector with fallback to default"""
        return self.limits.get(sector, self.limits.get("default", 0.10))