"""
POLARS ADAPTER
==============
Data format conversion between Pandas (TimeFolio) and Polars (Toraniko).

Toraniko expects data in long format:
- returns_df: | date | symbol | asset_returns |
- mkt_cap_df: | date | symbol | market_cap |
- sector_df:  | date | symbol | sector_1 | sector_2 | ... |
- style_df:   | date | symbol | style_1 | style_2 | ... |

TimeFolio uses wide format:
- returns: DataFrame with date index, ticker columns
- market_caps: DataFrame with date index, ticker columns

Author: TimeFolio-Toraniko Integration
Date: 2026-01-23
"""

import pandas as pd
import numpy as np
import polars as pl
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class PolarsAdapter:
    """
    Adapter for converting between Pandas (TimeFolio) and Polars (Toraniko) data formats.
    """
    
    def __init__(self, sector_mapping: Optional[Dict[str, str]] = None):
        """
        Initialize adapter.
        
        Args:
            sector_mapping: Dict mapping ticker -> sector name
        """
        self.sector_mapping = sector_mapping or {}
        self._sector_columns: Optional[List[str]] = None
    
    def set_sector_mapping(self, sector_mapping: Dict[str, str]) -> None:
        """Update sector mapping."""
        self.sector_mapping = sector_mapping
        self._sector_columns = None  # Reset cached columns
    
    # ==================== Pandas to Polars ====================
    
    def returns_to_polars(
        self,
        returns_df: pd.DataFrame,
        date_col: str = 'date',
        symbol_col: str = 'symbol',
        returns_col: str = 'asset_returns'
    ) -> pl.DataFrame:
        """
        Convert TimeFolio returns (wide format) to Toraniko format (long format).
        
        Args:
            returns_df: DataFrame with date index, ticker columns, return values
            
        Returns:
            Polars DataFrame with columns: | date | symbol | asset_returns |
        """
        if returns_df.empty:
            return pl.DataFrame(schema={date_col: pl.Date, symbol_col: pl.Utf8, returns_col: pl.Float64})
        
        # Reset index if date is index
        if isinstance(returns_df.index, pd.DatetimeIndex):
            returns_df = returns_df.reset_index()
            returns_df.columns = [date_col] + list(returns_df.columns[1:])
        
        # Melt to long format
        id_vars = [date_col] if date_col in returns_df.columns else []
        if not id_vars:
            # Date might be the first column with different name
            date_col_actual = returns_df.columns[0]
            id_vars = [date_col_actual]
        
        melted = returns_df.melt(
            id_vars=id_vars,
            var_name=symbol_col,
            value_name=returns_col
        )
        
        # Rename date column if needed
        if id_vars[0] != date_col:
            melted = melted.rename(columns={id_vars[0]: date_col})
        
        # Convert to Polars
        pl_df = pl.from_pandas(melted)
        
        # Ensure proper types
        pl_df = pl_df.with_columns([
            pl.col(date_col).cast(pl.Date),
            pl.col(symbol_col).cast(pl.Utf8),
            pl.col(returns_col).cast(pl.Float64)
        ])
        
        # Drop nulls
        pl_df = pl_df.drop_nulls(subset=[returns_col])
        
        logger.debug(f"Converted returns: {len(pl_df)} rows, {pl_df[symbol_col].n_unique()} symbols")
        return pl_df
    
    def market_caps_to_polars(
        self,
        market_caps_df: pd.DataFrame,
        date_col: str = 'date',
        symbol_col: str = 'symbol',
        cap_col: str = 'market_cap'
    ) -> pl.DataFrame:
        """
        Convert TimeFolio market caps (wide format) to Toraniko format (long format).
        
        Args:
            market_caps_df: DataFrame with date index, ticker columns, market cap values
            
        Returns:
            Polars DataFrame with columns: | date | symbol | market_cap |
        """
        if market_caps_df.empty:
            return pl.DataFrame(schema={date_col: pl.Date, symbol_col: pl.Utf8, cap_col: pl.Float64})
        
        # Reset index if date is index
        if isinstance(market_caps_df.index, pd.DatetimeIndex):
            market_caps_df = market_caps_df.reset_index()
            market_caps_df.columns = [date_col] + list(market_caps_df.columns[1:])
        
        # Melt to long format
        id_vars = [date_col] if date_col in market_caps_df.columns else [market_caps_df.columns[0]]
        
        melted = market_caps_df.melt(
            id_vars=id_vars,
            var_name=symbol_col,
            value_name=cap_col
        )
        
        if id_vars[0] != date_col:
            melted = melted.rename(columns={id_vars[0]: date_col})
        
        pl_df = pl.from_pandas(melted)
        
        pl_df = pl_df.with_columns([
            pl.col(date_col).cast(pl.Date),
            pl.col(symbol_col).cast(pl.Utf8),
            pl.col(cap_col).cast(pl.Float64)
        ])
        
        pl_df = pl_df.drop_nulls(subset=[cap_col])
        
        logger.debug(f"Converted market caps: {len(pl_df)} rows")
        return pl_df
    
    def create_sector_scores(
        self,
        symbols: List[str],
        dates: List[pd.Timestamp],
        date_col: str = 'date',
        symbol_col: str = 'symbol'
    ) -> pl.DataFrame:
        """
        Create one-hot encoded sector scores from sector mapping.
        
        Args:
            symbols: List of ticker symbols
            dates: List of dates
            
        Returns:
            Polars DataFrame with columns: | date | symbol | sector_1 | sector_2 | ... |
            Each sector column contains 0 or 1.
        """
        if not self.sector_mapping:
            logger.warning("No sector mapping provided, using single 'Market' sector")
            # Create single market sector
            rows = []
            for date in dates:
                for symbol in symbols:
                    rows.append({date_col: date, symbol_col: symbol, 'Market': 1})
            return pl.DataFrame(rows)
        
        # Get unique sectors
        sectors = sorted(set(self.sector_mapping.values()))
        self._sector_columns = sectors
        
        # Create one-hot encoding
        rows = []
        for date in dates:
            for symbol in symbols:
                row = {date_col: date, symbol_col: symbol}
                symbol_sector = self.sector_mapping.get(symbol, None)
                for sector in sectors:
                    row[sector] = 1 if sector == symbol_sector else 0
                rows.append(row)
        
        pl_df = pl.DataFrame(rows)
        
        # Cast types
        pl_df = pl_df.with_columns([
            pl.col(date_col).cast(pl.Date),
            pl.col(symbol_col).cast(pl.Utf8)
        ] + [pl.col(s).cast(pl.Int64) for s in sectors])
        
        logger.debug(f"Created sector scores: {len(sectors)} sectors, {len(pl_df)} rows")
        return pl_df
    
    def style_scores_to_polars(
        self,
        style_df: pd.DataFrame,
        date_col: str = 'date',
        symbol_col: str = 'symbol'
    ) -> pl.DataFrame:
        """
        Convert style scores DataFrame to Polars format.
        
        Args:
            style_df: DataFrame with columns [date, symbol, style_1, style_2, ...]
            
        Returns:
            Polars DataFrame with same structure
        """
        if style_df.empty:
            return pl.DataFrame()
        
        pl_df = pl.from_pandas(style_df)
        
        # Ensure date and symbol types
        if date_col in pl_df.columns:
            pl_df = pl_df.with_columns(pl.col(date_col).cast(pl.Date))
        if symbol_col in pl_df.columns:
            pl_df = pl_df.with_columns(pl.col(symbol_col).cast(pl.Utf8))
        
        return pl_df
    
    # ==================== Polars to Pandas ====================
    
    def polars_to_pandas_series(
        self,
        pl_df: pl.DataFrame,
        value_col: str,
        symbol_col: str = 'symbol',
        date: Optional[pd.Timestamp] = None
    ) -> pd.Series:
        """
        Convert Polars DataFrame to Pandas Series indexed by symbol.
        
        Args:
            pl_df: Polars DataFrame with symbol and value columns
            value_col: Name of value column
            symbol_col: Name of symbol column
            date: If provided, filter to this date first
            
        Returns:
            Pandas Series indexed by symbol
        """
        if pl_df.is_empty():
            return pd.Series(dtype=float)
        
        if date is not None and 'date' in pl_df.columns:
            pl_df = pl_df.filter(pl.col('date') == date)
        
        pdf = pl_df.select([symbol_col, value_col]).to_pandas()
        return pdf.set_index(symbol_col)[value_col]
    
    def polars_to_pandas_wide(
        self,
        pl_df: pl.DataFrame,
        value_col: str,
        date_col: str = 'date',
        symbol_col: str = 'symbol'
    ) -> pd.DataFrame:
        """
        Convert Polars long format to Pandas wide format.
        
        Args:
            pl_df: Polars DataFrame in long format
            value_col: Name of value column
            
        Returns:
            Pandas DataFrame with date index, symbol columns
        """
        if pl_df.is_empty():
            return pd.DataFrame()
        
        # Pivot to wide format
        wide_pl = pl_df.pivot(
            values=value_col,
            index=date_col,
            on=symbol_col
        )
        
        pdf = wide_pl.to_pandas()
        pdf = pdf.set_index(date_col)
        pdf.index = pd.to_datetime(pdf.index)
        
        return pdf
    
    def factor_returns_to_pandas(
        self,
        factor_returns_pl: pl.DataFrame,
        date_col: str = 'date'
    ) -> pd.DataFrame:
        """
        Convert Toraniko factor returns to Pandas DataFrame.
        
        Args:
            factor_returns_pl: Polars DataFrame with date and factor return columns
            
        Returns:
            Pandas DataFrame with date index, factor columns
        """
        if factor_returns_pl.is_empty():
            return pd.DataFrame()
        
        pdf = factor_returns_pl.to_pandas()
        if date_col in pdf.columns:
            pdf = pdf.set_index(date_col)
            pdf.index = pd.to_datetime(pdf.index)
        
        return pdf
    
    def residuals_to_pandas(
        self,
        residuals_pl: pl.DataFrame,
        date_col: str = 'date'
    ) -> pd.DataFrame:
        """
        Convert Toraniko residuals to Pandas DataFrame (wide format).
        
        Args:
            residuals_pl: Polars DataFrame with date and symbol residual columns
            
        Returns:
            Pandas DataFrame with date index, symbol columns
        """
        if residuals_pl.is_empty():
            return pd.DataFrame()
        
        pdf = residuals_pl.to_pandas()
        if date_col in pdf.columns:
            pdf = pdf.set_index(date_col)
            pdf.index = pd.to_datetime(pdf.index)
        
        return pdf
    
    # ==================== Utility Methods ====================
    
    def get_sector_columns(self) -> List[str]:
        """Get list of sector column names."""
        return self._sector_columns or []
    
    def prepare_toraniko_inputs(
        self,
        returns: pd.DataFrame,
        market_caps: pd.DataFrame,
        style_scores: pd.DataFrame,
        sector_mapping: Optional[Dict[str, str]] = None
    ) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """
        Prepare all inputs for Toraniko's estimate_factor_returns().
        
        Args:
            returns: TimeFolio returns (wide format)
            market_caps: TimeFolio market caps (wide format)
            style_scores: Style scores DataFrame (long format with date, symbol, style columns)
            sector_mapping: Optional sector mapping to override
            
        Returns:
            Tuple of (returns_df, mkt_cap_df, sector_df, style_df) in Polars format
        """
        if sector_mapping:
            self.set_sector_mapping(sector_mapping)
        
        # Convert returns and market caps
        returns_pl = self.returns_to_polars(returns)
        mkt_cap_pl = self.market_caps_to_polars(market_caps)
        
        # Get common symbols and dates
        symbols = list(set(returns_pl['symbol'].unique().to_list()) & 
                      set(mkt_cap_pl['symbol'].unique().to_list()))
        dates = returns_pl['date'].unique().to_list()
        
        # Create sector scores
        sector_pl = self.create_sector_scores(symbols, dates)
        
        # Convert style scores
        style_pl = self.style_scores_to_polars(style_scores)
        
        logger.info(f"Prepared Toraniko inputs: {len(symbols)} symbols, {len(dates)} dates")
        
        return returns_pl, mkt_cap_pl, sector_pl, style_pl


def create_adapter(sector_mapping: Optional[Dict[str, str]] = None) -> PolarsAdapter:
    """Factory function to create PolarsAdapter."""
    return PolarsAdapter(sector_mapping)
