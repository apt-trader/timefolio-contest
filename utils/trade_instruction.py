"""
Trade Instruction Generator

Generates professional institutional-level trade instructions by comparing
current portfolio with prior portfolio.
"""

import pandas as pd
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class TradeInstructionGenerator:
    """
    Generates trade instructions comparing current vs prior portfolio.
    
    Produces institutional-level instruction notes with:
    - New positions (BUY)
    - Closed positions (SELL)
    - Increased positions (INCREASE)
    - Decreased positions (DECREASE)
    - Unchanged positions (HOLD)
    """
    
    def __init__(
        self,
        db_path: str = "db/krx_data.db",
        output_dir: str = "output"
    ):
        self.db_path = db_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def _load_portfolio(self, filepath: Path) -> pd.DataFrame:
        """Load portfolio CSV file."""
        if not filepath.exists():
            return pd.DataFrame(columns=['ticker', 'weight', 'sector'])
        
        df = pd.read_csv(filepath)
        df['ticker'] = df['ticker'].astype(str).str.zfill(6)
        return df
    
    def _get_stock_names(self, tickers: List[str]) -> Dict[str, str]:
        """Fetch stock names from database."""
        try:
            conn = sqlite3.connect(self.db_path)
            placeholders = ','.join(['?' for _ in tickers])
            query = f"""
                SELECT DISTINCT code, name 
                FROM daily_prices 
                WHERE code IN ({placeholders})
            """
            df = pd.read_sql_query(query, conn, params=tickers)
            conn.close()
            df['code'] = df['code'].astype(str).str.zfill(6)
            return dict(zip(df['code'], df['name']))
        except Exception as e:
            logger.warning(f"Could not fetch stock names: {e}")
            return {}
    
    def _get_market_caps(self, tickers: List[str]) -> Dict[str, float]:
        """Fetch latest market caps from database."""
        try:
            conn = sqlite3.connect(self.db_path)
            placeholders = ','.join(['?' for _ in tickers])
            query = f"""
                SELECT code, market_cap 
                FROM daily_prices 
                WHERE date = (SELECT MAX(date) FROM daily_prices)
                AND code IN ({placeholders})
            """
            df = pd.read_sql_query(query, conn, params=tickers)
            conn.close()
            df['code'] = df['code'].astype(str).str.zfill(6)
            return dict(zip(df['code'], df['market_cap']))
        except Exception as e:
            logger.warning(f"Could not fetch market caps: {e}")
            return {}
    
    def _find_prior_portfolio(self, current_date: str) -> Optional[Path]:
        """Find the most recent portfolio file before current date."""
        current_dt = datetime.strptime(current_date, "%Y%m%d")
        
        portfolio_files = list(self.output_dir.glob("portfolio_*.csv"))
        prior_files = []
        
        for f in portfolio_files:
            try:
                date_str = f.stem.replace("portfolio_", "")
                file_dt = datetime.strptime(date_str, "%Y%m%d")
                if file_dt < current_dt:
                    prior_files.append((file_dt, f))
            except ValueError:
                continue
        
        if not prior_files:
            return None
        
        # Return most recent prior file
        prior_files.sort(key=lambda x: x[0], reverse=True)
        return prior_files[0][1]
    
    def generate_instructions(
        self,
        current_portfolio: pd.DataFrame,
        current_date: str,
        prior_portfolio: Optional[pd.DataFrame] = None
    ) -> str:
        """
        Generate trade instruction note.
        
        Args:
            current_portfolio: Current portfolio DataFrame
            current_date: Date string YYYYMMDD
            prior_portfolio: Optional prior portfolio DataFrame
            
        Returns:
            Formatted instruction note string
        """
        # Find prior portfolio if not provided
        if prior_portfolio is None:
            prior_path = self._find_prior_portfolio(current_date)
            if prior_path:
                prior_portfolio = self._load_portfolio(prior_path)
                prior_date = prior_path.stem.replace("portfolio_", "")
            else:
                prior_portfolio = pd.DataFrame(columns=['ticker', 'weight', 'sector'])
                prior_date = "N/A"
        else:
            prior_date = "Previous"
        
        # Normalize ticker format
        current_portfolio = current_portfolio.copy()
        current_portfolio['ticker'] = current_portfolio['ticker'].astype(str).str.zfill(6)
        
        # Get all tickers
        all_tickers = list(set(
            current_portfolio['ticker'].tolist() + 
            prior_portfolio['ticker'].tolist()
        ))
        
        # Fetch stock names and market caps
        stock_names = self._get_stock_names(all_tickers)
        market_caps = self._get_market_caps(all_tickers)
        
        # Create weight dictionaries
        current_weights = dict(zip(current_portfolio['ticker'], current_portfolio['weight']))
        prior_weights = dict(zip(prior_portfolio['ticker'], prior_portfolio['weight']))
        current_sectors = dict(zip(current_portfolio['ticker'], current_portfolio['sector']))
        prior_sectors = dict(zip(prior_portfolio['ticker'], prior_portfolio['sector']))
        
        # Classify actions
        new_positions = []      # BUY
        closed_positions = []   # SELL
        increased = []          # INCREASE
        decreased = []          # DECREASE
        unchanged = []          # HOLD
        
        threshold = 0.005  # 0.5% threshold for change
        
        for ticker in all_tickers:
            curr_w = current_weights.get(ticker, 0)
            prior_w = prior_weights.get(ticker, 0)
            sector = current_sectors.get(ticker) or prior_sectors.get(ticker, "?")
            name = stock_names.get(ticker, "")
            cap = market_caps.get(ticker, 0) / 1e12  # in trillion
            
            delta = curr_w - prior_w
            
            if prior_w == 0 and curr_w > 0:
                new_positions.append((ticker, name, sector, curr_w, cap))
            elif curr_w == 0 and prior_w > 0:
                closed_positions.append((ticker, name, sector, prior_w, cap))
            elif delta > threshold:
                increased.append((ticker, name, sector, prior_w, curr_w, delta, cap))
            elif delta < -threshold:
                decreased.append((ticker, name, sector, prior_w, curr_w, delta, cap))
            else:
                unchanged.append((ticker, name, sector, curr_w, cap))
        
        # Sort by weight
        new_positions.sort(key=lambda x: -x[3])
        closed_positions.sort(key=lambda x: -x[3])
        increased.sort(key=lambda x: -x[5])
        decreased.sort(key=lambda x: x[5])
        unchanged.sort(key=lambda x: -x[3])
        
        # Generate instruction note
        lines = []
        lines.append("=" * 80)
        lines.append("PORTFOLIO REBALANCING INSTRUCTION")
        lines.append("=" * 80)
        lines.append("")
        lines.append(f"Date:           {current_date[:4]}-{current_date[4:6]}-{current_date[6:]}")
        lines.append(f"Prior Date:     {prior_date[:4]}-{prior_date[4:6]}-{prior_date[6:]}" if prior_date != "N/A" else f"Prior Date:     {prior_date}")
        lines.append(f"Generated:      {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # Summary
        lines.append("-" * 80)
        lines.append("SUMMARY")
        lines.append("-" * 80)
        lines.append(f"  New Positions (BUY):      {len(new_positions)}")
        lines.append(f"  Closed Positions (SELL):  {len(closed_positions)}")
        lines.append(f"  Increased Positions:      {len(increased)}")
        lines.append(f"  Decreased Positions:      {len(decreased)}")
        lines.append(f"  Unchanged Positions:      {len(unchanged)}")
        lines.append(f"  Total Current Positions:  {len(current_portfolio)}")
        lines.append("")
        
        # Calculate turnover
        turnover = sum(abs(current_weights.get(t, 0) - prior_weights.get(t, 0)) for t in all_tickers) / 2
        lines.append(f"  Estimated Turnover:       {turnover:.1%}")
        lines.append("")
        
        # NEW POSITIONS (BUY)
        if new_positions:
            lines.append("-" * 80)
            lines.append("NEW POSITIONS - BUY")
            lines.append("-" * 80)
            lines.append(f"{'Ticker':<8} {'Name':<20} {'Sector':<6} {'Weight':>10} {'Mkt Cap':>12}")
            lines.append("-" * 60)
            for ticker, name, sector, weight, cap in new_positions:
                name_short = name[:18] if len(name) > 18 else name
                lines.append(f"{ticker:<8} {name_short:<20} {sector:<6} {weight:>9.2%} {cap:>10.2f}T")
            lines.append("")
        
        # CLOSED POSITIONS (SELL)
        if closed_positions:
            lines.append("-" * 80)
            lines.append("CLOSED POSITIONS - SELL (FULL EXIT)")
            lines.append("-" * 80)
            lines.append(f"{'Ticker':<8} {'Name':<20} {'Sector':<6} {'Prior Wt':>10} {'Mkt Cap':>12}")
            lines.append("-" * 60)
            for ticker, name, sector, weight, cap in closed_positions:
                name_short = name[:18] if len(name) > 18 else name
                lines.append(f"{ticker:<8} {name_short:<20} {sector:<6} {weight:>9.2%} {cap:>10.2f}T")
            lines.append("")
        
        # INCREASED POSITIONS
        if increased:
            lines.append("-" * 80)
            lines.append("INCREASED POSITIONS - ADD")
            lines.append("-" * 80)
            lines.append(f"{'Ticker':<8} {'Name':<20} {'Sector':<6} {'Prior':>8} {'Current':>8} {'Change':>8}")
            lines.append("-" * 60)
            for ticker, name, sector, prior_w, curr_w, delta, cap in increased:
                name_short = name[:18] if len(name) > 18 else name
                lines.append(f"{ticker:<8} {name_short:<20} {sector:<6} {prior_w:>7.2%} {curr_w:>7.2%} {delta:>+7.2%}")
            lines.append("")
        
        # DECREASED POSITIONS
        if decreased:
            lines.append("-" * 80)
            lines.append("DECREASED POSITIONS - REDUCE")
            lines.append("-" * 80)
            lines.append(f"{'Ticker':<8} {'Name':<20} {'Sector':<6} {'Prior':>8} {'Current':>8} {'Change':>8}")
            lines.append("-" * 60)
            for ticker, name, sector, prior_w, curr_w, delta, cap in decreased:
                name_short = name[:18] if len(name) > 18 else name
                lines.append(f"{ticker:<8} {name_short:<20} {sector:<6} {prior_w:>7.2%} {curr_w:>7.2%} {delta:>+7.2%}")
            lines.append("")
        
        # UNCHANGED POSITIONS
        if unchanged:
            lines.append("-" * 80)
            lines.append("UNCHANGED POSITIONS - HOLD")
            lines.append("-" * 80)
            lines.append(f"{'Ticker':<8} {'Name':<20} {'Sector':<6} {'Weight':>10}")
            lines.append("-" * 60)
            for ticker, name, sector, weight, cap in unchanged:
                name_short = name[:18] if len(name) > 18 else name
                lines.append(f"{ticker:<8} {name_short:<20} {sector:<6} {weight:>9.2%}")
            lines.append("")
        
        # FINAL PORTFOLIO
        lines.append("-" * 80)
        lines.append("FINAL PORTFOLIO COMPOSITION")
        lines.append("-" * 80)
        lines.append(f"{'Ticker':<8} {'Name':<20} {'Sector':<6} {'Weight':>10} {'Mkt Cap':>12}")
        lines.append("-" * 60)
        for _, row in current_portfolio.sort_values('weight', ascending=False).iterrows():
            ticker = row['ticker']
            name = stock_names.get(ticker, "")[:18]
            sector = row['sector']
            weight = row['weight']
            cap = market_caps.get(ticker, 0) / 1e12
            lines.append(f"{ticker:<8} {name:<20} {sector:<6} {weight:>9.2%} {cap:>10.2f}T")
        lines.append("-" * 60)
        lines.append(f"{'TOTAL':<8} {'':<20} {'':<6} {current_portfolio['weight'].sum():>9.2%}")
        lines.append("")
        
        # SECTOR ALLOCATION
        lines.append("-" * 80)
        lines.append("SECTOR ALLOCATION")
        lines.append("-" * 80)
        sector_weights = current_portfolio.groupby('sector')['weight'].sum().sort_values(ascending=False)
        for sector, weight in sector_weights.items():
            lines.append(f"  {sector:<20} {weight:>8.2%}")
        lines.append("")
        
        lines.append("=" * 80)
        lines.append("END OF INSTRUCTION")
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def save_instructions(
        self,
        current_portfolio: pd.DataFrame,
        current_date: str,
        filename: Optional[str] = None
    ) -> Path:
        """
        Generate and save trade instructions to file.
        
        Returns:
            Path to saved instruction file
        """
        instructions = self.generate_instructions(current_portfolio, current_date)
        
        if filename is None:
            filename = f"trade_instruction_{current_date}.txt"
        
        filepath = self.output_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(instructions)
        
        logger.info(f"Trade instructions saved to {filepath}")
        return filepath


def generate_trade_instructions(
    portfolio_path: str,
    output_dir: str = "output",
    db_path: str = "db/krx_data.db"
) -> str:
    """
    Convenience function to generate trade instructions from portfolio file.
    
    Args:
        portfolio_path: Path to current portfolio CSV
        output_dir: Output directory
        db_path: Database path
        
    Returns:
        Path to saved instruction file
    """
    portfolio = pd.read_csv(portfolio_path)
    date_str = Path(portfolio_path).stem.replace("portfolio_", "")
    
    generator = TradeInstructionGenerator(db_path=db_path, output_dir=output_dir)
    filepath = generator.save_instructions(portfolio, date_str)
    
    return str(filepath)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        portfolio_path = sys.argv[1]
    else:
        # Find latest portfolio
        output_dir = Path("output")
        portfolios = sorted(output_dir.glob("portfolio_*.csv"), reverse=True)
        if portfolios:
            portfolio_path = str(portfolios[0])
        else:
            print("No portfolio files found")
            sys.exit(1)
    
    filepath = generate_trade_instructions(portfolio_path)
    print(f"Instructions saved to: {filepath}")
    
    # Also print to console
    with open(filepath, 'r') as f:
        print(f.read())
