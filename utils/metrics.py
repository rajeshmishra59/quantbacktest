# quant_backtesting_project/utils/metrics.py
# Performance metrics calculate karne ke liye.

import pandas as pd
import numpy as np

def calculate_performance_metrics(trades_df: pd.DataFrame, equity_curve_df: pd.DataFrame, initial_capital: float):
    """
    Ek poore backtest run ke liye professional metrics calculate karta hai.
    """
    if trades_df.empty:
        return {"message": "No trades were executed."}

    # PnL Metrics
    total_pnl = trades_df['pnl'].sum()
    gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
    gross_loss = trades_df[trades_df['pnl'] < 0]['pnl'].sum()
    profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else np.inf
    
    # Trade Stats
    total_trades = len(trades_df)
    winning_trades = len(trades_df[trades_df['pnl'] > 0])
    losing_trades = total_trades - winning_trades
    win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
    
    # Equity Curve Metrics
    equity_curve = equity_curve_df['equity']
    total_return_pct = ((equity_curve.iloc[-1] - initial_capital) / initial_capital) * 100
    
    # Max Drawdown
    peak = equity_curve.cummax()
    drawdown = (equity_curve - peak) / peak
    max_drawdown_pct = abs(drawdown.min()) * 100
    
    # Sharpe Ratio (assuming daily data for simplicity, risk-free rate = 0)
    daily_returns = equity_curve.pct_change().dropna()
    sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() != 0 else 0

    return {
        "Total Return %": round(total_return_pct, 2),
        "Max Drawdown %": round(max_drawdown_pct, 2),
        "Sharpe Ratio": round(sharpe_ratio, 2),
        "Profit Factor": round(profit_factor, 2),
        "Win Rate %": round(win_rate, 2),
        "Total Trades": total_trades,
        "Total PnL": round(total_pnl, 2),
    }

