# quantbacktest/utils/metrics.py
import pandas as pd

class Metrics:
    @staticmethod
    def calculate(trade_log):
        if not trade_log:
            return {}
        df = pd.DataFrame(trade_log)
        total_pnl = df['pnl'].sum()
        win_rate = (df['pnl'] > 0).mean() * 100
        n_trades = len(df)
        max_drawdown = Metrics.max_drawdown(df['pnl'].cumsum())
        return {
            'total_pnl': total_pnl,
            'win_rate': win_rate,
            'n_trades': n_trades,
            'max_drawdown': max_drawdown
        }

    @staticmethod
    def max_drawdown(series):
        roll_max = series.cummax()
        drawdown = (series - roll_max)
        return drawdown.min()
