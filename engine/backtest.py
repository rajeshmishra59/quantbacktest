# quantbacktest/engine/backtest.py
import pandas as pd
from .portfolio import Portfolio

class BacktestEngine:
    def __init__(self, initial_cash=1_000_000):
        self.initial_cash = initial_cash
        self.portfolio = Portfolio(initial_cash)
        self.results = []
        self.run_id = None

    def run(self, strategy):
        df = strategy.df
        if df.empty or 'entry_signal' not in df.columns:
            strategy.log("No signals/data to process.")
            return
        in_position = False
        entry_price = None
        for idx, row in df.iterrows():
            signal = row.get('entry_signal', 'NONE')
            price = row['close']
            symbol = strategy.symbol

            if signal == 'LONG' and not in_position:
                self.portfolio.update_position(symbol, qty=1, price=price, side='buy')
                in_position = True
                entry_price = price
                strategy.log(f"LONG entry at {price} ({idx})")
            elif signal == 'SHORT' and in_position:
                self.portfolio.update_position(symbol, qty=1, price=price, side='sell')
                in_position = False
                pnl = price - entry_price
                strategy.log(f"SHORT/Exit at {price} ({idx}), P&L: {pnl}")

        self.results = self.portfolio.trade_log

    def get_results(self):
        return self.results
