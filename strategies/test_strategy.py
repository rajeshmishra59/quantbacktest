# strategies/test_strategy.py

from strategies.base_strategy import BaseStrategy
import pandas as pd

class TestStrategy(BaseStrategy):
    """
    Simple Test Strategy:
    - Buys if close > open (bullish bar)
    - Sells if close < open (bearish bar)
    - Trades 1 lot, with simple stoploss and target
    - Handles all None cases robustly
    - Exposes both 'signal_long' and 'signal_short' columns for vectorized engine
    """
    def __init__(self, df, symbol=None, logger=None, **kwargs):
        super().__init__(df, symbol, logger, **kwargs)

    def generate_signals(self):
        # Create signal columns for vectorized trade sim compatibility
        self.df['signal_long'] = self.df['close'] > self.df['open']
        self.df['signal_short'] = self.df['close'] < self.df['open']
        self.df['signal'] = None
        self.df.loc[self.df['signal_long'], 'signal'] = 'LONG'
        self.df.loc[self.df['signal_short'], 'signal'] = 'SHORT'

    def calculate_indicators(self):
        # No indicators for this simple test strategy
        pass

    def run(self):
        self.generate_signals()
        trades = []
        in_trade = False
        entry_price = None
        entry_time = None
        side = None
        stop_loss = None
        target = None
        qty = 1  # 1 lot per trade
        rr = 2   # Reward:Risk = 2

        for idx, row in self.df.iterrows():
            # Entry
            if not in_trade and row['signal'] in ('LONG', 'SHORT'):
                side = row['signal']
                entry_price = row['close'] if pd.notna(row['close']) else None
                entry_time = idx
                if entry_price is not None:
                    if side == 'LONG':
                        stop_loss = entry_price * 0.995
                        target = entry_price + (entry_price - stop_loss) * rr
                    else:  # SHORT
                        stop_loss = entry_price * 1.005
                        target = entry_price - (stop_loss - entry_price) * rr
                    in_trade = True
                continue

            # Exit logic (None-guarded)
            if in_trade and entry_price is not None and stop_loss is not None and target is not None:
                if side == 'LONG':
                    if pd.notna(row['low']) and row['low'] <= stop_loss:
                        trades.append({
                            'entry_time': entry_time,
                            'exit_time': idx,
                            'side': side,
                            'entry_price': entry_price,
                            'exit_price': stop_loss,
                            'pnl': (stop_loss - entry_price) * qty,
                            'exit_reason': 'SL'
                        })
                        in_trade = False
                    elif pd.notna(row['high']) and row['high'] >= target:
                        trades.append({
                            'entry_time': entry_time,
                            'exit_time': idx,
                            'side': side,
                            'entry_price': entry_price,
                            'exit_price': target,
                            'pnl': (target - entry_price) * qty,
                            'exit_reason': 'TARGET'
                        })
                        in_trade = False
                elif side == 'SHORT':
                    if pd.notna(row['high']) and row['high'] >= stop_loss:
                        trades.append({
                            'entry_time': entry_time,
                            'exit_time': idx,
                            'side': side,
                            'entry_price': entry_price,
                            'exit_price': stop_loss,
                            'pnl': (entry_price - stop_loss) * qty,
                            'exit_reason': 'SL'
                        })
                        in_trade = False
                    elif pd.notna(row['low']) and row['low'] <= target:
                        trades.append({
                            'entry_time': entry_time,
                            'exit_time': idx,
                            'side': side,
                            'entry_price': entry_price,
                            'exit_price': target,
                            'pnl': (entry_price - target) * qty,
                            'exit_reason': 'TARGET'
                        })
                        in_trade = False

            # End of day force exit (at last row or if time > 15:15)
            if in_trade and (idx == self.df.index[-1] or (isinstance(row.get('datetime', None), pd.Timestamp) and row['datetime'].time() > pd.to_datetime('15:15').time())):
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': idx,
                    'side': side,
                    'entry_price': entry_price,
                    'exit_price': row['close'] if pd.notna(row['close']) else entry_price,
                    'pnl': ((row['close'] - entry_price) if side == 'LONG' else (entry_price - row['close'])) * qty if pd.notna(row['close']) else 0.0,
                    'exit_reason': 'EOD'
                })
                in_trade = False

        self.trades = trades
        return self.df  # production grade

    def get_trades(self):
        return getattr(self, 'trades', [])
