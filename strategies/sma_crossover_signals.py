# strategies/sma_crossover_strategy.py
# NEW FILE: Implements SMA Crossover logic using the universal template.

import pandas as pd
import numpy as np
import pandas_ta as ta
from strategies.base_strategy import BaseStrategy
from typing import Optional

class SmaCrossoverStrategy(BaseStrategy):
    def __init__(self, df: pd.DataFrame, symbol: Optional[str] = None, logger=None, **kwargs):
        super().__init__(df, symbol=symbol, logger=logger, **kwargs)
        self.name = "SmaCrossoverStrategy"
        self.short_window = self.params.get('short_window', 20)
        self.long_window = self.params.get('long_window', 50)

    def calculate_indicators(self):
        tf_string = f'{self.primary_timeframe}min'
        df_tf = self.raw_df.resample(tf_string).agg(
            {'open':'first', 'high':'max', 'low':'min', 'close':'last', 'volume':'sum'}
        ).dropna()
        if len(df_tf) < self.long_window: self.df = pd.DataFrame(); return
        
        df_tf.ta.sma(length=self.short_window, append=True)
        df_tf.ta.sma(length=self.long_window, append=True)
        self.df = df_tf

    def generate_signals(self):
        df = self.df
        short_ma_col = f'SMA_{self.short_window}'
        long_ma_col = f'SMA_{self.long_window}'
        if df.empty or not all(c in df.columns for c in [short_ma_col, long_ma_col]):
            return None

        # Golden Cross (Buy) and Death Cross (Sell)
        long_cond = (df[short_ma_col] > df[long_ma_col]) & (df[short_ma_col].shift(1) <= df[long_ma_col].shift(1))
        short_cond = (df[short_ma_col] < df[long_ma_col]) & (df[short_ma_col].shift(1) >= df[long_ma_col].shift(1))

        df['stop_loss'] = np.nan
        df.loc[long_cond, 'stop_loss'] = df['low'] * 0.98  # 2% SL below the low
        df.loc[short_cond, 'stop_loss'] = df['high'] * 1.02 # 2% SL above the high

        return {'long_cond': long_cond, 'short_cond': short_cond}
