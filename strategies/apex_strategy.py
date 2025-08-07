# File: strategies/apex_strategy.py
# UPDATED for Universal Base Strategy Template

import pandas as pd
import numpy as np
from strategies.base_strategy import BaseStrategy
from typing import Optional, Dict, Any

class ApexStrategy(BaseStrategy):
    def __init__(self, df: pd.DataFrame, symbol: Optional[str] = None, logger=None, **kwargs):
        super().__init__(df, symbol=symbol, logger=logger, **kwargs)
        self.name = "ApexStrategy"
        self.squeeze_window = self.params.get('squeeze_window', 30)
        self.historical_window = self.params.get('historical_window', 200)
        self.volatility_ratio_threshold = self.params.get('volatility_ratio_threshold', 0.6)
        self.log(f"Initialized with squeeze_window={self.squeeze_window}")

    def calculate_indicators(self):
        tf_string = f'{self.primary_timeframe}min'
        self.df = self.raw_df.resample(tf_string).agg(
            {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
        ).dropna()
        self.log("Data resampled.")

    def generate_signals(self):
        df = self.df
        if len(df) < self.historical_window + self.squeeze_window:
            self.log("Not enough data for signal generation.", level='warning')
            return None

        atr_hist = df['high'].rolling(self.historical_window).max() - df['low'].rolling(self.historical_window).min()
        atr_squeeze = df['high'].rolling(self.squeeze_window).max() - df['low'].rolling(self.squeeze_window).min()
        is_squeeze = atr_squeeze < (atr_hist.shift(self.squeeze_window) * self.volatility_ratio_threshold)
        breakout_high = df['high'].rolling(self.squeeze_window).max().shift(1)
        breakout_low = df['low'].rolling(self.squeeze_window).min().shift(1)

        # Step 1: Generate boolean conditions
        long_cond = is_squeeze & (df['close'] > breakout_high)
        short_cond = is_squeeze & (df['close'] < breakout_low)

        # Step 2: Calculate Stop Loss
        df['stop_loss'] = np.nan
        df.loc[long_cond, 'stop_loss'] = breakout_low
        df.loc[short_cond, 'stop_loss'] = breakout_high
        
        # Step 3: Return conditions
        return {'long_cond': long_cond, 'short_cond': short_cond}
