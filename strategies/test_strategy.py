# strategies/test_strategy.py
# FINAL CORRECTED: Aligned with the Universal Base Strategy Template.

import pandas as pd
import numpy as np
from strategies.base_strategy import BaseStrategy
from typing import Optional

class TestStrategy(BaseStrategy):
    """
    Simple Test Strategy, now aligned with the universal template.
    - Buys if close > open (bullish bar)
    - Exits if close < open (bearish bar)
    """
    def __init__(self, df: pd.DataFrame, symbol: Optional[str] = None, logger=None, **kwargs):
        super().__init__(df, symbol=symbol, logger=logger, **kwargs)
        self.name = "TestStrategy"

    def calculate_indicators(self):
        """
        Resamples data to the primary timeframe. No other indicators needed.
        """
        if self.raw_df.empty:
            self.log("Raw data is empty.", "warning")
            self.df = pd.DataFrame()
            return
            
        tf_string = f'{self.primary_timeframe}min'
        self.df = self.raw_df.resample(tf_string).agg(
            {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
        ).dropna()

    def generate_signals(self):
        """
        Generates boolean conditions for long/short signals.
        """
        if self.df.empty:
            return None

        df = self.df
        
        # Step 1: Generate boolean conditions
        long_cond = df['close'] > df['open']
        short_cond = df['close'] < df['open']
        
        # Step 2: Calculate Stop Loss and Target
        df['stop_loss'] = np.nan
        df['target'] = np.nan # Optional, but good practice
        
        # For long trades
        df.loc[long_cond, 'stop_loss'] = df['low'] * 0.99
        
        # Step 3: Return conditions dictionary as required by the base template
        return {'long_cond': long_cond, 'short_cond': short_cond}
