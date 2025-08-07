# File: strategies/alphaone_strategy.py
# UPDATED for Universal Base Strategy Template

import pandas as pd
import numpy as np
import warnings
from strategies.base_strategy import BaseStrategy
from typing import Optional

warnings.simplefilter(action='ignore', category=FutureWarning)

class AlphaOneStrategy(BaseStrategy):
    def __init__(self, df: pd.DataFrame, symbol: Optional[str] = None, logger=None,
                 primary_timeframe: int = 15, **kwargs):

        super().__init__(df, symbol=symbol, logger=logger, primary_timeframe=primary_timeframe, **kwargs)
        self.name = "AlphaOneStrategy"
        # Parameters ko self.params se access karein
        self.streak_period_min = self.params.get('streak_period_min', 8)
        self.strong_candle_ratio = self.params.get('strong_candle_ratio', 0.7)
        self.volume_spike_multiplier = self.params.get('volume_spike_multiplier', 1.5)
        self.tp_rr_ratio = self.params.get('tp_rr_ratio', 2.0)
        self.log(f"Initialized for {self.symbol}.")

    def calculate_indicators(self):
        """
        Resamples raw 1-minute data to the strategy's primary timeframe.
        """
        # CORRECTED: Use self.raw_df as per the new template
        if self.raw_df.empty:
            self.df = pd.DataFrame()
            return

        # Use 'min' instead of 'T' for future compatibility
        tf_string = f'{self.primary_timeframe}min'
        self.df = self.raw_df.resample(tf_string).agg(
            {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
        ).dropna()

    def generate_signals(self):
        """
        Generates boolean conditions for long/short signals and calculates SL/TP.
        """
        df = self.df
        if df.empty or len(df) < self.streak_period_min + 1:
            return None

        # Indicator calculations
        is_down_streak = df['close'] < df['close'].shift(1)
        is_up_streak = df['close'] > df['close'].shift(1)
        is_green = df['close'] > df['open']
        is_red = df['close'] < df['open']
        candle_range = (df['high'] - df['low']).replace(0, np.nan)
        body_size = (df['close'] - df['open']).abs()
        is_strong = (body_size / candle_range) >= self.strong_candle_ratio
        volume_avg = df['volume'].rolling(window=20).mean()
        has_volume = df['volume'] > (volume_avg * self.volume_spike_multiplier)

        down_streak_active = is_down_streak.rolling(window=self.streak_period_min).sum() == self.streak_period_min
        up_streak_active = is_up_streak.rolling(window=self.streak_period_min).sum() == self.streak_period_min

        # Step 1: Generate boolean conditions
        long_cond = down_streak_active.shift(1) & is_green & is_strong & has_volume
        short_cond = up_streak_active.shift(1) & is_red & is_strong & has_volume

        # Step 2: Calculate Stop Loss and Target
        df['stop_loss'] = np.nan
        df['target'] = np.nan

        # For long trades
        risk_long = df['close'] - df['low']
        df.loc[long_cond, 'stop_loss'] = df['low']
        df.loc[long_cond, 'target'] = df['close'] + (risk_long * self.tp_rr_ratio)

        # For short trades
        risk_short = df['high'] - df['close']
        df.loc[short_cond, 'stop_loss'] = df['high']
        df.loc[short_cond, 'target'] = df['close'] - (risk_short * self.tp_rr_ratio)

        # Step 3: Return conditions dictionary
        return {'long_cond': long_cond, 'short_cond': short_cond}
