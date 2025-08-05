# File: numerouno_strategy.py (Final Corrected Version)
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import logging
from .base_strategy import BaseStrategy

class NumeroUnoStrategy(BaseStrategy):
    def __init__(self, df: pd.DataFrame, symbol: str = None, logger=None, primary_timeframe: int = 5,
                 pivot_lookback: int = 10, volume_ma_period: int = 20,
                 volume_spike_multiplier: float = 1.5, tp1_rr_ratio: float = 1.5,
                 tp2_rr_ratio: float = 3.0, pattern_confirmation_window: int = 60,
                 trailing_sl_pct: float = 0.5, **kwargs):

        super().__init__(df, symbol=symbol, logger=logger, primary_timeframe=primary_timeframe)
        self.name = "NumeroUno"
        self.primary_timeframe = primary_timeframe
        self.PIVOT_LOOKBACK, self.VOLUME_MA_PERIOD = pivot_lookback, volume_ma_period
        self.VOLUME_SPIKE_MULTIPLIER, self.TP1_RR_RATIO = volume_spike_multiplier, tp1_rr_ratio
        self.TP2_RR_RATIO, self.PATTERN_WINDOW = tp2_rr_ratio, pattern_confirmation_window
        self.TRAILING_SL_PCT = trailing_sl_pct
        self.log(f"NumeroUnoStrategy initialized for {self.symbol} with {self.primary_timeframe}-min timeframe.")

    def calculate_indicators(self):
        if self.df_1min_raw.empty: return
        tf_string = f'{self.primary_timeframe}T'
        self.df = self.df_1min_raw.resample(tf_string).agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
        if self.df.empty: return
        df = self.df
        high_peaks, _ = find_peaks(df['high'], distance=self.PIVOT_LOOKBACK)
        df['pivot_high'] = np.nan; df.iloc[high_peaks, df.columns.get_loc('pivot_high')] = df.iloc[high_peaks]['high']
        low_peaks, _ = find_peaks(-df['low'], distance=self.PIVOT_LOOKBACK)
        df['pivot_low'] = np.nan; df.iloc[low_peaks, df.columns.get_loc('pivot_low')] = df.iloc[low_peaks]['low']
        df['volume_avg'] = df['volume'].rolling(self.VOLUME_MA_PERIOD).mean()
        df['has_volume_spike'] = df['volume'] > (df['volume_avg'] * self.VOLUME_SPIKE_MULTIPLIER)

    def generate_signals(self):
        df = self.df
        df['entry_signal'], df['stop_loss'], df['tp1'], df['target'], df['trailing_sl_pct'] = 'NONE', np.nan, np.nan, np.nan, 0.0
        if 'pivot_high' not in df.columns or df.empty: return
        pivot_highs, pivot_lows = df.dropna(subset=['pivot_high']), df.dropna(subset=['pivot_low'])
        # (W-Pattern and H&S logic is unchanged)
        # ...