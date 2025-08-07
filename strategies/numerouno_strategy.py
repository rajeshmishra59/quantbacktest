# File: strategies/numerouno_strategy.py
# FINAL CORRECTED: Linter-friendly code to remove all type errors.

import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from strategies.base_strategy import BaseStrategy
from typing import Optional

class NumeroUnoStrategy(BaseStrategy):
    """
    Identifies 'W' (Double Bottom) patterns for long trades.
    """
    def __init__(self, df: pd.DataFrame, symbol: Optional[str] = None, logger=None, **kwargs):
        super().__init__(df, symbol=symbol, logger=logger, **kwargs)
        self.name = "NumeroUnoStrategy"
        self.pivot_lookback = self.params.get('pivot_lookback', 10)
        self.log(f"Initialized with pivot_lookback={self.pivot_lookback}")

    def calculate_indicators(self):
        tf_string = f'{self.primary_timeframe}min'
        self.df = self.raw_df.resample(tf_string).agg(
            {'open':'first', 'high':'max', 'low':'min', 'close':'last', 'volume':'sum'}
        ).dropna()

        df = self.df
        if df.empty: return

        # Find pivot points (peaks and troughs)
        high_peaks_indices, _ = find_peaks(df['high'], distance=self.pivot_lookback)
        low_peaks_indices, _ = find_peaks(-df['low'], distance=self.pivot_lookback)

        df['pivot_high'] = np.nan
        df['pivot_low'] = np.nan

        # --- CORRECTED LINTER-FRIENDLY METHOD ---
        # Get the actual index labels (timestamps) for the peaks
        high_peak_labels = df.index[high_peaks_indices]
        low_peak_labels = df.index[low_peaks_indices]

        # Use .loc with labels, which is clearer for the linter
        df.loc[high_peak_labels, 'pivot_high'] = df['high'].loc[high_peak_labels]
        df.loc[low_peak_labels, 'pivot_low'] = df['low'].loc[low_peak_labels]
        
        self.log("Pivot points calculated.")

    def generate_signals(self):
        df = self.df
        if 'pivot_low' not in df.columns or df.empty:
            return None

        pivot_highs = df.dropna(subset=['pivot_high'])
        pivot_lows = df.dropna(subset=['pivot_low'])

        long_cond = pd.Series(False, index=df.index)
        df['stop_loss'] = np.nan

        # W-Pattern (Double Bottom) for LONG
        for i in range(1, len(pivot_lows)):
            low2 = pivot_lows.iloc[i]
            low1 = pivot_lows.iloc[i-1]
            
            if abs(low1['pivot_low'] - low2['pivot_low']) / low1['pivot_low'] < 0.02:
                intervening_highs = pivot_highs[(pivot_highs.index > low1.name) & (pivot_highs.index < low2.name)]
                if not intervening_highs.empty:
                    neckline = intervening_highs['pivot_high'].max()
                    
                    breakout_candles = df[(df.index > low2.name) & (df['close'] > neckline)]
                    if not breakout_candles.empty:
                        entry_idx = breakout_candles.index[0]
                        long_cond.loc[entry_idx] = True
                        df.loc[entry_idx, 'stop_loss'] = low2['pivot_low']
        
        short_cond = pd.Series(False, index=df.index)
        
        return {'long_cond': long_cond, 'short_cond': short_cond}
