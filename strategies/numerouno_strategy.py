
# File: strategies/numerouno_strategy.py
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from strategies.base_strategy import BaseStrategy

class NumeroUnoStrategy(BaseStrategy):
    """
    Identifies classic chart patterns like 'W' (Double Bottom) for long trades
    and 'M' or 'Head and Shoulders' for short trades.
    """
    def __init__(self, symbol, initial_capital, logger):
        super().__init__(symbol, initial_capital, logger)
        self.pivot_lookback = 20  # Number of candles to look left and right for a pivot

    def add_indicators(self):
        """
        Identify pivot points which are the building blocks of chart patterns.
        """
        df = self.df
        high_peaks, _ = find_peaks(df['high'], distance=self.pivot_lookback, prominence=df['high'].std() * 0.5)
        low_peaks, _ = find_peaks(-df['low'], distance=self.pivot_lookback, prominence=df['low'].std() * 0.5)
        df['pivot_high'] = np.nan
        df.iloc[high_peaks, df.columns.get_loc('pivot_high')] = df.iloc[high_peaks]['high']
        df['pivot_low'] = np.nan
        df.iloc[low_peaks, df.columns.get_loc('pivot_low')] = df.iloc[low_peaks]['low']

    def generate_signals(self):
        df = self.df
        # Initialize boolean signal columns
        df['signal_long'], df['signal_short'] = False, False
        df['stop_loss'], df['tp1'], df['target'], df['trailing_sl_pct'] = np.nan, np.nan, np.nan, 0.0
        
        if 'pivot_high' not in df.columns or len(df) < 50:
            return

        pivot_highs = df.dropna(subset=['pivot_high'])
        pivot_lows = df.dropna(subset=['pivot_low'])
        
        if len(pivot_lows) < 2 or len(pivot_highs) < 1:
            return

        # --- W-Pattern (Double Bottom) for LONG ---
        last_lows = pivot_lows.tail(2)
        if len(last_lows) == 2:
            low1, low2 = last_lows.iloc[0], last_lows.iloc[1]
            # Check if lows are roughly at the same level
            if abs(low1['pivot_low'] - low2['pivot_low']) / low1['pivot_low'] < 0.02: # Within 2%
                # Find the intervening high
                intervening_highs = pivot_highs[(pivot_highs.index > low1.name) & (pivot_highs.index < low2.name)]
                if not intervening_highs.empty:
                    neckline = intervening_highs['pivot_high'].max()
                    # Check for breakout above the neckline
                    if df['close'].iloc[-1] > neckline and df['close'].iloc[-2] <= neckline:
                        sl = last_lows['pivot_low'].min()
                        risk = df['close'].iloc[-1] - sl
                        if risk > 0:
                            df.at[df.index[-1], 'signal_long'] = True
                            df.at[df.index[-1], 'stop_loss'] = sl
                            df.at[df.index[-1], 'target'] = df['close'].iloc[-1] + risk * 2.0
                            self.log(f"W-Pattern LONG Signal for {self.symbol}", "warning")

        # --- Head and Shoulders Pattern for SHORT ---
        if len(pivot_highs) >= 3 and len(pivot_lows) >= 2:
            last_highs = pivot_highs.tail(3)
            h1, h2_head, h3 = last_highs.iloc[0], last_highs.iloc[1], last_highs.iloc[2]
            # Check for H&S structure: head is the highest
            if h2_head['pivot_high'] > h1['pivot_high'] and h2_head['pivot_high'] > h3['pivot_high']:
                # Find the two intervening lows to form the neckline
                neckline_lows = pivot_lows[(pivot_lows.index > h1.name) & (pivot_lows.index < h3.name)]
                if len(neckline_lows) >= 2:
                    neckline_point1 = neckline_lows.iloc[0]
                    neckline_point2 = neckline_lows.iloc[-1]
                    # Check for breakdown below the lowest point of the neckline
                    neckline_level = min(neckline_point1['pivot_low'], neckline_point2['pivot_low'])
                    if df['close'].iloc[-1] < neckline_level and df['close'].iloc[-2] >= neckline_level:
                        sl = h2_head['pivot_high']
                        risk = sl - df['close'].iloc[-1]
                        if risk > 0:
                            df.at[df.index[-1], 'signal_short'] = True
                            df.at[df.index[-1], 'stop_loss'] = sl
                            df.at[df.index[-1], 'target'] = df['close'].iloc[-1] - risk * 2.0
                            self.log(f"H&S Pattern SHORT Signal for {self.symbol}", "warning")

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