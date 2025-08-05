# File: apex_strategy.py (Triangle Pattern Version)

import pandas as pd
import numpy as np
import logging
from .base_strategy import BaseStrategy
from scipy.signal import find_peaks

logger = logging.getLogger(__name__)

class ApexStrategy(BaseStrategy):
    """
    Apex Strategy (Triangle Pattern Breakout): Identifies contracting volatility
    which indicates a triangle, and trades the breakout.
    """
    def __init__(self, df: pd.DataFrame, symbol: str = None, logger=None,
                 primary_timeframe: int = 5, **kwargs):

        super().__init__(df, symbol=symbol, logger=logger, primary_timeframe=primary_timeframe)
        self.name = "Apex"
        self.primary_timeframe = primary_timeframe
        self.log(f"ApexStrategy (Triangle Pattern) initialized for {self.symbol} with {self.primary_timeframe} min TF.")


    def calculate_indicators(self):
        """
        Resamples data. No special indicators needed as this is a price action strategy.
        """
        if self.df_1min_raw.empty:
            self.df = pd.DataFrame()
            return

        tf_string = f'{self.primary_timeframe}T'
        self.df = self.df_1min_raw.resample(tf_string).agg(
            {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
        ).dropna()
        
    def generate_signals(self):
        """
        Detects a contracting range (triangle) and trades the breakout.
        """
        df = self.df
        df['entry_signal'], df['stop_loss'] = 'NONE', np.nan
        df['target'], df['target2'], df['target3'] = np.nan, np.nan, np.nan
        
        window = 30
        if len(df) < window + 100: # Need sufficient historical data to compare
            return

        # Simplified logic: Check for volatility contraction
        recent_candles = df.iloc[-window:]
        historical_candles = df.iloc[-200:-window]
        
        recent_avg_range = (recent_candles['high'] - recent_candles['low']).mean()
        historical_avg_range = (historical_candles['high'] - historical_candles['low']).mean()
        
        # If recent volatility is less than 60% of historical volatility, we have a squeeze
        if recent_avg_range < historical_avg_range * 0.6:
            breakout_high = recent_candles['high'].max()
            breakout_low = recent_candles['low'].min()
            current_close = df['close'].iloc[-1]
            
            # Check for a breakout from this contracted range
            if current_close > breakout_high:
                df.at[df.index[-1], 'entry_signal'] = 'LONG'
                sl = breakout_low
                risk = current_close - sl
                if risk > 0:
                    df.at[df.index[-1], 'stop_loss'] = sl
                    df.at[df.index[-1], 'target'] = current_close + (risk * 1.5)
                    df.at[df.index[-1], 'target2'] = current_close + (risk * 2.5)
                    df.at[df.index[-1], 'target3'] = current_close + (risk * 4.5)
                    self.log(f"Triangle Breakout LONG signal for {self.symbol}")
                
            elif current_close < breakout_low:
                df.at[df.index[-1], 'entry_signal'] = 'SHORT'
                sl = breakout_high
                risk = sl - current_close
                if risk > 0:
                    df.at[df.index[-1], 'stop_loss'] = sl
                    df.at[df.index[-1], 'target'] = current_close - (risk * 1.5)
                    df.at[df.index[-1], 'target2'] = current_close - (risk * 2.5)
                    df.at[df.index[-1], 'target3'] = current_close - (risk * 4.5)
                    self.log(f"Triangle Breakout SHORT signal for {self.symbol}")
