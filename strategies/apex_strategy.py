
# File: strategies/apex_strategy.py
import pandas as pd
import numpy as np
from strategies.base_strategy import BaseStrategy

class ApexStrategy(BaseStrategy):
    """
    Identifies a volatility contraction (squeeze) and trades the subsequent breakout.
    This pattern often resembles a triangle or wedge.
    """
    def __init__(self, symbol, initial_capital, logger):
        super().__init__(symbol, initial_capital, logger)
        self.squeeze_window = 30  # Look for contraction in the last 30 candles
        self.historical_window = 200 # Compare against the last 200 candles' volatility
        self.volatility_ratio_threshold = 0.6 # Squeeze is active if recent vol is <60% of historical

    def add_indicators(self):
        # This strategy calculates volatility dynamically, no standard indicators needed upfront.
        pass

    def generate_signals(self):
        """
        Detects a contracting range (triangle) and trades the breakout.
        """
        df = self.df
        # Initialize boolean signal columns
        df['signal_long'] = False
        df['signal_short'] = False
        df['stop_loss'] = np.nan
        df['target'], df['target2'], df['target3'] = np.nan, np.nan, np.nan
        
        # Ensure we have enough data to perform the comparison
        if len(df) < self.historical_window + self.squeeze_window:
            return

        # Define the time windows for comparison
        recent_candles = df.iloc[-self.squeeze_window:]
        historical_candles = df.iloc[-(self.historical_window + self.squeeze_window):-self.squeeze_window]
        
        # Calculate the average true range for both periods
        recent_avg_range = (recent_candles['high'] - recent_candles['low']).mean()
        historical_avg_range = (historical_candles['high'] - historical_candles['low']).mean()
        
        # If recent volatility is significantly less than historical, a squeeze is identified
        is_squeeze = recent_avg_range < (historical_avg_range * self.volatility_ratio_threshold)
        
        if is_squeeze:
            breakout_high = recent_candles['high'].max()
            breakout_low = recent_candles['low'].min()
            current_close = df['close'].iloc[-1]
            
            # Check for a breakout from this contracted range
            if current_close > breakout_high:
                df.at[df.index[-1], 'signal_long'] = True # Set boolean signal
                sl = breakout_low
                risk = current_close - sl
                if risk > 0:
                    df.at[df.index[-1], 'stop_loss'] = sl
                    df.at[df.index[-1], 'target'] = current_close + (risk * 1.5)
                    df.at[df.index[-1], 'target2'] = current_close + (risk * 2.5)
                    df.at[df.index[-1], 'target3'] = current_close + (risk * 4.5)
                    self.log(f"APEX Breakout LONG for {self.symbol} at {current_close:.2f}", "warning")
                
            elif current_close < breakout_low:
                df.at[df.index[-1], 'signal_short'] = True # Set boolean signal
                sl = breakout_high
                risk = sl - current_close
                if risk > 0:
                    df.at[df.index[-1], 'stop_loss'] = sl
                    df.at[df.index[-1], 'target'] = current_close - (risk * 1.5)
                    df.at[df.index[-1], 'target2'] = current_close - (risk * 2.5)
                    df.at[df.index[-1], 'target3'] = current_close - (risk * 4.5)
                    self.log(f"APEX Breakout SHORT for {self.symbol} at {current_close:.2f}", "warning")