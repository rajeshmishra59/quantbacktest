# quant_backtesting_project/utils/vbt_signal_generator.py
# Yeh file VectorBT ka istemal karke tezi se trading signals generate karegi.

import vectorbt as vbt
import pandas as pd

def generate_sma_signals(prices: pd.Series, short_window: int, long_window: int):
    """
    Ek simple Dual Moving Average Crossover strategy ke liye signals generate karta hai.

    Args:
        prices (pd.Series): Stock ke close prices ki series.
        short_window (int): Chote moving average ka period.
        long_window (int): Lambe moving average ka period.

    Returns:
        pd.DataFrame: 'entries' aur 'exits' signals ke saath ek DataFrame.
    """
    if prices.empty or len(prices) < long_window:
        return pd.DataFrame({'entries': [], 'exits': []})

    # VectorBT ka istemal karke tezi se moving averages calculate karein
    short_ma = vbt.MA.run(prices, window=short_window, short_name='short')
    long_ma = vbt.MA.run(prices, window=long_window, short_name='long')

    # Crossover signals generate karein
    entries = short_ma.ma_crossed_above(long_ma)
    exits = short_ma.ma_crossed_below(long_ma)
    
    return pd.DataFrame({'entries': entries, 'exits': exits})

