# strategies/rangebound_strategy.py
# UPDATED for Universal Base Strategy Template

import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Optional
from strategies.base_strategy import BaseStrategy

class RangeBoundStrategy(BaseStrategy):
    def __init__(self, df: pd.DataFrame, symbol: Optional[str] = None, logger=None, **kwargs):
        super().__init__(df, symbol=symbol, logger=logger, **kwargs)
        self.name = "RangeBoundStrategy"

    def calculate_indicators(self) -> None:
        tf_string = f"{self.primary_timeframe}min"
        df_tf = self.raw_df.resample(tf_string).agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        }).dropna().copy()
        if len(df_tf) < 20: self.df = pd.DataFrame(); return
        df_tf.ta.bbands(length=20, std=2, append=True)
        df_tf.ta.atr(length=14, append=True)
        df_tf.ta.adx(length=14, append=True)
        df_tf.ta.stoch(k=14, d=3, smooth_k=3, append=True)
        df_tf.ta.vwap(append=True)
        df_tf['vol_avg'] = df_tf['volume'].rolling(20, min_periods=20).mean()
        self.df = df_tf

    def generate_signals(self):
        df = self.df
        if df.empty or len(df) < 30: return None
        
        # --- Conditions ---
        is_trade_time = (df.index.to_series().dt.time >= pd.to_datetime("10:30").time()) & (df.index.to_series().dt.time <= pd.to_datetime("14:00").time())
        has_volume_spike = df['volume'] > 1.5 * df['vol_avg']
        
        long_cond = is_trade_time & has_volume_spike & (df['close'] <= df['BBl_20_2.0']) & (df['STOCHk_14_3_3'] <= 20) & (df['close'] > df['open'])
        short_cond = is_trade_time & has_volume_spike & (df['close'] >= df['BBu_20_2.0']) & (df['STOCHk_14_3_3'] >= 80) & (df['close'] < df['open'])

        # --- SL/TP ---
        df['stop_loss'] = np.nan
        risk = df['ATRr_14'] * np.where(df['ATRr_14'] >= 5, 0.8, 0.5)
        df.loc[long_cond, 'stop_loss'] = df['close'] - risk
        df.loc[short_cond, 'stop_loss'] = df['close'] + risk

        return {'long_cond': long_cond, 'short_cond': short_cond}
