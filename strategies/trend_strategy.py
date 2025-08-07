# strategies/trend_strategy.py
# UPDATED for Universal Base Strategy Template

import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Optional
from strategies.base_strategy import BaseStrategy

class TrendStrategy(BaseStrategy):
    def __init__(self, df: pd.DataFrame, symbol: Optional[str] = None, logger=None, **kwargs):
        super().__init__(df, symbol=symbol, logger=logger, **kwargs)
        self.name = "TrendStrategy"

    def calculate_indicators(self) -> None:
        tf_string = f"{self.primary_timeframe}min"
        df_tf = self.raw_df.resample(tf_string).agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        }).dropna().copy()
        if len(df_tf) < 30: self.df = pd.DataFrame(); return
        df_tf.ta.ema(length=9, append=True)
        df_tf.ta.ema(length=21, append=True)
        df_tf.ta.atr(length=14, append=True)
        df_tf.ta.rsi(length=14, append=True)
        df_tf.ta.adx(length=14, append=True)
        df_tf['vol_avg'] = df_tf['volume'].rolling(20, min_periods=20).mean()
        self.df = df_tf

    def generate_signals(self):
        df = self.df
        if df.empty or 'ADX_14' not in df.columns: return None

        df['trend'] = np.nan
        pre_market_window = df.between_time("09:15", "09:30")
        trend_def_candles = pre_market_window.groupby(pre_market_window.index.to_series().dt.date).tail(1)
        is_bullish = (trend_def_candles['EMA_9'] > trend_def_candles['EMA_21']) & (trend_def_candles['ADX_14'] >= 25)
        is_bearish = (trend_def_candles['EMA_9'] < trend_def_candles['EMA_21']) & (trend_def_candles['ADX_14'] >= 25)
        trend_map = pd.Series(np.nan, index=trend_def_candles.index)
        trend_map[is_bullish] = 'BULLISH'
        trend_map[is_bearish] = 'BEARISH'
        df['trend'] = trend_map
        df['trend'] = df.groupby(df.index.to_series().dt.date)['trend'].transform('ffill')

        is_trade_time = (df.index.to_series().dt.time >= pd.to_datetime("09:30").time()) & (df.index.to_series().dt.time < pd.to_datetime("15:00").time())
        is_volatile_enough = (df['ADX_14'] >= 25) & (df['ATRr_14'] >= 15)
        has_volume_spike = df['volume'] > 1.5 * df['vol_avg']
        is_bullish_trend = df['trend'] == 'BULLISH'
        rsi_pullback_long = (df['RSI_14'] >= 40) & (df['RSI_14'] <= 45)
        price_near_ema9 = abs(df['close'] - df['EMA_9']) <= 0.10 * df['ATRr_14']
        bullish_reversal = (df['close'] > df['open']) | ((df['close'] > df['open']) & (df.shift(1)['close'] < df.shift(1)['open']) & (df['close'] > df.shift(1)['open']))
        long_cond = is_trade_time & is_volatile_enough & has_volume_spike & is_bullish_trend & rsi_pullback_long & price_near_ema9 & bullish_reversal
        
        is_bearish_trend = df['trend'] == 'BEARISH'
        rsi_pullback_short = (df['RSI_14'] >= 55) & (df['RSI_14'] <= 60)
        bearish_reversal = (df['close'] < df['open']) | ((df['close'] < df['open']) & (df.shift(1)['close'] > df.shift(1)['open']) & (df['close'] < df.shift(1)['open']))
        short_cond = is_trade_time & is_volatile_enough & has_volume_spike & is_bearish_trend & rsi_pullback_short & price_near_ema9 & bearish_reversal

        df['stop_loss'] = np.nan
        df['target'] = np.nan
        atr_mult = np.select([df['ATRr_14'] < 20, df['ATRr_14'] > 50], [1.25, 2.0], default=1.5)
        risk = atr_mult * df['ATRr_14']
        df.loc[long_cond, 'stop_loss'] = df['close'] - risk
        df.loc[long_cond, 'target'] = df['close'] + (4 * risk)
        df.loc[short_cond, 'stop_loss'] = df['close'] + risk
        df.loc[short_cond, 'target'] = df['close'] - (4 * risk)

        return {'long_cond': long_cond, 'short_cond': short_cond}
