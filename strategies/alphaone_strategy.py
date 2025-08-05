# File: alphaone_strategy.py (Final Corrected Version)
import pandas as pd
import numpy as np
import warnings
import logging

from .base_strategy import BaseStrategy
from typing import Optional, Dict, Any
from .base_strategy import BaseStrategy


warnings.simplefilter(action='ignore', category=FutureWarning)

class AlphaOneStrategy(BaseStrategy):
    def __init__(
        self,
        df: pd.DataFrame,
        symbol: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
        primary_timeframe: int = 15,
        config_dict: Optional[Dict[str, Any]] = None,
        strategy_version: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            df,
            symbol=symbol,
            logger=logger,
            config_dict=config_dict,
            strategy_version=strategy_version,
            primary_timeframe=primary_timeframe,
            **kwargs
        )
        self.name = "AlphaOne"
        self.primary_timeframe = primary_timeframe
        self.STREAK_PERIOD_MIN = 8
        self.STRONG_CANDLE_RATIO = 0.7
        self.VOLUME_SPIKE_MULTIPLIER = 1.5
        self.TP1_RR_RATIO = 1.5
        self.TP2_RR_RATIO = 3.0
        
        self.log(f"AlphaOneStrategy initialized for {self.symbol}.")

    def calculate_indicators(self):
        if self.df_1min_raw.empty:
            self.df = pd.DataFrame()
            return

        tf_string = f'{self.primary_timeframe}T'
        self.df = self.df_1min_raw.resample(tf_string).agg(
            {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
        ).dropna()

    def generate_signals(self):
        df = self.df
        df['entry_signal'], df['exit_signal'], df['stop_loss'], df['target'], df['tp1'] = 'NONE', 'NONE', np.nan, np.nan, np.nan
        
        if df.empty or len(df) < self.STREAK_PERIOD_MIN + 1:
            return

        is_down_streak = df['close'] < df['close'].shift(1)
        is_up_streak = df['close'] > df['close'].shift(1)
        is_green = df['close'] > df['open']
        is_red = df['close'] < df['open']
        candle_range = (df['high'] - df['low']).replace(0, np.nan)
        body_size = (df['close'] - df['open']).abs()
        is_strong = (body_size / candle_range) >= self.STRONG_CANDLE_RATIO
        volume_avg = df['volume'].rolling(window=20).mean()
        has_volume = df['volume'] > (volume_avg * self.VOLUME_SPIKE_MULTIPLIER)

        down_streak_active = is_down_streak.rolling(window=self.STREAK_PERIOD_MIN).sum() == self.STREAK_PERIOD_MIN
        up_streak_active = is_up_streak.rolling(window=self.STREAK_PERIOD_MIN).sum() == self.STREAK_PERIOD_MIN

        long_entry_cond = down_streak_active.shift(1) & is_green & is_strong & has_volume
        short_entry_cond = up_streak_active.shift(1) & is_red & is_strong & has_volume

        df.loc[long_entry_cond, 'entry_signal'] = 'LONG'
        df.loc[long_entry_cond, 'stop_loss'] = df['low']
        risk_long = df['close'] - df['stop_loss']
        df.loc[long_entry_cond, 'tp1'] = df['close'] + (risk_long * self.TP1_RR_RATIO)
        df.loc[long_entry_cond, 'target'] = df['close'] + (risk_long * self.TP2_RR_RATIO)
        
        df.loc[short_entry_cond, 'entry_signal'] = 'SHORT'
        df.loc[short_entry_cond, 'stop_loss'] = df['high']
        risk_short = df['stop_loss'] - df['close']
        df.loc[short_entry_cond, 'tp1'] = df['close'] - (risk_short * self.TP1_RR_RATIO)
        df.loc[short_entry_cond, 'target'] = df['close'] - (risk_short * self.TP2_RR_RATIO)