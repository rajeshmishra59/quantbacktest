# File: strategies/sankhyaek_strategy.py
# UPDATED for Universal Base Strategy Template

import pandas as pd
import numpy as np
import pandas_ta as ta
from strategies.base_strategy import BaseStrategy
from typing import Optional

class SankhyaEkStrategy(BaseStrategy):
    def __init__(self, df: pd.DataFrame, symbol: Optional[str] = None, logger=None, **kwargs):
        super().__init__(df, symbol=symbol, logger=logger, **kwargs)
        self.name = "SankhyaEkStrategy"
        # Tweaked parameters
        self.bb_length = self.params.get('bb_length', 15)
        self.bb_std = self.params.get('bb_std', 2.5)
        self.rsi_period = self.params.get('rsi_period', 10)
        self.rsi_oversold = self.params.get('rsi_oversold', 25)
        self.rsi_overbought = self.params.get('rsi_overbought', 75)
        self.stop_loss_pct = self.params.get('stop_loss_pct', 0.015)  # 1.5%
        self.risk_reward_ratio = self.params.get('risk_reward_ratio', 2.0)
        self.max_trades_per_day = self.params.get('max_trades_per_day', 3)

    def calculate_indicators(self):
        tf_string = f'{self.primary_timeframe}min'
        resampled_df = self.raw_df.resample(tf_string).agg(
            {'open':'first', 'high':'max', 'low':'min', 'close':'last', 'volume':'sum'}
        ).dropna()
        if len(resampled_df) < self.bb_length: self.df = pd.DataFrame(); return
        resampled_df.ta.bbands(length=self.bb_length, std=self.bb_std, append=True)
        resampled_df.ta.rsi(length=self.rsi_period, append=True)
        self.df = resampled_df

    def generate_signals(self):
        df = self.df
        if df.empty:
            return None

        rsi_col = f'RSI_{self.rsi_period}'
        bbl_col = f'BBl_{self.bb_length}_{self.bb_std:.1f}'
        bbu_col = f'BBu_{self.bb_length}_{self.bb_std:.1f}'
        if not all(c in df.columns for c in [rsi_col, bbl_col, bbu_col]):
            return None

        # Add trend filter: 30-period moving average
        df['ma_trend'] = df['close'].rolling(30).mean()

        # Only go long if price is above trend (uptrend), short if below (downtrend)
        long_cond = (df['close'] < df[bbl_col]) & (df[rsi_col] < self.rsi_oversold) & (df['close'] > df['ma_trend'])
        short_cond = (df['close'] > df[bbu_col]) & (df[rsi_col] > self.rsi_overbought) & (df['close'] < df['ma_trend'])

        df['stop_loss'] = np.nan
        df['target'] = np.nan
        
        long_sl = df['close'] * (1 - self.stop_loss_pct)
        long_target = df['close'] + ((df['close'] - long_sl) * self.risk_reward_ratio)
        df.loc[long_cond, 'stop_loss'] = long_sl
        df.loc[long_cond, 'target'] = long_target

        short_sl = df['close'] * (1 + self.stop_loss_pct)
        short_target = df['close'] - ((short_sl - df['close']) * self.risk_reward_ratio)
        df.loc[short_cond, 'stop_loss'] = short_sl
        df.loc[short_cond, 'target'] = short_target
        
        return {'long_cond': long_cond, 'short_cond': short_cond}
