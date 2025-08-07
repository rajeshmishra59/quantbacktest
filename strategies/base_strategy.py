# strategies/base_strategy.py
# UNIVERSAL TEMPLATE: Compatible with both Backtesting and Paper Trading engines.

import pandas as pd
import numpy as np
import logging
from typing import Optional

class BaseStrategy:
    def __init__(self, df: pd.DataFrame, symbol: Optional[str] = None, logger=None, 
                 primary_timeframe: Optional[int] = None, **kwargs):
        self.name = self.__class__.__name__
        self.symbol = symbol
        self.primary_timeframe = primary_timeframe
        self.raw_df = df 
        self.df = pd.DataFrame() 
        self.logger = logger or logging.getLogger(__name__)
        self.params = kwargs

    def log(self, message: str, level: str = 'info'):
        if self.logger:
            log_func = getattr(self.logger, level, self.logger.info)
            log_func(f"[{self.name}][{self.symbol}] {message}")

    def calculate_indicators(self):
        raise NotImplementedError(f"'calculate_indicators' must be implemented in {self.name}")

    def generate_signals(self):
        """
        Child strategies must implement this.
        It should return a DataFrame with 'long_cond' and 'short_cond' (boolean) columns.
        """
        raise NotImplementedError(f"'generate_signals' must be implemented in {self.name}")

    def get_signal_df(self, long_cond, short_cond) -> pd.DataFrame:
        """
        UNIVERSAL METHOD: Generates all required signal formats.
        """
        df = self.df
        
        # 1. Boolean signals for Backtester
        df['entries'] = long_cond
        df['exits'] = short_cond
        
        # 2. String signals for Paper Trading Engine
        df['entry_signal'] = 'NONE'
        df.loc[long_cond, 'entry_signal'] = 'LONG'
        df.loc[short_cond, 'entry_signal'] = 'SHORT'
        
        # 3. Stop Loss and Target (must be calculated in child strategy)
        if 'stop_loss' not in df.columns: df['stop_loss'] = np.nan
        if 'target' not in df.columns: df['target'] = np.nan
        
        return df

    def run(self) -> pd.DataFrame:
        self.calculate_indicators()
        if not self.df.empty:
            # generate_signals ab sirf conditions return karega
            conditions = self.generate_signals()
            if conditions is not None:
                long_cond, short_cond = conditions['long_cond'], conditions['short_cond']
                # get_signal_df final DataFrame banayega
                return self.get_signal_df(long_cond, short_cond)
        return pd.DataFrame()
