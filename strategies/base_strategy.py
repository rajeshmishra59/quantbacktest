# strategies/base_strategy.py

import pandas as pd
import logging
from typing import Optional, Dict, Any

class BaseStrategy:
    """
    Standardized base for all trading strategies.
    Adds versioning and config for audit/reproducibility.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        df_15min: Optional[pd.DataFrame] = None,
        symbol: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
        primary_timeframe: int = 5,                # <--- Added!
        config_dict: Optional[Dict[str, Any]] = None,
        strategy_version: Optional[str] = None,
        **kwargs
    ):
        self.name: str = self.__class__.__name__
        self.symbol: Optional[str] = symbol
        self.df: pd.DataFrame = df
        self.df_1min_raw: pd.DataFrame = df
        self.df_15min: Optional[pd.DataFrame] = df_15min
        self.logger: logging.Logger = logger or logging.getLogger(__name__)
        self.config_dict: Dict[str, Any] = config_dict or {}
        self.strategy_version: str = strategy_version or "v1.0"
        self.primary_timeframe: int = primary_timeframe      # <--- Added!

    # ... rest as before ...

    def log(self, message: str, level: str = 'info') -> None:
        if self.logger:
            if level == 'info':
                self.logger.info(f"[{self.name}][{self.symbol}] {message}")
            elif level == 'debug':
                self.logger.debug(f"[{self.name}][{self.symbol}] {message}")
            elif level == 'warning':
                self.logger.warning(f"[{self.name}][{self.symbol}] {message}")
            elif level == 'error':
                self.logger.error(f"[{self.name}][{self.symbol}] {message}")

    def calculate_indicators(self) -> None:
        raise NotImplementedError("Each strategy must implement 'calculate_indicators'")

    def generate_signals(self) -> None:
        raise NotImplementedError("Each strategy must implement 'generate_signals'")

    def run(self) -> pd.DataFrame:
        self.calculate_indicators()
        self.generate_signals()
        return self.df
