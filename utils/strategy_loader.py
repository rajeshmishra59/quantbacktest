import os
import importlib.util
import inspect
from typing import Dict, Type
from strategies.base_strategy import BaseStrategy
import pandas as pd
import sqlite3

class StrategyLoader:
    """
    Simple OHLCV data loader for SQLite.
    You can modify db_path/table as needed.
    """
    def __init__(self, db_path="D:/AppDevelopment/AI Generated App/AlgoTradingLab_V1.3/data/market_data.db"):
        self.db_path = db_path

    def fetch_ohlcv(self, symbol, start_date, end_date, table="price_data_1min"):
        """
        Loads OHLCV data for a symbol and date range from SQLite.
        """
        query = f"""
            SELECT * FROM {table}
            WHERE symbol = ?
            AND DATE(timestamp) >= ?
            AND DATE(timestamp) <= ?
            ORDER BY timestamp
        """
        with sqlite3.connect(self.db_path) as con:
            df = pd.read_sql_query(query, con, params=(symbol, start_date, end_date))
        return df


def auto_discover_strategies(strategies_path: str = "strategies") -> Dict[str, Type[BaseStrategy]]:
    strat_map = {}
    for fname in os.listdir(strategies_path):
        if fname.endswith('_strategy.py') and fname != 'base_strategy.py':
            module_name = fname[:-3]
            file_path = os.path.join(strategies_path, fname)
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                try:
                    spec.loader.exec_module(module)
                except Exception as e:
                    print(f"[DISCOVERY] Could not import {fname}: {e}")
                    continue
                registered = False
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if (
                        issubclass(obj, BaseStrategy)
                        and obj is not BaseStrategy
                        and name.endswith("Strategy")
                    ):
                        # key = normalized module name before _strategy
                        strat_key = module_name.replace("_strategy", "").lower()
                        strat_map[strat_key] = obj
                        print(f"[DISCOVERY] Registered: {fname} => class {name}")
                        registered = True
                if not registered:
                    print(f"[DISCOVERY] {fname} SKIPPED: No subclass of BaseStrategy ending with 'Strategy'.")
    return strat_map
