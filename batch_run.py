# quantbacktest/batch_run.py

from config import NIFTY_50, DB_TABLE_NAME, DB_PATH
from data.loader import DataLoader
from engine.backtest import BacktestEngine
from utils.metrics import Metrics
from utils.results_db import init_results_db, save_run_metadata, save_all_trades
from utils.strategy_loader import auto_discover_strategies   # <-- NEW!

import os
import datetime
from typing import Type
import sys

def code_hash_from_file(filename: str) -> str:
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Strategy code file not found: {filename}")
    import hashlib
    with open(filename, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def batch_backtest(
    strategy_cls: Type, 
    symbols: list, 
    start_date: str, 
    end_date: str, 
    timeframe: int = 15, 
    initial_cash: int = 100_000, 
    db_path: str = DB_PATH, 
    table_name: str = DB_TABLE_NAME,
    strat_name: str = ""
):
    print(f"\nBatch backtest for strategy {strategy_cls.__name__}, symbols: {len(symbols)}, dates: {start_date} to {end_date}")
    init_results_db("backtest_results.db")

    code_file = os.path.join("strategies", f"{strat_name}_strategy.py")
    code_hash = code_hash_from_file(code_file)

    loader = DataLoader(db_path)
    success, fail = [], []

    for symbol in symbols:
        print(f"\n[{symbol}] Loading data...", end="")
        df_1min = loader.fetch_ohlcv(symbol, start_date, end_date, table=table_name)
        if df_1min.empty:
            print("NO DATA. Skipping.")
            fail.append(symbol)
            continue
        print("Running strategy...", end="")
        config = {
            "symbol": symbol, "start_date": start_date, "end_date": end_date,
            "timeframe": timeframe, "initial_cash": initial_cash
        }
        try:
            strat = strategy_cls(
                df_1min,
                symbol=symbol,
                primary_timeframe=timeframe,
                config_dict=config,
                strategy_version=code_hash
            )
            strat.calculate_indicators()
            strat.generate_signals()
            engine = BacktestEngine(initial_cash=initial_cash)
            engine.run(strat)
            results = engine.get_results()
            metrics = Metrics.calculate(results)
            run_id = save_run_metadata(
                strat.name, code_hash, symbol, timeframe, start_date, end_date,
                config, metrics, db_path="backtest_results.db"
            )
            save_all_trades(run_id, results, db_path="backtest_results.db")
            print(f" DONE. Trades: {len(results)}, run_id={run_id}")
            success.append(symbol)
        except Exception as e:
            print(f" ERROR: {e}")
            fail.append(symbol)

    print("\nBatch complete!")
    print(f"Success: {len(success)}/{len(symbols)} | Failed: {len(fail)}")
    if fail:
        print("Failed symbols:", fail)

if __name__ == "__main__":
    # --- Automated strategy discovery ---
    strat_map = auto_discover_strategies("strategies")
    print("Available strategies:", list(strat_map))
    strat_name = input(f"Enter strategy name for batch {tuple(strat_map.keys())}: ").strip().lower()
    if strat_name not in strat_map:
        print(f"Strategy '{strat_name}' not supported. Choose from {list(strat_map)}")
        sys.exit(1)

    timeframe = int(input("Enter timeframe (e.g., 5 or 15): ").strip())
    end_date = input("Enter end date (YYYY-MM-DD) [today]: ").strip() or datetime.date.today().strftime("%Y-%m-%d")
    start_date = input("Enter start date (YYYY-MM-DD) [5 years ago]: ").strip()
    if not start_date:
        start_date = (datetime.date.today() - datetime.timedelta(days=5*365)).strftime("%Y-%m-%d")

    symbols = NIFTY_50  # You may add a prompt here for custom symbol list if needed

    batch_backtest(
        strat_map[strat_name],
        symbols,
        start_date,
        end_date,
        timeframe=timeframe,
        strat_name=strat_name
    )
