# quantbacktest/main.py

from data.loader import DataLoader
from strategies.alphaone_strategy import AlphaOneStrategy
from strategies.apex_strategy import ApexStrategy
from strategies.numerouno_strategy import NumeroUnoStrategy
from engine.backtest import BacktestEngine
from utils.metrics import Metrics
from utils.results_db import init_results_db, save_run_metadata, save_all_trades

from config import DB_TABLE_NAME  # You should define this in your config.py

import os
from typing import Type

def code_hash_from_file(filename: str) -> str:
    import hashlib
    with open(filename, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def main():
    # ---- CONFIG ----
    db_path = r"D:\AppDevelopment\AI Generated App\AlgoTradingLab_V1.3\data\market_data.db"

    # ---- LOAD AND CHOOSE TABLE ----
    loader = DataLoader(db_path)
    available_tables = loader.list_tables()
    print("Available tables:", available_tables)

    table_name = input(f"Enter table name [{DB_TABLE_NAME}]: ").strip()
    if not table_name:
        table_name = DB_TABLE_NAME

    if table_name not in available_tables:
        print(f"Table '{table_name}' not found, using default '{DB_TABLE_NAME}'.")
        table_name = DB_TABLE_NAME

    # ---- STRATEGY/SYMBOL/TIMEFRAME INPUT ----
    strat_map = {
        "alphaone": AlphaOneStrategy,
        "apex": ApexStrategy,
        "numerouno": NumeroUnoStrategy
    }
    strat_name = input("Enter strategy name (alphaone, apex, numerouno): ").strip().lower()
    if strat_name not in strat_map:
        raise ValueError(f"Strategy '{strat_name}' not supported. Choose from {list(strat_map)}")

    StratClass: Type = strat_map[strat_name]

    symbol = input("Enter symbol: ").strip().upper()
    start_date = input("Enter start date (YYYY-MM-DD): ").strip()
    end_date = input("Enter end date (YYYY-MM-DD): ").strip()
    timeframe = int(input("Enter timeframe (in minutes): ").strip())
    initial_cash = 100_000

    # ---- ENGINE SETUP ----
    print("Setting up DB for results...")
    init_results_db("backtest_results.db")

    print(f"Loading data for {symbol} from {table_name}...")
    df_1min = loader.fetch_ohlcv(symbol, start_date, end_date, table=table_name)

    print("Instantiating strategy...")
    code_file = os.path.join("strategies", f"{strat_name}_strategy.py")
    code_hash = code_hash_from_file(code_file)
    strat_config = {
        "symbol": symbol,
        "start_date": start_date,
        "end_date": end_date,
        "timeframe": timeframe,
        "initial_cash": initial_cash
    }
    strat = StratClass(
        df_1min,
        symbol=symbol,
        primary_timeframe=timeframe,
        config_dict=strat_config,
        strategy_version=code_hash
    )

    print("Running strategy calculations...")
    strat.calculate_indicators()
    strat.generate_signals()

    print("Starting backtest engine...")
    engine = BacktestEngine(initial_cash=initial_cash)
    engine.run(strat)
    results = engine.get_results()
    metrics = Metrics.calculate(results)

    print("\n===== Backtest Results =====")
    print(f"Strategy: {strat.name} | Symbol: {symbol} | Timeframe: {timeframe}min")
    print("Metrics:", metrics)
    if results:
        print("Sample Trades:")
        for trade in results[:3]:
            print(trade)
    else:
        print("No trades generated.")

    # ---- Save to DB ----
    print("Saving results to DB...")
    run_id = save_run_metadata(
        strat.name, code_hash, symbol, timeframe, start_date, end_date,
        strat_config, metrics, db_path="backtest_results.db"
    )
    save_all_trades(run_id, results, db_path="backtest_results.db")
    print(f"Run saved with run_id={run_id}")

if __name__ == "__main__":
    main()
