# quant_backtesting_project/utils/results_db.py
# Yeh module backtest ke saare results ko ek alag database mein save karega.

import sqlite3
import pandas as pd
import json
from datetime import datetime

# config.py se DB_PATH import karein.
import sys, os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from config import RESULTS_DB_PATH # Hum config mein ek naya path add karenge

def init_results_db():
    """
    Results database aur zaroori tables banata hai.
    """
    with sqlite3.connect(RESULTS_DB_PATH) as con:
        # Har backtest run ki metadata store karne ke liye
        con.execute("""
            CREATE TABLE IF NOT EXISTS backtest_runs (
                run_id TEXT PRIMARY KEY,
                run_timestamp TEXT,
                strategy_name TEXT,
                symbol TEXT,
                timeframe TEXT,
                start_date TEXT,
                end_date TEXT,
                strategy_params TEXT,
                performance_summary TEXT
            )
        """)
        # Har run ke trades ko store karne ke liye
        con.execute("""
            CREATE TABLE IF NOT EXISTS trade_logs (
                trade_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                entry_timestamp TEXT,
                exit_timestamp TEXT,
                symbol TEXT,
                pnl REAL,
                FOREIGN KEY (run_id) REFERENCES backtest_runs (run_id)
            )
        """)
    print(f"Results database initialized at: {RESULTS_DB_PATH}")

def save_backtest_results(run_id: str, run_metadata: dict, trade_log: list, performance_summary: dict):
    """
    Ek poore backtest run ke results ko database mein save karta hai.
    """
    with sqlite3.connect(RESULTS_DB_PATH) as con:
        # 1. Metadata save karein
        meta_df = pd.DataFrame([run_metadata])
        meta_df.to_sql('backtest_runs', con, if_exists='append', index=False)
        
        # 2. Trades save karein
        if trade_log:
            trades_df = pd.DataFrame(trade_log)
            trades_df['run_id'] = run_id # Har trade mein run_id jodein
            trades_df.to_sql('trade_logs', con, if_exists='append', index=False)
            
    print(f"Successfully saved results for run_id: {run_id}")

# Pehli baar chalane ke liye DB initialize karein
if __name__ == '__main__':
    init_results_db()

