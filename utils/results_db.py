# quantbacktest/utils/results_db.py
import sqlite3
import json
from datetime import datetime

def init_results_db(db_path="backtest_results.db"):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''
    CREATE TABLE IF NOT EXISTS backtest_runs (
        run_id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        strategy TEXT,
        strategy_code_hash TEXT,
        symbol TEXT,
        timeframe INTEGER,
        start_date TEXT,
        end_date TEXT,
        config_json TEXT,
        metrics_json TEXT
    )
    ''')
    c.execute('''
    CREATE TABLE IF NOT EXISTS backtest_trades (
        trade_id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id INTEGER,
        symbol TEXT,
        entry_price REAL,
        exit_price REAL,
        qty INTEGER,
        pnl REAL,
        extra_info TEXT
    )
    ''')
    conn.commit()
    conn.close()

def save_run_metadata(strategy, code_hash, symbol, timeframe, start_date, end_date, config_dict, metrics_dict, db_path="backtest_results.db"):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute('''
        INSERT INTO backtest_runs (
            timestamp, strategy, strategy_code_hash, symbol, timeframe, start_date, end_date, config_json, metrics_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        now, strategy, code_hash, symbol, timeframe, start_date, end_date,
        json.dumps(config_dict), json.dumps(metrics_dict)
    ))
    run_id = c.lastrowid
    conn.commit()
    conn.close()
    return run_id

def save_all_trades(run_id, trades, db_path="backtest_results.db"):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    for trade in trades:
        c.execute('''
            INSERT INTO backtest_trades (
                run_id, symbol, entry_price, exit_price, qty, pnl, extra_info
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            run_id,
            trade.get("symbol"),
            trade.get("entry_price", 0),
            trade.get("exit_price", 0),
            trade.get("qty", 0),
            trade.get("pnl", 0),
            json.dumps(trade)
        ))
    conn.commit()
    conn.close()
