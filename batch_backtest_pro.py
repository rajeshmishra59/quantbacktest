# batch_backtest_pro.py

import sys
import os
import pandas as pd
import numpy as np
import traceback
import json
import datetime

from utils.ohlcv_resample import resample_ohlcv
from utils.results_db import save_all_trades, save_run_metadata, summarize_trades
from utils.strategy_loader import StrategyLoader, auto_discover_strategies
from config import NIFTY_50, DB_TABLE_NAME, DB_PATH
from datetime import datetime, timedelta

# ... (other imports, CLI/config setup etc, unchanged)

def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    essential = ['open', 'high', 'low', 'close']
    for col in essential:
        if col not in df.columns:
            raise ValueError(f"Missing OHLCV column: {col}")
    df = df.dropna(subset=essential)
    if 'volume' in df.columns:
        df['volume'] = df['volume'].fillna(0)
    return df

def json_safe(obj):
    """Recursively convert all Pandas/Numpy/NaN types to JSON-safe types."""
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [json_safe(x) for x in obj]
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, float) and (pd.isna(obj) or np.isnan(obj)):
        return None
    else:
        return obj

def batch_backtest(strategy_cls, symbols, start_date, end_date, table_name, timeframe, initial_cash=1_00_000, strat_name='unknown'):
    success, fail, all_trades = [], [], []
    code_hash = "testhash"
    loader = StrategyLoader()  # <--- This matches your original loader logic

    for symbol in symbols:
        try:
            print(f"\n[{symbol}] Loading data...", end="")
            df_1min = loader.fetch_ohlcv(symbol, start_date, end_date, table=table_name)
            if df_1min.empty:
                print("NO DATA. Skipping.")
                fail.append(symbol)
                continue
            df_1min = clean_ohlcv(df_1min)

            # --- RESAMPLING PATCH ---
            if timeframe > 1:
                df_tf = resample_ohlcv(df_1min, timeframe)
            else:
                df_tf = df_1min
            # --- END RESAMPLING PATCH ---

            print("Running strategy...", end="")
            config = {
                "symbol": symbol, "start_date": start_date, "end_date": end_date,
                "timeframe": timeframe, "initial_cash": initial_cash
            }
            strat = strategy_cls(
                df_tf,
                symbol=symbol,
                primary_timeframe=timeframe,
                config_dict=config,
                strategy_version=code_hash
            )
            strat.calculate_indicators()
            strat.generate_signals()
            trades = vectorized_trade_sim(strat)
            metrics = summarize_trades(trades)
            run_id = save_run_metadata(
                strat.name, code_hash, symbol, timeframe, start_date, end_date,
                config, metrics, db_path="backtest_results.db"
            )

            # --- JSON SAFE PATCH (MINIMAL) ---
            trades_jsonsafe = [json_safe(trade) for trade in trades]
            save_all_trades(run_id, trades_jsonsafe, db_path="backtest_results.db")
            for trade in trades_jsonsafe:
                if isinstance(trade, dict):  # guard for dict only
                    trade.update({
                        "run_id": run_id,
                        "symbol": symbol,
                        "strategy": strat.name
                    })
            all_trades.extend(trades_jsonsafe)
            # --- END PATCH ---

            print(f" DONE. Trades: {len(trades)}, run_id={run_id}")
            success.append(symbol)
        except Exception as e:
            print(f" ERROR: {e}")
            traceback.print_exc()
            fail.append(symbol)

    print("\nBatch completed.")
    print(f"Success: {len(success)}; Fail: {len(fail)}")
    if all_trades:
        df_trades = pd.DataFrame(all_trades)
        # Standardize columns for CSV, as you had before:
        cols = [
            'run_id', 'symbol', 'strategy', 'entry_time', 'exit_time', 'side', 
            'entry_price', 'exit_price', 'pnl', 'qty', 'stop_loss', 'target', 'exit_reason'
        ]
        extra_cols = [c for c in df_trades.columns if c not in cols]
        df_trades = df_trades[cols + extra_cols]
        TRADE_LOG_FILE = f"all_trades_{strat_name}_{timeframe}m.csv"
        df_trades.to_csv(TRADE_LOG_FILE, index=False)
        print(f"\nAll trades saved to {TRADE_LOG_FILE}")

# ... (rest of your code: CLI/user input, vectorized_trade_sim, etc. — UNCHANGED!)

def vectorized_trade_sim(strat):
    df = strat.df.copy()
    trades = []
    in_trade = False
    entry_idx = None
    entry_price = None
    stop_loss = None
    target = None
    qty = 1
    side = None
    entry_time = None

    for i, row in df.iterrows():
        if not in_trade:
            if row.get('signal_long', False):
                in_trade = True
                entry_idx = i
                entry_price = row['close']
                stop_loss = entry_price * 0.995
                target = entry_price * 1.01
                side = "LONG"
                entry_time = i
            elif row.get('signal_short', False):
                in_trade = True
                entry_idx = i
                entry_price = row['close']
                stop_loss = entry_price * 1.005
                target = entry_price * 0.99
                side = "SHORT"
                entry_time = i
        else:
            # Guard against None values
            if entry_price is None or stop_loss is None or target is None:
                in_trade = False
                continue

            if side == "LONG":
                if row['low'] <= stop_loss:
                    trades.append({
                        "entry_time": entry_time, "exit_time": i,
                        "side": side, "entry_price": entry_price, "exit_price": stop_loss,
                        "pnl": (stop_loss - entry_price) * qty, "qty": qty,
                        "stop_loss": stop_loss, "target": target, "exit_reason": "SL"
                    })
                    in_trade = False
                    entry_price = stop_loss = target = entry_time = None
                elif row['high'] >= target:
                    trades.append({
                        "entry_time": entry_time, "exit_time": i,
                        "side": side, "entry_price": entry_price, "exit_price": target,
                        "pnl": (target - entry_price) * qty, "qty": qty,
                        "stop_loss": stop_loss, "target": target, "exit_reason": "TARGET"
                    })
                    in_trade = False
                    entry_price = stop_loss = target = entry_time = None
            elif side == "SHORT":
                if row['high'] >= stop_loss:
                    trades.append({
                        "entry_time": entry_time, "exit_time": i,
                        "side": side, "entry_price": entry_price, "exit_price": stop_loss,
                        "pnl": (entry_price - stop_loss) * qty, "qty": qty,
                        "stop_loss": stop_loss, "target": target, "exit_reason": "SL"
                    })
                    in_trade = False
                    entry_price = stop_loss = target = entry_time = None
                elif row['low'] <= target:
                    trades.append({
                        "entry_time": entry_time, "exit_time": i,
                        "side": side, "entry_price": entry_price, "exit_price": target,
                        "pnl": (entry_price - target) * qty, "qty": qty,
                        "stop_loss": stop_loss, "target": target, "exit_reason": "TARGET"
                    })
                    in_trade = False
                    entry_price = stop_loss = target = entry_time = None
    # If still in trade at end
    if in_trade and entry_price is not None:
        trades.append({
            "entry_time": entry_time, "exit_time": df.index[-1],
            "side": side, "entry_price": entry_price, "exit_price": df.iloc[-1]['close'],
            "pnl": ((df.iloc[-1]['close'] - entry_price) * qty) if side == "LONG" else ((entry_price - df.iloc[-1]['close']) * qty),
            "qty": qty,
            "stop_loss": stop_loss, "target": target, "exit_reason": "CLOSE"
        })
    return trades


def get_valid_timeframe():
    while True:
        tf = input("Enter timeframe (e.g., 5 or 15): ").strip()
        try:
            tf = int(tf)
            if tf > 0:
                return tf
            else:
                print("Timeframe must be a positive integer.")
        except ValueError:
            print("Invalid input! Please enter an integer.")

def get_valid_date(prompt, default=None):
    while True:
        date_str = input(f"{prompt} [{default}]: ").strip() or default
        if not date_str:
            print("Date cannot be empty.")
            continue
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
            return date_str
        except (ValueError, TypeError):
            print("Please enter a valid date in YYYY-MM-DD format.")

if __name__ == "__main__":
    strat_map = auto_discover_strategies("strategies")
    print("Available strategies:", list(strat_map))
    strat_name = input(f"Enter strategy name for batch {tuple(strat_map.keys())}: ").strip().lower()
    if strat_name not in strat_map:
        raise ValueError(f"Strategy '{strat_name}' not found.")

    timeframe = get_valid_timeframe()

    today_str = datetime.today().strftime("%Y-%m-%d")
    five_years_ago_str = (datetime.today() - timedelta(days=5*365)).strftime("%Y-%m-%d")
    end_date = get_valid_date("Enter end date (YYYY-MM-DD)", default=today_str)
    start_date = get_valid_date("Enter start date (YYYY-MM-DD)", default=five_years_ago_str)

    # --- Symbol selection ---
    use_default = input("Use default NIFTY_50 symbols? (Y/n): ").strip().lower()
    if use_default in ("n", "no"):
        symbols_input = input("Enter comma-separated symbols (e.g. RELIANCE,TCS,INFY): ")
        symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]
    else:
        symbols = NIFTY_50

    print(f"\nBatch backtest for strategy {strat_name}, symbols: {len(symbols)}, dates: {start_date} to {end_date}\n")

    batch_backtest(
    strat_map[strat_name],
    symbols,
    start_date,
    end_date,
    "price_data_1min",  # <- Add your actual table name here!
    timeframe=timeframe,
    strat_name=strat_name
)
