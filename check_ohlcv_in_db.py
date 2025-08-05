# file: check_ohlcv_in_db.py

import sqlite3
import pandas as pd
import argparse

def quick_ohlcv_check(db_path, table_name, limit=100):
    """
    For each symbol: Check OHLCV columns, print stats and count bullish/bearish bars.
    """
    with sqlite3.connect(db_path) as conn:
        # Get columns
        cols = pd.read_sql_query(f"PRAGMA table_info({table_name})", conn)['name'].tolist()
        required = {"open", "high", "low", "close", "volume", "symbol"}
        missing = required - set(cols)
        if missing:
            print(f"❌ Table missing columns: {missing}")
            return
        symbols = pd.read_sql_query(f"SELECT DISTINCT symbol FROM {table_name}", conn)['symbol'].tolist()
        print(f"Symbols found: {symbols}\n")
        for symbol in symbols:
            print(f"== {symbol} ==")
            df = pd.read_sql_query(
                f"SELECT open, high, low, close, volume FROM {table_name} WHERE symbol=? LIMIT {limit}",
                conn, params=(symbol,)
            )
            if df.empty:
                print("  No data.")
                continue
            bullish = (df['close'] > df['open']).sum()
            bearish = (df['close'] < df['open']).sum()
            print(f"  Rows: {len(df)} | Bullish: {bullish} | Bearish: {bearish}")
            print("  open:", df['open'].describe().to_dict())
            print("  close:", df['close'].describe().to_dict())
            print("  high:", df['high'].describe().to_dict())
            print("  low:", df['low'].describe().to_dict())
            print("  volume:", df['volume'].describe().to_dict())
            print()
        print("=== Done ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=str, required=True, help="Path to SQLite DB file")
    parser.add_argument("--table", type=str, required=True, help="Table name to check")
    parser.add_argument("--limit", type=int, default=10, help="Max rows per symbol (default 10)")
    args = parser.parse_args()
    quick_ohlcv_check(args.db, args.table, args.limit)
