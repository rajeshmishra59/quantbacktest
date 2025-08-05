# file: market_db_check_terminal.py

import sqlite3
import pandas as pd
import argparse
import sys
from typing import List

def get_table_list(db_path: str) -> List[str]:
    """Return a list of table names in the given SQLite database."""
    try:
        with sqlite3.connect(db_path) as conn:
            tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            return [t[0] for t in tables]
    except Exception as e:
        print(f"❌ Error reading database tables: {e}")
        sys.exit(1)

def check_db_coverage(db_path: str, table_name: str) -> None:
    """
    Analyze the data coverage for each symbol in the given table of the SQLite DB.
    Prints symbol list, global and per-symbol data range, and potential missing data.
    """
    try:
        with sqlite3.connect(db_path) as conn:
            # Get column info and verify required columns
            columns = pd.read_sql_query(f"PRAGMA table_info({table_name})", conn)
            if not {'symbol', 'timestamp'}.issubset(set(columns['name'])):
                print(f"\n❌ Table '{table_name}' missing required columns. Columns found: {columns['name'].tolist()}")
                return
            query = f"""
                SELECT symbol, MIN(timestamp) AS min_date, MAX(timestamp) AS max_date, COUNT(*) AS rows
                FROM {table_name}
                GROUP BY symbol
            """
            df = pd.read_sql_query(query, conn)
    except Exception as e:
        print(f"\n❌ Error querying database: {e}")
        return

    if df.empty:
        print(f"\n❗ Table '{table_name}' has no data.")
        return

    # Robust date parsing
    for col in ['min_date', 'max_date']:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    if df['min_date'].isnull().any() or df['max_date'].isnull().any():
        print("⚠️ Warning: Some dates could not be parsed and are set to NaT.")

    # 1. Print all symbols
    symbols = sorted(df['symbol'].dropna().unique())
    print(f"\nSymbols in DB ({len(symbols)}):\n{', '.join(symbols)}\n")

    # 2. Global min/max
    global_min = df['min_date'].min()
    global_max = df['max_date'].max()
    if pd.isnull(global_min) or pd.isnull(global_max):
        print("❗ Error: Unable to determine global date range (date parse issue).")
        return
    print(f"Global Date Range: {global_min.date()}  to  {global_max.date()}\n")

    # 3. Each symbol coverage
    print("Symbol-wise Data Range:")
    for _, row in df.iterrows():
        min_m = row['min_date'].strftime('%Y-%m') if pd.notnull(row['min_date']) else "N/A"
        max_m = row['max_date'].strftime('%Y-%m') if pd.notnull(row['max_date']) else "N/A"
        min_date = row['min_date'].date() if pd.notnull(row['min_date']) else "N/A"
        max_date = row['max_date'].date() if pd.notnull(row['max_date']) else "N/A"
        print(f"  {row['symbol']:>12}: {min_date} to {max_date}  ({min_m} to {max_m}) [{row['rows']} rows]")

    # 4. Find missing data/gap symbols
    print("\nPotential Missing Data Symbols (not full coverage):")
    missing = df[(df['min_date'] > global_min) | (df['max_date'] < global_max)]
    if not missing.empty:
        for _, row in missing.iterrows():
            min_date = row['min_date'].date() if pd.notnull(row['min_date']) else "N/A"
            max_date = row['max_date'].date() if pd.notnull(row['max_date']) else "N/A"
            print(f"  {row['symbol']:>12}: {min_date} to {max_date}  (Rows: {row['rows']})")
    else:
        print("  None! All symbols cover the full range.")

def main():
    parser = argparse.ArgumentParser(description="Check coverage and health of a market data SQLite database.")
    parser.add_argument("--db", type=str, default="data/market_data.db", help="Path to SQLite DB file.")
    parser.add_argument("--table", type=str, default=None, help="Table name to check (default: auto-select if only one exists).")
    args = parser.parse_args()

    tables = get_table_list(args.db)
    if not tables:
        print("❌ No tables found in database!")
        return

    print("Available tables in DB:", tables)
    table = args.table
    if table and table not in tables:
        print(f"❌ Specified table '{table}' not found in DB. Please choose from: {tables}")
        return
    if not table:
        if len(tables) == 1:
            table = tables[0]
            print(f"Auto-selecting only table: {table}")
        else:
            print("Multiple tables found. Use --table argument to specify which to check.")
            return

    check_db_coverage(args.db, table)

if __name__ == "__main__":
    main()
