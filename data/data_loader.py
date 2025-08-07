# quant_backtesting_project/data/data_loader.py
# Yeh module database se data load karne ke liye zimmedar hai.

import sqlite3
import pandas as pd
import os

# config.py se DB_PATH import karein.
# Iske liye humein system path mein project root ko add karna hoga.
import sys
# __file__ is data_loader.py's path. os.path.dirname gives data/. os.path.dirname again gives project root.
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# CORRECTED IMPORT: Ab sahi variable import hoga
from config import MARKET_DATA_DB_PATH, DB_TABLE_NAME

class DataLoader:
    def __init__(self):
        """
        DataLoader ko initialize karta hai aur check karta hai ki DB file maujood hai ya nahi.
        """
        # CORRECTED PATH USAGE
        self.db_path = MARKET_DATA_DB_PATH
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"Database file not found at path: {self.db_path}")
        print(f"DataLoader initialized. Connecting to DB at: {self.db_path}")

    def fetch_data_for_symbol(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Ek specific symbol aur date range ke liye data fetch karta hai.
        """
        query = f"""
            SELECT timestamp, open, high, low, close, volume
            FROM {DB_TABLE_NAME}
            WHERE symbol = ?
            AND DATE(timestamp) BETWEEN ? AND ?
            ORDER BY timestamp ASC
        """
        try:
            with sqlite3.connect(self.db_path) as con:
                df = pd.read_sql_query(query, con, params=(symbol, start_date, end_date),
                                       parse_dates=['timestamp'], index_col='timestamp')

            if df.empty:
                print(f"Warning: No data found for {symbol} between {start_date} and {end_date}.")

            return df

        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()

# --- Example Usage ---
if __name__ == '__main__':
    print("Running DataLoader example...")
    loader = DataLoader()

    reliance_data = loader.fetch_data_for_symbol(
        symbol='RELIANCE',
        start_date='2023-01-01',
        end_date='2023-01-31'
    )

    if not reliance_data.empty:
        print("\nSuccessfully fetched data for RELIANCE:")
        print(reliance_data.head())
    else:
        print("\nCould not fetch data for RELIANCE for the given period.")