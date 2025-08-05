# quantbacktest/data/loader.py
import sqlite3
import pandas as pd

class DataLoader:
    def __init__(self, db_path):
        self.db_path = db_path

    def fetch_ohlcv(self, symbol, start_date, end_date, table='ohlcv'):
        conn = sqlite3.connect(self.db_path)
        query = f"""
        SELECT timestamp, open, high, low, close, volume
        FROM {table}
        WHERE symbol = ? AND timestamp BETWEEN ? AND ?
        ORDER BY timestamp ASC
        """
        df = pd.read_sql_query(query, conn, params=(symbol, start_date, end_date))
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        conn.close()
        return df

    def list_tables(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()
        return tables

    def list_columns(self, table):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({table});")
        columns = [row[1] for row in cursor.fetchall()]
        conn.close()
        return columns
