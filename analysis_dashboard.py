# quantbacktest/analysis_dashboard.py
# A new tool to analyze and compare all backtest results from the database.

import streamlit as st
import pandas as pd
import sqlite3
import json
import os

from config import RESULTS_DB_PATH

st.set_page_config(page_title="Backtest Analysis Dashboard", layout="wide")

st.title("🚀 Backtest Analysis Dashboard")

if not os.path.exists(RESULTS_DB_PATH):
    st.error(f"Database file not found at: {RESULTS_DB_PATH}")
    st.stop()

@st.cache_data
def load_results():
    """Loads all backtest runs from the database."""
    with sqlite3.connect(RESULTS_DB_PATH) as con:
        df = pd.read_sql_query("SELECT * FROM backtest_runs", con)
    return df

results_df = load_results()

if results_df.empty:
    st.warning("No backtest results found in the database yet.")
    st.stop()

# --- Data Processing ---
# Performance summary (JSON string) ko alag-alag columns mein convert karein
perf_summary_list = []
for summary_str in results_df['performance_summary']:
    try:
        perf_summary_list.append(json.loads(summary_str))
    except (json.JSONDecodeError, TypeError):
        perf_summary_list.append({})

perf_df = pd.DataFrame(perf_summary_list)

# Original DataFrame ke saath jodein
analysis_df = pd.concat([results_df.drop(columns=['performance_summary']), perf_df], axis=1)

# --- Dashboard Display ---
st.header("All Backtest Runs")

# Columns ko aache se arrange karein
display_cols = [
    'strategy_name', 'symbol', 'timeframe', 'Total Return %', 'Max Drawdown %',
    'Win Rate %', 'Profit Factor', 'Total Trades', 'Total PnL',
    'start_date', 'end_date', 'run_timestamp'
]
# Ensure all display columns exist, fill with None if not
for col in display_cols:
    if col not in analysis_df.columns:
        analysis_df[col] = None

st.dataframe(analysis_df[display_cols].sort_values(by='Total Return %', ascending=False), use_container_width=True)

st.header("Strategy Performance Summary")

# Strategy ke hisaab se results ko group karein
strategy_summary = analysis_df.groupby('strategy_name').agg(
    Avg_Return_pct=('Total Return %', 'mean'),
    Avg_Win_Rate_pct=('Win Rate %', 'mean'),
    Avg_Profit_Factor=('Profit Factor', 'mean'),
    Total_Runs=('run_id', 'count')
).sort_values(by='Avg_Return_pct', ascending=False)

st.dataframe(strategy_summary.style.format("{:.2f}"), use_container_width=True)

