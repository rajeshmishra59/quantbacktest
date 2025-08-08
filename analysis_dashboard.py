# quantbacktest/analysis_dashboard.py
# FINAL PROFESSIONAL VERSION: Walk-Forward results ko samajhne aur dikhane ke liye taiyaar.

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import json
import os
import plotly.express as px
import plotly.graph_objects as go

# config.py se zaroori variables import karein
from config import RESULTS_DB_PATH, INITIAL_CASH

st.set_page_config(page_title="Formula 1 Analysis", layout="wide")

st.title("🚀 Formula 1 - Professional Analysis Dashboard")
st.markdown("Yahan aap apne **Simple Optimization** aur **Walk-Forward** backtests ka vishleshan kar sakte hain.")

# --- Database Connection ---
@st.cache_data(ttl=300) # Data ko 5 minute ke liye cache karein
def load_all_data_from_db():
    """Database se backtest runs aur unke saare trades ko load karta hai."""
    if not os.path.exists(RESULTS_DB_PATH):
        return None, None # Agar file hi nahi hai to None return karein
    try:
        with sqlite3.connect(RESULTS_DB_PATH) as con:
            runs_df = pd.read_sql_query("SELECT * FROM backtest_runs", con)
            # trade_logs table shayad na ho, isliye try-except block
            try:
                trades_df = pd.read_sql_query("SELECT * FROM trade_logs", con, parse_dates=['entry_timestamp', 'exit_timestamp'])
            except sqlite3.OperationalError: # CORRECTED: Sahi exception ko pakdein
                trades_df = pd.DataFrame() # Agar table nahi hai to empty
        return runs_df, trades_df
    except Exception:
        # Kisi bhi anya error ke liye, empty dataframes return karein
        return pd.DataFrame(), pd.DataFrame()


runs_df, trades_df = load_all_data_from_db()

# --- Data Check (Sabse Zaroori Sudhaar) ---
# CORRECTED: Ab dono dataframes ko check karein
if runs_df is None or trades_df is None:
    st.error(f"Database file not found at: {RESULTS_DB_PATH}. Kripya pehle backtest runner chalayein.")
    st.stop()

if runs_df.empty:
    st.warning("Database mein abhi tak koi backtest results nahi hain. Naye results ke liye runner chalayein.")
    st.stop()

# --- Data Processing ---
perf_summary_list = [json.loads(s) for s in runs_df['performance_summary'] if s and s != 'null']
perf_df = pd.DataFrame(perf_summary_list) if perf_summary_list else pd.DataFrame()

# Check karein ki perf_df khali to nahi hai
if not perf_df.empty:
    # Ensure index alignment before concat
    runs_df.reset_index(drop=True, inplace=True)
    perf_df.reset_index(drop=True, inplace=True)
    analysis_df = pd.concat([runs_df.drop(columns=['performance_summary']), perf_df], axis=1).dropna(subset=['strategy_name'])
else:
    analysis_df = runs_df.dropna(subset=['strategy_name'])

analysis_df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Walk-Forward runs ko pehchanne ke liye ek naya column
if 'strategy_name' in analysis_df.columns:
    analysis_df['Run Type'] = np.where(analysis_df['strategy_name'].str.startswith('WF_'), 'Walk-Forward', 'Simple Opt.')
else:
    analysis_df['Run Type'] = 'Simple Opt.'


# --- Sidebar for Controls ---
st.sidebar.header("🔬 Analysis Controls")

# Unique values lene se pehle check karein ki column maujood hai
run_types = ['All'] + sorted(analysis_df['Run Type'].unique().tolist()) if 'Run Type' in analysis_df else ['All']
strategies = ['All'] + sorted(analysis_df['strategy_name'].unique().tolist()) if 'strategy_name' in analysis_df else ['All']
symbols = ['All'] + sorted(analysis_df['symbol'].unique().tolist()) if 'symbol' in analysis_df else ['All']

selected_run_type = st.sidebar.selectbox("Run Type Chunein:", run_types)
selected_strategies = st.sidebar.multiselect("Strategies Chunein:", strategies, default=['All'])
selected_symbols = st.sidebar.multiselect("Symbols Chunein:", symbols, default=['All'])

# Filter logic
filtered_df = analysis_df.copy()
if selected_run_type != 'All':
    filtered_df = filtered_df[filtered_df['Run Type'] == selected_run_type]
if 'All' not in selected_strategies:
    filtered_df = filtered_df[filtered_df['strategy_name'].isin(selected_strategies)]
if 'All' not in selected_symbols:
    filtered_df = filtered_df[filtered_df['symbol'].isin(selected_symbols)]

if filtered_df.empty:
    st.warning("Aapke chune gaye filters ke liye koi data nahi mila. Kripya apne selection ko badlein.")
    st.stop()

selected_run_ids = filtered_df['run_id'].tolist()
# CORRECTED: trades_df ke empty hone ka check
selected_trades_df = trades_df[trades_df['run_id'].isin(selected_run_ids)] if not trades_df.empty else pd.DataFrame()


# --- Main Dashboard with Tabs ---
tab1, tab2, tab3 = st.tabs(["📊 Performance Overview", "📈 Equity Curves", "📋 Trade Analysis"])

with tab1:
    st.header("Performance Metrics")
    display_cols = [
        'Run Type', 'strategy_name', 'symbol', 'timeframe', 'strategy_params', 'Total Return %', 'Max Drawdown %', 
        'Win Rate %', 'Profit Factor', 'Sharpe Ratio', 'Total Trades', 'Total PnL'
    ]
    # Dikhane se pehle columns ko check aur format karein
    display_df = filtered_df.copy()
    for col in display_cols:
        if col not in display_df.columns: display_df[col] = 'N/A'
        
    st.dataframe(
        display_df[display_cols].sort_values(by='Total Return %', ascending=False),
        use_container_width=True
    )

    st.header("🔥 Performance Heatmap")
    if not filtered_df.empty and 'Total Return %' in filtered_df.columns:
        # Pivot table banane se pehle NaN values ko handle karein
        heatmap_data = filtered_df.dropna(subset=['strategy_name', 'symbol', 'Total Return %']).pivot_table(
            index='strategy_name', 
            columns='symbol', 
            values='Total Return %', 
            aggfunc='max'
        )
        if not heatmap_data.empty:
            fig_heatmap = px.imshow(heatmap_data, text_auto=True, aspect="auto", color_continuous_scale='RdYlGn', title="Strategy vs. Symbol Heatmap (Best Return %)")
            st.plotly_chart(fig_heatmap, use_container_width=True)

with tab2:
    st.header("Equity Curve Comparison")
    fig_equity = go.Figure()
    if not selected_trades_df.empty:
        for _, row in filtered_df.iterrows():
            run_id = row['run_id']
            display_name = f"{row.get('strategy_name', 'N/A')} on {row.get('symbol', 'N/A')}"
            
            run_trades = selected_trades_df[selected_trades_df['run_id'] == run_id].sort_values(by='exit_timestamp')
            if not run_trades.empty:
                run_trades['equity'] = INITIAL_CASH + run_trades['pnl'].cumsum()
                fig_equity.add_trace(go.Scatter(x=run_trades['exit_timestamp'], y=run_trades['equity'], mode='lines', name=display_name))
    
    fig_equity.update_layout(title="Equity Curve Comparison", xaxis_title='Date', yaxis_title='Portfolio Equity (₹)', legend_title='Backtest Runs')
    st.plotly_chart(fig_equity, use_container_width=True)

with tab3:
    st.header("Individual Trade Analysis")
    if selected_trades_df.empty:
        st.warning("Chune gaye runs ke liye koi trades nahi mile.")
    else:
        fig_pnl_dist = px.histogram(selected_trades_df, x="pnl", nbins=100, title="PnL Distribution of All Trades", labels={'pnl': 'Profit/Loss per Trade (₹)'})
        st.plotly_chart(fig_pnl_dist, use_container_width=True)

    st.header("Detailed Trade Logs")
    if not selected_trades_df.empty:
        for _, row in filtered_df.iterrows():
            run_id = row['run_id']
            display_name = f"{row.get('strategy_name', 'N/A')} on {row.get('symbol', 'N/A')}"
            with st.expander(f"Trades for: {display_name}"):
                run_trades = selected_trades_df[selected_trades_df['run_id'] == run_id].copy()
                if run_trades.empty:
                    st.write("Is run mein koi trade nahi hua.")
                else:
                    run_trades['pnl'] = run_trades['pnl'].round(2)
                    st.dataframe(run_trades[['entry_timestamp', 'exit_timestamp', 'pnl']], use_container_width=True)
