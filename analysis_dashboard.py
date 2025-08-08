# quantbacktest/analysis_dashboard.py
# FINAL UPGRADE: Ek professional-grade interactive analysis tool.

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

st.title("🚀 Formula 1 - Advanced Backtest Analysis")
st.markdown("Yahan aap apne sabhi backtest results ka gehraai se vishleshan kar sakte hain.")

# --- Database Connection ---
@st.cache_data(ttl=300) # Data ko 5 minute ke liye cache karein
def load_all_data_from_db():
    """Database se backtest runs aur unke saare trades ko load karta hai."""
    if not os.path.exists(RESULTS_DB_PATH):
        return None, None
    try:
        with sqlite3.connect(RESULTS_DB_PATH) as con:
            runs_df = pd.read_sql_query("SELECT * FROM backtest_runs", con)
            trades_df = pd.read_sql_query("SELECT * FROM trade_logs", con, parse_dates=['entry_timestamp', 'exit_timestamp'])
        return runs_df, trades_df
    except sqlite3.Error:
        # Agar table maujood na ho to empty DataFrames return karein
        return pd.DataFrame(), pd.DataFrame()


runs_df, trades_df = load_all_data_from_db()

# --- CORRECTED & ROBUST DATA CHECK ---
if runs_df is None or trades_df is None:
    st.error(f"Database file not found at: {RESULTS_DB_PATH}. Kripya pehle backtest runner chalayein.")
    st.stop()

if runs_df.empty:
    st.warning("Database mein abhi tak koi backtest results nahi hain.")
    st.stop()

# --- Data Processing ---
perf_summary_list = [json.loads(s) for s in runs_df['performance_summary'] if s]
if not perf_summary_list:
    st.warning("Performance data corrupt ya missing hai. Kuch runs table mein nahi dikhenge.")
    perf_df = pd.DataFrame()
else:
    perf_df = pd.DataFrame(perf_summary_list)

analysis_df = pd.concat([runs_df.drop(columns=['performance_summary']), perf_df], axis=1).dropna(subset=['strategy_name'])

# --- YAHAN BADLAV KIYA GAYA HAI: Infinite values ko handle karein ---
analysis_df.replace([np.inf, -np.inf], np.nan, inplace=True)


# --- Sidebar for Granular Controls ---
st.sidebar.header("🔬 Analysis Controls")

strategies = ['All'] + sorted(analysis_df['strategy_name'].unique().tolist())
symbols = ['All'] + sorted(analysis_df['symbol'].unique().tolist())
timeframes = ['All'] + sorted(analysis_df['timeframe'].unique().tolist())

selected_strategies = st.sidebar.multiselect("Strategies Chunein:", strategies, default=['All'])
selected_symbols = st.sidebar.multiselect("Symbols Chunein:", symbols, default=['All'])
selected_timeframes = st.sidebar.multiselect("Timeframes Chunein:", timeframes, default=['All'])

# Filter logic
filtered_df = analysis_df.copy()
if 'All' not in selected_strategies:
    filtered_df = filtered_df[filtered_df['strategy_name'].isin(selected_strategies)]
if 'All' not in selected_symbols:
    filtered_df = filtered_df[filtered_df['symbol'].isin(selected_symbols)]
if 'All' not in selected_timeframes:
    filtered_df = filtered_df[filtered_df['timeframe'].isin(selected_timeframes)]

if filtered_df.empty:
    st.warning("Aapke chune gaye filters ke liye koi data nahi mila. Kripya apne selection ko badlein.")
    st.stop()

selected_run_ids = filtered_df['run_id'].tolist()
selected_trades_df = trades_df[trades_df['run_id'].isin(selected_run_ids)] if not trades_df.empty else pd.DataFrame()


# --- Main Dashboard with Tabs ---
tab1, tab2, tab3 = st.tabs(["📊 Performance Overview", "📈 Equity Curves", "📋 Trade Analysis"])

with tab1:
    st.header("Performance Metrics")
    st.markdown("Chune gaye backtest runs ke mukhya performance metrics.")
    
    display_cols = [
        'strategy_name', 'symbol', 'timeframe', 'Total Return %', 'Max Drawdown %', 
        'Win Rate %', 'Profit Factor', 'Sharpe Ratio', 'Total Trades', 'Total PnL'
    ]
    for col in display_cols:
        if col not in filtered_df.columns: filtered_df[col] = 'N/A'
        
    # --- YAHAN BADLAV KIYA GAYA HAI: Numbers ko format karein ---
    st.dataframe(
        filtered_df[display_cols]
        .sort_values(by='Total Return %', ascending=False)
        .style.format({
            "Total Return %": "{:.2f}",
            "Max Drawdown %": "{:.2f}",
            "Win Rate %": "{:.2f}",
            "Profit Factor": "{:.2f}",
            "Sharpe Ratio": "{:.2f}",
            "Total PnL": "{:,.2f}"
        }, na_rep="No Loss"), # Infinite values ab 'No Loss' dikhenge
        use_container_width=True
    )

    st.header("🔥 Performance Heatmap")
    st.markdown("Ek nazar mein dekhein kaun si strategy-symbol jodi sabse behtar hai (Total Return % ke adhaar par).")

    if not filtered_df.empty and 'Total Return %' in filtered_df.columns:
        heatmap_data = filtered_df.pivot_table(
            index='strategy_name', 
            columns='symbol', 
            values='Total Return %'
        )
        if not heatmap_data.empty:
            fig_heatmap = px.imshow(
                heatmap_data,
                text_auto=True,  # Enable text display
                aspect="auto",
                color_continuous_scale='RdYlGn',
                title="Strategy vs. Symbol Heatmap (Total Return %)"
            )
            fig_heatmap.update_traces(texttemplate='%{z:.2f}')  # Format numbers to 2 decimal places
            st.plotly_chart(fig_heatmap, use_container_width=True)
        else:
            st.info("Heatmap banane ke liye paryaapt data nahi hai.")

with tab2:
    st.header("Equity Curve Comparison")
    st.markdown("Chune gaye runs ki equity curve ki tulna.")

    fig_equity = go.Figure()
    if not trades_df.empty:
        for _, row in filtered_df.iterrows():
            run_id = row['run_id']
            display_name = f"{row['strategy_name']} on {row['symbol']} ({row['timeframe']})"
            run_trades = trades_df[trades_df['run_id'] == run_id].sort_values(by='exit_timestamp')
            if not run_trades.empty:
                run_trades['equity'] = INITIAL_CASH + run_trades['pnl'].cumsum()
                fig_equity.add_trace(go.Scatter(
                    x=run_trades['exit_timestamp'], 
                    y=run_trades['equity'], 
                    mode='lines', 
                    name=display_name
                ))
    
    fig_equity.update_layout(
        title="Equity Curve Comparison",
        xaxis_title='Date', 
        yaxis_title='Portfolio Equity (₹)', 
        legend_title='Backtest Runs'
    )
    st.plotly_chart(fig_equity, use_container_width=True)

with tab3:
    st.header("Individual Trade Analysis")
    
    if selected_trades_df.empty:
        st.warning("Chune gaye runs ke liye koi trades nahi mile.")
    else:
        st.markdown("Har trade se hone waale profit aur loss ka vitran.")
        fig_pnl_dist = px.histogram(
            selected_trades_df, 
            x="pnl", 
            nbins=100,
            title="PnL Distribution of All Trades",
            labels={'pnl': 'Profit/Loss per Trade (₹)'}
        )
        st.plotly_chart(fig_pnl_dist, use_container_width=True)

    st.header("Detailed Trade Logs")
    st.markdown("Har run ke liye saare individual trades ki jaankari.")
    
    if not trades_df.empty:
        for _, row in filtered_df.iterrows():
            run_id = row['run_id']
            display_name = f"{row['strategy_name']} on {row['symbol']} ({row['timeframe']})"
            with st.expander(f"Trades for: {display_name}"):
                run_trades = trades_df[trades_df['run_id'] == run_id].copy()
                if run_trades.empty:
                    st.write("Is run mein koi trade nahi hua.")
                else:
                    run_trades['pnl'] = run_trades['pnl'].round(2)
                    st.dataframe(run_trades[['entry_timestamp', 'exit_timestamp', 'pnl']], use_container_width=True)
