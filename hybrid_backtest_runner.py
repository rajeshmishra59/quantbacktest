# quant_backtesting_project/hybrid_backtest_runner.py
# UPGRADED: Now supports interactive batch testing using its own config.

import pandas as pd
import os
import time
from datetime import datetime
import json
import uuid
import traceback
import inspect
from tqdm import tqdm
import numpy as np # <-- Zaroori import add kar diya gaya hai

import config
from data.data_loader import DataLoader
from utils.results_db import save_backtest_results, init_results_db
from utils.strategy_loader import load_strategy
from utils.metrics import calculate_performance_metrics

def get_user_inputs():
    """User se backtest ke liye saare parameters leta hai, ab batch selection ke saath."""
    print("--- Backtest Configuration ---")
    
    # Symbol Selection
    print("\nSelect Symbol Input Method:")
    print("  1. Manual Entry (e.g., RELIANCE,TCS)")
    print("  2. Pre-defined Batch (e.g., NIFTY_50)")
    choice = input("Enter choice (1 or 2): ")
    
    symbols = []
    if choice == '1':
        symbols_str = input("Enter symbols (comma-separated): ")
        symbols = [s.strip().upper() for s in symbols_str.split(',')]
    elif choice == '2':
        print("Available Batches:")
        for i, batch_name in enumerate(config.SYMBOL_BATCHES.keys()):
            print(f"  {i+1}. {batch_name}")
        batch_choice_str = input(f"Enter batch number (1-{len(config.SYMBOL_BATCHES)}): ")
        try:
            batch_choice = int(batch_choice_str) - 1
            batch_name = list(config.SYMBOL_BATCHES.keys())[batch_choice]
            symbols = config.SYMBOL_BATCHES[batch_name]
            print(f"Selected batch '{batch_name}' with {len(symbols)} symbols.")
        except (ValueError, IndexError):
            print("Invalid choice. Exiting.")
            return [], [], [], None, None
    else:
        print("Invalid choice. Exiting.")
        return [], [], [], None, None

    timeframes_str = input("Enter timeframes (e.g., 5min,15min,1H): ")
    timeframes = [tf.strip() for tf in timeframes_str.split(',')]
    
    strategy_files = [f.replace('_strategy.py', '') for f in os.listdir('strategies') if f.endswith('_strategy.py') and not f.startswith('base')]
    print("\nAvailable Strategies:")
    for i, s_name in enumerate(strategy_files):
        print(f"  {i+1}. {s_name}")
    
    strategies_idx_str = input("Enter strategy numbers to test (comma-separated, e.g., 1,3): ")
    strategies_idx = [int(i.strip()) - 1 for i in strategies_idx_str.split(',')]
    strategies = [strategy_files[i] for i in strategies_idx]
    
    start_date = input("Enter Start Date (YYYY-MM-DD): ")
    end_date = input("Enter End Date (YYYY-MM-DD): ")
    
    return symbols, timeframes, strategies, start_date, end_date

def run_fast_backtest(symbol: str, timeframe: str, start_date: str, end_date: str, strategy_name: str, strategy_params: dict):
    run_id = f"{strategy_name}_{symbol}_{timeframe}_{uuid.uuid4().hex[:8]}"
    print(f"\n--- Starting Fast Backtest | Run ID: {run_id} ---")

    try:
        strategy_obj = load_strategy(strategy_name)
        if strategy_obj is None: return

        loader = DataLoader()
        prices_1min_df = loader.fetch_data_for_symbol(symbol, start_date, end_date)
        if prices_1min_df.empty: return

        print(f"Generating signals for {strategy_name}...")
        
        if 'H' in timeframe.upper():
            tf_value = int(''.join(filter(str.isdigit, timeframe))) * 60
        else:
            tf_value = int(''.join(filter(str.isdigit, timeframe)) or 1)

        strategy_instance = strategy_obj(df=prices_1min_df.copy(), symbol=symbol, primary_timeframe=tf_value, **strategy_params)
        signals = strategy_instance.run()
        if signals.empty:
            print("No signals generated. Skipping.")
            return

        trade_log = []
        cash = config.INITIAL_CASH
        equity_curve = [{'timestamp': prices_1min_df.index[0], 'equity': cash}]
        
        entry_signals = signals[signals['entries']].copy()

        for idx, entry_row in tqdm(entry_signals.iterrows(), total=len(entry_signals), desc=f"Simulating {strategy_name} on {symbol}"):
            entry_timestamp = idx
            stop_loss = entry_row['stop_loss']
            target = entry_row.get('target', np.nan)

            trade_data = prices_1min_df[prices_1min_df.index > entry_timestamp]
            if trade_data.empty: continue
            
            entry_price = trade_data.iloc[0]['open']
            
            capital_to_risk = cash * config.RISK_PER_TRADE_PCT
            risk_per_share = abs(entry_price - stop_loss)
            if risk_per_share == 0: continue
            qty = int(capital_to_risk / risk_per_share)
            if qty == 0: continue

            future_candles = trade_data.iloc[1:]
            exit_timestamp = None
            exit_price = 0

            for candle_idx, candle in future_candles.iterrows():
                if candle['low'] <= stop_loss:
                    exit_timestamp, exit_price = candle_idx, stop_loss
                    break
                if pd.notna(target) and candle['high'] >= target:
                    exit_timestamp, exit_price = candle_idx, target
                    break
                if candle_idx in signals.index and signals.loc[candle_idx]['exits']:
                    exit_timestamp, exit_price = candle_idx, candle['open']
                    break
            
            if exit_timestamp is None and not future_candles.empty:
                exit_timestamp, exit_price = future_candles.index[-1], future_candles.iloc[-1]['close']
            elif exit_timestamp is None:
                 continue

            pnl = (exit_price - entry_price) * qty
            brokerage = (entry_price * qty * (config.BROKERAGE_PCT/100)) + (exit_price * qty * (config.BROKERAGE_PCT/100))
            final_pnl = pnl - brokerage
            cash += final_pnl
            
            trade_log.append({'entry_timestamp': entry_timestamp, 'exit_timestamp': exit_timestamp, 'pnl': final_pnl})
            equity_curve.append({'timestamp': exit_timestamp, 'equity': cash})

        trades_df = pd.DataFrame(trade_log)
        equity_df = pd.DataFrame(equity_curve).set_index('timestamp').sort_index()
        performance_summary = calculate_performance_metrics(trades_df, equity_df, config.INITIAL_CASH)

        run_metadata = {
            "run_id": run_id, "run_timestamp": datetime.now().isoformat(),
            "strategy_name": strategy_name, "symbol": symbol, "timeframe": timeframe,
            "start_date": start_date, "end_date": end_date,
            "strategy_params": json.dumps(strategy_params),
            "performance_summary": json.dumps(performance_summary)
        }
        save_backtest_results(run_id, run_metadata, trade_log, performance_summary)
        print(f"--- Finished Backtest for {symbol} | Performance: {performance_summary} ---")

    except Exception as e:
        print(f"!!!!!! ERROR during backtest for {symbol} on {strategy_name} !!!!!!\n{traceback.format_exc()}")

if __name__ == '__main__':
    init_results_db()
    
    symbols_to_run, timeframes_to_run, strategies_to_run, start_date, end_date = get_user_inputs()
    
    if symbols_to_run and start_date and end_date: # Check if user made valid choices
        for strategy_name in strategies_to_run:
            for timeframe in timeframes_to_run:
                for symbol in symbols_to_run:
                    run_fast_backtest(symbol, timeframe, start_date, end_date, strategy_name, {})
    
    print("\n--- All Backtests Complete ---")
