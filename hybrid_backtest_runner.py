# quant_backtesting_project/hybrid_backtest_runner.py
# UPGRADED: Ab yeh multiprocessing ka istemaal karke backtests ko parallel mein chalata hai.

import pandas as pd
import os
import time
from datetime import datetime
import json
import uuid
import traceback
import itertools
import multiprocessing
from tqdm import tqdm
import numpy as np

# --- Zaroori Imports ---
import config
from data.data_loader import DataLoader
from utils.results_db import save_backtest_results, init_results_db
from utils.strategy_loader import load_strategy
from utils.metrics import calculate_performance_metrics
from engine.upgraded_portfolio import UpgradedPortfolio

def get_user_inputs():
    """User se backtest ke liye saare parameters leta hai."""
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

def run_backtest_task(task_info):
    """
    Yeh function ek single backtest task ko chalata hai.
    Ise hum parallel mein call karenge.
    """
    # Task info se saari jaankari nikalein
    symbol, timeframe, strategy_name, start_date, end_date, strategy_params, position_id = task_info
    
    run_id = f"{strategy_name}_{symbol}_{timeframe}_{uuid.uuid4().hex[:8]}"
    
    # Har process (worker) ka apna data loader hoga
    loader = DataLoader()
    prices_1min_df = loader.fetch_data_for_symbol(symbol, start_date, end_date)
    if prices_1min_df.empty:
        # Tqdm ke liye ek dummy message print karein
        tqdm.write(f"[{position_id}] No data for {symbol}. Skipping.")
        return

    # Strategy load aur signals generate karein
    strategy_obj = load_strategy(strategy_name)
    if strategy_obj is None: return

    if 'H' in timeframe.upper():
        tf_value = int(''.join(filter(str.isdigit, timeframe))) * 60
    else:
        tf_value = int(''.join(filter(str.isdigit, timeframe)) or 1)
    
    strategy_instance = strategy_obj(df=prices_1min_df.copy(), symbol=symbol, primary_timeframe=tf_value, **strategy_params)
    signals_df = strategy_instance.run()
    if signals_df.empty:
        tqdm.write(f"[{position_id}] No signals for {strategy_name} on {symbol}. Skipping.")
        return

    # Portfolio Engine Taiyaar Karein
    portfolio = UpgradedPortfolio(
        initial_cash=config.INITIAL_CASH,
        risk_per_trade_pct=config.RISK_PER_TRADE_PCT,
        max_daily_loss_pct=config.MAX_DAILY_LOSS_PCT,
        brokerage_pct=config.BROKERAGE_PCT,
        slippage_pct=config.SLIPPAGE_PCT
    )

    # Main Event Loop - Har task ka apna progress bar hoga
    progress_bar_desc = f"#{position_id}: {strategy_name[:10]} on {symbol[:10]}"
    
    # tqdm ka 'position' argument har progress bar ko uski sahi line par rakhta hai
    for timestamp, candle in tqdm(prices_1min_df.iterrows(), total=len(prices_1min_df), desc=progress_bar_desc, position=position_id):
        portfolio.on_new_day(timestamp)
        if portfolio.is_trading_halted_today: continue
        if symbol in portfolio.positions:
            portfolio.update_open_positions(timestamp, candle['low'], candle['high'])
        if timestamp in signals_df.index and signals_df.loc[timestamp]['entries']:
            signal_row = signals_df.loc[timestamp]
            portfolio.request_trade(timestamp, symbol, 'buy', candle['open'], signal_row.get('stop_loss'), signal_row.get('target'))
        elif timestamp in signals_df.index and signals_df.loc[timestamp]['exits']:
             if symbol in portfolio.positions:
                portfolio.request_trade(timestamp, symbol, 'sell', candle['open'])
        current_prices = pd.Series({symbol: candle['close']})
        portfolio.record_equity(timestamp, current_prices)

    # Backtest ke baad Final Hisaab-Kitaab
    if symbol in portfolio.positions:
        last_price = prices_1min_df.iloc[-1]['close']
        portfolio.request_trade(prices_1min_df.index[-1], symbol, 'sell', last_price)
    
    performance_summary = calculate_performance_metrics(pd.DataFrame(portfolio.trade_log), portfolio.equity_df, config.INITIAL_CASH)

    run_metadata = {
        "run_id": run_id, "run_timestamp": datetime.now().isoformat(),
        "strategy_name": strategy_name, "symbol": symbol, "timeframe": timeframe,
        "start_date": start_date, "end_date": end_date,
        "strategy_params": json.dumps(strategy_params),
        "performance_summary": json.dumps(performance_summary)
    }
    
    # DB mein save karein
    try:
        save_backtest_results(run_id, run_metadata, portfolio.trade_log, performance_summary)
        tqdm.write(f"--- ✔️ Finished & Saved: {progress_bar_desc} | Return: {performance_summary.get('Total Return %', 'N/A')}% ---")
    except Exception as e:
        tqdm.write(f"--- ❌ DB Error for {run_id}: {e} ---")


if __name__ == '__main__':
    # Yeh zaroori hai Windows par multiprocessing ke liye
    multiprocessing.freeze_support()
    
    init_results_db()
    
    symbols_to_run, timeframes_to_run, strategies_to_run, start_date, end_date = get_user_inputs()
    
    if symbols_to_run and start_date and end_date:
        # 1. Saare possible backtest combinations (tasks) ki ek list banayein
        all_combinations = list(itertools.product(symbols_to_run, timeframes_to_run, strategies_to_run))
        
        tasks_with_info = [
            (*combo, start_date, end_date, {}, i) 
            for i, combo in enumerate(all_combinations)
        ]
        
        print(f"\nTotal {len(tasks_with_info)} backtest tasks to run in parallel...")
        
        # 2. Multiprocessing Pool banayein
        # Yeh aapke computer ke sabhi available CPU cores ka istemaal karega
        num_processes = min(os.cpu_count() or 1, len(tasks_with_info))
        
        with multiprocessing.Pool(processes=num_processes) as pool:
            # pool.map har ek task ko 'run_backtest_task' function par chalayega
            # Har task ek alag CPU core par chalega
            list(pool.map(run_backtest_task, tasks_with_info))

    print("\n--- ✅ All Parallel Backtests Complete ---")
