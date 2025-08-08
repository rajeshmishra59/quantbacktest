# quant_backtesting_project/hybrid_backtest_runner.py
# FINAL UPGRADE: Ab yeh ek parameter optimizer hai jo saare cores ka istemaal karta hai.

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

def generate_param_combinations(strategy_name):
    """
    config.py se ek strategy ke liye saare parameter combinations banata hai.
    """
    param_config = config.STRATEGY_OPTIMIZATION_CONFIG.get(strategy_name, config.STRATEGY_OPTIMIZATION_CONFIG['default'])
    if not param_config:
        return [{}] # Agar config nahi hai to default params ke saath ek run

    keys, values = zip(*param_config.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    return combinations

def run_backtest_task(task_info):
    """
    Yeh function ek single backtest task ko chalata hai.
    Ise hum parallel mein call karenge.
    """
    # Default values for robust error message
    symbol, timeframe, strategy_name = None, None, None
    try:
        # Task info se saari jaankari nikalein
        symbol, timeframe, strategy_name, start_date, end_date, strategy_params, data_df = task_info
        
        run_id = f"{strategy_name}_{symbol}_{timeframe}_{uuid.uuid4().hex[:8]}"
        
        prices_1min_df = data_df.copy()
        if prices_1min_df.empty: return

        strategy_obj = load_strategy(strategy_name)
        if strategy_obj is None: return

        if 'H' in timeframe.upper():
            tf_value = int(''.join(filter(str.isdigit, timeframe))) * 60
        else:
            tf_value = int(''.join(filter(str.isdigit, timeframe)) or 1)
        
        strategy_instance = strategy_obj(df=prices_1min_df, symbol=symbol, primary_timeframe=tf_value, **strategy_params)
        signals_df = strategy_instance.run()
        if signals_df.empty: return

        # --- SPEED OPTIMIZATION ---
        prices_1min_df = prices_1min_df.join(signals_df[['entries', 'exits', 'stop_loss', 'target']])
        prices_1min_df[['entries', 'exits']] = prices_1min_df[['entries', 'exits']].fillna(False)
        # --- END OF OPTIMIZATION ---

        portfolio = UpgradedPortfolio(
            initial_cash=config.INITIAL_CASH,
            risk_per_trade_pct=config.RISK_PER_TRADE_PCT,
            max_daily_loss_pct=config.MAX_DAILY_LOSS_PCT,
            brokerage_pct=config.BROKERAGE_PCT,
            slippage_pct=config.SLIPPAGE_PCT
        )

        # Main Event Loop
        for candle in prices_1min_df.itertuples():
            timestamp = candle.Index
            portfolio.on_new_day(timestamp)
            if portfolio.is_trading_halted_today: continue
            if symbol in portfolio.positions:
                portfolio.update_open_positions(timestamp, candle.low, candle.high)
            
            if candle.entries:
                portfolio.request_trade(timestamp, symbol, 'buy', candle.open, candle.stop_loss, candle.target)
            elif candle.exits:
                 if symbol in portfolio.positions:
                    portfolio.request_trade(timestamp, symbol, 'sell', candle.open)
            
            current_prices = pd.Series({symbol: candle.close})
            portfolio.record_equity(timestamp, current_prices)

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
        
        save_backtest_results(run_id, run_metadata, portfolio.trade_log, performance_summary)
    
    except Exception as e:
        tqdm.write(f"ERROR in task {strategy_name}/{symbol}/{timeframe}: {e}")
        pass


if __name__ == '__main__':
    multiprocessing.freeze_support()
    init_results_db()
    
    symbols_to_run, timeframes_to_run, strategies_to_run, start_date, end_date = get_user_inputs()
    
    if symbols_to_run and start_date and end_date:
        
        print("\nPre-loading all required data... Please wait.")
        loader = DataLoader()
        data_cache = {
            symbol: loader.fetch_data_for_symbol(symbol, start_date, end_date)
            for symbol in tqdm(symbols_to_run, desc="Loading Data")
        }
        print("Data loading complete.")

        tasks_with_info = []
        position_counter = 0
        for strategy_name in strategies_to_run:
            param_combos = generate_param_combinations(strategy_name)
            for symbol in symbols_to_run:
                for timeframe in timeframes_to_run:
                    for params in param_combos:
                        tasks_with_info.append(
                            (symbol, timeframe, strategy_name, start_date, end_date, params, data_cache[symbol])
                        )
                        position_counter += 1
        
        print(f"\nTotal {len(tasks_with_info)} optimization tasks to run in parallel...")
        
        cpu_count = os.cpu_count() or 1
        num_processes = min(cpu_count, len(tasks_with_info))
        if num_processes == 0:
            print("No tasks to run.")
            exit()
            
        with multiprocessing.Pool(processes=num_processes) as pool:
            results = list(tqdm(pool.imap_unordered(run_backtest_task, tasks_with_info), total=len(tasks_with_info), desc="Optimizing Strategies"))

    print("\n--- ✅ All Optimization Backtests Complete ---")
