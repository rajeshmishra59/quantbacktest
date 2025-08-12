# simple_optimizer_runner.py
# KAISE KAREIN: Yeh file ka updated version hai jo optimization ke liye
# ek super-fast, vectorized P&L calculation ka istemal karta hai.
# NAYA: Ab yeh best parameters ko ek JSON file mein save bhi karta hai.

import pandas as pd
import os
from datetime import datetime
import json
import uuid
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


# --- Helper to convert NumPy to Python types ---
def to_py(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if hasattr(obj, 'isoformat'):
        return obj.isoformat()
    return obj


def get_user_inputs():
    """User se backtest ke liye saare parameters leta hai."""
    print("--- Simple Optimizer Configuration ---")
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
    """config.py se ek strategy ke liye saare parameter combinations banata hai."""
    param_config = config.STRATEGY_OPTIMIZATION_CONFIG.get(strategy_name, {})
    if not param_config:
        return [{}]
    keys, values = zip(*param_config.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    return combinations


# --- NAYA SUPER-FAST VECTORIZED TEST FUNCTION ---
def run_fast_vectorized_test(task_info):
    """
    Sirf optimization ke liye banaya gaya hai. Yeh P&L ka anuman (estimate) lagata hai
    bina har candle par loop kiye.
    """
    try:
        symbol, timeframe, strategy_name, start_date, end_date, strategy_params, data_df = task_info

        # --- Step 1: Signals Generate Karein (Yeh pehle se hi vectorized hai) ---
        strategy_obj = load_strategy(strategy_name)
        if not strategy_obj: return None

        if 'H' in timeframe.upper():
            tf_value = int(''.join(filter(str.isdigit, timeframe))) * 60
        else:
            tf_value = int(''.join(filter(str.isdigit, timeframe)) or 1)

        strategy_instance = strategy_obj(df=data_df.copy(), symbol=symbol, primary_timeframe=tf_value, **strategy_params)
        signals_df = strategy_instance.run()

        if signals_df.empty or 'entries' not in signals_df.columns:
            return {'Sharpe Ratio': 0.0, 'Total PnL': 0.0, 'Total Trades': 0}

        # --- Step 2: Vectorized P&L Calculation (Yahi asli jaadu hai) ---
        prices = signals_df['close']
        entries = signals_df['entries']
        exits = signals_df['exits']

        # positions naam ka ek naya column banayein
        # Jab entry signal aata hai (True), to 1 (long position)
        # Jab exit signal aata hai (True), to 0 (no position)
        # Baki samay pichli value (ffill)
        positions = pd.Series(np.nan, index=signals_df.index)
        positions[entries] = 1
        positions[exits] = 0
        positions = positions.ffill().fillna(0)

        # Har din ke returns calculate karein
        daily_returns = prices.pct_change()

        # Strategy ke returns (sirf tab jab position ho)
        strategy_returns = positions.shift(1) * daily_returns
        strategy_returns = strategy_returns.dropna()

        # --- Step 3: Performance Metrics Calculate Karein (Vectorized) ---
        total_pnl_series = (1 + strategy_returns).cumprod() - 1
        
        if strategy_returns.std() == 0:
            sharpe_ratio = 0.0
        else:
            sharpe_ratio = (strategy_returns.mean() / strategy_returns.std()) * np.sqrt(252)

        total_trades = (entries & (positions.shift(1) == 0)).sum()

        performance = {
            "Sharpe Ratio": sharpe_ratio,
            "Total PnL": total_pnl_series.iloc[-1] * config.INITIAL_CASH if not total_pnl_series.empty else 0,
            "Total Trades": int(total_trades)
        }
        return performance

    except Exception:
        return None


# --- PURANA, ACCURATE, LEKIN SLOW BACKTEST FUNCTION ---
# Iska istemal hum best parameters milne ke baad final validation ke liye kar sakte hain.
def run_backtest_task(task_info):
    """Yeh function ek single backtest task ko chalata hai."""
    try:
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

        prices_1min_df = prices_1min_df.join(signals_df[['entries', 'exits', 'stop_loss', 'target']])
        prices_1min_df['entries'] = prices_1min_df['entries'].astype('boolean').fillna(False)
        prices_1min_df['exits'] = prices_1min_df['exits'].astype('boolean').fillna(False)

        portfolio = UpgradedPortfolio(
            initial_cash=config.INITIAL_CASH,
            risk_per_trade_pct=config.RISK_PER_TRADE_PCT,
            max_daily_loss_pct=config.MAX_DAILY_LOSS_PCT,
            brokerage_pct=config.BROKERAGE_PCT,
            slippage_pct=config.SLIPPAGE_PCT
        )

        for candle in prices_1min_df.itertuples():
            timestamp = candle.Index
            portfolio.on_new_day(timestamp)
            if portfolio.is_trading_halted_today: continue
            if symbol in portfolio.positions:
                portfolio.update_open_positions(timestamp, candle.low, candle.high)
            if candle.entries:
                portfolio.request_trade(timestamp, symbol, 'buy', candle.open, candle.stop_loss, candle.target)
            elif candle.exits and symbol in portfolio.positions:
                portfolio.request_trade(timestamp, symbol, 'sell', candle.open)
            portfolio.record_equity(timestamp, pd.Series({symbol: candle.close}))

        if symbol in portfolio.positions:
            last_price = prices_1min_df.iloc[-1]['close']
            portfolio.request_trade(prices_1min_df.index[-1], symbol, 'sell', last_price)
        
        performance_summary = calculate_performance_metrics(pd.DataFrame(portfolio.trade_log), portfolio.equity_df, config.INITIAL_CASH)

        run_metadata = {
            "run_id": run_id, "run_timestamp": datetime.now().isoformat(),
            "strategy_name": strategy_name, "symbol": symbol, "timeframe": timeframe,
            "start_date": start_date, "end_date": end_date,
            "strategy_params": json.dumps({k: to_py(v) for k, v in strategy_params.items()}),
            "performance_summary": json.dumps({k: to_py(v) for k, v in performance_summary.items()})
        }
        
        save_backtest_results(run_id, run_metadata, portfolio.trade_log, performance_summary)
    
    except Exception as e:
        tqdm.write(f"ERROR in task {task_info[0]}/{task_info[2]}: {e}")
        pass


if __name__ == '__main__':
    multiprocessing.freeze_support()
    init_results_db()
    
    # Initialize best_params_per_combo here to avoid "possibly unbound" error
    best_params_per_combo = {}
    
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
        for strategy_name in strategies_to_run:
            param_combos = generate_param_combinations(strategy_name)
            for symbol in symbols_to_run:
                for timeframe in timeframes_to_run:
                    for params in param_combos:
                        tasks_with_info.append(
                            (symbol, timeframe, strategy_name, start_date, end_date, params, data_cache[symbol])
                        )
        
        print(f"\nTotal {len(tasks_with_info)} optimization tasks to run in parallel...")
        
        cpu_count = os.cpu_count() or 1
        num_processes = min(cpu_count, len(tasks_with_info))
        if num_processes == 0:
            print("No tasks to run.")
            exit()
        
        # --- NAYA, OPTIMIZED WORKFLOW ---
        best_params_per_combo = {}

        with multiprocessing.Pool(processes=num_processes) as pool:
            chunksize = max(1, len(tasks_with_info) // (num_processes * 4))
            # NAYE, FAST FUNCTION KO CALL KAREIN
            results = list(tqdm(
                pool.imap_unordered(run_fast_vectorized_test, tasks_with_info, chunksize=chunksize),
                total=len(tasks_with_info),
                desc="Optimizing All Strategies"
            ))

        # Sabhi results se best ko dhoondhein
        for i, perf in enumerate(results):
            if perf:
                task = tasks_with_info[i]
                strategy_name = task[2]
                symbol = task[0]
                combo_key = f"{strategy_name}_on_{symbol}"

                if combo_key not in best_params_per_combo or perf['Sharpe Ratio'] > best_params_per_combo[combo_key]['sharpe']:
                    best_params_per_combo[combo_key] = {
                        'params': task[5],
                        'sharpe': perf['Sharpe Ratio'],
                        'pnl': perf['Total PnL'],
                        'trades': perf['Total Trades']
                    }

    print("\n\n--- ✅ All Optimizations Complete ---")
    print("Found Best Parameter Combination for each Strategy/Symbol:")
    
    # Results ko behtar format mein dikhayein
    for combo, data in sorted(best_params_per_combo.items()):
        print(f"\n- For {combo}:")
        print(f"  - Best Parameters: {json.dumps(data['params'])}")
        print(f"  - Estimated Sharpe Ratio: {data['sharpe']:.2f}")
        print(f"  - Estimated PnL: ₹{data['pnl']:,.2f}")
        print(f"  - Total Trades: {data['trades']}")

    # --- YAHAN BADLAV KIYA GAYA HAI: Results ko file mein save karein ---
    results_filename = "best_optimizer_results.json"
    # NumPy types ko standard Python types mein convert karein
    for combo, data in best_params_per_combo.items():
        data['params'] = {k: to_py(v) for k, v in data['params'].items()}
        data['sharpe'] = to_py(data['sharpe'])
        data['pnl'] = to_py(data['pnl'])
        data['trades'] = to_py(data['trades'])

    with open(results_filename, 'w') as f:
        json.dump(best_params_per_combo, f, indent=4)
    
    print(f"\n\n✅ Best parameters saved to '{results_filename}'.")
    print("\nAb is file ka istemal karke `hybrid_backtest_runner.py` chalaayein.")
