# quant_backtesting_project/hybrid_backtest_runner.py
# FINAL PROFESSIONAL VERSION: Implements Walk-Forward Optimization.

import pandas as pd
import os
import time
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
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

# --- Helper Functions ---

def get_user_inputs():
    """User se backtest ke liye saare parameters leta hai."""
    print("--- Walk-Forward Backtest Configuration ---")
    # Symbol Selection
    print("\nSelect Symbol Input Method:")
    print("  1. Manual Entry (e.g., RELIANCE)")
    print("  2. Pre-defined Batch (e.g., SMALL_TEST)")
    choice = input("Enter choice (1 or 2): ")
    
    symbols = []
    if choice == '1':
        symbols_str = input("Enter a SINGLE symbol for Walk-Forward Analysis: ")
        symbols = [s.strip().upper() for s in symbols_str.split(',')][:1] # Sirf pehla symbol lein
    elif choice == '2':
        print("Available Batches:")
        for i, batch_name in enumerate(config.SYMBOL_BATCHES.keys()):
            print(f"  {i+1}. {batch_name}")
        batch_choice_str = input(f"Enter batch number (1-{len(config.SYMBOL_BATCHES)}): ")
        try:
            batch_choice = int(batch_choice_str) - 1
            batch_name = list(config.SYMBOL_BATCHES.keys())[batch_choice]
            symbols = config.SYMBOL_BATCHES[batch_name]
            print(f"Selected batch '{batch_name}'. Note: Walk-forward runs one symbol at a time.")
            symbols = symbols[:1] # Sirf pehla symbol lein
        except (ValueError, IndexError):
            print("Invalid choice. Exiting.")
            return None, None, None, None, None
    else:
        print("Invalid choice. Exiting.")
        return None, None, None, None, None

    timeframe = input("Enter a SINGLE timeframe (e.g., 15min): ")
    
    strategy_files = [f.replace('_strategy.py', '') for f in os.listdir('strategies') if f.endswith('_strategy.py') and not f.startswith('base')]
    print("\nAvailable Strategies:")
    for i, s_name in enumerate(strategy_files):
        print(f"  {i+1}. {s_name}")
    
    strategy_idx_str = input("Enter a SINGLE strategy number to test: ")
    strategy_idx = int(strategy_idx_str.strip()) - 1
    strategy_name = strategy_files[strategy_idx]
    
    start_date = input("Enter Full Period Start Date (YYYY-MM-DD): ")
    end_date = input("Enter Full Period End Date (YYYY-MM-DD): ")
    
    return symbols[0], timeframe, strategy_name, start_date, end_date


def generate_param_combinations(strategy_name):
    param_config = config.STRATEGY_OPTIMIZATION_CONFIG.get(strategy_name, {})
    if not param_config: return [{}]
    keys, values = zip(*param_config.items())
    return [dict(zip(keys, v)) for v in itertools.product(*values)]

def run_single_backtest(task_info):
    """
    Ek single backtest chalata hai. Iska istemaal optimization phase mein hota hai.
    """
    try:
        symbol, timeframe, strategy_name, strategy_params, data_df = task_info
        
        if data_df.empty: return None

        strategy_obj = load_strategy(strategy_name)
        if not strategy_obj: return None

        tf_value = int(''.join(filter(str.isdigit, timeframe)) or 1)
        if 'H' in timeframe.upper(): tf_value *= 60
        
        strategy_instance = strategy_obj(df=data_df.copy(), symbol=symbol, primary_timeframe=tf_value, **strategy_params)
        signals_df = strategy_instance.run()
        if signals_df.empty: return None

        prices_1min_df = data_df.join(signals_df[['entries', 'exits', 'stop_loss', 'target']])
        prices_1min_df[['entries', 'exits']] = prices_1min_df[['entries', 'exits']].fillna(False)

        portfolio = UpgradedPortfolio(config.INITIAL_CASH, config.RISK_PER_TRADE_PCT, config.MAX_DAILY_LOSS_PCT, config.BROKERAGE_PCT, config.SLIPPAGE_PCT)

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
        
        performance = calculate_performance_metrics(pd.DataFrame(portfolio.trade_log), portfolio.equity_df, config.INITIAL_CASH)
        
        return (performance, portfolio.trade_log)

    except Exception:
        return None

# --- Main Walk-Forward Logic ---
def run_walk_forward_analysis(symbol, timeframe, strategy_name, start_date, end_date):
    
    print("\n--- Starting Walk-Forward Analysis ---")
    print(f"Strategy: {strategy_name}, Symbol: {symbol}, Timeframe: {timeframe}")
    
    loader = DataLoader()
    full_data_df = loader.fetch_data_for_symbol(symbol, start_date, end_date)
    if full_data_df.empty:
        print("Could not load data for the full period.")
        return

    # Walk-Forward period calculations
    train_months = config.WALK_FORWARD_CONFIG['training_period_months']
    test_months = config.WALK_FORWARD_CONFIG['testing_period_months']
    
    current_date = pd.to_datetime(start_date)
    end_date_dt = pd.to_datetime(end_date)
    
    all_out_of_sample_trades = []
    
    while current_date + relativedelta(months=train_months + test_months) <= end_date_dt:
        
        train_start = current_date
        train_end = current_date + relativedelta(months=train_months)
        test_start = train_end
        test_end = test_start + relativedelta(months=test_months)
        
        print(f"\n--- Running Period: [Train: {train_start.date()} to {train_end.date()}] | [Test: {test_start.date()} to {test_end.date()}] ---")

        # 1. Optimization Phase (Training Data)
        train_data = full_data_df.loc[train_start:train_end]
        param_combos = generate_param_combinations(strategy_name)
        
        optimization_tasks = [(symbol, timeframe, strategy_name, params, train_data) for params in param_combos]
        
        best_params = {}
        best_performance_metric = -np.inf
        
        print(f"Optimizing {len(param_combos)} parameter combinations...")
        with multiprocessing.Pool(os.cpu_count()) as pool:
            results = list(tqdm(pool.imap_unordered(run_single_backtest, optimization_tasks), total=len(optimization_tasks), desc="Training"))
        
        for i, result in enumerate(results):
            if result:
                performance, _ = result
                metric_value = performance.get(config.WALK_FORWARD_CONFIG['optimization_metric'], -np.inf)
                if metric_value > best_performance_metric:
                    best_performance_metric = metric_value
                    best_params = param_combos[i]

        if not best_params:
            print("Could not find best parameters in training period. Skipping.")
            current_date += relativedelta(months=test_months)
            continue
            
        print(f"Found best parameters: {best_params} (Metric: {best_performance_metric:.2f})")

        # 2. Validation Phase (Out-of-Sample/Testing Data)
        print("Validating on out-of-sample data...")
        test_data = full_data_df.loc[test_start:test_end]
        validation_task = (symbol, timeframe, strategy_name, best_params, test_data)
        
        validation_result = run_single_backtest(validation_task)
        
        if validation_result:
            _, out_of_sample_trades = validation_result
            all_out_of_sample_trades.extend(out_of_sample_trades)
            print(f"Completed test period. Found {len(out_of_sample_trades)} trades.")
        else:
            print("No trades generated in test period.")

        # Move to the next window
        current_date += relativedelta(months=test_months)

    # 3. Final Report
    if not all_out_of_sample_trades:
        print("\n--- Walk-Forward Analysis Complete: No trades were generated in any test period. ---")
        return

    print("\n--- Walk-Forward Analysis Complete: Generating final report... ---")
    final_trades_df = pd.DataFrame(all_out_of_sample_trades)
    final_equity_df = pd.DataFrame([{'timestamp': pd.to_datetime(start_date), 'equity': config.INITIAL_CASH}])
    
    # Reconstruct final equity curve from all out-of-sample trades
    temp_equity = config.INITIAL_CASH
    equity_rows = []
    for _, trade in final_trades_df.sort_values(by='exit_timestamp').iterrows():
        temp_equity += trade['pnl']
        equity_rows.append({'timestamp': trade['exit_timestamp'], 'equity': temp_equity})
    
    if equity_rows:
        final_equity_df = pd.concat([final_equity_df, pd.DataFrame(equity_rows)], ignore_index=True)

    final_performance = calculate_performance_metrics(final_trades_df, final_equity_df, config.INITIAL_CASH)
    
    # Save the final robust result to the database
    run_id = f"WF_{strategy_name}_{symbol}_{timeframe}_{uuid.uuid4().hex[:8]}"
    run_metadata = {
        "run_id": run_id, "run_timestamp": datetime.now().isoformat(),
        "strategy_name": f"WF_{strategy_name}", # Add WF prefix
        "symbol": symbol, "timeframe": timeframe,
        "start_date": start_date, "end_date": end_date,
        "strategy_params": json.dumps({"walk_forward": True, "config": config.WALK_FORWARD_CONFIG}),
        "performance_summary": json.dumps(final_performance)
    }
    save_backtest_results(run_id, run_metadata, all_out_of_sample_trades, final_performance)
    
    print("\n--- FINAL ROBUST PERFORMANCE ---")
    print(json.dumps(final_performance, indent=4))
    print(f"Results saved to database with Run ID: {run_id}")


if __name__ == '__main__':
    multiprocessing.freeze_support()
    init_results_db()
    
    symbol, timeframe, strategy_name, start_date, end_date = get_user_inputs()
    
    if symbol and timeframe and strategy_name:
        run_walk_forward_analysis(symbol, timeframe, strategy_name, start_date, end_date)
    
    print("\n--- ✅ Script Finished ---")
