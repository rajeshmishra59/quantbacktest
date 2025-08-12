# quant_backtesting_project/hybrid_backtest_runner.py
# UPGRADED PROFESSIONAL VERSION: Ab ismein do mode hain:
# 1. Walk-Forward Optimization (Purana tarika)
# 2. Walk-Forward Validation (Naya tarika, jo JSON file se best params leta hai)
# NAYA: Validation mode ab results ko rank karke dikhata hai.

import pandas as pd
import os
import time
from datetime import datetime
from dateutil.relativedelta import relativedelta
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

# --- NAYA HELPER FUNCTION ---
def load_best_params():
    """best_optimizer_results.json file ko load karta hai."""
    filename = "best_optimizer_results.json"
    if not os.path.exists(filename):
        print(f"--- ERROR ---")
        print(f"'{filename}' file nahi mili.")
        print("Kripya pehle 'simple_optimizer_runner.py' chalayein taaki yeh file ban sake.")
        return None
    with open(filename, 'r') as f:
        return json.load(f)

# --- USER INPUT FUNCTION MEIN BADLAV ---
def get_user_inputs():
    """User se backtest ke liye mode aur parameters leta hai."""
    print("--- Hybrid Backtest Runner ---")
    print("Aap kaunsa test karna chahte hain?")
    print("\n  1. Walk-Forward OPTIMIZATION (Har window mein best params dhoondhega - Dheema)")
    print("  2. Walk-Forward VALIDATION (Pehle se mile best params ko test karega - Tez)")
    
    mode = input("Apna vikalp chunein (1 ya 2): ")

    if mode == '2':
        print("\n--- Walk-Forward Validation Mode ---")
        best_params_data = load_best_params()
        if not best_params_data:
            return None, None, None, None, None, None

        # --- YAHAN BADLAV KIYA GAYA HAI: Data ko DataFrame mein convert karein ---
        results_list = []
        for key, value in best_params_data.items():
            strategy_name, _, symbol = key.partition('_on_')
            results_list.append({
                'key': key,
                'Strategy': strategy_name,
                'Symbol': symbol,
                'Sharpe Ratio': value.get('sharpe', 0),
                'Total PnL': value.get('pnl', 0),
                'Total Trades': value.get('trades', 0),
                'params': value.get('params')
            })
        
        results_df = pd.DataFrame(results_list)

        print("\nAap results ko kaise rank karna chahte hain?")
        print("  1. Sharpe Ratio (sabse behtar)")
        print("  2. Total PnL")
        print("  3. Total Trades")
        sort_choice = input("Apna vikalp chunein (1, 2, ya 3): ")

        sort_column_map = {'1': 'Sharpe Ratio', '2': 'Total PnL', '3': 'Total Trades'}
        sort_by = sort_column_map.get(sort_choice, 'Sharpe Ratio')

        # DataFrame ko sort karein aur rank karein
        ranked_df = results_df.sort_values(by=sort_by, ascending=False).reset_index(drop=True)
        ranked_df.index = ranked_df.index + 1 # Rank 1 se shuru ho
        
        print(f"\n--- Optimizer se mile best combinations (Ranked by {sort_by}) ---")
        display_cols = ['Strategy', 'Symbol', 'Sharpe Ratio', 'Total PnL', 'Total Trades']
        print(ranked_df[display_cols].to_string())
        
        try:
            choice_idx = int(input("\nTest karne ke liye ek Rank Number chunein: "))
            selected_row = ranked_df.loc[choice_idx]
            
            strategy_name = selected_row['Strategy']
            symbol = selected_row['Symbol']
            fixed_params = selected_row['params']
            
            print(f"\nAapne chuna hai: Rank {choice_idx} -> {strategy_name} on {symbol}")
            print(f"In parameters ke saath test kiya jayega: {fixed_params}")
            
            timeframe = input("Enter timeframe (e.g., 5min, 15min): ")
            start_date = input("Enter Full Period Start Date (YYYY-MM-DD): ")
            end_date = input("Enter Full Period End Date (YYYY-MM-DD): ")
            
            return strategy_name, symbol, timeframe, start_date, end_date, fixed_params

        except (ValueError, IndexError, KeyError):
            print("Galt vikalp. Program band ho raha hai.")
            return None, None, None, None, None, None

    elif mode == '1':
        print("\n--- Walk-Forward Optimization Mode ---")
        # Purana user input logic
        strategy_files = [f.replace('_strategy.py', '') for f in os.listdir('strategies') if f.endswith('_strategy.py') and not f.startswith('base')]
        print("\nAvailable Strategies:")
        for i, s_name in enumerate(strategy_files): print(f"  {i+1}. {s_name}")
        strategy_idx = int(input("Optimize karne ke liye ek strategy chunein: ")) - 1
        strategy_name = strategy_files[strategy_idx]
        
        symbol = input("Enter a SINGLE symbol for Walk-Forward Analysis: ")
        timeframe = input("Enter a SINGLE timeframe (e.g., 15min): ")
        start_date = input("Enter Full Period Start Date (YYYY-MM-DD): ")
        end_date = input("Enter Full Period End Date (YYYY-MM-DD): ")
        return strategy_name, symbol.upper(), timeframe, start_date, end_date, None

    else:
        print("Galt vikalp.")
        return None, None, None, None, None, None


def generate_param_combinations(strategy_name):
    param_config = config.STRATEGY_OPTIMIZATION_CONFIG.get(strategy_name, {})
    if not param_config: return [{}]
    keys, values = zip(*param_config.items())
    return [dict(zip(keys, v)) for v in itertools.product(*values)]

def run_single_backtest(task_info):
    """Ek single backtest chalata hai."""
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
        prices_1min_df['entries'] = prices_1min_df['entries'].astype('boolean').fillna(False)
        prices_1min_df['exits'] = prices_1min_df['exits'].astype('boolean').fillna(False)
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

# --- MAIN LOGIC MEIN BADLAV ---
def run_walk_forward_analysis(strategy_name, symbol, timeframe, start_date, end_date, fixed_params=None):
    print(f"\n--- Starting Walk-Forward Analysis ---")
    print(f"Strategy: {strategy_name}, Symbol: {symbol}, Timeframe: {timeframe}")
    if fixed_params:
        print(f"Mode: VALIDATION with params: {fixed_params}")
    else:
        print(f"Mode: OPTIMIZATION")

    loader = DataLoader()
    full_data_df = loader.fetch_data_for_symbol(symbol, start_date, end_date)
    if full_data_df.empty:
        print("Data load nahi ho saka.")
        return

    # --- FIX FOR TIMEZONE ERROR ---
    # Ensure the DataFrame index is timezone-naive to match the slicing dates
    if isinstance(full_data_df.index, pd.DatetimeIndex) and full_data_df.index.tz is not None:
        full_data_df.index = full_data_df.index.tz_localize(None)

    train_months = config.WALK_FORWARD_CONFIG['training_period_months']
    test_months = config.WALK_FORWARD_CONFIG['testing_period_months']
    start_dt, end_date_dt = pd.to_datetime(start_date), pd.to_datetime(end_date)
    current_date, all_out_of_sample_trades = start_dt, []

    while current_date + relativedelta(months=train_months + test_months) <= end_date_dt:
        train_start, train_end = current_date, current_date + relativedelta(months=train_months)
        test_start, test_end = train_end, train_end + relativedelta(months=test_months)
        print(f"\n--- Running Period: [Train: {train_start.date()} to {train_end.date()}] | [Test: {test_start.date()} to {test_end.date()}] ---")

        best_params = {}
        # AGAR VALIDATION MODE HAI, TO OPTIMIZATION SKIP KAREIN
        if fixed_params:
            best_params = fixed_params
            print(f"Using pre-defined best parameters: {best_params}")
        else:
            # OPTIMIZATION MODE
            train_data = full_data_df.loc[train_start:train_end]
            param_combos = generate_param_combinations(strategy_name)
            optimization_tasks = [(symbol, timeframe, strategy_name, params, train_data) for params in param_combos]
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
                print("Training mein best parameters nahi mile. Skipping.")
                current_date += relativedelta(months=test_months)
                continue
            print(f"Is window ke best parameters: {best_params} (Metric: {best_performance_metric:.2f})")

        # Dono modes ke liye validation step
        print("Out-of-sample data par validate kiya ja raha hai...")
        test_data = full_data_df.loc[test_start:test_end]
        validation_task = (symbol, timeframe, strategy_name, best_params, test_data)
        validation_result = run_single_backtest(validation_task)
        if validation_result:
            _, out_of_sample_trades = validation_result
            all_out_of_sample_trades.extend(out_of_sample_trades)
            print(f"Test period poora hua. {len(out_of_sample_trades)} trades mile.")
        else:
            print("Test period mein koi trade nahi hua.")
        current_date += relativedelta(months=test_months)

    if not all_out_of_sample_trades:
        print("\n--- Walk-Forward Analysis Poora Hua: Koi trades nahi mile. ---")
        return

    # Final Report (sabke liye same)
    print("\n--- Walk-Forward Analysis Poora Hua: Final report ban rahi hai... ---")
    final_trades_df = pd.DataFrame(all_out_of_sample_trades)
    final_equity_df = pd.DataFrame([{'timestamp': pd.to_datetime(start_date), 'equity': config.INITIAL_CASH}])
    temp_equity = config.INITIAL_CASH
    equity_rows = []
    for _, trade in final_trades_df.sort_values(by='exit_timestamp').iterrows():
        temp_equity += trade['pnl']
        equity_rows.append({'timestamp': trade['exit_timestamp'], 'equity': temp_equity})
    if equity_rows:
        final_equity_df = pd.concat([final_equity_df, pd.DataFrame(equity_rows)], ignore_index=True)
    final_performance = calculate_performance_metrics(final_trades_df, final_equity_df, config.INITIAL_CASH)
    run_id = f"WF-VALIDATE_{strategy_name}_{symbol}_{timeframe}_{uuid.uuid4().hex[:8]}" if fixed_params else f"WF-OPTIMIZE_{strategy_name}_{symbol}_{timeframe}_{uuid.uuid4().hex[:8]}"
    run_metadata = {
        "run_id": run_id, "run_timestamp": datetime.now().isoformat(),
        "strategy_name": f"WF_{strategy_name}", "symbol": symbol, "timeframe": timeframe,
        "start_date": start_date, "end_date": end_date,
        "strategy_params": json.dumps(fixed_params if fixed_params else {"walk_forward_optimized": True}),
        "performance_summary": json.dumps(final_performance)
    }
    save_backtest_results(run_id, run_metadata, all_out_of_sample_trades, final_performance)
    print("\n--- FINAL ROBUST PERFORMANCE ---")
    print(json.dumps(final_performance, indent=4))
    print(f"Results database mein save ho gaye hain. Run ID: {run_id}")


if __name__ == '__main__':
    multiprocessing.freeze_support()
    init_results_db()
    
    user_input = get_user_inputs()
    if user_input and all(val is not None for val in user_input):
        strategy_name, symbol, timeframe, start_date, end_date, fixed_params = user_input
        run_walk_forward_analysis(strategy_name, symbol, timeframe, start_date, end_date, fixed_params=fixed_params)
    
    print("\n--- ✅ Script Finished ---")
