# quant_backtesting_project/hybrid_backtest_runner.py
# FINAL VERSION: Dynamic Strategy Loading, Advanced Metrics, aur Robust Error Handling ke saath.

import pandas as pd
import os
import time
from datetime import datetime
import json
import uuid
import traceback

# Project ke modules ko import karein
import config
from data.data_loader import DataLoader
from engine.upgraded_portfolio import UpgradedPortfolio
from utils.results_db import save_backtest_results, init_results_db
from utils.strategy_loader import load_strategy_signal_generator # NAYA: Strategy ko dynamically load karega
from utils.metrics import calculate_performance_metrics # NAYA: Advanced metrics ke liye

def resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """1-minute data ko kisi bhi timeframe mein resample karta hai."""
    if df.empty: return df
    resampling_rules = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
    return df.resample(timeframe).apply(resampling_rules).dropna()

def run_hybrid_backtest(symbol: str, timeframe: str, start_date: str, end_date: str, strategy_name: str, strategy_params: dict):
    """
    Ek poori tarah se modular aur robust backtest chalata hai.
    """
    run_id = f"{strategy_name}_{symbol}_{timeframe}_{uuid.uuid4().hex[:8]}"
    print(f"\n--- Starting Backtest | Run ID: {run_id} ---")

    try:
        # --- 1. Strategy Signal Generator Load Karein ---
        signal_generator = load_strategy_signal_generator(strategy_name)
        if signal_generator is None:
            print(f"Strategy '{strategy_name}' nahi mili. Skipping.")
            return

        # --- 2. Data Loading (1-min data) ---
        loader = DataLoader()
        prices_1min_df = loader.fetch_data_for_symbol(symbol, start_date, end_date)
        if prices_1min_df.empty:
            print(f"Data nahi mila for {symbol}. Skipping.")
            return

        # --- 3. Data Resampling & Signal Generation ---
        prices_tf_df = resample_ohlcv(prices_1min_df, timeframe)
        print(f"Data Resampled to {timeframe}. Generating signals...")
        signals = signal_generator(prices_tf_df, **strategy_params)
        signals = signals.reindex(prices_1min_df.index, method='ffill').fillna(False)

        # --- 4. Portfolio Simulation ---
        portfolio = UpgradedPortfolio(
            initial_cash=config.INITIAL_CASH,
            risk_per_trade_pct=config.RISK_PER_TRADE_PCT,
            max_daily_loss_pct=config.MAX_DAILY_LOSS_PCT,
            brokerage_pct=config.BROKERAGE_PCT,
            slippage_pct=config.SLIPPAGE_PCT
        )

        for i in range(1, len(prices_1min_df)):
            timestamp = prices_1min_df.index[i]
            current_price_info = prices_1min_df.iloc[i]
            
            portfolio.on_new_day(timestamp)

            if portfolio.is_trading_halted_today:
                portfolio.record_equity(timestamp, pd.Series({symbol: current_price_info['close']}))
                continue

            is_entry_signal = signals.iloc[i]['entries'] and not signals.iloc[i-1]['entries']
            is_exit_signal = signals.iloc[i]['exits'] and not signals.iloc[i-1]['exits']

            if is_entry_signal:
                entry_price = current_price_info['open']
                # Stop-loss ab strategy se aayega
                stop_loss = signals.iloc[i].get('stop_loss', entry_price * (1 - 0.02)) # Default 2% SL
                portfolio.request_trade(timestamp, symbol, 'buy', entry_price, stop_loss)

            elif is_exit_signal:
                exit_price = current_price_info['open']
                portfolio.request_trade(timestamp, symbol, 'sell', exit_price, 0)

            portfolio.record_equity(timestamp, pd.Series({symbol: current_price_info['close']}))

        # --- 5. Performance Metrics Calculate Karein ---
        trades_df = pd.DataFrame(portfolio.trade_log)
        equity_curve = portfolio.equity_df
        performance_summary = calculate_performance_metrics(trades_df, equity_curve, config.INITIAL_CASH)

        # --- 6. Results ko DB mein Save Karein ---
        run_metadata = {
            "run_id": run_id, "run_timestamp": datetime.now().isoformat(),
            "strategy_name": strategy_name, "symbol": symbol, "timeframe": timeframe,
            "start_date": start_date, "end_date": end_date,
            "strategy_params": json.dumps(strategy_params),
            "performance_summary": json.dumps(performance_summary)
        }
        save_backtest_results(run_id, run_metadata, portfolio.trade_log, performance_summary)
        print(f"--- Finished Backtest for {symbol} on {timeframe} ---")
        print(f"Performance: {performance_summary}")

    except Exception as e:
        print(f"!!!!!! ERROR during backtest for {symbol} !!!!!!")
        print(traceback.format_exc())


if __name__ == '__main__':
    init_results_db()

    # --- User Inputs for Batch Testing ---
    # (User input section waisa hi rahega, lekin ab hum strategy bhi select karwayenge)
    # For simplicity, yahan hardcode kar rahe hain.
    
    strategies_to_run = {
        'sma_crossover': config.STRATEGY_PARAMS['sma_crossover']
        # 'rsi_strategy': {'rsi_period': 14, 'entry': 30, 'exit': 70} # Aise aur strategies add kar sakte hain
    }
    symbols_to_run = ['RELIANCE', 'TCS']
    timeframes_to_run = ['15T', '1H']
    start_date = '2023-01-01'
    end_date = '2023-12-31'
    
    for strategy_name, params in strategies_to_run.items():
        for timeframe in timeframes_to_run:
            for symbol in symbols_to_run:
                run_hybrid_backtest(symbol, timeframe, start_date, end_date, strategy_name, params)
    
    print("\n--- Batch Run Complete ---")
