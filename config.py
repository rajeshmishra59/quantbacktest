# quant_backtesting_project/config.py
# FINAL VERSION: Ismein sabhi strategies ke liye Walk-Forward aur Optimization settings hain.

import os
import numpy as np

# --- Symbol Batches for Testing ---
SYMBOL_BATCHES = {
    "NIFTY_50": [
        "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "HINDUNILVR", "SBIN", "ITC", "BHARTIARTL", "KOTAKBANK",
        "LT", "ASIANPAINT", "HCLTECH", "BAJFINANCE", "MARUTI", "AXISBANK", "ULTRACEMCO", "NESTLEIND", "SUNPHARMA", "TITAN",
        "WIPRO", "TECHM", "POWERGRID", "COALINDIA", "NTPC", "GRASIM", "TATAMOTORS", "JSWSTEEL", "ADANIENT", "ADANIPORTS",
        "DIVISLAB", "ONGC", "HINDALCO", "SBILIFE", "BAJAJFINSV", "BRITANNIA", "HEROMOTOCO", "CIPLA", "EICHERMOT", "BPCL",
        "UPL", "INDUSINDBK", "APOLLOHOSP", "HDFCLIFE", "BAJAJ-AUTO", "TATASTEEL", "SHREECEM", "DRREDDY", "M&M", "IOC"
    ],
    "NIFTY_NEXT_50": [
        "AMBUJACEM", "AUROPHARMA", "BANKBARODA", "BERGEPAINT", "CANBK", "CHOLAFIN", "DABUR", "DLF", "GAIL",
        "GODREJCP", "ICICIGI", "ICICIPRULI", "IDFCFIRSTB", "IGL", "INDIGO", "MUTHOOTFIN", "NAUKRI", "PIDILITIND",
        "PNB", "RECLTD", "SAIL", "SRF", "TORNTPHARM", "TVSMOTOR", "UBL", "VEDL", "VOLTAS", "HAVELLS", "PEL", "PAGEIND"
    ],
    "SMALL_TEST": ["RELIANCE", "TCS", "INFY"] # Chote tests ke liye
}


# --- Database Configuration ---
MARKET_DATA_DB_PATH = r"D:\AppDevelopment\AI Generated App\AlgoTradingLab_V1.3\data\market_data.db"
DB_TABLE_NAME = "price_data_1min"
RESULTS_DB_PATH = os.path.join(os.path.dirname(__file__), 'data', 'backtest_results.db')

# --- Portfolio Configuration ---
INITIAL_CASH = 100000.0
RISK_PER_TRADE_PCT = 0.02
MAX_DAILY_LOSS_PCT = 0.05

# --- Cost Configuration ---
BROKERAGE_PCT = 0.05
SLIPPAGE_PCT = 0.02

# --- WALK-FORWARD OPTIMIZATION CONFIG ---
WALK_FORWARD_CONFIG = {
    "enabled": True,
    "training_period_months": 24,
    "testing_period_months": 6,
    "optimization_metric": "Sharpe Ratio"
}

# --- STRATEGY OPTIMIZATION CONFIG ---
STRATEGY_OPTIMIZATION_CONFIG = {
    "default": {},
    "apex": {
        "squeeze_window": [20, 40, 60],
        "volatility_ratio_threshold": [0.5, 0.6, 0.7]
    },
    "alphaone": {
        "streak_period_min": [6, 8, 10],
        "strong_candle_ratio": [0.7, 0.8],
        "volume_spike_multiplier": [1.5, 2.0],
        "tp_rr_ratio": [1.5, 2.0, 2.5]
    },
    "sankhyaek": {
        "bb_length": [15, 20, 25],
        "bb_std": [2.0, 2.5, 3.0],
        "rsi_period": [10, 12, 14],
        "rsi_oversold": [20, 25, 30],
        "rsi_overbought": [70, 75, 80],
        "stop_loss_pct": [0.01, 0.015, 0.02],
        "risk_reward_ratio": [1.5, 2.0, 2.5]
    },
    "numerouno": {
        "pivot_lookback": [5, 10, 15]
    },
    "rangebound": {
        "primary_timeframe": [15],
        "bb_length": [20, 30],
        "adx_window": [14, 20]
    },
    "trend": {
        "ema_short": [9, 12],
        "ema_long": [21, 26],
        "adx_period": [14, 20]
    },
    "sma_crossover": {
        "short_window": [10, 20],
        "long_window": [50, 100]
    },
    "test": {}
}

print("Configuration loaded with Walk-Forward and Optimization settings for all strategies.")