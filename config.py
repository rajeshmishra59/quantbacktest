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
# Yahan hum Walk-Forward analysis ke niyam tay karte hain.
WALK_FORWARD_CONFIG = {
    "enabled": True,
    "training_period_months": 24, # 24 mahine ke data par best parameters dhoondhega
    "testing_period_months": 6,   # Agle 6 mahine ke andekhe data par unhe test karega
    "optimization_metric": "Sharpe Ratio" # Best parameter dhoondhne ke liye is metric ka istemaal karega
}

# --- STRATEGY OPTIMIZATION CONFIG ---
# Har strategy ke liye "magic numbers" ki testing range.
STRATEGY_OPTIMIZATION_CONFIG = {
    "default": {}, # Agar kisi strategy ka config na ho to default
    
    "apex": {
        "squeeze_window": list(np.arange(20, 61, 20)), # 20, 40, 60
        "volatility_ratio_threshold": list(np.arange(0.5, 0.8, 0.1)) # 0.5, 0.6, 0.7
    },
    "alphaone": {
        "streak_period_min": [6, 8, 10],
        "tp_rr_ratio": [1.5, 2.0, 2.5]
    },
    "sankhyaek":{
        "bb_length": [20],
        "rsi_oversold": [30, 40, 45],
        "rsi_overbought": [55, 60, 70]
    },
    "numerouno": {
        "pivot_lookback": [5, 10, 15]
    },
    "rangebound": {
        "bb_length": [20, 30],
        "stoch_k": [14, 21]
    },
    "trend": {
        "ema_short": [9, 12],
        "ema_long": [21, 26],
        "adx_period": [14, 20]
    },
    "sma_crossover": { # File ka naam sma_crossover_signals.py hai, isliye 'sma_crossover'
        "short_window": [10, 20, 30],
        "long_window": [50, 100]
    },
    "test": {} # Iske liye koi optimization nahi
}

print("Configuration loaded with Walk-Forward and Optimization settings for all strategies.")
