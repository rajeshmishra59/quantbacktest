# quant_backtesting_project/config.py
# UPGRADED: Ab ismein strategy optimization ke liye ek dedicated section hai.

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

# --- STRATEGY OPTIMIZATION CONFIG ---
# Yahan hum har strategy ke liye parameters ki range define karte hain.
# Runner inhi ranges ka istemaal karke alag-alag combinations banayega.
# np.arange(start, stop, step) ka istemaal range banane ke liye kiya gaya hai.

STRATEGY_OPTIMIZATION_CONFIG = {
    "default": {}, # Default parameters agar kisi strategy ke liye config na ho
    "apex": {
        "squeeze_window": list(np.arange(20, 41, 10)),  # 20, 30, 40
        "historical_window": [200], # Ise constant rakhte hain
        "volatility_ratio_threshold": list(np.arange(0.5, 0.8, 0.1)) # 0.5, 0.6, 0.7
    },
    "alphaone": {
        "streak_period_min": [6, 8, 10],
        "tp_rr_ratio": [1.5, 2.0, 2.5]
    },
    "sankhyaek":{
        "bb_length": [20],
        "bb_std": [2.0],
        "rsi_oversold": [30, 40, 45],
        "rsi_overbought": [55, 60, 70]
    }
    # Aap yahan aur bhi strategies jod sakte hain...
}

print("Configuration loaded with Optimization settings.")
