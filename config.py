# quant_backtesting_project/config.py
# UPGRADED: Includes self-contained symbol lists for batch testing.

import os

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

# --- Strategy Default Parameters ---
STRATEGY_PARAMS = {}

print("Configuration loaded.")
