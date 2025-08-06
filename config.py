# quant_backtesting_project/config.py
# Yeh file project ki sabhi mukhya settings ko store karegi.

import os

# --- Database Configuration ---
# Market Data DB
MARKET_DATA_DB_PATH = r"D:\AppDevelopment\AI Generated App\AlgoTradingLab_V1.3\data\market_data.db"
DB_TABLE_NAME = "price_data_1min"

# Results DB - NAYA
RESULTS_DB_PATH = os.path.join(os.path.dirname(__file__), 'data', 'backtest_results.db')


# --- Portfolio Configuration ---
INITIAL_CASH = 100000.0
RISK_PER_TRADE_PCT = 0.02  # 2% risk per trade
MAX_DAILY_LOSS_PCT = 0.05  # 5% max loss per day

# --- Cost Configuration ---
BROKERAGE_PCT = 0.05  # 0.05% per trade
SLIPPAGE_PCT = 0.02   # 0.02% per trade

# --- Strategy Default Parameters ---
STRATEGY_PARAMS = {
    'sma_crossover': {
        'short_window': 20,
        'long_window': 50
    }
}

# --- Results Directory (ab istemal nahi hoga, DB use karenge) ---
# RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')

# Function to ensure data directory exists for results DB
def ensure_data_dir():
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

print("Configuration loaded.")
ensure_data_dir()
