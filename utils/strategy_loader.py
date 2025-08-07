# quant_backtesting_project/utils/strategy_loader.py
# CORRECTED: More robust loading for class-based strategies.

import importlib
import inspect
from strategies.base_strategy import BaseStrategy

def load_strategy(strategy_name: str):
    """
    Strategy ke naam se uski class ya signal generation function ko load karta hai.
    File ka naam strategy ke naam se match hona chahiye (e.g., 'trend' -> 'trend_strategy.py').
    """
    try:
        # Module ka naam strategy ke naam se banayein
        module_name = f"strategies.{strategy_name.lower()}_strategy"
        strategy_module = importlib.import_module(module_name)
        
        # Module ke andar BaseStrategy se inherit hui class ko dhoondhein
        for name, obj in inspect.getmembers(strategy_module, inspect.isclass):
            if issubclass(obj, BaseStrategy) and obj is not BaseStrategy:
                print(f"Successfully loaded CLASS strategy: {name}")
                return obj # Class ko return karein

    except ImportError as e:
        print(f"Error loading strategy '{strategy_name}': {e}")
        # Function-based strategies ke liye fallback (agar zaroorat pade)
        try:
            func_module_name = f"strategies.{strategy_name.lower()}_signals"
            strategy_module = importlib.import_module(func_module_name)
            if hasattr(strategy_module, 'generate_signals'):
                print(f"Successfully loaded FUNCTION strategy: {strategy_name}")
                return getattr(strategy_module, 'generate_signals')
        except ImportError:
            pass # Agar function bhi na mile to neeche error aa jayega

    print(f"Could not find a valid class or function for strategy '{strategy_name}'.")
    return None
