# quant_backtesting_project/utils/strategy_loader.py
# Yeh utility strategies folder se signal generation functions ko dynamically load karegi.

import importlib

def load_strategy_signal_generator(strategy_name: str):
    """
    Strategy ke naam se uski signal generation file aur function ko load karta hai.
    
    Convention: Har strategy ke liye 'strategies' folder mein ek file honi chahiye
    jiska naam '{strategy_name}_signals.py' ho aur uske andar ek function
    'generate_signals' ho.
    """
    try:
        module_path = f"strategies.{strategy_name}_signals"
        strategy_module = importlib.import_module(module_path)
        
        # Har module se 'generate_signals' function ko return karein
        return getattr(strategy_module, 'generate_signals')
    except (ImportError, AttributeError) as e:
        print(f"Error loading strategy '{strategy_name}': {e}")
        return None

