# quant_backtesting_project/engine/upgraded_portfolio.py
# Yeh aapke portfolio ka "supercharged" version hai jismein professional risk management shaamil hai.

import pandas as pd
import numpy as np

class UpgradedPortfolio:
    def __init__(self, initial_cash: float, risk_per_trade_pct: float, max_daily_loss_pct: float, 
                 brokerage_pct: float, slippage_pct: float):
        """
        Portfolio ko professional risk niyamon ke saath initialize karta hai.
        """
        # General Portfolio Stats
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions = {}  # symbol: {'qty': int, 'entry_price': float}
        self.trade_log = []

        # Equity curve ko DataFrame mein store karein
        self.equity_df = pd.DataFrame([{'timestamp': pd.Timestamp.min, 'equity': initial_cash}])

        # Risk Management Parameters
        self.RISK_PER_TRADE_PCT = risk_per_trade_pct
        self.MAX_DAILY_LOSS_PCT = max_daily_loss_pct
        self.BROKERAGE_PCT = brokerage_pct / 100
        self.SLIPPAGE_PCT = slippage_pct / 100

        # Daily Stats for Kill-Switch
        self.equity_at_day_start = initial_cash
        self.daily_pnl = 0.0
        self.is_trading_halted_today = False
        self.current_date = None

    def _apply_slippage(self, price, side):
        """Buy/Sell price par anumaanit slippage lagata hai."""
        if side == 'buy':
            return price * (1 + self.SLIPPAGE_PCT)
        else: # sell
            return price * (1 - self.SLIPPAGE_PCT)

    def _calculate_position_size(self, entry_price, stop_loss_price):
        """2% risk niyam ke anusaar position size nikalta hai."""
        risk_per_share = abs(entry_price - stop_loss_price)
        if risk_per_share <= 1e-6: # Zero ya bahut chote risk se bachein
            return 0
        
        capital_to_risk = self.get_current_equity() * self.RISK_PER_TRADE_PCT
        quantity = int(capital_to_risk / risk_per_share)
        return quantity

    def on_new_day(self, date):
        """Naye din ki shuruaat par daily stats reset karta hai."""
        if self.current_date != date.date():
            self.current_date = date.date()
            self.equity_at_day_start = self.get_current_equity()
            self.daily_pnl = 0.0
            self.is_trading_halted_today = False

    def request_trade(self, timestamp, symbol, side, entry_price, stop_loss_price):
        """
        Trade lene se pehle sabhi risk niyamon ki jaanch karta hai.
        """
        if self.is_trading_halted_today:
            return

        if side == 'buy':
            if symbol in self.positions: return

            quantity = self._calculate_position_size(entry_price, stop_loss_price)
            if quantity == 0: return
            
            actual_entry_price = self._apply_slippage(entry_price, 'buy')
            trade_cost = quantity * actual_entry_price
            brokerage = trade_cost * self.BROKERAGE_PCT
            
            if self.cash < trade_cost + brokerage: return

            self.cash -= (trade_cost + brokerage)
            self.positions[symbol] = {'qty': quantity, 'entry_price': actual_entry_price, 'entry_timestamp': timestamp}
            
        elif side == 'sell':
            if symbol not in self.positions: return
            
            position = self.positions[symbol]
            actual_exit_price = self._apply_slippage(entry_price, 'sell')
            
            proceeds = position['qty'] * actual_exit_price
            entry_cost = position['qty'] * position['entry_price']
            
            entry_brokerage = entry_cost * self.BROKERAGE_PCT
            exit_brokerage = proceeds * self.BROKERAGE_PCT
            total_brokerage = entry_brokerage + exit_brokerage

            pnl = (proceeds - entry_cost) - total_brokerage
            
            self.cash += proceeds - exit_brokerage
            self.daily_pnl += pnl
            
            self.trade_log.append({
                'entry_timestamp': position['entry_timestamp'],
                'exit_timestamp': timestamp,
                'symbol': symbol,
                'pnl': pnl
            })
            
            del self.positions[symbol]
            
            max_allowed_daily_loss = self.equity_at_day_start * self.MAX_DAILY_LOSS_PCT
            if self.daily_pnl < -max_allowed_daily_loss:
                self.is_trading_halted_today = True

    def get_current_equity(self, current_prices: pd.Series = None):
        """Portfolio ki vartamaan total value calculate karta hai."""
        current_value = self.cash
        if current_prices is not None:
            for symbol, pos in self.positions.items():
                current_value += pos['qty'] * current_prices.get(symbol, pos['entry_price'])
        # Agar current prices nahi diye, to sirf open positions ki entry value jodein
        else:
            for symbol, pos in self.positions.items():
                current_value += pos['qty'] * pos['entry_price']
        return current_value

    def record_equity(self, timestamp, current_prices: pd.Series):
        """Har candle ke ant mein portfolio ki equity ko record karta hai."""
        equity = self.get_current_equity(current_prices)
        new_row = pd.DataFrame([{'timestamp': timestamp, 'equity': equity}])
        self.equity_df = pd.concat([self.equity_df, new_row], ignore_index=True)

