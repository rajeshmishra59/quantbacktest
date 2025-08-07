# quant_backtesting_project/engine/upgraded_portfolio.py
# FINAL CORRECTED VERSION: Includes robust stop-loss and take-profit handling.

import pandas as pd
import numpy as np

class UpgradedPortfolio:
    def __init__(self, initial_cash: float, risk_per_trade_pct: float, max_daily_loss_pct: float,
                 brokerage_pct: float, slippage_pct: float):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        # Position dictionary ab stop-loss aur target bhi store karega
        self.positions = {}  # symbol: {'qty': int, 'entry_price': float, 'stop_loss': float, 'target': float}
        self.trade_log = []
        self.equity_df = pd.DataFrame([{'timestamp': pd.Timestamp.min, 'equity': initial_cash}])

        self.RISK_PER_TRADE_PCT = risk_per_trade_pct
        self.MAX_DAILY_LOSS_PCT = max_daily_loss_pct
        self.BROKERAGE_PCT = brokerage_pct / 100
        self.SLIPPAGE_PCT = slippage_pct / 100

        self.equity_at_day_start = initial_cash
        self.daily_pnl = 0.0
        self.is_trading_halted_today = False
        self.current_date = None

    def _apply_slippage(self, price, side):
        if side == 'buy':
            return price * (1 + self.SLIPPAGE_PCT)
        else: # sell
            return price * (1 - self.SLIPPAGE_PCT)

    def _calculate_position_size(self, entry_price, stop_loss_price):
        if abs(entry_price - stop_loss_price) <= 1e-6: return 0
        capital_to_risk = self.get_current_equity() * self.RISK_PER_TRADE_PCT
        risk_per_share = abs(entry_price - stop_loss_price)
        if risk_per_share == 0: return 0
        return int(capital_to_risk / risk_per_share)

    def on_new_day(self, date):
        if self.current_date != date.date():
            self.current_date = date.date()
            self.equity_at_day_start = self.get_current_equity()
            self.daily_pnl = 0.0
            self.is_trading_halted_today = False

    def request_trade(self, timestamp, symbol, side, price, stop_loss=0, target=0):
        if self.is_trading_halted_today:
            return

        if side == 'buy':
            if symbol in self.positions: return
            quantity = self._calculate_position_size(price, stop_loss)
            if quantity == 0: return
            actual_entry_price = self._apply_slippage(price, 'buy')
            trade_cost = quantity * actual_entry_price
            brokerage = trade_cost * self.BROKERAGE_PCT
            if self.cash < trade_cost + brokerage: return

            self.cash -= (trade_cost + brokerage)
            self.positions[symbol] = {
                'qty': quantity,
                'entry_price': actual_entry_price,
                'entry_timestamp': timestamp,
                'stop_loss': stop_loss,
                'target': target
            }

        elif side == 'sell':
            if symbol not in self.positions: return
            position = self.positions[symbol]
            actual_exit_price = self._apply_slippage(price, 'sell')
            proceeds = position['qty'] * actual_exit_price
            entry_cost = position['qty'] * position['entry_price']
            pnl = (actual_exit_price - position['entry_price']) * position['qty']
            
            # Calculate brokerage on entry and exit
            entry_brokerage = entry_cost * self.BROKERAGE_PCT
            exit_brokerage = proceeds * self.BROKERAGE_PCT
            total_brokerage = entry_brokerage + exit_brokerage
            
            final_pnl = pnl - total_brokerage
            
            self.cash += proceeds - exit_brokerage
            self.daily_pnl += final_pnl
            
            self.trade_log.append({
                'entry_timestamp': position['entry_timestamp'],
                'exit_timestamp': timestamp, 'symbol': symbol, 'pnl': final_pnl
            })
            del self.positions[symbol]
            max_allowed_daily_loss = self.equity_at_day_start * self.MAX_DAILY_LOSS_PCT
            if self.daily_pnl < -max_allowed_daily_loss:
                self.is_trading_halted_today = True

    from typing import Optional

    def get_current_equity(self, current_prices: Optional[pd.Series] = None):
        current_value = self.cash
        if current_prices is not None:
            for symbol, pos in self.positions.items():
                current_value += pos['qty'] * current_prices.get(symbol, pos['entry_price'])
        else:
            for symbol, pos in self.positions.items():
                current_value += pos['qty'] * pos['entry_price']
        return current_value

    def record_equity(self, timestamp, current_prices: pd.Series):
        equity = self.get_current_equity(current_prices)
        new_row = pd.DataFrame([{'timestamp': timestamp, 'equity': equity}])
        self.equity_df = pd.concat([self.equity_df, new_row], ignore_index=True)

    def update_open_positions(self, timestamp, current_low, current_high):
        """
        Checks if the stop-loss or take-profit for any open position has been hit.
        """
        positions_to_close = []
        for symbol, pos in self.positions.items():
            # For long positions
            if pos['qty'] > 0:
                if 'stop_loss' in pos and pd.notna(pos['stop_loss']) and current_low <= pos['stop_loss']:
                    positions_to_close.append({'symbol': symbol, 'price': pos['stop_loss'], 'reason': 'SL'})
                elif 'target' in pos and pd.notna(pos['target']) and current_high >= pos['target']:
                    positions_to_close.append({'symbol': symbol, 'price': pos['target'], 'reason': 'TP'})
            # Add logic for short positions if necessary

        for trade_to_close in positions_to_close:
            symbol = trade_to_close['symbol']
            price = trade_to_close['price']
            # Ensure position still exists before closing
            if symbol in self.positions:
                self.request_trade(timestamp, symbol, 'sell', price)
