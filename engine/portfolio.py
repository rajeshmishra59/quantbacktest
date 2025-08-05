# quantbacktest/engine/portfolio.py
class Portfolio:
    def __init__(self, initial_cash):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions = {}  # symbol: { 'qty': int, 'avg_price': float }
        self.trade_log = []
        self.total_pnl = 0.0

    def update_position(self, symbol, qty, price, side):
        if symbol not in self.positions:
            self.positions[symbol] = {'qty': 0, 'avg_price': 0}
        position = self.positions[symbol]
        if side == 'buy':
            total_cost = position['qty'] * position['avg_price'] + qty * price
            position['qty'] += qty
            position['avg_price'] = total_cost / position['qty'] if position['qty'] else 0
            self.cash -= qty * price
        elif side == 'sell':
            position['qty'] -= qty
            self.cash += qty * price
            if position['qty'] == 0:
                pnl = (price - position['avg_price']) * qty
                self.total_pnl += pnl
                self.trade_log.append({'symbol': symbol, 'qty': qty, 'exit_price': price, 'pnl': pnl})

    def get_equity(self):
        return self.cash + self.total_pnl
