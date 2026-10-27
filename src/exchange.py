"""Qlib adjusted units, with the existing cash, capacity and one-price guards."""
import numpy as np
from qlib.backtest.exchange import Exchange
from qlib.backtest.decision import Order


def normalized_capacity(volume, amount, close, fraction, already=0.):
    if not 0 < fraction <= 1:
        raise ValueError('capacity fraction must be in (0, 1]')
    if not all(np.isfinite(x) and x > 0 for x in [volume, amount, close]):
        return 0.
    # This project's provider stores volume in adjusted lots and amount in kCNY.
    return max(0., min(volume * 100. * fraction,
                       amount * 1000. * fraction / close) - already)


class AdjustedExchange(Exchange):
    def __init__(self, capacity_fraction=.05, **kwargs):
        self.capacity_fraction = capacity_fraction
        super().__init__(**kwargs)

    def _field(self, order, name):
        value = self.quote.get_data(order.stock_id, order.start_time, order.end_time,
                                    field=name, method='ts_data_last')
        return float(value) if value is not None else np.nan

    def check_order(self, order):
        if not super().check_order(order):
            return False
        high, low, close, previous = [self._field(order, f) for f in
                                      ['$high', '$low', '$close', 'Ref($close,1)']]
        if not all(np.isfinite(x) and x > 0 for x in [high, low, close, previous]):
            return False
        against = ((order.direction == Order.BUY and close > previous) or
                   (order.direction == Order.SELL and close < previous))
        return not (high == low and against)

    def _clip_amount_by_volume(self, order, dealt_order_amount):
        super()._clip_amount_by_volume(order, dealt_order_amount)
        cap = normalized_capacity(
            self._field(order, '$volume'), self._field(order, '$amount'),
            self._field(order, '$close'), self.capacity_fraction,
            float(dealt_order_amount.get(order.stock_id, 0.)))
        order.deal_amount = min(order.deal_amount, cap)
        if order.deal_amount < order.amount:
            factor = self._field(order, '$factor')
            order.deal_amount = (np.floor(order.deal_amount * factor / 100.) * 100. / factor
                                 if np.isfinite(factor) and factor > 0 else 0.)

    def _calc_trade_info_by_order(self, order, position, dealt_order_amount):
        price, value, cost = super()._calc_trade_info_by_order(
            order, position, dealt_order_amount)
        if position is not None and order.direction == Order.BUY:
            if self.impact_cost != 0:
                raise ValueError('This cash guard assumes zero impact_cost.')
            unit = self.trade_unit / order.factor
            while order.deal_amount > 0 and value + cost > position.get_cash() + 1e-7:
                order.deal_amount = max(0, order.deal_amount - unit)
                value = order.deal_amount * price
                cost = max(value * self.open_cost, self.min_cost) if value > 1e-5 else 0.
        if (order.direction == Order.BUY and order.stock_id.startswith('SH688')
                and order.deal_amount * order.factor < 199.9):
            order.deal_amount = 0.
            value = cost = 0.
        cap = normalized_capacity(
            self._field(order, '$volume'), self._field(order, '$amount'),
            self._field(order, '$close'), self.capacity_fraction,
            float(dealt_order_amount.get(order.stock_id, 0.)))
        if order.deal_amount > cap + 1e-7:
            raise ArithmeticError('Fill exceeds daily capacity.')
        if position is not None and order.direction == Order.SELL:
            if order.deal_amount > position.get_stock_amount(order.stock_id) + 1e-7:
                raise ArithmeticError('Fill exceeds the position.')
        return price, value, cost
