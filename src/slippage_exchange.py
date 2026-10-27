"""Directional price slippage, with original fees and close-based end-day valuation."""
import math
import numpy as np
from qlib.backtest.decision import Order
from .exchange import AdjustedExchange
from .slippage_math import adverse_price


class SlippageExchange(AdjustedExchange):
    def __init__(self, slippage_bps, **kwargs):
        if not math.isfinite(slippage_bps) or not 0 <= slippage_bps < 10000:
            raise ValueError('Slippage must be in [0, 10000) basis points.')
        self.slippage_bps = slippage_bps
        self.price_rejections = []
        super().__init__(**kwargs)

    def check_order(self, order):
        if not super().check_order(order):
            return False
        price = self.get_deal_price(order.stock_id, order.start_time, order.end_time, order.direction)
        low, high = self._field(order, '$low'), self._field(order, '$high')
        if not (np.isfinite(price) and np.isfinite(low) and np.isfinite(high)
                and low-1e-10 <= price <= high+1e-10):
            self.price_rejections.append({'date':str(order.start_time.date()),
                'instrument':order.stock_id,'direction':int(order.direction),
                'slipped_price':float(price),'low':float(low),'high':float(high)})
            return False
        return True

    def get_deal_price(self, stock_id, start_time, end_time, direction, method='ts_data_last'):
        reference = super().get_deal_price(stock_id, start_time, end_time, direction, method)
        if reference is None or not np.isfinite(reference) or reference <= 0:
            return reference
        side = 'buy' if direction == Order.BUY else 'sell'
        return adverse_price(float(reference), side, self.slippage_bps)

    def _clip_amount_by_volume(self, order, dealt_order_amount):
        super()._clip_amount_by_volume(order, dealt_order_amount)
        # Actual notional must still fit the original amount cap.
        price = self.get_deal_price(order.stock_id, order.start_time, order.end_time, order.direction)
        amount = self._field(order, '$amount')
        if not (np.isfinite(price) and price > 0 and np.isfinite(amount) and amount > 0):
            order.deal_amount = 0.
            return
        already = float(dealt_order_amount.get(order.stock_id, 0.))
        cap = max(0., amount * 1000. * self.capacity_fraction / price - already)
        previous = order.deal_amount
        order.deal_amount = min(order.deal_amount, cap)
        if order.deal_amount < previous:
            factor = self._field(order, '$factor')
            order.deal_amount = (np.floor(order.deal_amount * factor / 100.) * 100. / factor
                                 if np.isfinite(factor) and factor > 0 else 0.)
