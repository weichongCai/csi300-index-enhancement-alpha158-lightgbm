"""Versioned paper execution rules and conservative auction funding."""
from types import SimpleNamespace
from src.open_execution import CausalExecutionExchange
from qlib.backtest.decision import Order

NEXT_OPEN='next_open_prefunded_v1'
LEGACY_CLOSE='legacy_next_close_v1'


def execution_rule(row):
    return row.get('execution_policy',LEGACY_CLOSE)


def resize_buys_to_cash(orders,cash,prices,fee_rate,min_fee,fraction=.95):
    buys=[o for o in orders if o.direction==Order.BUY]
    budget=max(0.,cash)*fraction/len(buys) if buys else 0.
    for order in buys:
        order.amount=max(0.,budget-min_fee)/(1+fee_rate)/prices[order.stock_id]
    return [o for o in orders if o.amount>0]


class PrefundedOpenExchange(CausalExecutionExchange):
    def __init__(self,*,cash_before_auction,**kwargs):
        self.remaining_buy_cash=float(cash_before_auction)
        super().__init__(execution='open',**kwargs)

    def _calc_trade_info_by_order(self,order,position,dealt_order_amount):
        if order.direction==Order.BUY and position is not None:
            real=position
            class CashView:
                def get_cash(view):return min(real.get_cash(),self.remaining_buy_cash)
                def __getattr__(view,name):return getattr(real,name)
            price,value,cost=super()._calc_trade_info_by_order(order,CashView(),dealt_order_amount)
            self.remaining_buy_cash=max(0.,self.remaining_buy_cash-value-cost)
            return price,value,cost
        return super()._calc_trade_info_by_order(order,position,dealt_order_amount)
