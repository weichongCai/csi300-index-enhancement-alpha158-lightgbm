import unittest
import numpy as np
import pandas as pd
from types import SimpleNamespace
from qlib.backtest.decision import Order
from src.open_execution import CausalExecutionExchange,PREVIOUS,LAG_AMOUNT,LAG_VOLUME


class Quote:
    def __init__(self,values):self.values=values
    def get_all_stock(self):return ['SH600000']
    def get_data(self,*args,field=None,**kwargs):return self.values[field]


class OpeningTests(unittest.TestCase):
    def market(self):
        m=object.__new__(CausalExecutionExchange)
        m.execution='open';m.slippage_bps=3.;m.capacity_fraction=.05
        m.open_cost=.0005;m.close_cost=.0015;m.min_cost=5.;m.trade_unit=100
        m.quote=Quote({'$open':10.,'$close':11.,'$high':12.,'$low':9.,'$factor':1.,
            LAG_AMOUNT:1e6,LAG_VOLUME:1e6,'$volume':1.,'$amount':1.})
        return m

    def test_open_limits_do_not_use_future_close(self):
        m=self.market();m.quote_df=pd.DataFrame({'$open':[10.2,11.,9.],PREVIOUS:[10.,10.,10.],'$close':[100.,1.,100.]})
        m._update_limit(.095);buy=m.quote_df.limit_buy.copy();sell=m.quote_df.limit_sell.copy()
        m.quote_df['$close']=[1.,100.,1.];m._update_limit(.095)
        pd.testing.assert_series_equal(buy,m.quote_df.limit_buy);pd.testing.assert_series_equal(sell,m.quote_df.limit_sell)
        self.assertEqual(list(buy),[False,True,False]);self.assertEqual(list(sell),[False,False,True])

    def test_fill_ignores_current_day_close_high_low_volume(self):
        m=self.market();order=Order(stock_id='SH600000',amount=1000,direction=Order.BUY,start_time=pd.Timestamp('2020-01-02'),end_time=pd.Timestamp('2020-01-02'))
        a=m._calc_trade_info_by_order(order,None,{})
        for field in ['$close','$high','$low','$volume','$amount']:m.quote.values[field]=np.nan
        b=m._calc_trade_info_by_order(order,None,{})
        np.testing.assert_array_equal(a,b)

    def test_full_exit_preserves_fractional_adjusted_holding(self):
        m=self.market();m.quote.values['$factor']=.9063
        position=SimpleNamespace(check_stock=lambda c:True,get_stock_amount=lambda c:110.3405074885,get_cash=lambda:10000.)
        order=Order(stock_id='SH600000',amount=110.3405074885,direction=Order.SELL,start_time=pd.Timestamp('2020-01-02'),end_time=pd.Timestamp('2020-01-02'))
        m._calc_trade_info_by_order(order,position,{})
        self.assertEqual(order.deal_amount,110.3405074885)

    def test_buy_checks_minimum_commission_and_cash(self):
        m=self.market();position=SimpleNamespace(get_cash=lambda:1005.)
        order=Order(stock_id='SH600000',amount=100,direction=Order.BUY,start_time=pd.Timestamp('2020-01-02'),end_time=pd.Timestamp('2020-01-02'))
        _,value,cost=m._calc_trade_info_by_order(order,position,{})
        self.assertEqual(order.deal_amount,0.)
        self.assertLessEqual(value+cost,1005.)
