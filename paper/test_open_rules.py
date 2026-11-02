import unittest,tempfile,json
from pathlib import Path
from types import SimpleNamespace
import pandas as pd
from qlib.backtest.decision import Order
from paper import cli
from paper.open_rules import PrefundedOpenExchange,NEXT_OPEN,LEGACY_CLOSE,execution_rule,resize_buys_to_cash
from src.open_execution import LAG_AMOUNT,LAG_VOLUME


class Quote:
    def get_all_stock(self):return ['SH600000']
    def get_data(self,*args,field=None,**kwargs):return {'$open':10.,'$factor':1.,LAG_AMOUNT:1e6,LAG_VOLUME:1e6}[field]


class PaperOpenTests(unittest.TestCase):
    def order(self):return Order(stock_id='SH600000',amount=1000,direction=Order.BUY,start_time=pd.Timestamp('2020-01-02'),end_time=pd.Timestamp('2020-01-02'))

    def test_same_auction_sale_cash_cannot_fund_buys(self):
        m=object.__new__(PrefundedOpenExchange);m.execution='open';m.slippage_bps=3.;m.capacity_fraction=.05
        m.open_cost=.0005;m.close_cost=.0015;m.min_cost=5.;m.quote=Quote();m.remaining_buy_cash=1100.
        position=SimpleNamespace(get_cash=lambda:100000.)
        o=self.order();_,value,cost=m._calc_trade_info_by_order(o,position,{})
        self.assertEqual(o.deal_amount,100.)
        self.assertLessEqual(value+cost,1100.)
        o=self.order();_,value,cost=m._calc_trade_info_by_order(o,position,{})
        self.assertEqual(o.deal_amount,0.)

    def test_plan_budget_uses_cash_not_expected_sales(self):
        orders=[self.order(),self.order()]
        sized=resize_buys_to_cash(orders,1000.,{'SH600000':10.},.0005,5.)
        total=sum(o.amount*10+max(o.amount*10*.0005,5) for o in sized)
        self.assertLessEqual(total,950.)
        self.assertEqual(resize_buys_to_cash([self.order()],0.,{'SH600000':10.},.0005,5.),[])

    def test_legacy_pending_plan_keeps_original_execution(self):
        self.assertEqual(execution_rule({}),LEGACY_CLOSE)
        self.assertEqual(execution_rule({'execution_policy':NEXT_OPEN}),NEXT_OPEN)

    def test_upgrade_is_append_only_idempotent_and_exact_policy(self):
        base=Path(__file__).resolve().parents[1]/'runs/paper-tests';base.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as folder:
            ledger=cli.Ledger(Path(folder)/'ledger.sqlite')
            policy=cli.current_policy();old={k:v for k,v in policy.items() if k not in ['src/backtest.py','src/open_execution.py','paper/open_rules.py']}
            old['paper/cli.py']='legacy-cli-hash'
            row={'asof':'2026-10-30','policy':old};ledger.save(row['asof'],row)
            payload=ledger.db.execute('SELECT payload FROM days').fetchone()[0];ledger.db.close()
            cli.register_upgrade(folder);cli.register_upgrade(folder)
            ledger=cli.Ledger(Path(folder)/'ledger.sqlite')
            self.assertEqual(ledger.db.execute('SELECT payload FROM days').fetchone()[0],payload)
            self.assertEqual(ledger.db.execute('SELECT COUNT(*) FROM rule_upgrades').fetchone()[0],1)
            self.assertTrue(cli.upgrade_authorized(ledger,row,policy))
            self.assertFalse(cli.upgrade_authorized(ledger,row,{**policy,'config.json':'changed'}))
            ledger.db.close()
