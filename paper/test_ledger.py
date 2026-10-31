import importlib.util,sqlite3,tempfile,unittest
from pathlib import Path
spec=importlib.util.spec_from_file_location('paper_cli',Path(__file__).with_name('cli.py'))
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class LedgerTests(unittest.TestCase):
    def test_previous_close_plan_and_holding_age(self):
        import pandas as pd
        from types import SimpleNamespace
        from qlib.backtest.position import Position
        from src.orders import decide_orders
        class Signal:
            def get_signal(self,**kwargs):return pd.Series({'SH600000':0.,'SH600001':1.})
        pos=Position(cash=100.,position_dict={'SH600000':{'amount':100.,'price':10.,'count_day':0}})
        strategy=SimpleNamespace(trade_calendar=module.Calendar('2026-10-30','2026-11-02'),signal=Signal(),
            trade_position=pos,prior_closes=pd.DataFrame([{'SH600000':10.,'SH600001':20.}],index=[pd.Timestamp('2026-10-30')]),
            n_drop=1,topk=1,hold_thresh=1,risk_degree=.95,timing=[],
            trade_exchange=SimpleNamespace(open_cost=.0005,close_cost=.0015,min_cost=5.))
        self.assertEqual(decide_orders(strategy).get_decision(),[])
        pos.position['SH600000']['count_day']=1
        orders=decide_orders(strategy).get_decision()
        self.assertEqual([o.direction for o in orders],[0,1])
        self.assertTrue(all(o.start_time==pd.Timestamp('2026-11-02') for o in orders))
        self.assertLess(orders[1].amount*20*1.0005+5,1100)

    def test_duplicate_date_is_rejected_and_reopen_keeps_state(self):
        base=Path(__file__).resolve().parents[1]/'runs/paper-tests';base.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as folder:
            path=Path(folder)/'ledger.sqlite';ledger=module.Ledger(path)
            ledger.save('2026-10-30',{'asof':'2026-10-30','cash':100.})
            with self.assertRaises(sqlite3.IntegrityError):ledger.save('2026-10-30',{'cash':0})
            ledger.db.rollback();ledger.db.close()
            reopened=module.Ledger(path)
            self.assertEqual(reopened.latest()['cash'],100.)
            self.assertEqual(reopened.db.execute('SELECT COUNT(*) FROM days').fetchone()[0],1)
            reopened.db.close()
    def test_invalid_numeric_state_cannot_be_serialized(self):
        with self.assertRaises(ValueError):module.encode({'cash':float('nan')})

if __name__=='__main__':unittest.main()
