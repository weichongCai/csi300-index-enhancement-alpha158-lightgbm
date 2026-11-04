import unittest
import numpy as np
import pandas as pd
from qlib.backtest.decision import Order
from src.industry_style_risk import factor_variance_change,factor_covariance,asof_indices,cross_standardize,broad_sector
from src.turnover_style_gate import planned_turnover,cap_exemption


class IndustryStyleTests(unittest.TestCase):
    def test_count_above_50_does_not_exempt_paired_replacement(self):
        paired=[dict(forced=True,mandatory_exit=False,new='b')]
        self.assertEqual(cap_exemption(paired,53),'none')
        self.assertEqual(cap_exemption([dict(forced=True,mandatory_exit=True,new='b')],53),'mandatory_exit_or_pure_trim')
        self.assertEqual(cap_exemption([dict(forced=True,mandatory_exit=False,new=None)],53),'mandatory_exit_or_pure_trim')
        self.assertEqual(cap_exemption([],0),'initial_investment')

    def test_sector_mapping_handles_historical_text_and_modern_codes(self):
        self.assertEqual(broad_sector('制造业-计算机通信'),broad_sector('C39计算机、通信和其他电子设备制造业'))
        self.assertEqual(broad_sector('金融保险业-银行业'),'J')
        self.assertEqual(broad_sector(None),'UNKNOWN')
        self.assertEqual(broad_sector('社会服务业-旧分类'),'LEGACY_SOCIAL')

    def test_whole_active_variance_matches_marginal_replacement(self):
        x={'a':np.array([1.5,.4,1.]),'b':np.array([.9,-.1,0.]),'c':np.array([1.1,.2,1.])}
        f=np.array([[.0002,.00001,0.],[.00001,.0001,-.00002],[0.,-.00002,.0003]])
        d={'a':.0004,'b':.0002,'c':.0003};w={'a':.2,'b':.79}
        def var(weights):
            a=sum((v*x[c] for c,v in weights.items()),np.zeros(3))-np.array([1.,0.,0.])
            return a@f@a+sum(v*v*d[c] for c,v in weights.items())
        actual=factor_variance_change(w,x,f,d,'a','c')
        self.assertAlmostEqual(actual,var({'c':.2,'b':.79})-var(w),places=15)

    def test_covariance_is_psd_and_future_data_cannot_change_past(self):
        rng=np.random.default_rng(7);f=rng.normal(0,.01,(320,5));f[12]=np.nan
        a=factor_covariance(f);f[201:]=999;b=factor_covariance(f)
        np.testing.assert_array_equal(a[:201],b[:201])
        self.assertGreaterEqual(np.linalg.eigvalsh(a).min(),-1e-15)

    def test_month_end_industry_is_not_backfilled_to_month_start(self):
        snapshots=pd.to_datetime(['2020-01-31','2020-02-28'])
        dates=pd.to_datetime(['2020-01-20','2020-02-03','2020-02-28'])
        np.testing.assert_array_equal(asof_indices(snapshots,dates),[-1,0,1])

    def test_standardization_uses_only_current_members(self):
        x=np.arange(30,dtype=float);members=np.arange(30)<25
        a,_=cross_standardize(x,members);x[25:]=100000;b,_=cross_standardize(x,members)
        np.testing.assert_array_equal(a[:25],b[:25])

    def test_turnover_counts_both_sides_then_divides_by_two(self):
        day=pd.Timestamp('2020-01-01')
        orders=[Order(stock_id='a',amount=100,start_time=day,end_time=day,direction=Order.SELL),
                Order(stock_id='b',amount=50,start_time=day,end_time=day,direction=Order.BUY)]
        self.assertAlmostEqual(planned_turnover(orders,{'a':10.,'b':20.},100000),.01)


if __name__=='__main__':unittest.main()
