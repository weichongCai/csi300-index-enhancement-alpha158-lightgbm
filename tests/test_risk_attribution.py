import unittest
import numpy as np
import pandas as pd
from src.risk_attribution import episodes, linked_contribution, variance_attribution


class RiskAttributionTests(unittest.TestCase):
    def test_initial_capital_peak_and_unrecovered_drawdown(self):
        dates = pd.date_range('2016-01-04', periods=4)
        e = episodes(pd.Series([.9, 1.1, .88, .99], index=dates))
        self.assertAlmostEqual(e.iloc[0].depth, -.2)
        self.assertTrue(pd.isna(e.iloc[0].recovery))
        initial = e[e.peak == '2015-12-31'].iloc[0]
        self.assertEqual(initial.recovery, '2016-01-05')
        self.assertAlmostEqual(initial.depth, -.1)

    def test_drawdown_recovery_can_equal_previous_peak(self):
        dates = pd.date_range('2020-01-01', periods=3)
        e = episodes(pd.Series([1.2, 1.08, 1.2], index=dates))
        self.assertEqual(len(e), 1)
        self.assertEqual(e.iloc[0].recovery, '2020-01-03')

    def test_linking_costs_is_not_simple_return_addition(self):
        index = pd.date_range('2016-01-04', periods=3)
        r = pd.Series([.1, -.2, .05], index=index)
        b = pd.Series([.03, -.1, .02], index=index)
        daily = pd.DataFrame({'model_nav': (1+r).cumprod(), 'benchmark_return': b,
                              'relative_nav': (1+r).cumprod()/(1+b).cumprod()})
        c = pd.DataFrame({'market': r+.02, 'commission': -.02}, index=index)
        contribution, _, _ = linked_contribution(c, daily, '2015-12-31', '2016-01-05')
        self.assertAlmostEqual(contribution.sum(), -.12)
        self.assertAlmostEqual(contribution.commission, -.042)
        self.assertNotAlmostEqual(contribution.sum(), r.iloc[:2].sum())

    def test_relative_linking_uses_geometric_relative_returns(self):
        index = pd.date_range('2016-01-04', periods=2)
        rp, rb = pd.Series([.1, -.2], index=index), pd.Series([.05, -.1], index=index)
        daily = pd.DataFrame({'model_nav': (1+rp).cumprod(), 'benchmark_return': rb,
                              'relative_nav': (1+rp).cumprod()/(1+rb).cumprod()})
        c = pd.DataFrame({'market': rp+.01, 'commission': -.01}, index=index)
        contribution, _, _ = linked_contribution(c, daily, '2015-12-31', '2016-01-05', True)
        self.assertAlmostEqual(contribution.sum(), .88/.945-1)

    def test_covariance_risk_contribution_allows_diversification(self):
        x = np.array([-.02, -.01, .01, .02])
        c = pd.DataFrame({'market': x, 'hedge': -.5*x, 'fixed_cost': -.001})
        result = variance_attribution(c, c.sum(axis=1)).set_index('component')
        self.assertAlmostEqual(result.variance_share.sum(), 1.)
        self.assertLess(result.loc['hedge','variance_share'], 0.)
        self.assertAlmostEqual(result.loc['fixed_cost','variance_share'], 0.)


if __name__ == '__main__':
    unittest.main()
