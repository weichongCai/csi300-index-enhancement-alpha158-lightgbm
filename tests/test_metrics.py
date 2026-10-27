import unittest
import numpy as np
import pandas as pd
from src.metrics import drawdown, ir


class MetricConventionTests(unittest.TestCase):
    def test_initial_trading_fee_is_a_drawdown(self):
        np.testing.assert_allclose(drawdown([.999, 1.1, .99]), [-.001, 0., -.1])

    def test_information_ratio_uses_daily_active_risk(self):
        active = pd.Series([.01, -.01, .02, 0.])
        self.assertAlmostEqual(ir(active), 6.148170459575759)

    def test_equal_daily_returns_have_undefined_ir(self):
        self.assertTrue(np.isnan(ir(pd.Series([.01, .01, .01]))))


if __name__ == '__main__':
    unittest.main()
