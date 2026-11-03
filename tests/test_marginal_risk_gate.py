import unittest
import numpy as np
import pandas as pd
from src.marginal_risk_gate import calibration_schedule,variance_change,conditional_risk


class MarginalGateTests(unittest.TestCase):
    def test_calibration_never_uses_unmatured_future_labels(self):
        dates=pd.bdate_range('2020-01-01',periods=200)
        slopes=pd.DataFrame({'slope':np.sin(np.arange(194))*.001+.002,'label_exit':dates[6:]},index=dates[:-6])
        cutoff=dates[120]
        a=calibration_schedule(slopes,[cutoff])
        slopes.loc[slopes.label_exit>cutoff,'slope']=999
        b=calibration_schedule(slopes,[cutoff])
        pd.testing.assert_frame_equal(a,b)
        self.assertLessEqual(a.max_label_exit.iloc[0],cutoff)

    def test_marginal_risk_matches_full_single_factor_variance(self):
        w={'a':.3,'b':.69};beta={'a':1.4,'b':.8,'c':1.0};specific={'a':.0004,'b':.0001,'c':.0002};m=.0001
        def variance(weights):return (sum(v*beta[c] for c,v in weights.items())-1)**2*m+sum(v*v*specific[c] for c,v in weights.items())
        new={'c':.3,'b':.69}
        self.assertAlmostEqual(variance_change(w,beta,specific,m,'a','c'),variance(new)-variance(w))

    def test_risk_is_causal_and_does_not_shrink_benchmark_beta(self):
        rng=np.random.default_rng(71);m=rng.normal(0,.01,400);stock=1.5*m
        beta,var,_=conditional_risk(stock,m)
        self.assertAlmostEqual(beta[250],1.5,places=10)
        stock[251:]=999;m[251:]=-999
        beta2,var2,_=conditional_risk(stock,m)
        np.testing.assert_array_equal(beta[:251],beta2[:251]);np.testing.assert_array_equal(var[:251],var2[:251])
