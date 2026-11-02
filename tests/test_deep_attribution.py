import unittest
import numpy as np
import pandas as pd
from src.deep_attribution import cross_section,STYLES


class CrossSectionAttributionTests(unittest.TestCase):
    def frame(self,n=120):
        rng=np.random.default_rng(2301)
        frame=pd.DataFrame(rng.normal(size=(n,3)),columns=STYLES)
        frame['beta']=rng.uniform(.5,1.5,n)
        frame['beta_known']=1
        return frame

    def test_recovers_known_factor_returns_and_market_offset(self):
        frame=self.frame();z=(frame[STYLES]-frame[STYLES].mean())/frame[STYLES].std()
        coefficients=np.array([.003,-.002,.001])
        returns=frame.beta*.02+.0005+z@coefficients
        _,coef,residual,_=cross_section(frame,returns,.02)
        np.testing.assert_allclose(coef,np.r_[.0005,coefficients],atol=1e-13)
        np.testing.assert_allclose(residual,0.,atol=1e-13)

    def test_does_not_use_imputed_beta_or_missing_styles(self):
        frame=self.frame();frame.loc[:9,'beta_known']=0;frame.loc[10:14,STYLES[0]]=np.nan
        fit=cross_section(frame,pd.Series(0.,index=frame.index),.01)
        self.assertEqual(len(fit[0]),105)
        self.assertTrue((fit[0].index>=15).all())

    def test_does_not_fit_when_complete_universe_is_too_small(self):
        frame=self.frame(99)
        self.assertIsNone(cross_section(frame,pd.Series(0.,index=frame.index),.01))

    def test_rejects_unidentified_collinear_factor_design(self):
        frame=self.frame();frame[STYLES[1]]=frame[STYLES[0]]
        self.assertIsNone(cross_section(frame,pd.Series(0.,index=frame.index),.01))
