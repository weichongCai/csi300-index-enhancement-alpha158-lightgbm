import unittest
import numpy as np
import pandas as pd
from src.active_risk_system import estimate_risk,risk_numbers,optimize_weights,POLICY


class ActiveRiskSystemTests(unittest.TestCase):
    def test_future_returns_do_not_change_risk_or_weights(self):
        rng=np.random.default_rng(80);dates=pd.bdate_range('2018-01-01',periods=350)
        b=pd.Series(rng.normal(0,.01,350),index=dates)
        r=pd.DataFrame(b.to_numpy()[:,None]+rng.normal(0,.008,(350,30)),index=dates)
        asof=dates[270]; codes=list(r.columns)
        cov,beta,_,_=estimate_risk(r,b,asof,codes)
        r.loc[r.index>asof]=999;b.loc[b.index>asof]=-999
        cov2,beta2,_,_=estimate_risk(r,b,asof,codes)
        np.testing.assert_array_equal(cov,cov2);np.testing.assert_array_equal(beta,beta2)
        q=np.full(30,.99/30)
        a,_=optimize_weights(q,q,cov,beta)
        a2,_=optimize_weights(q,q,cov2,beta2)
        np.testing.assert_array_equal(a,a2)

    def test_cash_is_a_benchmark_shortfall_not_a_free_risk_hedge(self):
        covariance=np.array([[.0001,.0001],[.0001,.0001]])
        te,rc=risk_numbers(np.array([.5]),covariance)
        self.assertAlmostEqual(te,.005*np.sqrt(252))
        self.assertAlmostEqual(rc.sum(),te**2)

    def test_constraints_and_covariance_euler(self):
        n=40; covariance=np.full((n+1,n+1),.00008)+np.eye(n+1)*.00001
        q=np.full(n,.99/n);previous=q.copy();beta=np.ones(n)
        weights,_=optimize_weights(q,previous,covariance,beta)
        te,rc=risk_numbers(weights,covariance)
        self.assertAlmostEqual(weights.sum(),.99,places=6)
        self.assertLessEqual(weights.max(),.040002)
        self.assertLessEqual(te,.100002)
        self.assertAlmostEqual(rc.sum(),te**2)

    def test_infeasible_constraints_raise_without_relaxation(self):
        n=30;q=np.full(n,.99/n)
        with self.assertRaises(ValueError):
            optimize_weights(q,q,np.eye(n+1)*.0001,np.zeros(n))

    def test_soft_risk_budget_cannot_turn_infeasibility_into_cash_timing(self):
        n=30;q=np.full(n,.99/n)
        covariance=np.full((n+1,n+1),.00008)+np.eye(n+1)*.003
        w,_=optimize_weights(q,q,covariance,np.ones(n))
        te,_=risk_numbers(w,covariance)
        self.assertGreater(te,POLICY['annual_te_limit'])
        self.assertAlmostEqual(w.sum(),.99,places=6)

    def test_missing_history_is_not_filled(self):
        dates=pd.bdate_range('2020-01-01',periods=200)
        r=pd.DataFrame({'a':np.sin(np.arange(200))*.01},index=dates)
        r.iloc[:100]=np.nan
        with self.assertRaises(ValueError):
            estimate_risk(r,pd.Series(np.cos(np.arange(200))*.01,index=dates),dates[-1],['a'])

    def test_partial_and_stale_holdings_have_explicit_psd_risk_reserves(self):
        rng=np.random.default_rng(22);dates=pd.bdate_range('2018-01-01',periods=300)
        b=pd.Series(rng.normal(0,.01,300),index=dates)
        r=pd.DataFrame({'full':b+rng.normal(0,.008,300),
            'partial':b+rng.normal(0,.008,300),'stale':b.copy(),'stale2':b.copy()},index=dates)
        r.loc[dates[:100],'partial']=np.nan
        r.loc[dates[40:],'stale']=np.nan;r.loc[dates[40:],'stale2']=np.nan
        c,beta,h,_=estimate_risk(r,b,dates[-1],list(r.columns),allow_reserve=True)
        self.assertEqual(h.attrs['reserve_codes'],['stale','stale2'])
        self.assertEqual(h.attrs['risk_kind']['partial'],'partial_history_market_extension')
        self.assertTrue(h.stale.isna().all())
        self.assertGreaterEqual(np.linalg.eigvalsh(c).min(),-1e-12)
        self.assertAlmostEqual(c[2,3]-beta[2]*beta[3]*c[-1,-1],1/252)

    def test_unknown_target_can_be_forced_to_exit_without_relaxing_other_limits(self):
        n=31;q=np.r_[np.full(30,.99/30),0.];previous=q.copy();previous[0]-=.01;previous[-1]=.01
        cov=np.full((n+1,n+1),.00008)+np.eye(n+1)*.00001
        upper=np.r_[np.full(30,.04),0.]
        w,_=optimize_weights(q,previous,cov,np.ones(n),upper_bounds=upper)
        self.assertLess(w[-1],2e-6);self.assertAlmostEqual(w.sum(),.99,places=6)

    def test_risk_optimization_cannot_sacrifice_reference_rank_budget(self):
        n=40;signal=np.linspace(0,1,n);q=np.r_[np.zeros(10),np.full(30,.99/30)]
        cov=np.full((n+1,n+1),.00008)+np.diag(np.r_[np.linspace(.00001,.0002,n),.00001])
        w,_=optimize_weights(q,q,cov,np.ones(n),signal=signal)
        self.assertGreaterEqual(signal@w,signal@q-2e-6)

    def test_unattainable_reference_has_explicit_feasible_signal_frontier(self):
        n=40;signal=np.linspace(0,1,n);q=np.r_[np.zeros(30),np.full(10,.99/10)]
        previous=np.full(n,.99/n)
        cov=np.full((n+1,n+1),.00008)+np.eye(n+1)*.00001
        w,_,details=optimize_weights(q,previous,cov,np.ones(n),signal=signal,bootstrap=True,return_details=True)
        self.assertGreater(details['rank_reference_feasibility_shortfall'],0)
        self.assertGreaterEqual(signal@w,details['required_rank_exposure']-2e-6)
        self.assertAlmostEqual(w.sum(),.99,places=6)

    def test_monitor_uses_candidate_covariance_not_only_held_names(self):
        from src.active_risk_report import monitor
        rng=np.random.default_rng(70);dates=pd.bdate_range('2018-01-01',periods=300)
        b=pd.Series(rng.normal(0,.01,300),index=dates)
        r=pd.DataFrame({c:b+rng.normal(0,s,300) for c,s in [('a',.01),('b',.02),('c',.06)]})
        asof=dates[-2];holdings=pd.DataFrame({'date':[asof,asof],'instrument':['a','b'],'weight':[.4,.59]})
        forecast=pd.Series([1.,2.,3.],index=pd.MultiIndex.from_product([[asof],['a','b','c']],names=['datetime','instrument']))
        curves=pd.DataFrame({'relative_drawdown':[0.]},index=[asof])
        watch,_=monitor(holdings,r,b,curves,forecast)
        c,_,_,_=estimate_risk(r,b,asof,['a','b','c'],allow_reserve=True)
        te,_=risk_numbers(np.array([.4,.59,0.]),c)
        self.assertAlmostEqual(watch.predicted_te.iloc[0],te,places=12)


if __name__=='__main__':unittest.main()
