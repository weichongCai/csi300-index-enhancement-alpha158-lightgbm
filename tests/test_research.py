import unittest
import numpy as np
import pandas as pd
from src.research_math import paired_ic,by_adjust,circular_bootstrap_mean
from src.linear_reference import fit_ridge,predict_ridge
from src.research_build import trailing_beta


class ResearchTests(unittest.TestCase):
    def test_pairwise_ranks_target_after_missing_deletion(self):
        x=np.array([[1,1],[np.nan,2],[3,2],[4,np.nan],[5,4]],float)
        y=np.array([5,1,3,2,4],float)
        actual,n=paired_ic(x,y,rank=True,minimum=3)
        expected=[pd.Series(x[:,j]).corr(pd.Series(y),method='spearman') for j in range(2)]
        np.testing.assert_allclose(actual,expected);np.testing.assert_equal(n,[4,4])

    def test_BY_monotone_and_declared_family(self):
        p=np.array([.01,.04,.03,np.nan]);m=4;h=sum(1/np.arange(1,m+1))
        expected=np.array([.01*m*h,.04*m*h/3,.04*m*h/3,1.])
        np.testing.assert_allclose(by_adjust(p),expected)

    def test_bootstrap_constant_and_reproducible(self):
        x=np.column_stack([np.arange(47),np.full(47,3.)]);x[0,1]=np.nan
        a=circular_bootstrap_mean(x,20,201,123)
        np.testing.assert_array_equal(a,circular_bootstrap_mean(x,20,201,123))
        np.testing.assert_allclose(a[:,1],3.)
        self.assertTrue(np.ptp(a[:,0])>0)

    def test_circular_blocks_match_direct_resampling_including_tail(self):
        x=np.arange(13,dtype=float);size=5;reps=11;seed=4
        rng=np.random.default_rng(seed);starts=rng.integers(len(x),size=(reps,3))
        expected=[]
        for row in starts:
            indices=np.concatenate([(s+np.arange(size))%len(x) for s in row])[:len(x)]
            expected.append(x[indices].mean())
        np.testing.assert_allclose(circular_bootstrap_mean(x,size,reps,seed)[:,0],expected)

    def test_ridge_training_only_and_closed_form(self):
        x=pd.DataFrame({'a':[1.,2.,3.,99999.],'b':[np.nan,1.,2.,-99999.]})
        y=np.array([0.,1.,2.]);positions=np.arange(3)
        model=fit_ridge(x,y,positions,penalty=1.,clip=5.)
        self.assertEqual(model['mean'],[2.,1.5])
        from src.linear_reference import transform
        z=transform(x.iloc[:3].to_numpy(),np.array(model['mean']),np.array(model['scale']),5.)
        a=np.column_stack([z,np.ones(3)]);penalty=np.diag([1.,1.,0.])
        expected=np.linalg.solve(a.T@a/3+penalty,a.T@y/3)
        np.testing.assert_allclose(predict_ridge(x,model,positions),a@expected)
        x.loc[3]=[1e20,-1e20]
        np.testing.assert_allclose(fit_ridge(x,y,positions)['coefficient'],model['coefficient'])

    def test_beta_is_causal_and_matches_linear_stock(self):
        market=np.sin(np.arange(500)/8)*.01
        stock=1.6*market+.0003
        beta,known=trailing_beta(stock,market)
        np.testing.assert_allclose(beta[known],1.6,rtol=1e-10)
        changed=stock.copy();changed[351:]=np.arange(149)
        np.testing.assert_array_equal(trailing_beta(changed,market)[0][:351],beta[:351])


if __name__=='__main__':unittest.main()
