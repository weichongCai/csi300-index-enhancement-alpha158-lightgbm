"""Descriptive research statistics; these functions do not select trading parameters."""
import numpy as np
import pandas as pd


def paired_ic(values, target, rank=False, minimum=30):
    """Pairwise deletion precedes BOTH ranks, including target ranks; average ties."""
    frame=pd.DataFrame(values).replace([np.inf,-np.inf],np.nan)
    target=np.asarray(target,dtype=float)
    valid=frame.notna() & np.isfinite(target)[:,None]
    x=frame.where(valid)
    y=pd.DataFrame(np.broadcast_to(target[:,None],x.shape)).where(valid)
    if rank:
        x=x.rank(method='average');y=y.rank(method='average')
    n=valid.sum().to_numpy()
    a=x.to_numpy();b=y.to_numpy()
    with np.errstate(invalid='ignore',divide='ignore'):
        ac=a-np.nanmean(a,axis=0);bc=b-np.nanmean(b,axis=0)
        result=np.nansum(ac*bc,axis=0)/np.sqrt(np.nansum(ac*ac,axis=0)*np.nansum(bc*bc,axis=0))
    result[n<minimum]=np.nan
    return result,n


def circular_bootstrap_mean(values, length=20, replications=19999, seed=20261027):
    """Fixed circular blocks, exactly n observations per resample, missing-aware means."""
    x=np.asarray(values,dtype=float)
    if x.ndim==1:x=x[:,None]
    n,width=x.shape
    if not 1<=length<=n:raise ValueError('Invalid block length.')
    good=np.isfinite(x)
    def block_sum(v,size):
        if size==0:return np.zeros((n,width))
        z=np.concatenate([v,v[:size-1]],axis=0)
        cs=np.concatenate([np.zeros((1,width)),np.cumsum(z,axis=0)])
        return cs[size:]-cs[:-size]
    sums=block_sum(np.where(good,x,0),length);counts=block_sum(good,length)
    k,remainder=divmod(n,length)
    tail_sum=block_sum(np.where(good,x,0),remainder);tail_count=block_sum(good,remainder)
    rng=np.random.default_rng(seed);result=np.full((replications,width),np.nan)
    for start in range(0,replications,128):
        size=min(128,replications-start)
        indices=rng.integers(n,size=(size,k+(remainder>0)))
        total=sums[indices[:,:k]].sum(axis=1);num=counts[indices[:,:k]].sum(axis=1)
        if remainder:
            total+=tail_sum[indices[:,-1]];num+=tail_count[indices[:,-1]]
        np.divide(total,num,out=result[start:start+size],where=num>0)
    return result


def by_adjust(pvalues):
    """Benjamini-Yekutieli adjusted p-values, all declared hypotheses counted."""
    p=np.asarray(pvalues,dtype=float);p=np.where(np.isfinite(p),p,1.)
    m=len(p);order=np.argsort(p,kind='stable');harmonic=np.sum(1/np.arange(1,m+1))
    adjusted=np.minimum.accumulate((p[order]*m*harmonic/np.arange(1,m+1))[::-1])[::-1]
    result=np.empty(m);result[order]=np.minimum(adjusted,1.)
    return result


def circular_bootstrap_ir(values,length=20,replications=19999,seed=20261027):
    x=np.asarray(values,dtype=float)
    if not np.isfinite(x).all():raise ValueError('IR input must be finite.')
    moments=circular_bootstrap_mean(np.column_stack([x,x*x]),length,replications,seed)
    variance=(moments[:,1]-moments[:,0]**2)*len(x)/(len(x)-1)
    result=moments[:,0]/np.sqrt(np.maximum(variance,1e-30))*np.sqrt(252)
    return np.quantile(result,[.025,.5,.975])
