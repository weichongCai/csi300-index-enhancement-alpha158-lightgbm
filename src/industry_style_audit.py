"""Independent truncated-history checks of risk descriptors and estimates."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import pandas as pd
from .industry_style_risk import RiskArrays,cross_standardize,broad_sector
from .research_build import read_bin

def audit(data,risk,industry_source,out):
    data,risk,industry_source,out=map(Path,[data,risk,industry_source,out])
    r=RiskArrays(risk)
    cal=pd.DatetimeIndex(pd.read_csv(data/'calendars/day.txt',header=None)[0]);cal=cal[cal<='2025-12-31']
    close=pd.DataFrame({c:read_bin(data,c,'close',cal).astype(float) for c in r.codes},index=cal)
    amount=pd.DataFrame({c:read_bin(data,c,'amount',cal).astype(float) for c in r.codes},index=cal)
    rr=close.pct_change(fill_method=None)
    raw=pd.read_parquet(industry_source);raw.snapshot_date=pd.to_datetime(raw.snapshot_date)
    raw['instrument']=raw.code.str.replace('.','',regex=False).str.upper();raw['sector']=raw.industry.map(broad_sector)
    metadata=json.loads((risk/'metadata.json').read_text(encoding='utf-8'));sectors=metadata['industry_sectors']
    intervals=pd.read_csv(data/'instruments/csi300.txt',sep='\t',header=None,names=['code','start','end'],parse_dates=['start','end'])
    f=pd.read_csv(risk/'factor_returns.csv',index_col=0,parse_dates=True);fit=pd.read_csv(risk/'fit_audit.csv',parse_dates=['date','exposure_date'])
    assert (fit.exposure_date<fit.date).all()
    industry=pd.read_parquet(risk/'industry_codes.parquet');available=pd.read_parquet(risk/'industry_available_dates.parquet')
    checks=0;industry_checks=0
    specific_checks=0
    sample=np.unique(np.linspace(500,len(r.dates)-1,25).astype(int))
    for t in sample:
        day=r.dates[t];prior=r.dates[t-1]
        members=set(intervals.loc[intervals.start.le(prior)&intervals.end.ge(prior),'code']);member=np.array([c in members for c in r.codes])
        # Rebuild every stock's raw descriptors using a strictly truncated price history.
        history=close.loc[:prior];ar=amount.loc[:prior]
        descriptors=[history.iloc[-22]/history.iloc[-127]-1,history.pct_change(fill_method=None).tail(60).std(ddof=1),
                     np.log(ar.where(ar>0).tail(20).mean())]
        # These dates have complete 60/20 windows for valid names; preserve exact min-period policy.
        descriptors[1]=descriptors[1].where(history.pct_change(fill_method=None).tail(60).count()>=40)
        descriptors[2]=descriptors[2].where(ar.where(ar>0).tail(20).count()>=15)
        for k,x in enumerate(descriptors,1):
            z,valid=cross_standardize(x.to_numpy(),member)
            np.testing.assert_allclose(z,r.x[t-1,:,k],atol=1e-9,rtol=1e-8)
        good=member&r.valid[t-1]&rr.loc[day].reindex(r.codes).notna().to_numpy()
        x=r.x[t-1,good,1:];y=rr.loc[day].reindex(r.codes).to_numpy()[good]-r.x[t-1,good,0]*f.loc[day].iloc[0]
        expected=np.linalg.lstsq(np.r_[x,1e-2*np.eye(x.shape[1])],np.r_[y,np.zeros(x.shape[1])],rcond=None)[0]
        np.testing.assert_allclose(expected,f.loc[day].iloc[1:].to_numpy(),atol=1e-8,rtol=1e-7)
        past=f.iloc[max(0,t-251):t+1];ok=np.isfinite(past).all(axis=1);values=past.to_numpy()[ok]
        ages=np.arange(len(past)-1,-1,-1)[ok];w=2.**(-ages/60);w/=w.sum();mean=w@values
        centered=values-mean;cov=(centered*w[:,None]).T@centered
        expected_cov=.9*cov+.1*np.diag(np.diag(cov))+1e-12*np.eye(len(mean))
        np.testing.assert_allclose(expected_cov,r.cov[t],atol=1e-12,rtol=1e-7)
        for j in np.linspace(0,len(r.codes)-1,5).astype(int):
            positions=np.arange(max(1,t-251),t+1);coeff=f.iloc[positions].to_numpy()
            realized=rr.reindex(r.dates).iloc[positions,j].to_numpy()
            v=realized-np.einsum('nk,nk->n',r.x[positions-1,j,:],coeff)
            valid=np.isfinite(v)&r.valid[positions-1,j]
            if valid.sum()<60 or not r.valid[t,j]:expected_specific=1/252.
            else:
                weight=2.**(-(t-positions[valid])/60);weight/=weight.sum();v=v[valid]
                expected_specific=max(float(weight@(v*v)-(weight@v)**2),1e-10)
            np.testing.assert_allclose(expected_specific,r.d[t,j],atol=1e-12,rtol=1e-7)
            specific_checks+=1
        checks+=1
    for c in r.codes[::17]:
        g=raw.loc[raw.instrument.eq(c)].sort_values('snapshot_date')
        for t in sample[::3]:
            day=r.dates[t];past=g.loc[g.snapshot_date.le(day)]
            if len(past):
                row=past.iloc[-1];assert industry.loc[day,c]==sectors.index(row.sector);assert available.loc[day,c]==row.snapshot_date
            else:assert industry.loc[day,c]==sectors.index('UNKNOWN')
            industry_checks+=1
    assert min(np.linalg.eigvalsh(c).min() for c in r.cov[::25])>=-1e-12
    result=dict(truncated_history_style_and_factor_regressions_recomputed=checks,trailing_covariances_independently_recomputed=checks,
        specific_variances_independently_recomputed=specific_checks,
        asof_industry_snapshots_independently_checked=industry_checks,all_regressions_use_previous_close_exposures=True,
        sampled_covariances_PSD=True,source_file_sha256=hashlib.sha256((industry_source).read_bytes()).hexdigest())
    out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for x in ['data-dir','risk-cache','industry-source','output']:p.add_argument('--'+x,type=Path,required=True)
    a=p.parse_args();audit(a.data_dir,a.risk_cache,a.industry_source,a.output)
