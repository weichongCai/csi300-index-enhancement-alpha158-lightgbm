"""Public-data industry/price-style risk model, explicitly NOT MSCI Barra.

Conditional benchmark anchor avoids inventing historical benchmark holdings.
Fit each realized return with previous-close exposures; risk at t uses data <=t.
"""
from pathlib import Path
import hashlib,json,argparse,urllib.request
import numpy as np
import pandas as pd
from .marginal_risk_gate import conditional_risk,ew_sum
from .research_build import read_bin

PUBLIC_INDUSTRY=dict(url='https://github.com/noblechang/a-share-data/releases/download/FREE_DATA_v0/a-share-raw-v0.tar',
    start=924553216,end=927545198,sha256='01d79998888900ef737b774cab942031b0f892c67942f31bd84694faaa269575')


def download_industry(destination):
    """Retrieve exactly the previously checked public Parquet member, not a full TAR."""
    destination=Path(destination)
    if destination.exists():raise FileExistsError(destination)
    request=urllib.request.Request(PUBLIC_INDUSTRY['url'],headers={'Range':f"bytes={PUBLIC_INDUSTRY['start']}-{PUBLIC_INDUSTRY['end']}"})
    with urllib.request.urlopen(request,timeout=120) as response:
        if response.status!=206:raise ValueError('Server did not honor exact Range; refusing full-archive download')
        raw=response.read(PUBLIC_INDUSTRY['end']-PUBLIC_INDUSTRY['start']+2)
    if len(raw)!=PUBLIC_INDUSTRY['end']-PUBLIC_INDUSTRY['start']+1:raise ValueError('Unexpected member length')
    if hashlib.sha256(raw).hexdigest()!=PUBLIC_INDUSTRY['sha256']:raise ValueError('Source hash changed; manual source review required')
    destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(raw)

BROAD_ALIASES={
 '农、林、牧、渔业':'A','农林牧渔业':'A','采矿业':'B','采掘业':'B','制造业':'C',
 '电力、热力、燃气及水生产和供应业':'D','电力煤气及水的生产和供应业':'D','建筑业':'E',
 '批发和零售业':'F','批发和零售贸易':'F','交通运输、仓储和邮政业':'G','交通运输仓储业':'G',
 '住宿和餐饮业':'H','信息传输、软件和信息技术服务业':'I','信息技术业':'I',
 '金融业':'J','金融保险业':'J','房地产业':'K','租赁和商务服务业':'L','科学研究和技术服务业':'M',
 '水利、环境和公共设施管理业':'N','社会服务业':'LEGACY_SOCIAL','教育':'P',
 '卫生和社会工作':'Q','文化、体育和娱乐业':'R','传播与文化产业':'R','综合':'S','综合类':'S'}


def broad_sector(label):
    label=str(label) if pd.notna(label) else ''
    if len(label)>1 and label[0] in 'ABCDEFGHIJKLMNOPQRS' and label[1].isdigit():return label[0]
    return BROAD_ALIASES.get(label.split('-')[0],'UNKNOWN')


def cross_standardize(x, members):
    """Five-MAD clipping, equal-member mean/std, finite observed values only."""
    v=np.asarray(x,float);ok=np.asarray(members,bool)&np.isfinite(v)
    if ok.sum()<20:return np.zeros(len(v)),np.zeros(len(v),bool)
    med=np.median(v[ok]);mad=np.median(abs(v[ok]-med))
    clipped=np.clip(v,med-5*1.4826*mad,med+5*1.4826*mad) if mad>0 else v.copy()
    mean=clipped[ok].mean();std=clipped[ok].std(ddof=1)
    if std<1e-12:return np.zeros(len(v)),np.isfinite(v)
    return np.where(np.isfinite(v),(clipped-mean)/std,0.),np.isfinite(v)


def factor_variance_change(weights,exposures,cov,specific,old,new):
    """Exact whole-model active-variance change of an equal-weight replacement."""
    a=sum((w*exposures[c] for c,w in weights.items()),np.zeros(cov.shape[0]))
    a[0]-=1. # benchmark exposure is exactly [1, 0, ...]
    w=weights.get(old,0.);d=w*(exposures[new]-exposures[old])
    return float(2*a@cov@d+d@cov@d+w*w*(specific[new]-specific[old]))


def asof_indices(snapshots,dates):
    snapshots=np.asarray(snapshots,dtype='datetime64[ns]')
    if len(snapshots)>1 and (snapshots[1:]<=snapshots[:-1]).any():raise ValueError('Snapshot dates must be unique and increasing')
    return snapshots.searchsorted(np.asarray(dates,dtype='datetime64[ns]'),side='right')-1


def factor_covariance(factors):
    """Causal rolling EW covariance, PSD-preserving diagonal shrinkage."""
    factors=np.asarray(factors,float);k=factors.shape[1]
    good=np.isfinite(factors).all(axis=1);z=np.where(good[:,None],factors,0.)
    mass=ew_sum(good.astype(float));means=np.column_stack([ew_sum(z[:,j]) for j in range(k)])/np.maximum(mass[:,None],1e-30)
    cov=np.zeros((len(factors),k,k))
    for i in range(k):
        for j in range(i,k):
            v=ew_sum(z[:,i]*z[:,j])/np.maximum(mass,1e-30)-means[:,i]*means[:,j]
            cov[:,i,j]=v;cov[:,j,i]=v
    for t in range(len(factors)):
        cov[t]=.9*cov[t]+.1*np.diag(np.diag(cov[t]))+1e-12*np.eye(k)
    return cov


def build(root,data,industry_source,out):
    root,data,out=map(Path,[root,data,out]);out.mkdir(parents=True,exist_ok=False)
    calendar=pd.DatetimeIndex(pd.read_csv(data/'calendars/day.txt',header=None)[0]);calendar=calendar[calendar<='2025-12-31']
    dates=calendar[calendar>='2014-01-01'];loc=calendar.get_indexer(dates)
    intervals=pd.read_csv(data/'instruments/csi300.txt',sep='\t',header=None,names=['code','start','end'],parse_dates=['start','end'])
    intervals=intervals.loc[intervals.end.ge(dates[0])&intervals.start.le(dates[-1])]
    codes=sorted(intervals.code.unique());n=len(codes);slots={c:i for i,c in enumerate(codes)}
    membership=np.zeros((len(dates),n),bool)
    for row in intervals.itertuples():membership[:,slots[row.code]]|=(dates>=row.start)&(dates<=row.end)
    close=pd.DataFrame({c:read_bin(data,c,'close',calendar).astype(float) for c in codes},index=calendar)
    amount=pd.DataFrame({c:read_bin(data,c,'amount',calendar).astype(float) for c in codes},index=calendar)
    returns=close.pct_change(fill_method=None).reindex(dates).to_numpy()
    momentum=(close.shift(21)/close.shift(126)-1).reindex(dates).to_numpy()
    volatility=close.pct_change(fill_method=None).rolling(60,min_periods=40).std().reindex(dates).to_numpy()
    liquidity=np.log(amount.where(amount>0).rolling(20,min_periods=15).mean()).reindex(dates).to_numpy()
    raw=pd.read_parquet(industry_source);raw.snapshot_date=pd.to_datetime(raw.snapshot_date);raw.updateDate=pd.to_datetime(raw.updateDate)
    raw=raw.loc[raw.snapshot_date.le(dates[-1])].copy()
    assert raw.updateDate.le(raw.snapshot_date).all()
    assert not raw.duplicated(['snapshot_date','code']).any()
    raw['instrument']=raw.code.str.replace('.','',regex=False).str.upper()
    # Broad historical CSRC sectors, not the changing 400+ compound leaf labels.
    raw['sector']=raw.industry.map(broad_sector)
    sectors=sorted(set(raw.sector)|{'UNKNOWN'});sector_slots={s:i for i,s in enumerate(sectors)}
    industry=np.full((len(dates),n),sector_slots['UNKNOWN'],np.int16)
    availability=np.full((len(dates),n),np.datetime64('NaT'),dtype='datetime64[ns]')
    for c,g in raw.groupby('instrument'):
        if c not in slots:continue
        g=g.sort_values('snapshot_date');p=asof_indices(g.snapshot_date,dates);valid=p>=0
        industry[valid,slots[c]]=g.sector.map(sector_slots).to_numpy()[p[valid]]
        availability[valid,slots[c]]=g.snapshot_date.to_numpy()[p[valid]]
    known=~np.isnat(availability)
    assert (availability[known]<=np.broadcast_to(dates.to_numpy()[:,None],availability.shape)[known]).all()
    exposures=np.zeros((len(dates),n,4+len(sectors)))
    raw_valid=np.ones((len(dates),n),bool)
    for t in range(len(dates)):
        for k,x in enumerate([momentum,volatility,liquidity],1):
            exposures[t,:,k],good=cross_standardize(x[t],membership[t]);raw_valid[t]&=good
        exposures[t,np.arange(n),4+industry[t]]=1.
    official=pd.read_csv(root/'results/risk_management/risk_benchmark_returns.csv',index_col=0,parse_dates=True).benchmark_return
    market=official.reindex(calendar).to_numpy();rfull=close.pct_change(fill_method=None)
    for c,j in slots.items():exposures[:,j,0]=conditional_risk(rfull[c].to_numpy(),market)[0][loc]
    m=official.reindex(dates).to_numpy();k=exposures.shape[2]
    factors=np.full((len(dates),k),np.nan);residual=np.full((len(dates),n),np.nan);fit_rows=[]
    for t in range(1,len(dates)):
        good=membership[t-1]&raw_valid[t-1]&np.isfinite(returns[t])
        if good.sum()<100 or not np.isfinite(m[t]):continue
        x=exposures[t-1,good,1:];y=returns[t,good]-exposures[t-1,good,0]*m[t]
        # Tiny ridge stabilizes empty/rare industry columns; no future training.
        f=np.linalg.solve(x.T@x+1e-4*np.eye(k-1),x.T@y)
        factors[t]=np.r_[m[t],f]
        usable=raw_valid[t-1]&np.isfinite(returns[t])
        residual[t,usable]=returns[t,usable]-exposures[t-1,usable]@factors[t]
        fit_rows.append(dict(date=dates[t],exposure_date=dates[t-1],members=int(good.sum()),unknown_industries=int((industry[t-1,good]==sector_slots['UNKNOWN']).sum())))
    good=np.isfinite(factors).all(axis=1)
    cov=factor_covariance(factors)
    specific=np.full((len(dates),n),1/252.)
    for j in range(n):
        good_r=np.isfinite(residual[:,j]);a=np.where(good_r,residual[:,j],0.);mass_r=ew_sum(good_r.astype(float))
        count=pd.Series(good_r).rolling(252,min_periods=1).sum().to_numpy()
        var=np.maximum(ew_sum(a*a)/np.maximum(mass_r,1e-30)-(ew_sum(a)/np.maximum(mass_r,1e-30))**2,1e-10)
        specific[:,j]=np.where((count>=60)&raw_valid[:,j],var,1/252.)
    ready=pd.Series(good).rolling(252,min_periods=1).sum().to_numpy()>=126
    np.savez_compressed(out/'risk_arrays.npz',dates=dates.to_numpy(),codes=np.asarray(codes),exposures=exposures,covariance=cov,specific=specific,ready=ready,raw_valid=raw_valid)
    pd.DataFrame(factors,index=dates,columns=['Market','Momentum','Volatility','TradedValue',*sectors]).to_csv(out/'factor_returns.csv')
    pd.DataFrame(fit_rows).to_csv(out/'fit_audit.csv',index=False)
    pd.DataFrame(industry,index=dates,columns=codes).to_parquet(out/'industry_codes.parquet')
    pd.DataFrame(availability,index=dates,columns=codes).to_parquet(out/'industry_available_dates.parquet')
    metadata=dict(model='Conditional benchmark plus historical CSRC broad industries and three price/volume styles; NOT Barra/CNE5',
        styles=['126-minus-21-day momentum','60-day volatility, at least40','log20-day traded value, at least15'],
        standardization='current historical CSI300 members; equal weighted, 5MAD clip; no cap proxy',
        industry_source_sha256=hashlib.sha256(Path(industry_source).read_bytes()).hexdigest(),industry_sectors=sectors,
        fit='equal-member cross-sectional residual regression with prior-close exposures; ridge1e-4',
        covariance='past252 EW half60, 10% diagonal shrinkage',specific='past252 EW half60, minimum60; unknown1 annual volatility',
        benchmark_exposure='[1,0,...] conditional observed benchmark anchor; no invented benchmark weights',
        covariance_ready_minimum_days=126,ready_start=str(dates[np.flatnonzero(ready)[0]].date()),
        limitations=['Not official Barra','No size/value/growth/leverage descriptors','No actual benchmark industry/style holdings',
        'Historical classifications are archived monthly snapshots, not certified original vintages','Constant styles over the next day assumed','Residual covariance assumed diagonal'],
        development_period_reused=True)
    (out/'metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('INDUSTRY_STYLE_RISK_READY',k,len(codes),metadata['ready_start'],flush=True)


class RiskArrays:
    def __init__(self,path):
        a=np.load(Path(path)/'risk_arrays.npz');self.dates=pd.DatetimeIndex(a['dates']);self.codes=list(a['codes'])
        self.slots={c:i for i,c in enumerate(self.codes)};self.x=a['exposures'];self.cov=a['covariance'];self.d=a['specific'];self.ready=a['ready'];self.valid=a['raw_valid']
    def at(self,day):
        t=self.dates.get_loc(day)
        return ({c:self.x[t,i] for c,i in self.slots.items()},self.cov[t],dict(zip(self.codes,self.d[t])),bool(self.ready[t]))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--industry-source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--download-public-industry',action='store_true')
    a=p.parse_args()
    if a.download_public_industry:download_industry(a.industry_source)
    build(Path(__file__).resolve().parents[1],a.data_dir,a.industry_source,a.output)
