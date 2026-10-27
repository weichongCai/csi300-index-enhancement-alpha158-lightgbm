"""Build diagnostic sufficient statistics from trusted local feature caches and Qlib bins.

No output of this module changes either frozen model or the trading strategy.
"""
from pathlib import Path
import gc,json,pickle
import numpy as np
import pandas as pd
from scipy.signal import lfilter
from .research_math import paired_ic


def read_bin(data,code,field,calendar):
    path=Path(data)/'features'/code.lower()/f'{field}.day.bin'
    out=np.full(len(calendar),np.nan,dtype=np.float32)
    if not path.exists():return out
    with path.open('rb') as stream:
        first=int(np.fromfile(stream,dtype='<f4',count=1)[0])
        values=np.fromfile(stream,dtype='<f4',count=max(0,len(calendar)-first))
    if first<len(calendar):out[first:first+len(values)]=values
    return out


def returns(close):
    a=np.asarray(close,dtype=float);v=np.full(len(a),np.nan)
    ok=np.isfinite(a[1:])&np.isfinite(a[:-1])&(a[1:]>0)&(a[:-1]>0)
    v[1:][ok]=a[1:][ok]/a[:-1][ok]-1
    return v


def trailing_beta(stock,market):
    good=np.isfinite(stock)&np.isfinite(market);decay=2**(-1/60)
    def ew(v):
        z=lfilter([1.],[1.,-decay],v)
        if len(z)>252:z[252:]-=decay**252*z[:-252].copy()
        return z
    c=np.cumsum(good);n=c.copy();n[252:]-=c[:-252]
    x=np.where(good,market,0);y=np.where(good,stock,0);w=ew(good)
    with np.errstate(divide='ignore',invalid='ignore'):
        cov=ew(x*y)/w-ew(x)*ew(y)/w**2;var=ew(x*x)/w-(ew(x)/w)**2
    known=(n>=126)&(var>1e-16)&np.isfinite(cov)&np.isfinite(var)
    beta=np.ones(len(stock));beta[known]=cov[known]/var[known]
    return beta,known


def price_panel(data,index,cutoff):
    cal=pd.DatetimeIndex(pd.read_csv(Path(data)/'calendars/day.txt',header=None)[0])
    cal=cal[cal<=cutoff]  # Never load any 2026 observations for this diagnostic.
    dates=index.get_level_values('datetime');codes=index.get_level_values('instrument')
    positions=cal.get_indexer(dates)
    if (positions<0).any():raise ValueError('Feature dates outside allowed calendar.')
    market=read_bin(data,'SH000300','close',cal);mr=returns(market)
    result=np.full((len(index),7),np.nan)
    for code in sorted(set(codes)):
        selected=np.flatnonzero(codes==code);p=positions[selected]
        close=read_bin(data,code,'close',cal);r=returns(close)
        beta,known=trailing_beta(r,mr)
        amount=read_bin(data,code,'amount',cal).astype(float)*1000
        v=pd.Series(r).rolling(60,min_periods=40).std(ddof=1).to_numpy()*np.sqrt(252)
        momentum=pd.Series(close).shift(21)/pd.Series(close).shift(252)-1
        liquidity=np.log(pd.Series(np.where(amount>0,amount,np.nan)).rolling(20,min_periods=15).mean())
        result[selected,1]=beta[p];result[selected,2]=known[p]
        result[selected,3]=v[p];result[selected,4]=momentum.to_numpy()[p]
        result[selected,5]=liquidity.to_numpy()[p]
        mature=p+6<len(cal);ok=p[mature]
        # Preserve float32 ratio arithmetic used in the original target.
        stock5=close[ok+6]/close[ok+1]-np.float32(1.)
        market5=market[ok+6]/market[ok+1]-np.float32(1.)
        result[selected[mature],0]=stock5-beta[ok]*market5
        result[selected,6]=mature
    return pd.DataFrame(result,index=index,columns=['residual5','beta','beta_known','volatility_60',
        'momentum_252_21','log_mean_amount_20','label_mature'])


def build(root,data,early_features,late_features,destination):
    root,destination=Path(root),Path(destination)
    if destination.exists():raise FileExistsError(destination)
    destination.mkdir(parents=True)
    config=json.loads((root/'research_config.json').read_text(encoding='utf-8'))
    early=pd.read_parquet(early_features)
    with Path(late_features).open('rb') as stream:late=pickle.load(stream)
    features=pd.concat([early,late]).sort_index();del early,late;gc.collect()
    dates=features.index.get_level_values('datetime')
    features=features.loc[(dates>=config['signal_start'])&(dates<=config['signal_end'])]
    if len(features.columns)!=158 or features.index.has_duplicates:raise ValueError('Unexpected feature panel.')
    frozen=pd.read_parquet(root/'data/frozen_forecast.parquet').iloc[:,0].reindex(features.index)
    ridge=pd.read_parquet(root/'results/research/linear_reference/frozen_ridge_forecast.parquet').iloc[:,0].reindex(features.index)
    panel=price_panel(data,features.index,config['label_price_cutoff'])
    panel.drop(columns=['residual5','label_mature']).to_parquet(destination/'risk_proxy_panel.parquet')
    print('PRICE_LABEL_RISK_PANEL_READY',len(features),flush=True)
    ic_rows=[];rank_rows=[];coverage=[];groups=[];correlations=[];corr_dates=[]
    names=list(features.columns)+['MODEL_LIGHTGBM','MODEL_RIDGE']
    selected=config['illustrative_features']+['MODEL_LIGHTGBM','MODEL_RIDGE']
    mature_dates=pd.DatetimeIndex(sorted(set(features.index.get_level_values('datetime')[panel.label_mature.eq(1)])))
    monthly_last=set(pd.Series(mature_dates,index=mature_dates).resample('ME').last().dropna().values)
    for day,frame in features.groupby(level='datetime',sort=True):
        pp=panel.loc[frame.index]
        if not pp.label_mature.eq(1).all():continue
        frame=frame.copy();frame['MODEL_LIGHTGBM']=frozen.loc[frame.index].to_numpy()
        frame['MODEL_RIDGE']=ridge.loc[frame.index].to_numpy()
        target=pp.residual5.to_numpy()
        ic,n=paired_ic(frame.to_numpy(),target,minimum=30)
        rank,_=paired_ic(frame.to_numpy(),target,rank=True,minimum=30)
        ic_rows.append([day,*ic]);rank_rows.append([day,*rank]);coverage.append([day,len(frame),*n])
        for name in selected:
            x=frame[name].to_numpy();ok=np.isfinite(x)&np.isfinite(target)
            # Percentile ranks preserve ties; equal factors may leave empty groups.
            ranks=pd.Series(x[ok]).rank(method='average',pct=True).to_numpy()
            bucket=np.minimum(np.ceil(ranks*5),5).astype(int)
            for q in range(1,6):
                y=target[ok][bucket==q]
                groups.append([day,name,q,len(y),float(y.mean()) if len(y) else np.nan])
        if day.to_datetime64() in monthly_last:
            correlations.append(frame.iloc[:,:158].replace([np.inf,-np.inf],np.nan).corr(method='spearman').to_numpy())
            corr_dates.append(day)
        if len(ic_rows)%250==0:print('DAILY_FACTOR_DIAGNOSTICS',len(ic_rows),flush=True)
    for filename,rows,columns in [('daily_ic',ic_rows,names),('daily_rank_ic',rank_rows,names),
            ('daily_pair_counts',coverage,['universe_count']+names)]:
        pd.DataFrame(rows,columns=['date']+columns).set_index('date').to_csv(destination/f'{filename}.csv')
    pd.DataFrame(groups,columns=['date','factor','quintile','pairs','mean_forward_residual5']).to_csv(
        destination/'quintile_forward_means.csv',index=False)
    corrs=np.asarray(correlations)
    pd.DataFrame(np.nanmean(corrs,axis=0),index=names[:158],columns=names[:158]).to_csv(destination/'mean_signed_correlation.csv')
    pd.DataFrame(np.nanmean(abs(corrs),axis=0),index=names[:158],columns=names[:158]).to_csv(destination/'mean_absolute_correlation.csv')
    metadata={'feature_rows':len(features),'feature_count':158,'mature_signal_days':len(ic_rows),
        'first_mature_signal':str(ic_rows[0][0].date()),'last_mature_signal':str(ic_rows[-1][0].date()),
        'last_label_exit':'2025-12-31','correlation_sample_dates':[str(d.date()) for d in corr_dates],
        'target':'unstandardized beta-residual 5-day return; same IC/ranking as daily standardized target',
        'quintiles_are_overlapping_forward_means_not_tradable_returns':True,
        'future_prices_only_in_evaluation_labels':True,'strategy_or_model_changed':False}
    (destination/'inputs.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8',newline='\n')
    print('RESEARCH_SUFFICIENT_STATISTICS_READY',flush=True)
