"""Recompute diagnostics and figures from saved, auditable sufficient statistics."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .metrics import evaluate,ir
from .research_math import circular_bootstrap_mean,circular_bootstrap_ir,by_adjust


def save(root,destination):
    root,destination=Path(root),Path(destination)
    if destination.exists():raise FileExistsError(destination)
    destination.mkdir(parents=True)
    source=root/'results/research';factor=source/'factors'
    cfg=json.loads((root/'research_config.json').read_text(encoding='utf-8'))
    rank=pd.read_csv(factor/'daily_rank_ic.csv',index_col=0,parse_dates=True)
    pearson=pd.read_csv(factor/'daily_ic.csv',index_col=0,parse_dates=True)
    counts=pd.read_csv(factor/'daily_pair_counts.csv',index_col=0,parse_dates=True)
    params=cfg['bootstrap'];boot=circular_bootstrap_mean(rank.to_numpy(),params['block_length'],params['replications'],params['seed'])
    observed=rank.mean().to_numpy()
    # Center bootstrap at its expected circular-block mean, then form a two-sided null.
    p=(1+(abs(boot-observed)>=abs(observed)).sum(axis=0))/(len(boot)+1)
    p[~np.isfinite(observed)]=1.
    statistics=pd.DataFrame({'factor':rank.columns,'mean_PearsonIC':pearson.mean().to_numpy(),
        'mean_RankIC':observed,'daily_RankIC_std':rank.std(ddof=1).to_numpy(),
        'ICIR_unannualized':(rank.mean()/rank.std(ddof=1)).to_numpy(),
        'finite_days':rank.count().to_numpy(),'positive_day_fraction':(rank.gt(0).sum()/rank.count()).to_numpy(),
        'mean_pair_coverage':counts[rank.columns].div(counts.universe_count,axis=0).mean().to_numpy(),
        'block_CI_low':np.nanquantile(boot,.025,axis=0),'block_CI_high':np.nanquantile(boot,.975,axis=0),
        'block_null_p_two_sided':p})
    statistics['BY_q_158']=np.nan
    statistics.loc[:157,'BY_q_158']=by_adjust(p[:158])
    statistics['BY_reject_5pct_158']=statistics.BY_q_158.le(.05)
    statistics.to_csv(destination/'factor_statistics.csv',index=False)
    annual=rank.groupby(rank.index.year).mean();annual.to_csv(destination/'annual_mean_rank_ic.csv',index_label='year')
    corr=pd.read_csv(factor/'mean_absolute_correlation.csv',index_col=0)
    signed=pd.read_csv(factor/'mean_signed_correlation.csv',index_col=0)
    pairs=[(corr.index[i],corr.columns[j],corr.iloc[i,j],signed.iloc[i,j])
        for i in range(len(corr)) for j in range(i+1,len(corr))]
    pd.DataFrame(pairs,columns=['factor_a','factor_b','mean_abs_monthly_Spearman','mean_signed_monthly_Spearman']).sort_values(
        'mean_abs_monthly_Spearman',ascending=False).to_csv(destination/'correlation_pairs.csv',index=False)
    accounts={
        'LightGBM_no_slippage':root/'results/daily_report.csv',
        'LightGBM_3bp':root/'results/slippage_3bp/slippage_daily_report.csv',
        'Ridge_3bp':source/'linear_reference/daily_report.csv'}
    rows=[];curves={};annual_models=[];bootstrap_ir=[];active_series={}
    for name,path in accounts.items():
        report=pd.read_csv(path,index_col=0,parse_dates=True,float_precision='round_trip')
        metric,years,daily=evaluate(root,report)
        rows.append(dict(model=name,**metric));annual_models.append(years.assign(model=name).reset_index())
        curves[name]=daily.model_nav;curves[name+'_relative']=daily.relative_nav
        active_series[name]=daily.active_return.to_numpy()
        curves['benchmark']=daily.benchmark_nav
        if name!='LightGBM_no_slippage':
            ci=circular_bootstrap_ir(daily.active_return.to_numpy(),params['block_length'],params['replications'],params['seed'])
            bootstrap_ir.append({'model':name,'pooled_active_IR':metric['pooled_net_active_ir'],
                'circular_block_percentile_CI_low':ci[0],'circular_block_percentile_CI_high':ci[2]})
    metrics=pd.DataFrame(rows);metrics.to_csv(destination/'model_comparison.csv',index=False)
    pd.concat(annual_models).to_csv(destination/'model_annual.csv',index=False)
    pd.DataFrame(bootstrap_ir).to_csv(destination/'pooled_IR_block_CI.csv',index=False)
    x=active_series['LightGBM_3bp'];y=active_series['Ridge_3bp']
    moments=circular_bootstrap_mean(np.column_stack([x,x*x,y,y*y]),params['block_length'],params['replications'],params['seed'])
    def sample_ir(mean,square):
        return mean/np.sqrt(np.maximum((square-mean**2)*len(x)/(len(x)-1),1e-30))*np.sqrt(252)
    difference=sample_ir(moments[:,0],moments[:,1])-sample_ir(moments[:,2],moments[:,3])
    rank_difference=boot[:,-2]-boot[:,-1]
    pd.DataFrame([{'contrast':'LightGBM_3bp minus Ridge_3bp, pooled active IR',
        'point_estimate':ir(pd.Series(x))-ir(pd.Series(y)),
        'paired_block_CI_low':np.quantile(difference,.025),'paired_block_CI_high':np.quantile(difference,.975)},
        {'contrast':'LightGBM minus Ridge, mean daily RankIC',
         'point_estimate':observed[-2]-observed[-1],
         'paired_block_CI_low':np.quantile(rank_difference,.025),'paired_block_CI_high':np.quantile(rank_difference,.975)}]).to_csv(
             destination/'paired_model_differences.csv',index=False)
    pd.DataFrame(curves).to_csv(destination/'model_curves.csv',index_label='date')
    diagnostics=execution_and_risk(root,destination)
    summary={'research_period_is_reused_development':True,'models_promoted':False,
        'factor_hypotheses':158,'BY_5pct_rejections':int(statistics.iloc[:158].BY_reject_5pct_158.sum()),
        'minimum_empirical_p':1/(params['replications']+1),'bootstrap':params,
        'IR_interval_is_pooled_not_mean_annual':True,
        'FDR_does_not_correct_prior_adaptive_strategy_search':True,
        'confidence_intervals_do_not_prove_alpha':True,'full_Barra_or_industry_neutrality_claimed':False,
        **diagnostics}
    (destination/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    plots(root,destination)
    return metrics


def execution_and_risk(root,destination):
    cfg=json.loads((root/'config.json').read_text(encoding='utf-8'))
    path=root/'results/slippage_3bp'
    report=pd.read_csv(path/'slippage_daily_report.csv',index_col=0,parse_dates=True,float_precision='round_trip')
    _,_,daily=evaluate(root,report)
    orders=pd.read_parquet(path/'slippage_orders.parquet');orders['date']=pd.to_datetime(orders.date)
    fills=orders.loc[orders.value.gt(1e-5)].copy()
    previous=report.account.shift(1).fillna(cfg['initial_cash'])
    execution=pd.DataFrame(index=report.index)
    for field,out in [('value','two_sided_turnover'),('cost','commission_to_prior_NAV'),('slippage_loss','price_slippage_attribution_to_prior_NAV')]:
        execution[out]=fills.groupby('date')[field].sum().reindex(report.index,fill_value=0)/previous
    holdings=pd.read_parquet(root/'results/research/execution/holdings.parquet')
    holdings['date']=pd.to_datetime(holdings.date)
    for date,h in holdings.groupby('date'):
        w=h.weight.to_numpy();hh=np.sum(w*w)
        execution.loc[date,'held_names']=len(h)
        execution.loc[date,'stock_weight']=w.sum()
        execution.loc[date,'largest_name_weight']=w.max()
        execution.loc[date,'effective_stock_names']=w.sum()**2/hh
    execution.to_csv(destination/'daily_execution.csv',index_label='date')
    yearly=execution.groupby(execution.index.year).agg({'two_sided_turnover':'sum',
        'commission_to_prior_NAV':'sum','price_slippage_attribution_to_prior_NAV':'sum',
        'held_names':'mean','stock_weight':'mean','largest_name_weight':'max','effective_stock_names':'mean'})
    yearly.to_csv(destination/'annual_execution.csv',index_label='year')
    traded=fills.groupby(['date','instrument']).agg(value=('value','sum'),quantity=('filled_amount','sum'),
        amount_cap=('daily_amount_cap','first'),volume_cap=('daily_volume_cap','first'))
    stress=[]
    for scale in [.5,1,2,5]:
        breach=(traded.value*scale>traded.amount_cap+1e-5)|(traded.quantity*scale>traded.volume_cap+1e-6)
        stress.append({'trade_path_scale':scale,'scaled_INITIAL_cash_CNY':cfg['initial_cash']*scale,
            'stock_day_pairs':len(traded),'breached_stock_days':int(breach.sum()),
            'breached_notional_fraction':float(traded.value.loc[breach].sum()/traded.value.sum()),
            'p95_day_amount_participation':float((traded.value*scale/(traded.amount_cap/.05)).quantile(.95))})
    pd.DataFrame(stress).to_csv(destination/'static_capacity_stress.csv',index=False)
    proxies=pd.read_parquet(root/'results/research/factors/risk_proxy_panel.parquet')
    for field in ['volatility_60','momentum_252_21','log_mean_amount_20']:
        x=proxies[field].replace([np.inf,-np.inf],np.nan)
        grouped=x.groupby(level='datetime')
        proxies[field]=(x-grouped.transform('mean'))/grouped.transform('std')
    # Map previous close holdings to the next trading day; no post-trade weights.
    next_day=dict(zip(report.index[:-1],report.index[1:]))
    joined=holdings.rename(columns={'date':'datetime'}).merge(proxies.reset_index(),
        on=['datetime','instrument'],how='left',validate='one_to_one')
    joined['trade_date']=joined.datetime.map(next_day)
    joined=joined.loc[joined.trade_date.notna()].copy()
    grouped=joined.groupby('trade_date');stock_weight=grouped.weight.sum()
    exposure=pd.DataFrame({'signal_date':grouped.datetime.first(),'stock_weight':stock_weight})
    for field in ['beta','volatility_60','momentum_252_21','log_mean_amount_20']:
        joined['_covered']=joined.weight.where(joined[field].notna(),0)
        joined['_weighted']=joined[field]*joined.weight
        coverage=joined.groupby('trade_date')._covered.sum()/stock_weight
        exposure[field+'_covered_weight_fraction']=coverage
        exposure[field]=joined.groupby('trade_date')._weighted.sum().where(coverage>=.98)
    joined['_unknown_beta_weight']=joined.weight.where(joined.beta_known.eq(0),0)
    exposure['unknown_beta_weight_fraction']=joined.groupby('trade_date')._unknown_beta_weight.sum()/stock_weight
    exposure.to_csv(destination/'prior_day_risk_exposures.csv')
    # A descriptive one-market regression is not a full risk-model attribution.
    x=daily.benchmark_return.iloc[1:].to_numpy();y=daily.active_return.iloc[1:].to_numpy()
    coefficient=np.linalg.lstsq(np.column_stack([np.ones(len(x)),x]),y,rcond=None)[0]
    residual=y-coefficient[0]-coefficient[1]*x
    pd.DataFrame([{'daily_intercept':coefficient[0],'intercept_times_252':coefficient[0]*252,
        'active_beta':coefficient[1],'portfolio_beta_approx':1+coefficient[1],
        'residual_volatility_annualized':np.std(residual,ddof=1)*np.sqrt(252),
        'not_pure_alpha_or_Barra_attribution':True}]).to_csv(destination/'descriptive_market_regression.csv',index=False)
    levels=pd.read_csv(root/'data/official_index_levels.csv',index_col=0,parse_dates=True).H00300
    prior_trend=(levels/levels.rolling(200,min_periods=200).mean()-1).shift(1).reindex(report.index)
    prior_vol=(levels.pct_change(fill_method=None).rolling(60,min_periods=60).std()*np.sqrt(252)).shift(1).reindex(report.index)
    regimes=[]
    for trend in ['above_200d','below_200d']:
        for vol in ['vol_at_most_25pct','vol_over_25pct']:
            mask=(prior_trend.ge(0) if trend=='above_200d' else prior_trend.lt(0))&(
                prior_vol.le(.25) if vol=='vol_at_most_25pct' else prior_vol.gt(.25))
            active=daily.active_return.loc[mask]
            regimes.append({'prior_trend':trend,'prior_volatility':vol,'days':len(active),
                'conditional_active_mean_times_252':float(active.mean()*252),
                'conditional_active_IR':ir(active),'discontinuous_subset_not_NAV_or_OOS':True})
    pd.DataFrame(regimes).to_csv(destination/'conditional_regimes.csv',index=False)
    return {'mean_annual_two_sided_turnover':float(yearly.two_sided_turnover.mean()),
        'mean_annual_commission_fraction_sum':float(yearly.commission_to_prior_NAV.mean()),
        'mean_annual_slippage_attribution_fraction_sum':float(yearly.price_slippage_attribution_to_prior_NAV.mean()),
        'capacity_stress_is_static_path_not_actual_AUM_backtest':True,
        'risk_proxies_reference_equal_weight_universe_not_index_weights':True}


def plots(root,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    curves=pd.read_csv(out/'model_curves.csv',index_col=0,parse_dates=True)
    fig,axes=plt.subplots(2,1,figsize=(12,8),sharex=True)
    for name,color in [('LightGBM_no_slippage','#999999'),('LightGBM_3bp','#186b96'),('Ridge_3bp','#c77830')]:
        axes[0].plot(curves.index,curves[name],label=name,color=color)
        axes[1].plot(curves.index,curves[name+'_relative'],label=name,color=color)
    axes[0].plot(curves.index,curves.benchmark,color='#555',label='CSI300 total return')
    axes[0].set_yscale('log');axes[0].set_ylabel('NAV (log scale)');axes[0].legend()
    axes[0].set_title('Fixed models | same trading rules | 2016–2025 REUSED development')
    axes[1].set_ylabel('NAV / benchmark');axes[1].axhline(1,color='#aaa',lw=.5)
    for ax in axes:ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'model_comparison.png',dpi=160);plt.close(fig)
    rank=pd.read_csv(root/'results/research/factors/daily_rank_ic.csv',index_col=0,parse_dates=True)
    selected=['KMID','KUP2','ROC20','RSV20','MODEL_LIGHTGBM','MODEL_RIDGE']
    fig,axes=plt.subplots(2,1,figsize=(12,8),sharex=True)
    rank[selected].cumsum().plot(ax=axes[0]);axes[0].set_ylabel('Cumulative RankIC (NOT PnL)')
    rank[['MODEL_LIGHTGBM','MODEL_RIDGE']].rolling(120,min_periods=90).mean().plot(ax=axes[1])
    axes[1].axhline(0,color='#aaa',lw=.5);axes[1].set_ylabel('120-day mean RankIC')
    axes[0].set_title('Daily cross-sectional diagnostics | overlapping 5-day evaluation labels')
    for ax in axes:ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'factor_ic.png',dpi=160);plt.close(fig)
    groups=pd.read_csv(root/'results/research/factors/quintile_forward_means.csv')
    g=groups.groupby(['factor','quintile']).mean_forward_residual5.mean().unstack('quintile')
    fig,ax=plt.subplots(figsize=(12,5));(g*100).T.plot(ax=ax,marker='o')
    ax.set_ylabel('Mean forward beta-residual 5-day return (%)');ax.set_xticks(range(1,6))
    ax.set_xlabel('Feature percentile quintile, LOW to HIGH (ties preserved)')
    ax.set_title('Descriptive forward means; NOT tradable portfolio returns');ax.grid(alpha=.2)
    ax.legend(loc='center left',bbox_to_anchor=(1.01,.5),fontsize=9)
    fig.tight_layout();fig.savefig(out/'factor_quintiles.png',dpi=160);plt.close(fig)
    execution=pd.read_csv(out/'daily_execution.csv',index_col=0,parse_dates=True)
    risk=pd.read_csv(out/'prior_day_risk_exposures.csv',index_col=0,parse_dates=True)
    fig,axes=plt.subplots(4,1,figsize=(12,11),sharex=True)
    execution.two_sided_turnover.rolling(60).mean().plot(ax=axes[0],color='#186b96')
    axes[0].set_ylabel('60d avg two-sided turnover')
    execution[['held_names','effective_stock_names']].plot(ax=axes[1]);axes[1].set_ylabel('Stock count')
    risk[['beta','volatility_60','momentum_252_21','log_mean_amount_20']].rolling(60,min_periods=40).mean().plot(ax=axes[2])
    axes[2].set_ylabel('60d avg: beta / style z');axes[2].legend(ncol=2,fontsize=8)
    risk[[c for c in risk if c.endswith('_covered_weight_fraction')]].rolling(60,min_periods=40).mean().plot(ax=axes[3],legend=False)
    axes[3].axhline(.98,color='#aaa',lw=.7,ls='--');axes[3].set_ylabel('60d avg covered weight')
    axes[0].set_title('LightGBM 3bp | holdings, turnover and prior-day risk proxies')
    for ax in axes:ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'execution_risk.png',dpi=160);plt.close(fig)
    corr=pd.read_csv(root/'results/research/factors/mean_absolute_correlation.csv',index_col=0)
    fig,ax=plt.subplots(figsize=(11,10));im=ax.imshow(corr.to_numpy(),vmin=0,vmax=1,cmap='viridis',interpolation='nearest')
    ticks=np.arange(0,len(corr),8)
    ax.set_xticks(ticks,corr.columns[ticks],rotation=90,fontsize=8)
    ax.set_yticks(ticks,corr.index[ticks],fontsize=8)
    ax.set_title('Alpha158 | mean absolute monthly cross-sectional Spearman correlation')
    fig.colorbar(im,ax=ax,label='Mean absolute correlation',shrink=.8)
    fig.tight_layout();fig.savefig(out/'factor_correlation.png',dpi=160);plt.close(fig)
