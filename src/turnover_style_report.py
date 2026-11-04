"""Compare every predeclared risk/turnover case, including failures."""
from pathlib import Path
import argparse,json,shutil
import numpy as np
import pandas as pd
from .metrics import ir,drawdown
from .active_risk_report import independent_audit
from .marginal_gate_report import audit_score_and_cost
from .industry_style_risk import RiskArrays
from .research_build import read_bin
from .risk_attribution import episodes

LABELS={'current_replay':'Current best','turnover_6pct':'Turnover budget 6%',
        'industry_style':'Industry + styles','industry_style_turnover_6pct':'Industry + styles + budget'}


def audit_risk_decisions(root,data,path,risk,cache):
    decisions=pd.read_parquet(path/'gate_decisions.parquet');decisions['signal_date']=pd.to_datetime(decisions.signal_date)
    cal=pd.DatetimeIndex(pd.read_csv(data/'calendars/day.txt',header=None)[0]);cal=cal[cal<='2025-12-31']
    orders=pd.read_parquet(path/'orders.parquet');orders['date']=pd.to_datetime(orders.date)
    orders['flow']=orders.filled_amount*np.where(orders.direction.eq(1),1.,-1.)
    a=pd.read_csv(path/'daily_report.csv',index_col=0,parse_dates=True)
    q=orders.pivot_table(index='date',columns='instrument',values='flow',aggfunc='sum').reindex(a.index).fillna(0.).cumsum()
    h=pd.read_parquet(path/'holdings.parquet');h['date']=pd.to_datetime(h.date);held={d:set(g.instrument) for d,g in h.groupby('date')}
    prices=pd.DataFrame({c:read_bin(data,c,'close',cal).astype(float) for c in q.columns},index=cal)
    beta=pd.read_parquet(cache/'risk_beta.parquet');specific=pd.read_parquet(cache/'risk_specific.parquet')
    mv=pd.read_csv(cache/'market_variance.csv',index_col=0,parse_dates=True).iloc[:,0]
    dates=sorted(set(decisions.signal_date)&set(a.index));chosen={dates[i] for i in np.unique(np.linspace(0,len(dates)-1,30).astype(int))}
    checks=0
    for day,g in decisions.groupby('signal_date'):
        if day not in chosen:continue
        marks=prices.loc[:day].ffill().iloc[-1]
        w={c:q.loc[day,c]*marks[c]/a.account.loc[day] for c in held[day]}
        x,f,d,ready=risk.at(day)
        for row in g.itertuples():
            incoming=row.new if pd.notna(row.new) else None
            if incoming is None:delta=0.
            else:
                wo=w.get(row.old,0.)
                if path.name.startswith('industry_style') and ready:
                    active=np.zeros(f.shape[0]);active[0]=-1.
                    for c,v in w.items():active+=v*x[c]
                    after=active+wo*(x[incoming]-x[row.old])
                    delta=after@f@after-active@f@active+wo*wo*(d[incoming]-d[row.old])
                else:
                    bb=beta.loc[day];dd=specific.loc[day];pb=sum(v*bb[c] for c,v in w.items());change=wo*(bb[incoming]-bb[row.old])
                    delta=(2*(pb-1)*change+change*change)*mv.loc[day]+wo*wo*(dd[incoming]-dd[row.old])
            np.testing.assert_allclose(row.risk_variance_change,delta,atol=1e-12,rtol=1e-7)
            checks+=1
            # Gate utilities were evaluated sequentially before dropping a trailing
            # accepted prefix to satisfy the budget. Rebuild that proposal path.
            if row.forced or (incoming is not None and row.marginal_utility>0):
                moved=w.pop(row.old,0.)
                if incoming:w[incoming]=moved
    return checks


def save(root,data,source,riskpath,cache,out):
    root,data,source,riskpath,cache,out=map(Path,[root,data,source,riskpath,cache,out])
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);risk=RiskArrays(riskpath)
    benchmark=pd.read_csv(root/'results/next_open_3bp/report/next_open_daily.csv',index_col=0,parse_dates=True).benchmark_return
    bnav=(1+benchmark).cumprod();metrics=[];annual=[];curves={};audits={};riskmon=[];budgetrows=[];episode_rows=[]
    calendar=pd.DatetimeIndex(pd.read_csv(data/'calendars/day.txt',header=None)[0]);calendar=calendar[calendar<='2025-12-31']
    close=pd.DataFrame({c:read_bin(data,c,'close',calendar).astype(float) for c in risk.codes},index=calendar)
    stock_returns=close.pct_change(fill_method=None)
    factor_names=list(pd.read_csv(riskpath/'factor_returns.csv',nrows=0).columns[1:]);rcrows=[]
    for case in LABELS:
        path=source/case;a=pd.read_csv(path/'daily_report.csv',index_col=0,parse_dates=True,float_precision='round_trip')
        assert a.index.equals(benchmark.index)
        net=a['return']-a.cost;nav=a.account/1e8;active=net-benchmark
        np.testing.assert_allclose(nav,(1+net).cumprod(),rtol=1e-9,atol=1e-10)
        curve=pd.DataFrame(dict(NAV=nav,benchmark_NAV=bnav,relative_NAV=nav/bnav,net_return=net,active_return=active,
                                relative_DD=drawdown(nav/bnav),absolute_DD=drawdown(nav)))
        curve.to_csv(out/f'{case}_daily.csv');curves[case]=curve
        ep=episodes(curve.relative_NAV).head(5);ep.insert(0,'case',case);episode_rows.extend(ep.to_dict('records'))
        orders=pd.read_parquet(path/'orders.parquet');orders['date']=pd.to_datetime(orders.date)
        prior=a.account.shift(1).fillna(1e8);turn=orders.groupby('date').value.sum().reindex(a.index,fill_value=0)/prior
        slip=(orders.filled_amount*(orders.price-orders.reference_price)*np.where(orders.direction.eq(1),1.,-1.)).groupby(orders.date).sum().reindex(a.index,fill_value=0)/prior
        year_ir=[]
        for year,g in curve.groupby(curve.index.year):
            val=ir(g.active_return);year_ir.append(val);annual.append(dict(case=case,year=year,excess_IR=val,
                portfolio_return=(1+g.net_return).prod()-1,benchmark_return=(1+benchmark.loc[g.index]).prod()-1))
        metrics.append(dict(case=case,mean_annual_excess_IR=np.mean(year_ir),pooled_excess_IR=ir(active),CAGR=nav.iloc[-1]**.1-1,
            relative_max_DD=curve.relative_DD.min(),absolute_max_DD=curve.absolute_DD.min(),realized_TE=active.std(ddof=1)*np.sqrt(252),
            annual_two_way_turnover=turn.sum()/10,annual_commission_fraction=a.cost.sum()/10,annual_slippage_fraction=slip.sum()/10,
            mean_stock_exposure=(a.value/a.account).mean()))
        decision=pd.read_parquet(path/'gate_decisions.parquet');decision['signal_date']=pd.to_datetime(decision.signal_date)
        exits=pd.to_datetime(decision.max_label_exit);assert (exits<=decision.signal_date).all()
        np.testing.assert_allclose(decision.marginal_utility,decision.expected_edge-decision.cost_fraction-decision.risk_penalty,atol=1e-12)
        rejected=decision.get('turnover_rejected',pd.Series(False,index=decision.index)).fillna(False).astype(bool)
        np.testing.assert_array_equal(decision.accepted,(decision.forced|decision.marginal_utility.gt(0))&~rejected)
        budgets=pd.read_csv(path/'turnover_plans.csv',parse_dates=['trade_date','signal_date'])
        regular=budgets.exception.eq('none')
        if '6pct' in case:assert budgets.loc[regular,'planned_one_way_turnover'].le(.06+1e-12).all()
        for row in budgets.itertuples():budgetrows.append(dict(case=case,trade_date=row.trade_date,signal_date=row.signal_date,
            planned_one_way_turnover=row.planned_one_way_turnover,actual_one_way_turnover=turn.loc[row.trade_date]/2,
            exception=row.exception,has_cap='6pct' in case))
        audits[case]=dict(ledger=independent_audit(a,orders,data),
            score_and_cost=audit_score_and_cost(root,data,path,cache),
            independent_whole_factor_variance_checks=audit_risk_decisions(root,data,path,risk,cache),
            mature_labels=True,utility_reconciled=True,ordinary_planned_cap_verified=True if '6pct' in case else None,
            turnover_filtered_proposals=int(rejected.sum()),exception_days=budgets.loc[~regular,'exception'].value_counts().to_dict())
        h=pd.read_parquet(path/'holdings.parquet');h['date']=pd.to_datetime(h.date)
        quarter_dates=set(h.date.groupby(h.date.dt.to_period('Q')).max())
        for day,g in h.groupby('date'):
            x,f,d,ready=risk.at(day);w=dict(zip(g.instrument,g.weight));b=np.zeros(f.shape[0]);b[0]=-1.
            for c,v in w.items():b+=v*x[c]
            common=b@f@b;spec=sum(v*v*d[c] for c,v in w.items());variance=max(common+spec,1e-14)
            record=dict(case=case,date=day,predicted_TE=np.sqrt(252*variance),portfolio_beta=b[0]+1,
                        common_variance_share=common/variance,specific_variance_share=spec/variance,
                        covariance_ready=ready,unknown_style_weight=sum(v for c,v in w.items() if not risk.valid[risk.dates.get_loc(day),risk.slots[c]]))
            record.update({f'exposure_{factor_names[i]}':b[i] for i in range(1,4)})
            position=calendar.get_loc(day)
            if position+1<len(calendar):
                following=calendar[position+1];r=stock_returns.loc[following,list(w)]
                if r.notna().all() and following in benchmark.index:
                    outcome=sum(w[c]*r[c] for c in w)-benchmark.loc[following]
                    record.update(validation_date=following,frozen_inventory_next_active=outcome,
                                  standardized_next_active=outcome/np.sqrt(variance))
            riskmon.append(record)
            if day in quarter_dates:
                rc=b*(f@b)*252
                for name,v in zip(factor_names,rc):rcrows.append(dict(case=case,date=day,factor=name,annual_variance_contribution=v))
                rcrows.append(dict(case=case,date=day,factor='Specific',annual_variance_contribution=spec*252))
        if case=='current_replay':
            original=pd.read_csv(root/'results/net_edge_improvement/account/daily_report.csv',index_col=0,parse_dates=True,float_precision='round_trip')
            np.testing.assert_allclose(a.account,original.account,rtol=1e-12,atol=1e-5)
    m=pd.DataFrame(metrics).set_index('case');m.to_csv(out/'metrics.csv');pd.DataFrame(annual).to_csv(out/'annual.csv',index=False)
    pd.DataFrame(episode_rows).to_csv(out/'relative_drawdowns.csv',index=False)
    windowrows=[]
    for record in episode_rows:
        if record['depth']!=min(x['depth'] for x in episode_rows if x['case']==record['case']):continue
        start=pd.Timestamp(record['peak']);end=pd.Timestamp(record['trough'])
        for case,g in curves.items():
            selected=(g.index>start)&(g.index<=end);model=(1+g.loc[selected,'net_return']).prod()-1
            bench=(1+benchmark.loc[selected]).prod()-1
            windowrows.append(dict(window_from=record['case'],peak=start,trough=end,case=case,
                portfolio_return=model,benchmark_return=bench,relative_return=(1+model)/(1+bench)-1))
    pd.DataFrame(windowrows).to_csv(out/'drawdown_window_comparison.csv',index=False)
    pd.DataFrame(riskmon).to_csv(out/'risk_monitor.csv',index=False);pd.DataFrame(rcrows).to_csv(out/'risk_contributions.csv',index=False)
    monitor=pd.DataFrame(riskmon);validation=[]
    for case,g in monitor.groupby('case'):
        z=g.standardized_next_active.dropna();validation.append(dict(case=case,validation_days=len(z),
            frozen_inventory_standardized_RMS=float(np.sqrt(np.mean(z*z))),
            frozen_inventory_abs_z_above_1_96_fraction=float((abs(z)>1.96).mean()),
            mean_predicted_TE=float(g.predicted_TE.mean()),mean_unknown_style_weight=float(g.unknown_style_weight.mean())))
    pd.DataFrame(validation).to_csv(out/'risk_validation.csv',index=False)
    pd.DataFrame(budgetrows).to_csv(out/'turnover_budget_audit.csv',index=False)
    # Reused development sample: compare each variant, no cherry-picked CI.
    rng=np.random.default_rng(20261104);uncertainty={};base=curves['current_replay'].active_return.to_numpy();n=len(base)
    starts=rng.integers(0,n,size=(1999,int(np.ceil(n/20))));sample=(starts[:,:,None]+np.arange(20)).reshape(1999,-1)[:,:n]%n
    bootstrap_base=base[sample];base_ir=bootstrap_base.mean(axis=1)/bootstrap_base.std(axis=1,ddof=1)*np.sqrt(252)
    for case in list(LABELS)[1:]:
        v=curves[case].active_return.to_numpy()[sample];delta=v.mean(axis=1)/v.std(axis=1,ddof=1)*np.sqrt(252)-base_ir
        uncertainty[case]=dict(pooled_IR_difference=float(m.loc[case,'pooled_excess_IR']-m.loc['current_replay','pooled_excess_IR']),
            paired_20day_block_CI95=np.quantile(delta,[.025,.975]).tolist(),replications=1999,selection_corrected=False)
    (out/'audit.json').write_text(json.dumps(audits,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (out/'uncertainty.json').write_text(json.dumps(uncertainty,indent=2)+'\n',encoding='utf-8')
    for file in ['plan.json']:shutil.copy2(source/file,out/file)
    for file in ['metadata.json','factor_returns.csv','fit_audit.csv']:shutil.copy2(riskpath/file,out/('risk_'+file))
    plot(curves,pd.DataFrame(riskmon),pd.DataFrame(budgetrows),out)
    print(m.to_string(),flush=True)


def plot(curves,monitor,budgets,out):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.grid':True,'grid.alpha':.2})
    fig,axes=plt.subplots(3,1,figsize=(12,10),sharex=True)
    for c,g in curves.items():
        axes[0].plot(g.index,g.NAV,label=LABELS[c],lw=1.3)
        axes[1].plot(g.index,g.relative_NAV,label=LABELS[c],lw=1.3)
        axes[2].plot(g.index,100*g.relative_DD,lw=1.)
    axes[0].plot(g.index,g.benchmark_NAV,color='gray',ls='--',label='CSI300 total return')
    axes[0].set_ylabel('Portfolio NAV');axes[0].legend(ncol=2);axes[1].set_ylabel('Relative NAV');axes[2].set_ylabel('Relative drawdown (%)')
    axes[0].set_title('2016–2025 | same frozen alpha, next open, 3bp per side | reused development sample')
    fig.tight_layout();fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(12,9),sharex=True)
    for case,g in monitor.groupby('case',sort=False):axes[0].plot(g.date,100*g.predicted_TE.rolling(20).mean(),label=LABELS[case],lw=1)
    axes[0].set_ylabel('Predicted TE (%)');axes[0].legend(ncol=2)
    for case,g in budgets.groupby('case',sort=False):axes[1].plot(g.trade_date,100*g.actual_one_way_turnover.rolling(60).mean(),label=LABELS[case],lw=1)
    axes[1].set_ylabel('60d avg one-way turnover (%)')
    for case,g in monitor.groupby('case',sort=False):axes[2].plot(g.date,g.portfolio_beta.rolling(20).mean(),lw=1)
    axes[2].axhline(1,color='gray',ls='--');axes[2].set_ylabel('Estimated portfolio beta')
    axes[0].set_title('Conditional risk estimates are diagnostics, not guaranteed realized risk')
    fig.tight_layout();fig.savefig(out/'risk_turnover.png',dpi=160);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for x in ['data-dir','source','risk-cache','calibration-cache','output']:p.add_argument('--'+x,type=Path,required=True)
    a=p.parse_args();save(Path(__file__).resolve().parents[1],a.data_dir,a.source,a.risk_cache,a.calibration_cache,a.output)
