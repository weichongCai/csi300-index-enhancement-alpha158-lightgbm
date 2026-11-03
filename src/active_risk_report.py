"""Audit, risk monitoring and transparent comparison for active-risk experiments."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import pandas as pd
from .metrics import ir,drawdown
from .active_risk_system import estimate_risk,risk_numbers,POLICY


def independent_audit(account,orders,data):
    from .research_build import read_bin
    orders=orders.copy();orders['date']=pd.to_datetime(orders.date)
    orders['flow']=orders.filled_amount*np.where(orders.direction.eq(1),1.,-1.)
    flows=orders.pivot_table(index='date',columns='instrument',values='flow',aggfunc='sum').reindex(account.index).fillna(0.)
    quantity=flows.cumsum()
    if quantity.to_numpy().min() < -1e-6:raise ArithmeticError('short stock ledger')
    cal=pd.DatetimeIndex(pd.read_csv(data/'calendars/day.txt',header=None)[0]);cal=cal[cal<='2025-12-31']
    slots=cal.get_indexer(account.index);mark_columns={}
    for code in quantity.columns:
        close=read_bin(data,code,'close',cal).astype(float)
        opening=read_bin(data,code,'open',cal)
        mark_columns[code]=pd.Series(np.where(np.isfinite(opening)&(opening>0),close,np.nan)).ffill().to_numpy()[slots]
    marks=pd.DataFrame(mark_columns,index=account.index)
    active=quantity.abs().gt(1e-6)
    if not (marks.notna()|~active).all().all():raise ArithmeticError('unmarked holding')
    stock=(quantity*marks).where(active,0.).sum(axis=1)
    np.testing.assert_allclose(stock,account.value,rtol=1e-9,atol=.02)
    np.testing.assert_allclose(stock+account.cash,account.account,rtol=1e-9,atol=.02)
    return dict(days=len(account),maximum_absolute_account_difference_CNY=float(abs(stock+account.cash-account.account).max()),
                quantity_ledger_nonnegative=True,marks_match=True)


def monitor(holdings,stock_returns,benchmark,curves,forecast):
    wide=holdings.pivot(index='date',columns='instrument',values='weight').fillna(0.).sort_index()
    wide.index=pd.to_datetime(wide.index)
    wide=wide.reindex(curves.index,fill_value=0.)
    rows=[];contributions=[]
    for asof,weights in wide.iterrows():
        weights=weights.loc[weights>1e-8]
        row={'asof':asof, 'stock_weight':weights.sum(), 'max_stock_weight':weights.max(),
             'effective_stock_count':weights.sum()**2/(weights@weights) if len(weights) else np.nan,
             'relative_drawdown_known':float(curves.loc[asof,'relative_drawdown'])}
        try:
            members=forecast.xs(asof,level='datetime').index
            h=stock_returns.loc[:asof,members].tail(POLICY['window'])
            qualified=h.columns[h.notna().sum().eq(POLICY['window'])]
            codes=sorted(set(qualified)|set(weights.index))
            w=weights.reindex(codes,fill_value=0.).to_numpy()
            cov,beta,hist,shrink=estimate_risk(stock_returns,benchmark,asof,codes,allow_reserve=True)
            te,rc=risk_numbers(w,cov)
            scenario_history=hist[[*weights.index,'__benchmark']].dropna()
            scenario=scenario_history.iloc[:,:-1].to_numpy()@weights.to_numpy()-scenario_history.iloc[:,-1].to_numpy()
            q=float(np.quantile(scenario,.05)) if len(scenario)>=126 else np.nan
            row.update(status='estimated',predicted_te=te,beta=float(beta@w),shrinkage=shrink,
                VaR95=-q,ES95=-float(scenario[scenario<=q].mean()) if np.isfinite(q) else np.nan,
                worst_past_day=float(scenario.min()) if len(scenario)>=126 else np.nan,
                unestimable_stock_weight=float(weights.reindex(hist.attrs['reserve_codes']).sum()),
                history_end=hist.index.max(), history_observations=len(hist))
            row['linear_market_up10_active_pnl']=(row['beta']-1)*.10
            row['linear_market_down10_active_pnl']=-(row['beta']-1)*.10
            row['volatility_1_5x_TE_stress']=te*1.5
            pos=stock_returns.index.get_loc(asof)
            if pos+1<len(stock_returns):
                tomorrow=stock_returns.index[pos+1]
                rr=stock_returns.loc[tomorrow,weights.index]
                if rr.notna().all() and np.isfinite(benchmark.get(tomorrow,np.nan)):
                    # Frozen closing inventory, before hypothetical next-day trades/fees.
                    realized=float(rr@weights-benchmark.loc[tomorrow])
                    row.update(validation_date=tomorrow,frozen_inventory_next_active=realized,
                        standardized_next_active=realized/(te/np.sqrt(252)),VaR95_exception=bool(realized < q) if np.isfinite(q) else np.nan)
            if asof==wide.index[-1] or asof.strftime('%m-%d') in ['12-31']:
                for c,value in zip([*codes,'__benchmark'],rc):
                    contributions.append(dict(asof=asof,instrument=c,annual_variance_contribution=value,
                        annual_TE_contribution=value/te if te>0 else np.nan))
        except ValueError as error:
            row.update(status=f'not estimated: {error}')
        # Alert thresholds do not generate separate daily trades; weekly optimization is the response policy.
        row['te_alert']=row.get('predicted_te',0)>POLICY['annual_te_limit']+1e-6
        row['beta_alert']=row.get('beta',1)<POLICY['beta_lower'] or row.get('beta',1)>POLICY['beta_upper']
        row['weight_drift_alert']=weights.max()>.045
        row['relative_dd_warning']=row['relative_drawdown_known']<-.06
        row['relative_dd_escalation']=row['relative_drawdown_known']<-.08
        rows.append(row)
    return pd.DataFrame(rows).set_index('asof'),pd.DataFrame(contributions)


def cached_monitor(holdings,stocks,benchmark,daily,forecast,cache,input_paths):
    if cache is None:return monitor(holdings,stocks,benchmark,daily,forecast)
    digests=[hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in [*input_paths,Path(__file__),Path(__file__).with_name('active_risk_system.py')]]
    key=hashlib.sha256(json.dumps(digests).encode()).hexdigest()
    dest=Path(cache)/key;meta=dest/'checksums.json'
    if meta.exists():
        expected=json.loads(meta.read_text(encoding='utf-8'))
        if all(hashlib.sha256((dest/n).read_bytes()).hexdigest()==h for n,h in expected.items()):
            return pd.read_parquet(dest/'watch.parquet'),pd.read_parquet(dest/'contributions.parquet')
    watch,rc=monitor(holdings,stocks,benchmark,daily,forecast)
    dest.mkdir(parents=True,exist_ok=True)
    watch.to_parquet(dest/'watch.parquet');rc.to_parquet(dest/'contributions.parquet',index=False)
    meta.write_text(json.dumps({n:hashlib.sha256((dest/n).read_bytes()).hexdigest() for n in ['watch.parquet','contributions.parquet']},indent=2)+'\n',encoding='utf-8',newline='\n')
    return watch,rc


def save(root,data,source,out,monitor_cache=None):
    root,data,source,out=[Path(p).resolve() for p in [root,data,source,out]]
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    benchmark=pd.read_csv(root/'results/next_open_3bp/report/next_open_daily.csv',index_col=0,parse_dates=True).benchmark_return
    stock_returns=pd.read_parquet(source/'risk_stock_returns.parquet')
    forecast=pd.read_parquet(root/'data/frozen_forecast.parquet').iloc[:,0]
    risk_benchmark=pd.read_csv(source/'risk_benchmark_returns.csv',index_col=0,parse_dates=True).benchmark_return
    metrics=[];annual=[];curves={};checks={};monitoring=[];windows=[]
    cases={'published_daily':root/'results/next_open_3bp/open',
           'weekly_equal':source/'weekly_equal','optimized':source/'optimized'}
    for case,path in cases.items():
        account=pd.read_csv(path/'daily_report.csv',index_col=0,parse_dates=True,float_precision='round_trip')
        net=account['return']-account.cost;nav=account.account/1e8
        bm=benchmark.reindex(account.index);bnav=(1+bm).cumprod()
        daily=pd.DataFrame(dict(model_return=net,benchmark_return=bm,active_return=net-bm,
            model_nav=nav,benchmark_nav=bnav,relative_nav=nav/bnav,
            model_drawdown=drawdown(nav),relative_drawdown=drawdown(nav/bnav)))
        daily.to_csv(out/f'{case}_daily.csv',index_label='date');curves[case]=daily
        orders=pd.read_parquet(path/'orders.parquet')
        prior=account.account.shift(1).fillna(1e8)
        turnover=orders.assign(date=pd.to_datetime(orders.date)).groupby('date').value.sum().reindex(account.index,fill_value=0)/prior
        slip=orders.assign(date=pd.to_datetime(orders.date),slip=orders.filled_amount*(orders.price-orders.reference_price)*np.where(orders.direction.eq(1),1.,-1.)).groupby('date').slip.sum().reindex(account.index,fill_value=0)/prior
        years=[]
        for y,g in daily.groupby(daily.index.year):
            val=dict(case=case,year=y,net_active_ir=ir(g.active_return),model_return=(1+g.model_return).prod()-1,
                benchmark_return=(1+g.benchmark_return).prod()-1,relative_return=(1+g.model_return).prod()/(1+g.benchmark_return).prod()-1)
            annual.append(val);years.append(val['net_active_ir'])
        metrics.append(dict(case=case,mean_annual_excess_IR=np.mean(years),pooled_excess_IR=ir(net-bm),
            CAGR=nav.iloc[-1]**.1-1,max_absolute_DD=drawdown(nav).min(),max_relative_DD=drawdown(nav/bnav).min(),
            annual_TE=(net-bm).std(ddof=1)*np.sqrt(252),mean_stock_exposure=(account.value/account.account).mean(),
            annual_two_way_turnover=turnover.sum()/10,annual_commission_fraction=account.cost.sum()/10,
            annual_slippage_fraction=slip.sum()/10))
        for name,peak,end in [('2025_relative','2025-05-30','2025-09-03'),('2022_absolute','2021-12-31','2022-04-26'),('2023_24_relative','2023-07-11','2024-02-05')]:
            pr=float(nav.loc[end]/nav.loc[peak]-1);br=float(bnav.loc[end]/bnav.loc[peak]-1)
            windows.append(dict(case=case,window=name,peak=peak,end=end,portfolio_return=pr,benchmark_return=br,relative_return=(1+pr)/(1+br)-1))
        if case!='published_daily':
            checks[case]=independent_audit(account,orders,data)
            decisions=pd.read_csv(path/'decisions.csv')
            planned=decisions.loc[decisions.status.eq('planned')]
            assert (pd.to_datetime(planned.history_end)<=pd.to_datetime(planned.signal_date)).all()
            checks[case]['planned_rebalances']=len(planned)
            checks[case]['skipped_rebalances']=int((~decisions.status.eq('planned')).sum())
            checks[case]['skip_reasons']=decisions.loc[~decisions.status.eq('planned'),'status'].value_counts().to_dict()
            checks[case]['risk_inputs_never_after_signal']=True
            if case=='optimized':
                np.testing.assert_allclose(planned.TE_budget_excess,np.maximum(0.,planned.target_predicted_te-.1),atol=1e-10)
                assert planned.target_max_weight.le(.040002).all()
                assert planned.target_beta.between(.849998,1.150002).all()
                assert (planned.target_rank_exposure>=planned.required_rank_exposure-2e-6).all()
                checks[case]['reference_rank_unattainable_plans']=int(planned.rank_reference_feasibility_shortfall.gt(2e-6).sum())
                checks[case]['TE_soft_budget_exceeded_plans']=int(planned.TE_budget_excess.gt(1e-6).sum())
                checks[case]['maximum_TE_soft_budget_excess']=float(planned.TE_budget_excess.max())
            targets=pd.read_parquet(path/'targets.parquet')
            summed=targets.groupby('trade_date').annual_active_variance_contribution.sum()
            implied=planned.set_index('trade_date').target_predicted_te**2
            np.testing.assert_allclose(summed.reindex(implied.index),implied,atol=1e-10,rtol=1e-7)
            checks[case]['target_risk_contributions_reconcile']=True
        holdings=pd.read_parquet(path/'holdings.parquet');holdings['date']=pd.to_datetime(holdings.date)
        watch,rc=cached_monitor(holdings,stock_returns,risk_benchmark,daily,forecast,monitor_cache,
            [path/'holdings.parquet',path/'daily_report.csv',source/'risk_stock_returns.parquet',source/'risk_benchmark_returns.csv',root/'data/frozen_forecast.parquet',root/'results/next_open_3bp/report/next_open_daily.csv'])
        watch.to_csv(out/f'{case}_risk_monitor.csv');rc.to_csv(out/f'{case}_risk_contributions.csv',index=False)
        flags=['te_alert','beta_alert','weight_drift_alert','relative_dd_warning','relative_dd_escalation']
        alerts=watch.loc[watch[flags].any(axis=1)|~watch.status.eq('estimated')].copy()
        alerts['response']='review at next scheduled optimization; DD alerts require diagnosis, no automatic sell-to-cash'
        alerts.to_csv(out/f'{case}_alerts.csv')
        frozen=watch.dropna(subset=['frozen_inventory_next_active','standardized_next_active'])
        valid=frozen.dropna(subset=['VaR95_exception'])
        summary=dict(case=case,risk_estimated_days=int(watch.status.eq('estimated').sum()),
            unavailable_days=int((~watch.status.eq('estimated')).sum()),
            mean_predicted_TE=watch.predicted_te.mean(),TE_alert_days=int(watch.te_alert.sum()),
            beta_alert_days=int(watch.beta_alert.sum()),weight_drift_alert_days=int(watch.weight_drift_alert.sum()),
            relative_DD_warning_days=int(watch.relative_dd_warning.sum()),
            relative_DD_escalation_days=int(watch.relative_dd_escalation.sum()),
            unestimable_holding_days=int(watch.unestimable_stock_weight.gt(1e-8).sum()),
            maximum_unestimable_stock_weight=watch.unestimable_stock_weight.max(),
            historical_VaR_available_days=int(watch.VaR95.notna().sum()),
            frozen_inventory_validation_days=len(frozen),
            VaR_validation_days=len(valid),VaR95_exception_rate=valid.VaR95_exception.astype(bool).mean(),
            standardized_frozen_return_std=frozen.standardized_next_active.std(ddof=1))
        monitoring.append(summary)
        print('RISK_REPORT_AUDITED',case,flush=True)
    metrics=pd.DataFrame(metrics);metrics.to_csv(out/'metrics.csv',index=False)
    pd.DataFrame(annual).to_csv(out/'annual.csv',index=False)
    pd.DataFrame(windows).to_csv(out/'episode_comparison.csv',index=False)
    pd.DataFrame(monitoring).to_csv(out/'risk_calibration.csv',index=False)
    from .risk_attribution import episodes
    from .research_math import circular_bootstrap_mean
    episodes_all=[];eras=[];uncertainty=[]
    for case,d in curves.items():
        eps=episodes(d.relative_nav);eps['case']=case;episodes_all.append(eps)
        for name,begin,end in [('2016_2020','2016-01-01','2020-12-31'),('2021_2025','2021-01-01','2025-12-31')]:
            z=d.loc[begin:end]
            eras.append(dict(case=case,period=name,pooled_excess_IR=ir(z.active_return),
                relative_return=(1+z.model_return).prod()/(1+z.benchmark_return).prod()-1,
                annual_TE=z.active_return.std(ddof=1)*np.sqrt(252)))
    pd.concat(episodes_all,ignore_index=True).to_csv(out/'relative_drawdown_episodes.csv',index=False)
    pd.DataFrame(eras).to_csv(out/'era_comparison.csv',index=False)
    for other in ['published_daily','weekly_equal']:
        a=curves['optimized'].active_return.to_numpy();b=curves[other].active_return.to_numpy()
        m=circular_bootstrap_mean(np.column_stack([a,a*a,b,b*b]),length=20,replications=4999,seed=20261103)
        n=len(a)
        sampled_a=m[:,0]/np.sqrt(np.maximum((m[:,1]-m[:,0]**2)*n/(n-1),1e-30))*np.sqrt(252)
        sampled_b=m[:,2]/np.sqrt(np.maximum((m[:,3]-m[:,2]**2)*n/(n-1),1e-30))*np.sqrt(252)
        ci=np.quantile(sampled_a-sampled_b,[.025,.975])
        uncertainty.append(dict(comparison='optimized minus '+other,pooled_IR_difference=ir(pd.Series(a))-ir(pd.Series(b)),
            paired_block_bootstrap_low95=ci[0],paired_block_bootstrap_high95=ci[1],
            block_length=20,replications=4999,seed=20261103,development_selection_corrected=False))
    pd.DataFrame(uncertainty).to_csv(out/'paired_IR_uncertainty.csv',index=False)
    (out/'checks.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'published_daily':'#a0a0a0','weekly_equal':'#da8c29','optimized':'#167e8d'}
    labels={'published_daily':'Published daily Top50/Drop5','weekly_equal':'Weekly Top50 equal weight','optimized':'Weekly risk / cost optimization'}
    fig,axes=plt.subplots(3,1,figsize=(13,11),sharex=True)
    for case,daily in curves.items():
        axes[0].plot(daily.index,daily.model_nav,label=labels[case],color=colors[case])
        axes[1].plot(daily.index,daily.relative_nav,color=colors[case])
        axes[2].plot(daily.index,daily.relative_drawdown*100,color=colors[case])
    axes[0].plot(daily.index,daily.benchmark_nav,label='CSI300 total return',color='#333333',ls='--')
    axes[0].set_yscale('log');axes[0].legend(fontsize=8)
    for ax,title in zip(axes,['After-cost NAV (log)','NAV / benchmark','Relative drawdown (%)']):ax.set_ylabel(title);ax.grid(alpha=.2)
    axes[0].set_title('2016–2025 REUSED development | frozen alpha | next open | 3bp adverse prices')
    fig.tight_layout();fig.savefig(out/'curves.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(13,10),sharex=True)
    for case in cases:
        watch=pd.read_csv(out/f'{case}_risk_monitor.csv',index_col=0,parse_dates=True)
        axes[0].plot(watch.index,watch.predicted_te*100,color=colors[case],label=labels[case],lw=.7)
        axes[1].plot(watch.index,watch.beta,color=colors[case],lw=.7)
        realized=curves[case].active_return.rolling(60,min_periods=40).std(ddof=1)*np.sqrt(252)
        axes[2].plot(realized.index,realized*100,color=colors[case],lw=.8)
    axes[0].axhline(10,ls='--',color='red',lw=.8);axes[0].legend(fontsize=8)
    for b in [.85,1.15]:axes[1].axhline(b,ls='--',color='red',lw=.8)
    for ax,title in zip(axes,['Closing inventory forecast TE (%)','Closing inventory beta','Actual net active 60-day TE (%)']):ax.set_ylabel(title);ax.grid(alpha=.2)
    axes[0].set_title('Risk limits apply to targets; fills and weekly drift can breach them')
    fig.tight_layout();fig.savefig(out/'risk_monitor.png',dpi=150);plt.close(fig)
    files={str(p.relative_to(root)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest()
           for p in [root/'data/frozen_forecast.parquet',source/'policy.json',source/'risk_stock_returns.parquet',source/'risk_benchmark_returns.csv',source/'official_benchmark_history.json']}
    (out/'provenance.json').write_text(json.dumps(files,indent=2)+'\n',encoding='utf-8',newline='\n')
    write_report(metrics,pd.DataFrame(monitoring),checks,pd.DataFrame(windows),pd.DataFrame(uncertainty),out)
    print(metrics.to_string(index=False),flush=True)


def write_report(metrics,monitoring,checks,windows,uncertainty,out):
    names={'published_daily':'原日频Top50/Drop5','weekly_equal':'周频Top50等权','optimized':'分层信号预算＋风险/成本优化'}
    m=metrics.set_index('case');b=m.loc['published_daily'];o=m.loc['optimized']
    improvement=o.max_relative_DD-b.max_relative_DD
    verdict=('全期最大相对回撤较原策略减轻' if improvement>0 else '未能降低全期最大相对回撤')
    lines=['# 组合优化与主动风险管理实验','',
        f'**本轮结论：{verdict}。** 优化组合与原账户的最大相对回撤差为{improvement*100:+.2f}个百分点，年化收益差为{(o.CAGR-b.CAGR)*100:+.2f}个百分点，全期超额IR差为{o.pooled_excess_IR-b.pooled_excess_IR:+.3f}。这些变化同时报告，不仅选取有利窗口。此模块为离线研究风控系统，不自动替换原模型或模拟盘。','',
        '2016-01-04—2025-12-31，共2,430个交易日、初始资金1亿元。冻结原LightGBM及预测，次日开盘执行，每边3bp成交价滑点；基准为H00300全收益指数，首日仍沿用原开盘实验的价格指数开收盘比例近似。','',
        '## 一、全期账户与成本','',
        '| 方案 | 平均年度超额IR | 全期超额IR | CAGR | 最大绝对回撤 | 最大相对回撤 | 实现TE | 平均股票仓位 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for c,r in m.iterrows():
        lines.append(f'| {names[c]} | {r.mean_annual_excess_IR:.3f} | {r.pooled_excess_IR:.3f} | {r.CAGR:.2%} | {r.max_absolute_DD:.2%} | {r.max_relative_DD:.2%} | {r.annual_TE:.2%} | {r.mean_stock_exposure:.2%} |')
    lines+=['','![净值、相对净值及超额回撤](curves.png)','','| 方案 | 年均双边换手 | 年均基础费用日占比加总 | 年均滑点日占比加总 |',
        '| --- | ---: | ---: | ---: |']
    for c,r in m.iterrows():lines.append(f'| {names[c]} | {r.annual_two_way_turnover:.2f}倍 | {r.annual_commission_fraction:.2%} | {r.annual_slippage_fraction:.2%} |')
    lines+=['','费用占比按每日前净值计算，不直接从CAGR相减。周频完全重选50只股票可能比日频只换5只的账户换手更高；频率低不等于成本低。',
        '','','## 二、原超额回撤的线索与设计对应','',
        '2025-05-30至2025-09-03，原账户上涨6.13%，基准上涨18.09%，相对回撤10.13%。已有精确链接的代理归因显示：市场敏感度偏离−2.90个百分点、价格/全收益基准差−1.64个百分点、波动风格−2.41个百分点、已覆盖股票残余−1.62个百分点、买卖当日日内增量合计−1.04个百分点、基础费用及滑点−1.60个百分点，另有共同项等抵消。它是模型依赖的事后解释，不是因果证明。',
        '', '因此新设计同时考虑双向beta、总跟踪误差和成本；没有仅在亏损后削减股票仓位。预测和评价统一采用全收益基准。详细账目见[原回撤归因](../../deep_attribution/README.md)。','',
        '## 三、独立比较旧回撤窗口','',
        '| 原账户选定窗口 | 方案 | 策略区间收益 | 基准区间收益 | 相对区间收益 |',
        '| --- | --- | ---: | ---: | ---: |']
    for _,r in windows.iterrows():lines.append(f"| {r['window']} | {names[r['case']]} | {r.portfolio_return:.2%} | {r.benchmark_return:.2%} | {r.relative_return:.2%} |")
    lines+=['','这些日期由原策略的事后回撤确定，只用于诊断；优化组合自身峰谷见[完整相对回撤表](relative_drawdown_episodes.csv)，不能将这些窗口当作优化组合的最大回撤。','',
        '## 四、风控系统及优先级','',
        '- 新买入候选须具备252日完整历史；使用信号日及此前数据，实际旧持仓始终保留在风险账本。',
        '- 完整历史子集用联合收缩协方差；部分历史旧持仓用市场条件扩展；无法估计的持仓使用明确标注的100%年化特质波动准备，尝试退出但保留未成交风险。准备不是损失上界。',
        '- 硬约束：股票目标99%、单股不超过4%、beta在0.85—1.15、每次双边计划换手不超过40%（首次建仓例外），不合格旧持仓不可新增。',
        '- 先计算硬约束下可达到的最高排序暴露。能保留参考Top50排序暴露时保留；不能达到时要求可行前沿并记录参考缺口。排序预算不等于预期收益保证。',
        '- 再权衡参考权重偏离、预测主动方差、线性交易成本和TE预算超出量。10%TE是软预算，超出显式记录，不能用不建仓躲过下跌冒充改进。',
        '- 每周首个交易日按前收盘计划，次日开盘执行。目标低于1bp的数值尾仓编译为完整退出/不新买；实际成交仍受整手、费用、现金和前20日量额容量限制。',
        '- 收盘基于实际持仓、同一类候选风险宇宙重新估计TE/beta；逐日生成风险快照、市场±10%与波动1.5倍情景、相对回撤−6%预警/−8%升级记录。预警不触发额外事后择时交易，自动交易响应仍按周执行。','',
        '![事前风险与实际主动波动](risk_monitor.png)','','## 五、风险估计与校准的边界','',
        '| 方案 | TE超预算天数 | beta超限天数 | 无法估计风险持仓天数 | 最大准备项股票权重 | 可验证历史VaR天数 | VaR95突破率 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for _,r in monitoring.iterrows():
        lines.append(f"| {names[r['case']]} | {int(r.TE_alert_days)} | {int(r.beta_alert_days)} | {int(r.unestimable_holding_days)} | {r.maximum_unestimable_stock_weight:.2%} | {int(r.VaR_validation_days)} | {r.VaR95_exception_rate:.2%} |")
    lines+=['','TE/beta目标约束与收盘实际持仓不同；跳空、未成交、价格漂移和每周复核间隔均可能导致实际超限。10%是预算，不能声称实际TE永不超过10%。',
        '', 'VaR检验使用收盘后冻结该库存的下一日假想主动收益，不计下一日换仓和费用；实际账户的净60日TE另画。共同历史不足或未来单股收益缺失的日期不参加VaR检验；准备项有风险估计不等于有可验证VaR。有效天数少时突破率不能证明风险模型可靠。',
        '', 'Euler证券/基准风险贡献精确加总到预测TE²，但含负基准项，不是完整Barra行业/风格归因，也没有证实行业中性。历史波动和VaR不能保证未来最大回撤。','',
        '## 六、实现核对与统计不确定性','',
        '| 实验账户 | 成功目标 | 跳过调仓 | 最大独立账簿差异（元） |',
        '| --- | ---: | ---: | ---: |']
    for c,v in checks.items():lines.append(f"| {names[c]} | {v['planned_rebalances']} | {v['skipped_rebalances']} | {v['maximum_absolute_account_difference_CNY']:.9f} |")
    lines+=['','每次跳过的原因保存在checks.json和decisions.csv。已核对逐笔价格、费用、现金、容量、信号前一交易日、数量账簿、非负持仓、目标约束和风险贡献加总。','',
        '| 全期IR差值 | 点估计 | 配对块Bootstrap描述性95%区间 |',
        '| --- | ---: | --- |']
    for _,r in uncertainty.iterrows():lines.append(f"| {r.comparison} | {r.pooled_IR_difference:+.3f} | [{r.paired_block_bootstrap_low95:+.3f}, {r.paired_block_bootstrap_high95:+.3f}] |")
    lines+=['','固定20日圆形块、4,999次、种子20261103。该区间不能校正整个项目的策略选择。规则经过开发期可用性、可行性和信号损失反馈后修订；没有参数网格搜索不等于没有过拟合风险。2016—2020和2021—2025拆分结果均是开发子区间，见era_comparison.csv。','',
        '## 七、方法来源与复现','',
        '单期风险/成本决策参考[Boyd等，Multi-Period Trading via Convex Optimization](https://web.stanford.edu/~boyd/papers/cvx_portfolio.html)；协方差收缩参考[Ledoit与Wolf](https://ledoit.net/honey_abstract.htm)，实现采用[scikit-learn缩放单位阵目标](https://scikit-learn.org/stable/modules/generated/sklearn.covariance.LedoitWolf.html)。这是借鉴方法，没有复现原论文的全部实证或MSCI商业Barra模型。','',
        '完整数学、缺失风险、约束冲突、执行与岗位讨论见[组合优化与风险管理](../../../docs/组合优化与风险管理.md)。',
        '', '```bash','pip install -r requirements-risk.txt',
        'python -m src.active_risk_system --data-dir /path/to/qlib_data --historical-benchmark results/risk_management/official_benchmark_history.json --output runs/risk-replay',
        'python -m src.active_risk_report --data-dir /path/to/qlib_data --source runs/risk-replay --output runs/risk-replay/report','```','',
        '原始股票行情仍需另备。报告复算使用保存的收益风险输入，并通过原始行情独立重建数量账簿；未重放盘口、集合竞价资金冻结和动态冲击。历史风险数据本身也不是逐时点供应商快照。原模型和原发布账户没有被本实验覆盖。','']
    (out/'README.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--monitor-cache',type=Path)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();save(Path(__file__).resolve().parents[1],args.data_dir,args.source,args.output,args.monitor_cache)
