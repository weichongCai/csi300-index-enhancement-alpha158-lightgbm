"""Audit and report all disclosed daily marginal-utility gate variants."""
from pathlib import Path
import argparse,json,hashlib,shutil
import numpy as np
import pandas as pd
from .metrics import ir,drawdown
from .risk_attribution import episodes
from .active_risk_report import independent_audit


def audit_score_and_cost(root,data,path,source):
    from .research_build import read_bin
    f=pd.read_parquet(root/'data/frozen_forecast.parquet').iloc[:,0]
    cal=pd.DatetimeIndex(pd.read_csv(data/'calendars/day.txt',header=None)[0]);cal=cal[cal<='2025-12-31']
    codes=sorted(f.index.get_level_values('instrument').unique())
    close=pd.DataFrame({c:read_bin(data,c,'close',cal).astype(float) for c in codes},index=cal)
    opening=pd.DataFrame({c:read_bin(data,c,'open',cal).astype(float) for c in codes},index=cal)
    marks=close.where(opening.gt(0)&np.isfinite(opening)).ffill()
    o=pd.read_parquet(path/'orders.parquet');o['date']=pd.to_datetime(o.date)
    o['flow']=o.filled_amount*np.where(o.direction.eq(1),1.,-1.)
    a=pd.read_csv(path/'daily_report.csv',index_col=0,parse_dates=True)
    q=o.pivot_table(index='date',columns='instrument',values='flow',aggfunc='sum').reindex(a.index).fillna(0.).cumsum()
    h=pd.read_parquet(path/'holdings.parquet');h['date']=pd.to_datetime(h.date)
    held={d:set(g.instrument) for d,g in h.groupby('date')}
    decisions=pd.read_parquet(path/'gate_decisions.parquet');decisions['signal_date']=pd.to_datetime(decisions.signal_date)
    regular=decisions.loc[~decisions.forced];count=0
    for day,g in regular.groupby('signal_date'):
        score=f.xs(day,level='datetime').dropna();prices=close.loc[day].copy()
        for c in held.get(day,set()):
            if not np.isfinite(prices[c]) or prices[c]<=0:prices[c]=marks.loc[day,c]
        z=(score-score.mean())/score.std(ddof=1)
        np.testing.assert_allclose(z.reindex(g.old),g.z_old,rtol=1e-11,atol=1e-13)
        np.testing.assert_allclose(z.reindex(g.new),g.z_new,rtol=1e-11,atol=1e-13)
        value=q.loc[day].reindex(g.old).to_numpy()*prices.reindex(g.old).to_numpy()
        expected=np.maximum(.0015,5/value)+np.maximum(.0005,5/value)+.0006
        np.testing.assert_allclose(expected,g.cost_fraction,rtol=1e-9,atol=1e-13)
        count+=len(g)
    risk_checks=0
    if path.name.startswith('risk_'):
        beta=pd.read_parquet(source/'risk_beta.parquet');specific=pd.read_parquet(source/'risk_specific.parquet')
        mv=pd.read_csv(source/'market_variance.csv',index_col=0,parse_dates=True).iloc[:,0]
        dates=sorted(decisions.signal_date.unique());chosen={dates[i] for i in np.unique(np.linspace(0,len(dates)-1,20).astype(int))}
        for day,g in decisions.groupby('signal_date'):
            if day not in chosen:continue
            prices=close.loc[day].copy()
            for c in held[day]:
                if not np.isfinite(prices[c]) or prices[c]<=0:prices[c]=marks.loc[day,c]
            weights={c:float(q.loc[day,c]*prices[c]/a.account.loc[day]) for c in held[day]}
            bb=beta.loc[day].to_dict();dd=specific.loc[day].to_dict()
            for row in g.itertuples():
                wo=weights.get(row.old,0.);incoming=row.new if pd.notna(row.new) else None
                pb=sum(w*bb.get(c,1.) for c,w in weights.items())
                change=wo*(bb.get(incoming,1.)-bb.get(row.old,1.))
                delta=((2*(pb-1)*change+change*change)*mv.loc[day]+wo*wo*(dd.get(incoming,1/252)-dd.get(row.old,1/252))) if incoming else 0.
                np.testing.assert_allclose([delta,10*delta/max(wo,1e-20)],[row.risk_variance_change,row.risk_penalty],atol=1e-11,rtol=1e-7)
                risk_checks+=1
                if row.accepted:
                    moved=weights.pop(row.old,0.)
                    if incoming:weights[incoming]=moved
    return dict(regular_frozen_score_and_prior_value_cost_checks=count,independent_marginal_risk_checks=risk_checks)


def audit_calibration(root,data,source):
    from .research_build import read_bin
    s=pd.read_csv(source/'mature_label_slopes.csv',index_col=0,parse_dates=True)
    s['label_exit']=pd.to_datetime(s.label_exit)
    schedule=pd.read_csv(source/'calibration_schedule.csv',index_col=0,parse_dates=True)
    schedule['max_label_exit']=pd.to_datetime(schedule.max_label_exit)
    assert (schedule.max_label_exit.dropna()<=schedule.max_label_exit.dropna().index).all()
    forecast=pd.read_parquet(root/'data/frozen_forecast.parquet').iloc[:,0]
    cal=pd.DatetimeIndex(pd.read_csv(data/'calendars/day.txt',header=None)[0]);cal=cal[cal<='2025-12-31']
    rng=np.random.default_rng(20261103);sample=sorted(set([0,len(s)-1,*rng.choice(len(s),18,replace=False)]))
    opens={};audited=0
    for k in sample:
        day=s.index[k];score=forecast.xs(day,level='datetime');z=(score-score.mean())/score.std(ddof=1)
        slot=cal.get_loc(day);assert cal[slot+6]==s.label_exit.iloc[k]
        for c in score.index:
            if c not in opens:opens[c]=read_bin(data,c,'open',cal).astype(float)
        entry=np.array([opens[c][slot+1] for c in score.index]);exit=np.array([opens[c][slot+6] for c in score.index])
        with np.errstate(divide='ignore',invalid='ignore'):y=exit/entry-1
        valid=np.isfinite(y)&np.isfinite(z)&(entry>0)&(exit>0)
        x=z.to_numpy()[valid];y=y[valid];x=x-x.mean();y=y-y.mean()
        np.testing.assert_allclose((x@y)/(x@x),s.slope.iloc[k],rtol=1e-11,atol=1e-14);audited+=1
    eligible=schedule.index[schedule.observations.ge(60)]
    chosen=eligible[np.unique(np.linspace(0,len(eligible)-1,25).astype(int))]
    for day in chosen:
        history=s.loc[(s.label_exit<=day)&s.slope.notna()].tail(126)
        a=history.slope.to_numpy();w=2.**(np.arange(len(a))/60);w/=w.sum();mean=w@a
        e=(a-mean)*w;v=e@e
        for lag in range(1,6):v+=2*(1-lag/6)*(e[lag:]@e[:-lag])
        se=np.sqrt(max(v,0)*len(a)/(len(a)-1))
        np.testing.assert_allclose([mean,se,max(0.,mean-se)],schedule.loc[day,['slope','SE','conservative_slope']].to_numpy(dtype=float),rtol=1e-11,atol=1e-14)
    return dict(sampled_daily_slopes_independently_recomputed=audited,rolling_calibrations_independently_recomputed=len(chosen),
                all_mature_label_exits_at_or_before_signal=True),schedule


def save(root,data,sources,out):
    root,data,out=[Path(p).resolve() for p in [root,data,out]];sources=[Path(p).resolve() for p in sources]
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    cases={}
    for source in sources:
        for p in source.glob('*/daily_report.csv'):
            if p.parent.name in cases:raise ValueError('Duplicate ablation case')
            cases[p.parent.name]=(p.parent,source)
    expected={'baseline_replay','net_edge','risk_net_edge','net_edge_mean','risk_net_edge_mean'}
    if set(cases)!=expected:raise ValueError(f'All five disclosed cases required, found {set(cases)}')
    base_path=root/'results/next_open_3bp/open'
    original=pd.read_csv(base_path/'daily_report.csv',index_col=0,parse_dates=True,float_precision='round_trip')
    benchmark=pd.read_csv(root/'results/next_open_3bp/report/next_open_daily.csv',index_col=0,parse_dates=True).benchmark_return
    bnav=(1+benchmark).cumprod();rows=[];annual=[];daily={};audits={};costseries={};turnseries={}
    calibration_proof,schedule=audit_calibration(root,data,sources[0])
    for source in sources[1:]:
        assert (source/'calibration_schedule.csv').read_bytes()==(sources[0]/'calibration_schedule.csv').read_bytes()
        assert (source/'mature_label_slopes.csv').read_bytes()==(sources[0]/'mature_label_slopes.csv').read_bytes()
    for name,(path,source) in cases.items():
        a=pd.read_csv(path/'daily_report.csv',index_col=0,parse_dates=True,float_precision='round_trip')
        orders=pd.read_parquet(path/'orders.parquet');orders['date']=pd.to_datetime(orders.date)
        net=a['return']-a.cost;nav=a.account/1e8;active=net-benchmark
        z=pd.DataFrame(dict(net_return=net,benchmark_return=benchmark,active_return=active,NAV=nav,benchmark_NAV=bnav,
            relative_NAV=nav/bnav,absolute_DD=drawdown(nav),relative_DD=drawdown(nav/bnav)))
        prior=a.account.shift(1).fillna(1e8)
        slip=(orders.filled_amount*(orders.price-orders.reference_price)*np.where(orders.direction.eq(1),1.,-1.)).groupby(orders.date).sum().reindex(a.index,fill_value=0)/prior
        turn=orders.groupby('date').value.sum().reindex(a.index,fill_value=0)/prior
        costseries[name]=a.cost+slip;turnseries[name]=turn
        years=[]
        for y,g in z.groupby(z.index.year):
            val=ir(g.active_return);years.append(val);annual.append(dict(case=name,year=y,net_excess_IR=val,
                portfolio_return=(1+g.net_return).prod()-1,benchmark_return=(1+g.benchmark_return).prod()-1))
        rows.append(dict(case=name,mean_annual_excess_IR=np.mean(years),pooled_excess_IR=ir(active),CAGR=nav.iloc[-1]**.1-1,
            absolute_max_DD=z.absolute_DD.min(),relative_max_DD=z.relative_DD.min(),realized_TE=active.std(ddof=1)*np.sqrt(252),
            mean_stock_exposure=(a.value/a.account).mean(),annual_two_way_turnover=turn.sum()/10,
            annual_commission_fraction=a.cost.sum()/10,annual_slippage_fraction=slip.sum()/10,
            annual_arithmetic_gross_active=(active+a.cost+slip).mean()*252,annual_arithmetic_net_active=active.mean()*252))
        daily[name]=z
        if name=='baseline_replay':
            np.testing.assert_allclose(a.account,original.account,rtol=1e-12,atol=1e-5)
        else:
            decisions=pd.read_parquet(path/'gate_decisions.parquet');decisions['signal_date']=pd.to_datetime(decisions.signal_date)
            decisions['trade_date']=pd.to_datetime(decisions.trade_date);decisions['max_label_exit']=pd.to_datetime(decisions.max_label_exit)
            assert (decisions.max_label_exit<=decisions.signal_date).all()
            regular=decisions.loc[~decisions.forced]
            fit=schedule.reindex(regular.signal_date)
            coefficient=np.maximum(fit.slope.to_numpy(),0) if name.endswith('_mean') else fit.conservative_slope.to_numpy()
            np.testing.assert_allclose(regular.expected_edge,coefficient*(regular.z_new-regular.z_old),atol=1e-13,rtol=1e-10)
            np.testing.assert_allclose(regular.marginal_utility,regular.expected_edge-regular.cost_fraction-regular.risk_penalty,atol=1e-13)
            assert (regular.accepted.to_numpy()==regular.marginal_utility.gt(0).to_numpy()).all()
            first=decisions.trade_date.min();warm=a.index<first
            np.testing.assert_allclose(a.account.loc[warm],original.account.loc[warm],atol=1e-5,rtol=1e-12)
            audits[name]=dict(gate_records=len(decisions),regular_proposals=len(regular),regular_accepted=int(regular.accepted.sum()),
                forced_proposals=int(decisions.forced.sum()),past_only_calibration_verified=True,warmup_identical=True,
                decision_utility_reconciles=True)
    metrics=pd.DataFrame(rows).set_index('case').sort_index();base=metrics.loc['baseline_replay']
    criteria=(metrics.CAGR>=base.CAGR)&(metrics.mean_annual_excess_IR>=max(1.3,base.mean_annual_excess_IR))&(metrics.pooled_excess_IR>=base.pooled_excess_IR)&(metrics.relative_max_DD>base.relative_max_DD)
    eligible=metrics.index[criteria].tolist()
    if not eligible:raise ValueError('No candidate meets joint improvement criteria')
    selected=max(eligible,key=lambda c:metrics.loc[c,'relative_max_DD'])
    account_path,source=cases[selected]
    a=pd.read_csv(account_path/'daily_report.csv',index_col=0,parse_dates=True,float_precision='round_trip')
    audits[selected]['independent_quantity_ledger']=independent_audit(a,pd.read_parquet(account_path/'orders.parquet'),data)
    audits[selected].update(audit_score_and_cost(root,data,account_path,source))
    audits['baseline_replay']=dict(exactly_matches_published_account=True,independent_quantity_ledger=independent_audit(original,pd.read_parquet(base_path/'orders.parquet'),data))
    account=out/'account';shutil.copytree(account_path,account)
    for f in ['calibration_schedule.csv','mature_label_slopes.csv']:shutil.copyfile(source/f,out/f)
    pd.DataFrame(annual).to_csv(out/'annual.csv',index=False);metrics.to_csv(out/'all_ablation_metrics.csv')
    for name in ['baseline_replay',selected]:daily[name].to_csv(out/f'{name}_daily.csv',index_label='date')
    pd.concat([episodes(daily[name].relative_NAV).assign(case=name) for name in ['baseline_replay',selected]],ignore_index=True).to_csv(out/'relative_drawdown_episodes.csv',index=False)
    audit=dict(calibration=calibration_proof,case_audits=audits,all_five_variants_disclosed=True,
        independent_holdout_claimed=False,benchmark_first_open_approximation_unchanged=True,
        causal_open_exchange_and_3bp_account_validation_passed=True)
    (out/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    selection=dict(selected_case=selected,selection_period='2016-2025 reused development',acceptance_criteria=
        ['CAGR at least baseline','mean annual excess IR at least baseline and 1.3','pooled excess IR at least baseline','smaller maximum relative drawdown'],
        frozen_model_and_forecast_unchanged=True,portfolio_rule='original daily Top50/Drop5; positive calibrated marginal edge net of costs',
        calibration=dict(window=126,minimum_days=60,half_life=60,target='t+1 to t+6 open return; cross-sectional demeaned',applied_coefficient='max(0, past rolling mean slope)'),
        automatic_factor_risk_penalty_enabled=selected.startswith('risk_'),uncertainty_haircut_enabled=False,paper_account_changed=False,
        tie_break='smallest maximum relative drawdown among candidates meeting every acceptance criterion',
        future_returns_used_for_current_decision=False,independent_out_of_sample_proof=False,
        forecast_sha256=hashlib.sha256((root/'data/frozen_forecast.parquet').read_bytes()).hexdigest())
    (out/'selected_case.json').write_text(json.dumps(selection,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    from .research_math import circular_bootstrap_mean
    aa=daily[selected].active_return.to_numpy();bb=daily['baseline_replay'].active_return.to_numpy();n=len(aa)
    moments=circular_bootstrap_mean(np.column_stack([aa,aa*aa,bb,bb*bb]),20,4999,20261103)
    def from_mom(m,v):return m/np.sqrt(np.maximum((v-m*m)*n/(n-1),1e-30))*np.sqrt(252)
    difference=from_mom(moments[:,0],moments[:,1])-from_mom(moments[:,2],moments[:,3]);ci=np.quantile(difference,[.025,.975])
    uncertainty=dict(pooled_IR_difference=float(metrics.loc[selected,'pooled_excess_IR']-base.pooled_excess_IR),
        paired_20day_block_bootstrap_95pct=[float(ci[0]),float(ci[1])],replications=4999,seed=20261103,selection_bias_corrected=False)
    (out/'uncertainty.json').write_text(json.dumps(uncertainty,indent=2)+'\n',encoding='utf-8',newline='\n')
    plot_and_write(metrics,daily,selected,audit,uncertainty,costseries,turnseries,out)
    print(metrics.to_string(),flush=True)


def plot_and_write(metrics,daily,selected,audit,uncertainty,costseries,turnseries,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels={'baseline_replay':'Original daily Top50/Drop5',selected:'Calibrated risk / cost utility gate'}
    colors={'baseline_replay':'#8b8b8b',selected:'#167e8d'}
    fig,axes=plt.subplots(3,1,figsize=(12,10),sharex=True)
    for c in labels:
        d=daily[c];axes[0].plot(d.index,d.NAV,label=labels[c],color=colors[c]);axes[1].plot(d.index,d.relative_NAV,color=colors[c]);axes[2].plot(d.index,d.relative_DD*100,color=colors[c])
    axes[0].plot(d.index,d.benchmark_NAV,label='CSI300 total return',color='#333',ls='--');axes[0].set_yscale('log');axes[0].legend(fontsize=8)
    for ax,t in zip(axes,['Net NAV (log)','NAV / benchmark','Relative drawdown (%)']):ax.set_ylabel(t);ax.grid(alpha=.2)
    axes[0].set_title('2016–2025 reused development | same frozen alpha | next open | 3bp each side')
    fig.tight_layout();fig.savefig(out/'comparison.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(12,9),sharex=True)
    for c in labels:
        d=daily[c];axes[0].plot(d.index,costseries[c].rolling(252,min_periods=126).sum()*100,color=colors[c],label=labels[c])
        axes[1].plot(d.index,turnseries[c].rolling(252,min_periods=126).sum(),color=colors[c])
        axes[2].plot(d.index,d.active_return.rolling(60,min_periods=40).std(ddof=1)*np.sqrt(252)*100,color=colors[c])
    axes[0].legend(fontsize=8)
    for ax,t in zip(axes,['Trailing 252d cost sum (%)','Trailing 252d two-way turnover','Actual 60d active TE (%)']):ax.set_ylabel(t);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'cost_and_risk.png',dpi=150);plt.close(fig)
    b=metrics.loc['baseline_replay'];s=metrics.loc[selected]
    lines=['# 五日收益校准与边际净效用换股','',
        '**该候选在2016—2025开发回测中同时改善收益、超额IR和最大相对回撤。** 原LightGBM、历史股票池、日频Top50/Drop5候选机制、买入预算、下一日开盘执行及每边3bp滑点保持原样；仅增加换股净收益门槛。没有降低股票总仓位，也没有使用未来未成熟标签。','',
        '| 指标 | 原日频模型 | 边际净收益候选 |','| --- | ---: | ---: |']
    for key,title,fmt in [('mean_annual_excess_IR','平均年度超额IR','.3f'),('pooled_excess_IR','全期超额IR','.3f'),('CAGR','年化收益','.2%'),('relative_max_DD','最大相对回撤','.2%'),('absolute_max_DD','最大绝对回撤','.2%'),('realized_TE','实现跟踪误差','.2%'),('mean_stock_exposure','平均股票仓位','.2%'),('annual_two_way_turnover','年均双边换手','.2f')]:
        lines.append(f'| {title} | {format(b[key],fmt)} | {format(s[key],fmt)} |')
    lines+=['','![净值和相对回撤](comparison.png)','','## 一、为什么这次保留原选股机制','',
        '此前周频风险预算优化同时改变了选股实现、频率和候选准入，未通过整体检验。本轮保留原模型每次最多替换5只股票的机制，只决定一笔候选换股是否值得支付成本。关闭门槛的完整账户已与原发布账户逐日复现一致。','',
        '原规则可能在预测分数差很小时仍然换仓。分数不是收益单位，因此先用过去已成熟的样本校准“分数差对应多少五日收益差”，再与费用和滑点比较。该方法借鉴[Boyd等的交易成本与风险统一决策框架](https://web.stanford.edu/~boyd/papers/cvx_portfolio.html)，是原候选换股集上的局部净收益选择，不宣称全市场全局最优。','',
        '## 二、从分数到可比较的收益单位','',
        '每日将当期股票分数横截面标准化为z。标签为次日开盘至第六日开盘的五日股票收益，去除该信号日股票横截面均值。逐日用带截距OLS估计该标签对z的斜率；每天至少100对有效样本。',
        '', '信号日t只使用退出开盘已经发生、退出日≤t的历史标签。对最近126个有效斜率按60日半衰期加权，得到非负均值系数；至少60个成熟信号日后才启用门槛，之前完全使用原规则。也记录HAC五阶标准误作为估计不确定性，最终候选不额外扣除一倍标准误。','',
        '```text','预期五日换股收益差 = max(0, 历史加权斜率均值) × (z新股票 − z旧股票)',
        '换股成本门槛 = 卖出费用 + 买入费用 + 两边滑点',
        '主动风险扣减 = (风险厌恶4 / 2) × 5 × 换股引起的日主动方差变化 / 被替换持仓权重',
        '普通候选换股仅在：预期收益差 − 成本门槛 − 主动风险扣减 > 0 时执行','```','',
        '通常费用为卖出15bp、买入5bp，两边滑点合计6bp，共26bp；最低5元费用按旧持仓计划价值折算。预测成本是计划近似，真实费用仍逐笔扣除，滑点进入成交价而不重复收费。退出股票池、修正持仓数量等必要处置单独记录，已有空缺席位仍按原规则补齐。','',
        '## 三、成本、风险与全部方法比较','',
        '![换手、成本与实现主动波动](cost_and_risk.png)','',
        '| 方案 | 平均年度超额IR | CAGR | 最大相对回撤 |','| --- | ---: | ---: | ---: |']
    names={'baseline_replay':'原规则精确重放','net_edge':'一倍标准误折减＋净收益门槛','risk_net_edge':'一倍标准误折减＋风险扣减','net_edge_mean':'历史均值＋净收益门槛','risk_net_edge_mean':'历史均值＋风险扣减（选中）'}
    for c,r in metrics.iterrows():lines.append(f'| {names[c]} | {r.mean_annual_excess_IR:.3f} | {r.CAGR:.2%} | {r.relative_max_DD:.2%} |')
    lines+=['','风险扣减使用未收缩基准beta的条件单市场＋对角残余模型，比较换股前后五日主动方差，风险厌恶系数固定4。它假设残余独立，会遗漏行业共振；最终版本启用该连续风险扣减，未启用前一失败候选的10%TE或beta硬约束。降低风险的换股可以得到正向效用，不等于无条件减少股票仓位。在同时达到基线收益、两类IR和相对回撤要求的版本中，选择最大相对回撤最小的一组；全部变体仍披露。','',
        '## 四、检查与不确定性','',
        f"- 独立重算{audit['calibration']['sampled_daily_slopes_independently_recomputed']}个横截面原始开盘收益斜率，以及{audit['calibration']['rolling_calibrations_independently_recomputed']}个滚动校准点。所有可用标签的退出日期不晚于信号日。",
        '- 已核对门槛效用计算、接受/拒绝、成熟时点、与原规则完全相同的预热期、逐笔费用/滑点、现金、量额容量和数量账簿。',
        f"- 新账户2,430天独立账簿最大金额差异为{audit['case_audits'][selected]['independent_quantity_ledger']['maximum_absolute_account_difference_CNY']:.9f}元。",
        f"- 全期IR差值为{uncertainty['pooled_IR_difference']:+.3f}；20日配对圆形块Bootstrap、4,999次的描述性95%区间为[{uncertainty['paired_20day_block_bootstrap_95pct'][0]:+.3f}, {uncertainty['paired_20day_block_bootstrap_95pct'][1]:+.3f}]。",'',
        '改进幅度有限，区间若覆盖零不能声称统计显著。四个门槛变体及此前风险预算试验都使用了开发区间，存在选择偏差，Bootstrap没有校正这些选择；尚未获得独立样本外证明。没有未来未成熟标签，不等于没有过拟合风险。','',
        '原Qlib复权单位账簿、同次执行先卖后买、9.5%简化方向限制和前20日量额5%容量代理继续保留；没有重放集合竞价冻结资金、订单队列或动态冲击。基准仍为H00300全收益，首日开盘采用原实验的近似，所有比较保持同口径。','',
        '## 五、复现','', '```bash',
        'python -m src.marginal_risk_gate --data-dir /path/to/qlib_data --output runs/marginal-gate',
        'python -m src.marginal_gate_report --data-dir /path/to/qlib_data --source runs/marginal-gate --output runs/marginal-gate-report',
        '```','', '保存的account为选中候选的完整账户，原模型及原账户继续保留。模拟盘未自动切换。','']
    (out/'README.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data-dir',type=Path,required=True);p.add_argument('--source',type=Path,action='append',required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    save(Path(__file__).resolve().parents[1],args.data_dir,args.source,args.output)
