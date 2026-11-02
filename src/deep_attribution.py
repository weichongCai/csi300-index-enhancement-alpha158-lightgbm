"""Inventory-exact realized attribution, with a small cross-sectional style proxy.

No forecast, portfolio, or trading parameter is changed. Factor returns are ex post.
"""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import pandas as pd
from .research_build import read_bin
from .risk_attribution import variance_attribution

STYLES=['volatility_60','momentum_252_21','log_mean_amount_20']
NAMES={'market':'历史beta×市场','benchmark_gap':'价格/全收益基准差','common':'横截面共同项','volatility_60':'波动风格','momentum_252_21':'动量风格',
       'log_mean_amount_20':'流动性风格','covered_residual':'已覆盖股票残余','uncovered':'未覆盖持仓损益',
       'buy_increment':'买入当日日内增量','sell_increment':'卖出当日日内增量','commission':'基础费用','slippage':'滑点'}


def cross_section(frame,returns,market):
    x=frame[['beta',*STYLES]].copy().replace([np.inf,-np.inf],np.nan)
    for f in STYLES:x[f]=(x[f]-x[f].mean())/x[f].std(ddof=1)
    x=x.join(returns.rename('r')).dropna()
    x=x.loc[frame.beta_known.reindex(x.index).eq(1)]
    if len(x)<100:return None
    design=np.column_stack([np.ones(len(x)),x[STYLES]])
    condition=np.linalg.cond(design)
    if not np.isfinite(condition) or condition>1e6:return None
    y=x.r.to_numpy()-x.beta.to_numpy()*market
    coef=np.linalg.lstsq(design,y,rcond=None)[0]
    residual=y-design@coef
    return x,coef,residual,condition


def build(root,data,out):
    root,data,out=map(Path,[root,data,out])
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    old=root/'results/risk_attribution'
    d=pd.read_csv(old/'daily_attribution.csv',index_col=0,parse_dates=True,float_precision='round_trip')
    o=pd.read_parquet(root/'results/next_open_3bp/open/orders.parquet');o['date']=pd.to_datetime(o.date)
    o['signed']=o.filled_amount*np.where(o.direction.eq(1),1.,-1.)
    dq=o.pivot_table(index='date',columns='instrument',values='signed',aggfunc='sum').reindex(d.index).fillna(0.)
    q=dq.cumsum();qp=q.shift(1).fillna(0.)
    calendar=pd.DatetimeIndex(pd.read_csv(data/'calendars/day.txt',header=None)[0]);calendar=calendar[calendar<='2025-12-31']
    panel=pd.read_parquet(root/'results/research/factors/risk_proxy_panel.parquet')
    codes=sorted(set(panel.index.get_level_values('instrument'))|set(q.columns))
    pos=calendar.get_indexer(d.index)
    close=pd.DataFrame({c:read_bin(data,c,'close',calendar).astype(float)[pos] for c in codes},index=d.index)
    op=pd.DataFrame({c:read_bin(data,c,'open',calendar).astype(float)[pos] for c in q.columns},index=d.index)
    marks=close[q.columns].where(op.gt(0)&np.isfinite(op)).ffill()
    prev=marks.shift(1);effective=op.where(op.gt(0)&np.isfinite(op),prev)
    carry=(qp*(marks-prev)).where(qp.abs().gt(1e-6),0.).fillna(0.)
    increment=(dq*(marks-effective)).where(dq.abs().gt(1e-6),0.).fillna(0.)
    report=pd.read_csv(root/'results/next_open_3bp/open/daily_report.csv',index_col=0,parse_dates=True,float_precision='round_trip')
    assets=report.account.shift(1).fillna(1e8)
    carry=carry.div(assets,axis=0);increment=increment.div(assets,axis=0)
    np.testing.assert_allclose(carry.sum(axis=1)+increment.sum(axis=1),d.gross_return,atol=1e-10)
    market_close=read_bin(data,'SH000300','close',calendar).astype(float)
    market_return=pd.Series(market_close,index=calendar).pct_change(fill_method=None).reindex(d.index)
    returns=close.pct_change(fill_method=None)
    parts=pd.DataFrame(0.,index=d.index,columns=NAMES)
    parts['buy_increment']=increment.where(dq.gt(0),0.).sum(axis=1)
    parts['sell_increment']=increment.where(dq.lt(0),0.).sum(axis=1)
    parts['commission']=d.commission;parts['slippage']=d.slippage
    fitstats=[];exposures=[]
    for i,day in enumerate(d.index):
        before=d.index[i-1] if i else None
        fit=None
        if before is not None and before in panel.index.get_level_values('datetime'):
            fit=cross_section(panel.xs(before,level='datetime'),returns.loc[day],market_return.loc[day])
        if fit is None:
            parts.loc[day,'uncovered']=carry.loc[day].sum()
            fitstats.append({'date':day,'signal_date':before,'pairs':0,'condition':np.nan,'weight_coverage':0.,'gross_carry_covered':0.})
            continue
        x,coef,residual,condition=fit
        # Carry belongs to inventory present BEFORE opening orders.
        w=(qp.loc[day]*prev.loc[day]/assets.loc[day]).fillna(0.)
        stock_total=w.sum();covered=x.index.intersection(w.index)
        wc=w.reindex(x.index).fillna(0.)
        pieces=pd.DataFrame(index=x.index)
        pieces['market']=wc*x.beta*market_return.loc[day]
        pieces['common']=wc*coef[0]
        for j,f in enumerate(STYLES):pieces[f]=wc*x[f]*coef[j+1]
        pieces['covered_residual']=wc*residual
        # Unavailable account marks or stock returns cannot be silently zero-filled.
        actual=carry.loc[day].reindex(x.index).fillna(0.)
        mark_bridge=actual-pieces.sum(axis=1)
        parts.loc[day,list(pieces)]=pieces.sum(axis=0)
        parts.loc[day,'uncovered']=carry.loc[day].drop(covered).sum()+mark_bridge.sum()
        fitted=x.beta.to_numpy()*market_return.loc[day]+np.column_stack([np.ones(len(x)),x[STYLES]])@coef
        r2=1-np.var(x.r.to_numpy()-fitted)/np.var(x.r.to_numpy()) if x.r.var()>0 else np.nan
        fitstats.append({'date':day,'signal_date':before,'pairs':len(x),'condition':condition,
                         'weight_coverage':float(wc.sum()/stock_total) if stock_total>0 else np.nan,
                         'gross_carry_covered':actual.sum(),'cross_section_R2':r2,
                         'mean_beta_covered':float((wc*x.beta).sum()/wc.sum()) if wc.sum()>0 else np.nan,
                         'max_mark_bridge_abs':float(mark_bridge.abs().max()),
                         'market_price_return':market_return.loc[day],
                         **{f'return_{f}':coef[j] for j,f in enumerate(['common',*STYLES])}})
        exposures.append({'date':day,**{f:float((wc*x[f]).sum()) for f in ['beta',*STYLES]},'covered_weight':wc.sum()})
    np.testing.assert_allclose(parts.sum(axis=1),d.model_return,atol=1e-10)
    parts.to_csv(out/'daily_components.csv',index_label='date',lineterminator='\n')
    fitframe=pd.DataFrame(fitstats).set_index('date');fitframe['market_price_return']=market_return
    fitframe.to_csv(out/'factor_fit_and_coverage.csv',lineterminator='\n')
    pd.DataFrame(exposures).set_index('date').to_csv(out/'covered_inventory_exposures.csv',lineterminator='\n')
    # Long-form transaction table records actual one-day effects, not future holding counterfactuals.
    o['same_day_gross_increment']=o.signed*(marks.stack().reindex(pd.MultiIndex.from_frame(o[['date','instrument']])).to_numpy()-o.reference_price)
    o['slippage_loss']=o.signed*(o.price-o.reference_price)
    o['same_day_increment_net_of_trade_cost']=o.same_day_gross_increment-o.cost-o.slippage_loss
    o.to_parquet(out/'trade_diagnostics.parquet',index=False)
    pricepaths=[data/'features'/c.lower()/f'{f}.day.bin' for c in codes for f in ['close']]
    pricepaths += [data/'features'/c.lower()/'open.day.bin' for c in q.columns]
    sources=[old/'daily_attribution.csv',old/'stock_net_return_contributions.parquet',
             root/'results/next_open_3bp/open/orders.parquet',root/'results/next_open_3bp/open/daily_report.csv',
             root/'results/research/factors/risk_proxy_panel.parquet']
    proof={'source_sha256':{p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
           'raw_price_sha256':{p.relative_to(data).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in pricepaths if p.exists()},
           'input_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()},
           'daily_max_abs_return_reconciliation_error':float(abs(parts.sum(axis=1)-d.model_return).max()),
           'model_changed':False,'benchmark_price_model':'SH000300; performance H00300 total return',
           'factor_returns_are_ex_post':True,'fit':'equal-weight OLS of beta-adjusted stock close-to-close return; three standardized style proxies + intercept; beta_known required; min100 complete names; condition<=1e6',
           'missing_style_or_fit':'actual uncovered inventory PnL retained explicitly; no imputation or renormalization'}
    (out/'provenance.json').write_text(json.dumps(proof,indent=2)+'\n',encoding='utf-8',newline='\n')
    return generate(root,out,out)


def generate(root,inputs,out):
    root,inputs,out=map(Path,[root,inputs,out])
    proof=json.loads((inputs/'provenance.json').read_text())
    for name,digest in proof['input_sha256'].items():assert hashlib.sha256((inputs/name).read_bytes()).hexdigest()==digest,name
    for name,digest in proof['source_sha256'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
    if out!=inputs:out.mkdir(parents=True,exist_ok=False)
    parts=pd.read_csv(inputs/'daily_components.csv',index_col=0,parse_dates=True,float_precision='round_trip')
    d=pd.read_csv(root/'results/risk_attribution/daily_attribution.csv',index_col=0,parse_dates=True,float_precision='round_trip')
    fit=pd.read_csv(inputs/'factor_fit_and_coverage.csv',index_col=0,parse_dates=True)
    exposure=pd.read_csv(inputs/'covered_inventory_exposures.csv',index_col=0,parse_dates=True)
    trade=pd.read_parquet(inputs/'trade_diagnostics.parquet')
    stock=pd.read_parquet(root/'results/risk_attribution/stock_net_return_contributions.parquet')
    windows=[('2022下跌','2021-12-31','2022-04-26'),('2025落后','2025-05-30','2025-09-03'),
             ('2023—2024落后','2023-07-11','2024-02-05'),('2022—2024修复','2022-04-26','2024-09-30')]
    summaries=[];monthly=[];names=[];tails=[];executions=[]
    for label,peak,end in windows:
        mask=(d.index>peak)&(d.index<=end);g=d.loc[mask];c=parts.loc[mask];f=fit.loc[mask]
        w=d.model_nav.shift(1).loc[mask]/d.model_nav.loc[peak]
        wr=d.relative_nav.shift(1).loc[mask]/d.relative_nav.loc[peak]/(1+g.benchmark_return)
        absolute=c.mul(w,axis=0).sum()
        relative=c.copy();relative['market']-=f.market_price_return
        relative['benchmark_gap']=f.market_price_return-g.benchmark_return
        relative=relative.mul(wr,axis=0).sum()
        p_return=d.model_nav.loc[end]/d.model_nav.loc[peak]-1
        b_return=d.benchmark_nav.loc[end]/d.benchmark_nav.loc[peak]-1
        rel_return=(1+p_return)/(1+b_return)-1
        np.testing.assert_allclose(absolute.sum(),p_return,atol=1e-10)
        np.testing.assert_allclose(relative.sum(),rel_return,atol=1e-10)
        summaries.append({'window':label,'peak':peak,'end':end,'days':len(g),'portfolio_return':p_return,
                          'benchmark_return':b_return,'relative_return':rel_return,
                          'mean_coverage':f.weight_coverage.mean(),'minimum_coverage':f.weight_coverage.min(),
                          'days_coverage_below_98pct':int(f.weight_coverage.lt(.98).sum()),
                          'fit_pairs_median':f.pairs.median(),'fit_condition_max':f.condition.max(),
                          **{'mean_covered_'+k:exposure.reindex(g.index)[k].div(exposure.reindex(g.index).covered_weight).mean() for k in ['beta',*STYLES]},
                          **{'abs_'+k:absolute[k] for k in NAMES},**{'rel_'+k:relative[k] for k in NAMES}})
        for month,idx in g.groupby(g.index.to_period('M')).groups.items():
            z=g.loc[idx];pa=(1+z.model_return).prod()-1;ba=(1+z.benchmark_return).prod()-1
            linked_abs=c.loc[idx].mul(w.loc[idx],axis=0).sum()
            rr=c.loc[idx].copy();rr['market']-=f.loc[idx,'market_price_return']
            rr['benchmark_gap']=f.loc[idx,'market_price_return']-z.benchmark_return
            linked_rel=rr.mul(wr.loc[idx],axis=0).sum()
            monthly.append({'window':label,'month':str(month),'portfolio_return':pa,'benchmark_return':ba,
                            'relative_return':(1+pa)/(1+ba)-1,
                            **{'abs_'+k:linked_abs[k] for k in NAMES},**{'rel_'+k:linked_rel[k] for k in NAMES}})
        contribution=stock.loc[mask].mul(w,axis=0).sum()
        for code,value in contribution.items():names.append({'window':label,'instrument':code,'portfolio_PnL_contribution':value})
        for mode,criterion in [('absolute','model_return'),('active','active_return')]:
            for day,row in g.nsmallest(10,criterion).iterrows():
                tails.append({'window':label,'sort':mode,'date':str(day.date()),'net_return':row.model_return,
                              'benchmark_return':row.benchmark_return,'arithmetic_active':row.active_return,
                              'linked_abs_loss':row.model_return*w.loc[day],
                              **{k:parts.loc[day,k] for k in NAMES}})
        t=trade.loc[(trade.date>peak)&(trade.date<=end)].copy()
        assets=pd.read_csv(root/'results/next_open_3bp/open/daily_report.csv',index_col=0,parse_dates=True).account.shift(1).fillna(1e8)
        t['scale']=t.date.map(w/assets.loc[mask])
        for direction in [1,0]:
            z=t.loc[(t.direction==direction)&t.value.gt(0)]
            executions.append({'window':label,'direction':direction,'fills':len(z),'notional_CNY':z.value.sum(),
                               'same_day_increment_positive_fraction':z.same_day_gross_increment.gt(0).mean(),
                               'linked_gross_increment':(z.same_day_gross_increment*z.scale).sum(),
                               'linked_commission':-(z.cost*z.scale).sum(),
                               'linked_slippage':-(z.slippage_loss*z.scale).sum()})
    summary=pd.DataFrame(summaries);month=pd.DataFrame(monthly);stocks=pd.DataFrame(names);tail=pd.DataFrame(tails);execution=pd.DataFrame(executions)
    for name,frame in [('window_attribution',summary),('monthly_attribution',month),('stock_contributions',stocks),('tail_days',tail),('execution_summary',execution)]:
        frame.to_csv(out/f'{name}.csv',index=False,lineterminator='\n')
    risks=[]
    for target in ['absolute','active']:
        c=parts.iloc[1:].copy();total=d.model_return.iloc[1:]
        if target=='active':
            c['market']-=fit.market_price_return.iloc[1:]
            c['benchmark_gap']=fit.market_price_return.iloc[1:]-d.benchmark_return.iloc[1:]
            total=total-d.benchmark_return.iloc[1:]
        risk=variance_attribution(c,total);risk['target']=target;risks.append(risk)
    pd.concat(risks).to_csv(out/'realized_risk_contributions.csv',index=False,lineterminator='\n')
    plot(d,parts,summary,month,stocks,fit,out)
    write_report(summary,month,stocks,tail,execution,fit,out)
    print(summary[['window','portfolio_return','relative_return','mean_coverage']].to_string(index=False),flush=True)


def plot(d,parts,summary,month,stocks,fit,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','DejaVu Sans'],'axes.unicode_minus':False})
    for i in [0,1]:
        row=summary.iloc[i];mask=(d.index>row.peak)&(d.index<=row.end);g=d.loc[mask]
        curve=d.loc[(d.index>=row.peak)&(d.index<=row.end)]
        fig,axes=plt.subplots(3,1,figsize=(13,10),gridspec_kw={'height_ratios':[1.5,1,1]})
        axes[0].plot(curve.index,curve.model_nav/d.model_nav.loc[row.peak],label='策略',color='#176a91')
        axes[0].plot(curve.index,curve.benchmark_nav/d.benchmark_nav.loc[row.peak],label='沪深300全收益',color='#999')
        axes[0].legend();axes[0].set_title(row['window']+'：区间起点净值=1');axes[0].grid(alpha=.2)
        prefix='abs_' if i==0 else 'rel_'
        vals=np.array([row[prefix+k] for k in NAMES])*100
        axes[1].bar(range(len(NAMES)),vals,color=['#176a91' if v>=0 else '#b65450' for v in vals])
        axes[1].set_xticks(range(len(NAMES)),list(NAMES.values()),rotation=25,ha='right',fontsize=8)
        axes[1].set_ylabel('绝对贡献（百分点）' if i==0 else '相对贡献（百分点）')
        for j,v in enumerate(vals):axes[1].text(j,v,f'{v:+.2f}',ha='center',va='bottom' if v>=0 else 'top',fontsize=8)
        z=month[month.window==row['window']];keys=list(NAMES)
        heat=z[[prefix+k for k in keys]].to_numpy()*100
        limit=max(abs(heat).max(),.1)
        im=axes[2].imshow(heat.T,aspect='auto',cmap='RdBu',vmin=-limit,vmax=limit)
        axes[2].set_yticks(range(len(keys)),list(NAMES.values()),fontsize=8)
        axes[2].set_xticks(range(len(z)),z.month,rotation=30)
        fig.colorbar(im,ax=axes[2],label='对整段回撤的链接贡献（百分点）')
        fig.tight_layout();fig.savefig(out/f'episode_{2022 if i==0 else 2025}.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(13,8))
    for ax,label in zip(axes,summary.window.iloc[:2]):
        s=stocks[stocks.window==label].sort_values('portfolio_PnL_contribution')
        z=pd.concat([s.head(10),s.tail(5)])
        ax.bar(z.instrument,z.portfolio_PnL_contribution*100,color=['#b65450' if v<0 else '#228371' for v in z.portfolio_PnL_contribution])
        ax.tick_params(axis='x',rotation=40);ax.set_ylabel('组合损益贡献（百分点）');ax.set_title(label+'：最差10只与最好5只（不是个股主动Alpha）');ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(out/'stock_winners_losers.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(13,7),sharex=True)
    axes[0].plot(fit.index,fit.weight_coverage,color='#176a91');axes[0].axhline(.98,color='#999',ls='--');axes[0].set_ylabel('前收盘持仓覆盖率')
    axes[1].plot(fit.index,fit.cross_section_R2.rolling(60,min_periods=40).mean(),label='横截面R²的60日均值')
    axes[1].plot(fit.index,fit.condition.rolling(60,min_periods=40).mean()/10,label='条件数60日均值 / 10')
    axes[1].legend();axes[0].set_title('代理归因的数据覆盖与回归稳定性；未覆盖损益单独保留')
    for ax in axes:ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'coverage_and_fit.png',dpi=150);plt.close(fig)


def write_report(summary,month,stocks,tail,execution,fit,out):
    lines=['# 持仓、风格、交易与回撤的详细归因','','本报告进一步解释上一版“非市场毛分量”。对象仍为2016—2025冻结LightGBM、t+1开盘、每边3bp滑点的同一账户；没有修改模型、交易、持仓或净值。','',
           '## 先说明：哪些是账目事实，哪些依赖模型','',
           '- **账目恒等式**：逐笔交易、逐股损益、费用、滑点、买卖当日增量和峰谷回撤，可与原账户精确核对。',
           '- **代理解释**：历史beta、波动、动量、流动性及横截面共同项来自简化归因模型。它们解释同一损益的不同部分，数值依赖所选因子和拟合方法。',
           '- **仍无法识别**：真实历史指数权重下的行业配置/选股效应、规模/价值/盈利风格，以及完整Barra特质风险。没有用现在行业分类冒充历史分类，也不将未解释项称为纯Alpha。','',
           '## 一、损益分解是怎样算出来的','',
           '对每只股票和每一天，以复权单位计算：','',
           '```text','旧持仓收益 = 昨日股数 ×（今日收盘估值 − 昨日收盘估值）',
           '买卖当日增量 = 股数变动 ×（今日收盘估值 − 今日开盘参考价）',
           '实际净损益 = 旧持仓收益 + 买入当日增量 + 卖出当日增量 − 基础费用 − 滑点',
           '```','',
           '这与“旧持仓隔夜损益 + 开盘后持仓日内损益 − 成本”完全等价。买入当日增量为正表示新增股份当日赚到钱；卖出当日增量为负表示卖出的股份后来当日上涨，形成机会损失。它只比较当天，不证明不换仓长期更好，不能累计成独立可交易策略。','',
           '再将旧持仓收益拆为历史beta对应的市场分量、横截面共同项、波动风格、动量风格、流动性风格、已覆盖股票残余和未覆盖持仓损益。开盘换仓增量独立保留，避免用收盘新持仓权重倒推全天风险。','',
           '## 二、逐段贡献明细','',
           '| 分量 | 2022峰谷：绝对贡献 | 2025峰谷：相对贡献 |','| --- | ---: | ---: |']
    for k,label in NAMES.items():lines.append(f'| {label} | {summary.iloc[0]["abs_"+k]*100:+.3f}个百分点 | {summary.iloc[1]["rel_"+k]*100:+.3f}个百分点 |')
    a,b=summary.iloc[0],summary.iloc[1]
    lines+=['',f'合计分别为{a.portfolio_return:.2%}和{b.relative_return:.2%}。绝对贡献与相对贡献分母不同，不横向直接比较。相对市场行按价格指数衡量beta偏离；价格指数与全收益基准的差异另列，不混进选股或beta失配。',
            '', '### 2022年：系统性下跌为主，换仓和成本加深回撤','',
            f'市场分量为{a.abs_market*100:.2f}个百分点，共同项{a.abs_common*100:.2f}个百分点。三类风格合计{(a.abs_volatility_60+a.abs_momentum_252_21+a.abs_log_mean_amount_20)*100:+.2f}个百分点：波动风格拖累，被动量与流动性分量部分抵消。已覆盖股票残余贡献{a.abs_covered_residual*100:+.2f}个百分点。因此，这个代理模型不支持把整次回撤简单归结为“模型个股选择全面失效”。',
            '', f'开盘买入新增股份当日贡献{a.abs_buy_increment*100:.2f}个百分点，卖出避免部分后续当日下跌、贡献{a.abs_sell_increment*100:+.2f}个百分点；两者合计{(a.abs_buy_increment+a.abs_sell_increment)*100:.2f}个百分点，再叠加{(a.abs_commission+a.abs_slippage)*100:.2f}个百分点交易成本。卖出在当天起到了部分防御作用，买入和成本抵消了这部分收益。',
            '', '### 2025年：低敏感度、低波动暴露和交易拖累同时出现','',
            f'相对回撤中，按价格指数衡量的市场敏感度偏离贡献{b.rel_market*100:.2f}个百分点，价格/全收益基准差贡献{b.rel_benchmark_gap*100:.2f}个百分点，波动风格贡献{b.rel_volatility_60*100:.2f}个百分点，已覆盖股票残余贡献{b.rel_covered_residual*100:.2f}个百分点。此前看到的低beta不是全部原因，风格和股票残余仍占重要部分。',
            '', f'买入新增股份当日贡献{b.rel_buy_increment*100:+.2f}个百分点，卖出当日增量贡献{b.rel_sell_increment*100:.2f}个百分点，说明该阶段确实存在成交当天“卖后上涨”的总体机会损失。两项净增量与交易成本合计{(b.rel_buy_increment+b.rel_sell_increment+b.rel_commission+b.rel_slippage)*100:.2f}个百分点。这里不能推出“所有卖单都错了”或“长期不换股更好”。',
            '', '原模型预测五日收盘残差收益，实际在次日开盘换仓；当天增量不是模型完整预测期限。标签与执行持有期不一致是需要独立检验的假设，现有归因不能把它直接认定为根因。','',
            '![2022详细分解](episode_2022.png)','','![2025详细分解](episode_2025.png)','',
            '## 三、回撤在哪些月份形成','',
            '| 阶段 | 月份 | 策略收益 | 基准收益 | 当月相对收益 | 对整段回撤的贡献 |',
            '| --- | --- | ---: | ---: | ---: | ---: |']
    for _,r in month[month.window.isin(summary.window.iloc[:3])].iterrows():
        prefix='abs_' if r.window==summary.window.iloc[0] else 'rel_'
        val=sum(r[prefix+k] for k in NAMES)
        lines.append(f'| {r.window} | {r.month} | {r.portfolio_return:.2%} | {r.benchmark_return:.2%} | {r.relative_return:.2%} | {val*100:+.3f}个百分点 |')
    lines += ['','月份按区间边界截取，首尾月不一定是完整自然月。“对整段贡献”使用整段起点的净值链接，能够相加；当月收益各自复利，不能简单相加。','',
              '## 四、哪些股票拖累，是否是一两只股票造成的','',
              '![个股正负贡献](stock_winners_losers.png)','']
    for _,window in summary.iloc[:2].iterrows():
        s=stocks[stocks.window==window.window].sort_values('portfolio_PnL_contribution')
        neg=s[s.portfolio_PnL_contribution<0];pos=s[s.portfolio_PnL_contribution>0]
        lines += [f'### {window.window}', '',f'发生负贡献的股票{len(neg)}只，合计{neg.portfolio_PnL_contribution.sum()*100:.2f}个百分点；正贡献股票{len(pos)}只，合计{pos.portfolio_PnL_contribution.sum()*100:+.2f}个百分点。两者相加等于策略区间收益{window.portfolio_return:.2%}，并非相对收益。','',
                  '| 股票 | 组合净损益贡献 |','| --- | ---: |']
        for _,r in s.head(10).iterrows():lines.append(f'| {r.instrument} | {r.portfolio_PnL_contribution*100:+.3f}个百分点 |')
    lines+=['','在上涨中落后时，拖累不只来自亏损股票，也可能来自没有持有涨幅更大的股票。缺少历史指数个股权重，不能从“组合最差股票”表反推出全部跑输原因。','',
            '## 五、买错还是卖飞：只看成交当天的可核对增量','',
              '本表各区间均为对起点组合资产的**绝对损益贡献**；2025主表使用相对净值，链接权重不同，所以数字不完全相同。','',
              '| 阶段 | 方向 | 有成交订单数 | 当日增量为正的比例 | 毛增量 | 基础费用 | 滑点 |',
            '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for _,r in execution.iterrows():lines.append(f'| {r.window} | {"买入" if r.direction==1 else "卖出"} | {int(r.fills)} | {r.same_day_increment_positive_fraction:.1%} | {r.linked_gross_increment*100:+.3f}pp | {r.linked_commission*100:.3f}pp | {r.linked_slippage*100:.3f}pp |')
    lines+=['','这里的“卖出增量为正”指卖掉之后当日下跌，避免了部分损失；不是卖出盈利比例，也不是策略胜率。成交额、最低手续费、实际成交数量都按原订单记录，没有按计划数量替代成交数量。','',
            '## 六、最差交易日与尾部集中度','']
    for _,r in summary.iloc[:2].iterrows():
        z=tail[(tail.window==r.window)&(tail['sort']=='absolute')]
        lines += [f'### {r.window}：按策略单日净收益排序','',f'最差10天对区间起点资产的损益贡献合计{z.linked_abs_loss.sum()*100:.2f}个百分点。它是损失日集中程度，不是10天独立策略回撤。','',
                  '| 日期 | 策略单日收益 | 基准单日收益 | 算术超额 | 买卖当日毛增量 |',
                  '| --- | ---: | ---: | ---: | ---: |']
        for _,zrow in z.head(5).iterrows():lines.append(f'| {zrow.date} | {zrow.net_return:.2%} | {zrow.benchmark_return:.2%} | {zrow.arithmetic_active:.2%} | {(zrow.buy_increment+zrow.sell_increment):.2%} |')
    lines+=['','## 七、为什么恢复花了很长时间','',
            f'2022-04-26谷底至2024-09-30首次恢复前高，策略累计上涨{summary.iloc[3].portfolio_return:.2%}，基准同期{summary.iloc[3].benchmark_return:.2%}。下跌28.71%需要上涨约40.27%才能回本；实际恢复日的上涨可能越过前高，所以实现的53.20%修复涨幅不必等于最低回本门槛。恢复期间不是持续下跌，而是反弹、再次回落与再修复交替；月度表保留整个路径。','',
            '| 修复阶段分量 | 对谷底起点资产的贡献 |','| --- | ---: |']
    for k,label in NAMES.items():lines.append(f'| {label} | {summary.iloc[3]["abs_"+k]*100:+.3f}个百分点 |')
    lines+=['','## 八、回归方法、覆盖率和解释边界','',
            '每天使用前一交易日历史沪深300成员的beta、60日波动、252−21日动量及20日成交额对数；后三项按该日前成员横截面均值/样本标准差转成z分数。beta必须有至少126对有效历史样本；回退为1的样本不进入该回归。','',
            '| 暴露 | 具体计算 | 如何理解 |','| --- | --- | --- |',
            '| beta | 252日配对收益、指数半衰期60日，至少126对观测 | 对SH000300价格收益的历史敏感度 |',
            '| 波动 | 60日收盘收益样本标准差×sqrt(252)，至少40日 | z<0为低于当日成员平均波动 |',
            '| 动量 | close(t−21)/close(t−252)−1 | 跳过最近一月的长期涨幅代理 |',
            '| 流动性 | log(20日平均人民币成交额)，至少15日 | 正z代表成交额较大，不等于盘口冲击已被度量 |','',
            '例如2025阶段波动暴露为负，且波动风格收益对组合贡献为负，说明该代理模型将部分落后关联到低波动持仓；不能把这个系数直接解释成投资者风险偏好变化的因果证据。三个风格一起回归，未强行正交化，系数是控制其余代理后的条件关联。','',
            '```text','r_i,t − beta_i,t−1 × r_SH000300,t',
            '    = a_t + z_vol,i,t−1 × f_vol,t + z_mom,i,t−1 × f_mom,t',
            '          + z_liq,i,t−1 × f_liq,t + epsilon_i,t','```','',
            '用至少100只有完整暴露及当日收益的股票做等权OLS，并记录设计矩阵条件数（要求≤1e6）。因子收益在当天结束后估计，仅用于事后解释，绝不用于原交易决策。共同项不是无风险收益，残余不是纯个股Alpha；行业、规模、估值等遗漏风险可能落入任意代理分量。价格指数用于历史beta和市场归因，全收益指数用于账户评价；两者差异另列。价格/全收益差反映基准口径，不是可以通过调模型消除的交易损失。','',
            '| 区间 | 平均持仓权重覆盖 | 最低覆盖 | 低于98%的天数 | 回归样本数中位数 | 最大条件数 |',
            '| --- | ---: | ---: | ---: | ---: | ---: |']
    for _,r in summary.iterrows():lines.append(f'| {r.window} | {r.mean_coverage:.2%} | {r.minimum_coverage:.2%} | {int(r.days_coverage_below_98pct)} / {int(r.days)} | {r.fit_pairs_median:.0f} | {r.fit_condition_max:.2f} |')
    lines+=['','前收盘已覆盖持仓的平均暴露如下（每天先除以已覆盖股票权重，再取时间均值；用于描述，不用于放大贡献）：','',
            '| 区间 | beta | 波动z | 动量z | 流动性z |','| --- | ---: | ---: | ---: | ---: |']
    for _,r in summary.iterrows():lines.append(f'| {r.window} | {r.mean_covered_beta:.3f} | {r.mean_covered_volatility_60:.3f} | {r.mean_covered_momentum_252_21:.3f} | {r.mean_covered_log_mean_amount_20:.3f} |')
    lines+=['','未覆盖持仓损益按账目原值单列，**不补零、不放大剩余股票权重、不把全部损益强行分给风格**。因此这是部分覆盖的代理归因，不是完整Barra。风险表使用实际分量与净收益的协方差贡献，允许负值，不能假设各分量彼此独立。','',
            '![覆盖与拟合诊断](coverage_and_fit.png)','','## 九、全期风险方差来自哪里','',
            '风险归因与回撤损益归因不是同一问题。下表计算每个日收益分量与总净收益（或日净超额）的协方差，除以对应总方差；排除首个建仓日。各列合计100%，可以出现负贡献，说明该分量对波动有抵消作用。费用的方差贡献接近0不代表费用没有拖累收益。','',
            '| 分量 | 绝对收益方差贡献 | 净超额收益方差贡献 |','| --- | ---: | ---: |']
    risk=pd.read_csv(out/'realized_risk_contributions.csv').pivot(index='component',columns='target',values='variance_share')
    for k,label in NAMES.items():lines.append(f'| {label} | {risk.loc[k,"absolute"]:.2%} | {risk.loc[k,"active"]:.2%} |')
    lines += ['','这比上一版单市场回归更细，但不是唯一划分。新增风格、共同项和独立开盘换仓增量后，部分原“市场/非市场”会重新分配；总收益、总风险和净值没有改变。不能把此表的残余视为没有任何系统性风险，也不能当作未来风险预测。','',
            '## 十、归因对应的研究问题（本轮不修改策略）','',
            '| 观察到的问题 | 下一步可检验的方案 | 为什么不能直接据此改参数 |','| --- | --- | --- |',
            '| 2022市场敏感度较高，2025上涨阶段敏感度偏低 | 以历史基准为参照约束beta漂移，而非单方向压低仓位 | 单纯低beta可能降低熊市损失，也可能扩大上涨中的相对回撤 |',
            '| 波动风格在两个重点区间有负贡献 | 在组合构建中监控波动代理与历史基准暴露偏离 | 低波动或高波动都不是永远占优，缺少真实基准权重时不能声称风格中性 |',
            '| 买卖当日净增量不足覆盖成本 | 检验换股缓冲和经训练段校准的预期收益/成本门槛 | 原模型分数没有收益率单位，不能直接拿分差与bp比较 |',
            '| 原五日收盘标签与开盘执行存在持有区间差异 | 独立检验与实际执行时点一致的预测标签 | 成交当天归因不能证明完整五日信号失效，需要重新切分与验证 |',
            '| 2017之后模型一直冻结 | 预先固定滚动更新方案，在新时间段检验稳定性 | 不能用已见回撤挑更新日期或只保留胜出的模型 |',
            '| 缺少历史行业和真实指数权重 | 补充时点一致的数据后做行业配置/选择分解 | 不能用当前行业分类或等权股票池替代真实历史指数 |','',
            '价格/全收益基准差是口径解释，不能通过改用价格指数来美化结果。完整归因提高问题定位能力，并不证明上述改进必然提高样本外收益。','',
            '## 十一、可复算材料','',
            '`daily_components.csv`保留12个归因槽位（绝对收益中基准口径差为0，相对收益中显式启用）；`window_attribution.csv`为绝对和相对两种精确链接；`monthly_attribution.csv`为月度路径；`stock_contributions.csv`为个股贡献；`tail_days.csv`为最差收益/超额日；`execution_summary.csv`及`trade_diagnostics.parquet`保留成交增量；`realized_risk_contributions.csv`给出全期风险方差贡献。','',
            '```bash','python -m src.deep_attribution report --input results/deep_attribution --output runs/deep-report',
            'python -m src.deep_attribution build --data-dir /path/to/qlib_data --output runs/deep-rebuild','```','',
            '新输出目录必须不存在。输入和原始行情指纹、逐日净收益核对误差见`provenance.json`。这轮提供的是对既有回测的更细解释，没有用归因结果修改模型；2016—2025仍是复用开发区间。方法参考：[MOSEK因子模型说明](https://docs.mosek.com/portfolio-cookbook/factormodels.html)。','']
    (out/'README.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='action',required=True)
    for action in ['build','report']:
        a=sub.add_parser(action);a.add_argument('--output',type=Path,required=True)
        a.add_argument('--data-dir' if action=='build' else '--input',type=Path,required=True)
    args=p.parse_args();root=Path(__file__).resolve().parents[1]
    if args.action=='build':build(root,args.data_dir,args.output)
    else:generate(root,args.input,args.output)
