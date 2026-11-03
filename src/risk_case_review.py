"""Exact-link a fixed diagnostic window; ex-post market regression is explanatory only."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd


def review(root,source,out):
    root,source,out=[Path(p).resolve() for p in [root,source,out]]
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    bm=pd.read_csv(root/'results/next_open_3bp/report/next_open_daily.csv',index_col=0,parse_dates=True).benchmark_return
    from .risk_attribution import episodes
    candidate=pd.read_csv(source/'optimized/daily_report.csv',index_col=0,parse_dates=True,float_precision='round_trip')
    worst=episodes(candidate.account/1e8/(1+bm).cumprod()).iloc[0]
    peak=worst.peak;trough=worst.trough
    rows=[];components=[]
    cases={'published_daily':root/'results/next_open_3bp/open','weekly_equal':source/'weekly_equal','optimized':source/'optimized'}
    for case,path in cases.items():
        a=pd.read_csv(path/'daily_report.csv',index_col=0,parse_dates=True,float_precision='round_trip')
        o=pd.read_parquet(path/'orders.parquet');o['date']=pd.to_datetime(o.date)
        prior=a.account.shift(1).fillna(1e8)
        slip=(o.filled_amount*(o.price-o.reference_price)*np.where(o.direction.eq(1),1.,-1.)).groupby(o.date).sum().reindex(a.index,fill_value=0.)/prior
        net=a['return']-a.cost;gross=net+a.cost+slip
        net_active=net-bm
        window=(a.index>peak)&(a.index<=trough)
        beta=gross.loc[window].cov(bm.loc[window])/bm.loc[window].var(ddof=1)
        d=pd.DataFrame({'market_mismatch':(beta-1)*bm,'nonmarket_gross':gross-beta*bm,'commission':-a.cost,'slippage':-slip})
        np.testing.assert_allclose(d.sum(axis=1),net_active,atol=1e-12)
        relative=a.account/1e8/(1+bm).cumprod()
        linked=d.mul(relative.shift(1)/relative.loc[peak]/(1+bm),axis=0).loc[window].sum()
        actual=float(relative.loc[trough]/relative.loc[peak]-1)
        np.testing.assert_allclose(linked.sum(),actual,atol=1e-12)
        rows.append(dict(case=case,peak=peak,trough=trough,relative_return=actual,ex_post_gross_beta=beta,
            annual_arithmetic_gross_active=float((gross-bm).mean()*252),
            annual_arithmetic_net_active=float(net_active.mean()*252),annual_arithmetic_cost=float((a.cost+slip).mean()*252)))
        components.extend(dict(case=case,component=k,relative_contribution=v) for k,v in linked.items())
    rows=pd.DataFrame(rows);rows.to_csv(out/'window_and_alpha_cost.csv',index=False)
    pd.DataFrame(components).to_csv(out/'relative_contributions.csv',index=False)
    dec=pd.read_csv(source/'optimized/decisions.csv');planned=dec.loc[dec.status.eq('planned')]
    summary=dict(planned_rebalances=len(planned),skipped_rebalances=int((~dec.status.eq('planned')).sum()),
        reference_unattainable_plans=int(planned.rank_reference_feasibility_shortfall.gt(2e-6).sum()),
        TE_budget_exceeded_plans=int(planned.TE_budget_excess.gt(1e-6).sum()),
        turnover_cap_binding_noninitial_plans=int(planned.iloc[1:].target_two_way_turnover.ge(.4-2e-6).sum()))
    (out/'constraint_diagnostics.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8',newline='\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    values=pd.DataFrame(components).pivot(index='component',columns='case',values='relative_contribution')*100
    ax=values.plot.bar(figsize=(11,5),color=['#167e8d','#999999','#da8c29'])
    ax.axhline(0,color='black',lw=.7);ax.set_ylabel('Linked relative contribution (percentage points)')
    ax.set_title(f'{peak} to {trough} | fixed ex-post market regression | NOT pure alpha')
    ax.tick_params(axis='x',rotation=0);ax.figure.tight_layout();ax.figure.savefig(out/'failure_window.png',dpi=150);plt.close(ax.figure)
    labels={'published_daily':'原日频模型','weekly_equal':'周频等权','optimized':'分层风控候选'}
    lines=['# 风控实验复盘：为何局部有效，整体没有通过','',
        f"该候选未替换原策略。2025旧回撤窗口虽改善，但候选自身最深相对回撤为{worst.depth:.2%}，峰谷是{peak}至{trough}，恢复日期为{worst.recovery}。原策略十年最大相对回撤为10.13%。",'',
        '## 一、成本节约补不回毛超额流失','',
        '| 方案 | 年化算术毛超额均值 | 年化算术成本 | 年化算术净超额均值 |','| --- | ---: | ---: | ---: |']
    for _,r in rows.iterrows():lines.append(f"| {labels[r['case']]} | {r.annual_arithmetic_gross_active:.2%} | {r.annual_arithmetic_cost:.2%} | {r.annual_arithmetic_net_active:.2%} |")
    lines+=['','这是日均值×252的同口径加减，不是CAGR分解。毛超额指在已成交数量路径上加回费用/滑点的会计分量，不是可独立交易的无成本账户。','',
        '## 二、新最大回撤发生在另一个阶段','',
        '| 同一固定窗口 | 相对收益 | 区间毛收益描述性beta |','| --- | ---: | ---: |']
    for _,r in rows.iterrows():lines.append(f"| {labels[r['case']]} | {r.relative_return:.2%} | {r.ex_post_gross_beta:.3f} |")
    comp=pd.DataFrame(components).pivot(index='case',columns='component',values='relative_contribution')
    c=comp.loc['optimized'];old=comp.loc['published_daily']
    window_plans=planned.loc[pd.to_datetime(planned.trade_date).gt(peak)&pd.to_datetime(planned.trade_date).le(trough)]
    lines+=['',f"候选在该段的精确链接贡献为：市场敏感度偏离{c.market_mismatch*100:+.2f}个百分点、非市场毛分量{c.nonmarket_gross*100:+.2f}个百分点、基础费用{c.commission*100:+.2f}个百分点、滑点{c.slippage*100:+.2f}个百分点；合计{c.sum()*100:+.2f}个百分点。原模型相同窗口分别为{old.market_mismatch*100:+.2f}、{old.nonmarket_gross*100:+.2f}、{old.commission*100:+.2f}、{old.slippage*100:+.2f}个百分点。",
        '',f"该窗口成功发布目标的模型beta均值为{window_plans.target_beta.mean():.3f}，与事后实现的区间beta不是同一统计量。目标约束、每周复核、成交偏差及风险估计误差共同构成局限；这里不能只凭一个beta差值归因于某一项。"]
    lines+=['','![候选最差窗口的相对损益](failure_window.png)','',
        '将毛收益对同一窗口的全收益基准做事后单市场回归，拆成市场偏离、非市场毛分量、基础费用和滑点，再按相对净值精确链接。贡献合计对应各账户相同窗口相对收益。beta使用整个诊断窗口估计，仅用于事后解释，没有进入交易。非市场包含遗漏风格、行业、个股、交易路径和数据影响，不能叫纯选股Alpha。','',
        '## 三、约束冲突挤占了组合选择空间','',
        f"- 共{summary['planned_rebalances']}次发布目标、{summary['skipped_rebalances']}次跳过；跳过时保留真实持仓。",
        f"- {summary['reference_unattainable_plans']}次在硬约束下不能达到Top50参考排序暴露；只能执行已记录的可行前沿，不能声称实际保留全部信号。",
        f"- {summary['TE_budget_exceeded_plans']}次目标仍超过10%软TE预算。预算优先级和可行性限制意味着不能宣传成严格10%TE控制。",
        f"- 非初始目标中，{summary['turnover_cap_binding_noninitial_plans']}次触及40%双边换手上限。换手约束、beta范围和参考排序预算存在明显冲突。",'',
        '这组证据支持“组合约束改变了原信号的实现、成本节约不足以抵消毛超额流失”，但不能分别证明某一个约束是唯一原因。要识别单项作用，需要另设预注册消融实验；本轮没有继续追逐开发期最优参数。','',
        '## 四、保留哪些成果，暂不采用哪些规则','',
        '保留实际持仓风险快照、数据缺口准备、订单账簿审计、可行性检查、成本拆分和报警记录。原日频模型继续作为当前研究基线，分层优化不自动进入模拟盘。',
        '', '下一步值得验证的假设包括：保持原Top50/Drop5信号形成机制以隔离组合层改动；使用过去已成熟样本校准分数的预期收益单位；对约束做独立消融并校准风险预测。它们是后续研究问题，不是本轮已实现的能力，也不能保证提高IR或降低回撤。','']
    (out/'README.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')
    print(rows.to_string(index=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();review(Path(__file__).resolve().parents[1],args.source,args.output)
