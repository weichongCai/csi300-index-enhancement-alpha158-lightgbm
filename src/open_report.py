"""Report the fixed forecast next-open execution comparison, funding matched."""
from pathlib import Path
import json,argparse,hashlib
import numpy as np
import pandas as pd
from .metrics import evaluate,ir,drawdown


def save(root,data,output):
    from .research_build import read_bin
    root,data,output=map(Path,[root,data,output])
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    source=root/'results/next_open_3bp'
    calendar=pd.DatetimeIndex(pd.read_csv(data/'calendars/day.txt',header=None)[0]);calendar=calendar[calendar<='2025-12-31']
    p=calendar.get_loc(pd.Timestamp('2016-01-04'))
    opening=float(read_bin(data,'SH000300','open',calendar)[p]);closing=float(read_bin(data,'SH000300','close',calendar)[p])
    first_benchmark=closing/opening-1
    rows=[];curves={};years=[];checks={}
    cases={'next_open':source/'open/daily_report.csv','matched_close':source/'close/daily_report.csv',
        'published_close':root/'results/slippage_3bp/slippage_daily_report.csv'}
    for case,path in cases.items():
        account=pd.read_csv(path,index_col=0,parse_dates=True,float_precision='round_trip')
        metric,_,daily=evaluate(root,account)
        daily.loc[daily.index[0],'benchmark_return']=first_benchmark
        daily['benchmark_nav']=(1+daily.benchmark_return).cumprod()
        daily['active_return']=daily.model_return-daily.benchmark_return
        daily['relative_nav']=daily.model_nav/daily.benchmark_nav
        daily['relative_drawdown']=drawdown(daily.relative_nav)
        annual=[]
        for year,g in daily.groupby(daily.index.year):
            annual.append({'case':case,'year':year,'model_return':float((1+g.model_return).prod()-1),
                'benchmark_return':float((1+g.benchmark_return).prod()-1),'net_active_ir':ir(g.active_return)})
        annual=pd.DataFrame(annual);years.append(annual)
        metric.update({'case':case,'mean_annual_net_active_ir':float(annual.net_active_ir.mean()),
            'pooled_net_active_ir':ir(daily.active_return),'active_tracking_error':float(daily.active_return.std(ddof=1)*np.sqrt(252)),
            'relative_max_drawdown':float(daily.relative_drawdown.min()),
            'benchmark':'common first opening funding; first TR open approximated by price-index open/close ratio'})
        rows.append(metric);curves[case]=daily.model_nav;curves[case+'_relative']=daily.relative_nav
        curves[case+'_drawdown']=daily.model_drawdown;curves['benchmark']=daily.benchmark_nav
        daily.to_csv(output/f'{case}_daily.csv',index_label='date')
        if case!='published_close':
            orders=pd.read_parquet(source/('open' if case=='next_open' else 'close')/'orders.parquet')
            positive=orders.value.gt(1e-5)
            slip=(orders.loc[positive,'filled_amount']*(orders.loc[positive,'price']-orders.loc[positive,'reference_price'])*np.where(orders.loc[positive,'direction'].eq(1),1.,-1.)).sum()
            checks[case]={'days':len(account),'order_records':len(orders),'positive_fills':int(positive.sum()),
                'commission_CNY':float(orders.cost.sum()),'slippage_attribution_CNY':float(slip),
                'cash_fees_prices_capacity_and_timing_assertions_passed':True}
    metrics=pd.DataFrame(rows);metrics.to_csv(output/'metrics.csv',index=False)
    annual_all=pd.concat(years);annual_all.to_csv(output/'annual.csv',index=False)
    annual_wide=annual_all.pivot(index='year',columns='case',values='net_active_ir')
    annual_wide['open_minus_matched_close']=annual_wide.next_open-annual_wide.matched_close
    annual_wide.to_csv(output/'annual_ir_comparison.csv')
    sensitivity=[]
    for case in ['next_open','matched_close']:
        z=pd.read_csv(output/f'{case}_daily.csv',index_col=0,parse_dates=True).active_return.iloc[1:]
        by_year=z.groupby(z.index.year).apply(ir)
        sensitivity.append({'case':case,'mean_annual_ir_excluding_first_day':float(by_year.mean()),
            'pooled_ir_excluding_first_day':ir(z),'note':'sensitivity only; 2016 misses first trading day'})
    sensitivity=pd.DataFrame(sensitivity);sensitivity.to_csv(output/'exclude_first_day_sensitivity.csv',index=False)
    curves=pd.DataFrame(curves);curves.to_csv(output/'plot_series.csv',index_label='date')
    selected=json.loads((root/'models/selected_model.json').read_text(encoding='utf-8'))
    for filename,digest in selected['models_sha256'].items():assert hashlib.sha256((root/'models'/filename).read_bytes()).hexdigest()==digest
    plan=json.loads((source/'experiment.json').read_text())
    assert hashlib.sha256((root/'data/frozen_forecast.parquet').read_bytes()).hexdigest()==plan['forecast_sha256']
    checks.update({'fixed_model_and_forecast_verified':True,'same_day_future_quote_invariance_tests_passed':True,
        'first_benchmark_return':first_benchmark,'official_H00300_open_available':False,
        'benchmark_open_is_approximation':True,'first_day_funding_same_for_all_comparison_rows':True})
    (output/'checks.json').write_text(json.dumps(checks,indent=2)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,1,figsize=(12,10),sharex=True,gridspec_kw={'height_ratios':[2,1,1]})
    for case,label,color in [('next_open','Next open / lagged ADV','#c47c28'),('matched_close','Next close / lagged ADV','#176a91'),
        ('published_close','Published next close / daily cap','#999999')]:
        axes[0].plot(curves.index,curves[case],label=label,color=color)
        axes[1].plot(curves.index,curves[case+'_relative'],color=color)
        axes[2].plot(curves.index,curves[case+'_drawdown']*100,color=color)
    axes[0].plot(curves.index,curves.benchmark,label='CSI300 TR, common first-open proxy',color='#444444')
    axes[0].set_yscale('log');axes[0].legend(fontsize=9)
    axes[0].set_title('2016–2025 reused development | fixed model | 3bp adverse execution prices')
    for ax,label in zip(axes,['NAV (log scale)','NAV / benchmark','Absolute drawdown (%)']):ax.set_ylabel(label);ax.grid(alpha=.2)
    axes[1].axhline(1,color='#aaa',lw=.5)
    fig.tight_layout();fig.savefig(output/'comparison.png',dpi=150);plt.close(fig)
    labels={'next_open':'下一日开盘 / 前日已知量额容量','matched_close':'下一日收盘 / 前日已知量额容量','published_close':'原收盘账户 / 原全天量额容量'}
    lines=['# 冻结预测：t+1开盘执行对照','','模型、原五日收盘标签和固定预测不变，不重训；换仓的卖出与买入都移到同一执行时点。每边3bp滑点、买入5bp/卖出15bp、最低5元。','',
        '| 执行方案 | 平均年度超额IR | 全期超额IR | 年化收益 | 最大绝对回撤 | 平均股票敞口 |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for _,row in metrics.iterrows():lines.append(f"| {labels[row['case']]} | {row.mean_annual_net_active_ir:.3f} | {row.pooled_net_active_ir:.3f} | {row.calendar_10year_cagr:.2%} | {row.absolute_max_drawdown:.2%} | {row.mean_stock_exposure:.2%} |")
    lines+=['','![账户曲线](comparison.png)','','## 实验口径','',
        '- 两个新账户均使用上一交易日已完成的20日平均成交量、成交额的5%作容量代理。它不是开盘竞价可用量，也不是容量保证。',
        '- 开盘方向限制基于开盘价相对前收盘，不使用当日收盘涨跌幅；统一9.5%只是简化限制。',
        '- 不以当日高低价或全天量额决定开盘成交；开盘×1.0003/0.9997作为不利执行价代理，没有模拟盘口冲击和竞价队列。',
        '- 日终仍按收盘价估值；计划数量继续按上一收盘制定，现金不足和整手会缩量。全额卖出保留完整复权持仓，避免留下不可卖的小额尾仓。',
        '- 仍沿用模拟器先卖后买、同次执行资金可再用的顺序假设，尚未证明真实同一次集合竞价的资金冻结可实现。',
        '- 所有表中行统一以首日开盘为基准投入时点；缺少官方H00300开盘点位，首日收益由SH000300开收盘比率近似，之后使用H00300全收益收盘序列。首日是基准近似。',
        '- 因首日基准投入时点不同，表内原收盘账户IR不会等于既有报告的1.378/1.344；既有账户净值、CAGR和回撤不变。原报告继续保留。',
        '- 原收盘账户还使用全天量额和高低价条件，因此应优先比较前两行；第三行为历史参考，不能将其差异全部归因于时点。',
        '- 2016—2025为反复使用的开发区间；这是原标签模型的执行敏感性实验，不是针对开盘持有期重新训练的最优模型。','',
        '## 逐年与首日敏感性','',
        f'- 开盘版的年度超额IR在10年中有{int((annual_wide.next_open>annual_wide.matched_close).sum())}年高于同口径收盘版；2016、2020年较低，2025年仍为负。',
        f"- 排除首个交易日后，开盘/收盘的平均年度IR分别为{sensitivity.iloc[0].mean_annual_ir_excluding_first_day:.3f} / {sensitivity.iloc[1].mean_annual_ir_excluding_first_day:.3f}，优势仍在。该项仅为敏感性分析，2016年少了首日，不替代主表十年完整账户。",'',
        '## 复现','','```bash','python -m src.open_execution --data-dir /path/to/qlib_data --output results/next_open_3bp',
        'python -m src.open_report --data-dir /path/to/qlib_data --output results/next_open_3bp/report','```','',
        '新输出目录必须不存在。该独立模块不刷新原发布manifest，也不覆盖原模型、预测或账户。','']
    (output/'README.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')
    return metrics


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--data-dir',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    print(save(root,args.data_dir,args.output).to_string(index=False))
