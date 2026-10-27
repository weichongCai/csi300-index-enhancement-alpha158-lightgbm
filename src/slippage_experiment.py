"""Independent fixed 3bp adverse-price scenario, without retraining or extra debits."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from .metrics import evaluate

SLIPPAGE_BPS = 3.


def run_experiment(root, data_dir, destination):
    from .backtest import run
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError('Choose a new experiment output directory.')
    folder = destination / 'account'
    report = run(root, data_dir, folder, slippage_bps=SLIPPAGE_BPS)
    return save_comparison(root, report, pd.read_csv(folder/'orders.csv'),
        pd.read_csv(folder/'signal_timing.csv'), pd.read_csv(folder/'slippage_price_rejections.csv'), destination)


def saved_report(root, destination):
    source = Path(root)/'results/slippage_3bp'
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError('Choose a new report output directory.')
    return save_comparison(root,
        pd.read_csv(source/'slippage_daily_report.csv',index_col=0,parse_dates=True,float_precision='round_trip'),
        pd.read_parquet(source/'slippage_orders.parquet'),
        pd.read_csv(source/'signal_timing.csv'),pd.read_csv(source/'price_rejections.csv'),destination)


def save_comparison(root, report, orders, timing, rejections, destination):
    root, destination = Path(root), Path(destination)
    base = pd.read_csv(root/'results/daily_report.csv',index_col=0,parse_dates=True)
    if not report.index.equals(base.index):
        raise ValueError('Baseline and slippage calendars differ.')
    cfg = json.loads((root/'config.json').read_text(encoding='utf-8'))
    selected = json.loads((root/'models/selected_model.json').read_text(encoding='utf-8'))
    for name, expected in selected['models_sha256'].items():
        if hashlib.sha256((root/'models'/name).read_bytes()).hexdigest()!=expected:
            raise ValueError('Frozen model changed.')
    bm, ba, bd = evaluate(root,base)
    sm, sa, sd = evaluate(root,report)
    positive = orders.value.gt(1e-5)
    fills = orders.loc[positive]
    factor = np.where(fills.direction.eq(1),1.0003,.9997)
    np.testing.assert_allclose(fills.price,fills.reference_close*factor,rtol=1e-12,atol=1e-10)
    np.testing.assert_allclose(fills.value,fills.filled_amount*fills.price,rtol=1e-12,atol=1e-7)
    if not ((fills.price >= fills.day_low-1e-10)&(fills.price <= fills.day_high+1e-10)).all():
        raise ArithmeticError('Execution price is outside the observed daily range.')
    fee_rate=np.where(orders.direction.eq(1),cfg['open_cost'],cfg['close_cost'])
    fee=np.where(positive,np.maximum(orders.value*fee_rate,cfg['min_cost']),0.)
    np.testing.assert_allclose(orders.cost,fee,rtol=1e-12,atol=1e-7)
    orders=orders.copy()
    orders['slippage_loss']=np.where(positive,orders.filled_amount*(orders.price-orders.reference_close)
                                     *np.where(orders.direction.eq(1),1.,-1.),0.)
    if orders.slippage_loss.lt(-1e-7).any():
        raise ArithmeticError('Slippage improved an execution price.')
    grouped=fills.groupby(['date','instrument']).agg(
        actual_amount=('value','sum'),amount_cap=('daily_amount_cap','first'),
        actual_quantity=('filled_amount','sum'),volume_cap=('daily_volume_cap','first'))
    if ((grouped.actual_amount>grouped.amount_cap+1e-5)|
        (grouped.actual_quantity>grouped.volume_cap+1e-6)).any():
        raise ArithmeticError('Actual executions exceed daily amount/volume capacity.')
    dated=orders.assign(date=pd.to_datetime(orders.date))
    fees=dated.groupby('date').cost.sum().reindex(report.index,fill_value=0.)
    buys=dated.loc[dated.direction.eq(1)].groupby('date').value.sum().reindex(report.index,fill_value=0.)
    sells=dated.loc[dated.direction.eq(0)].groupby('date').value.sum().reindex(report.index,fill_value=0.)
    previous_cash=report.cash.shift(1).fillna(cfg['initial_cash'])
    np.testing.assert_allclose(report.cash,previous_cash-buys+sells-fees,rtol=1e-10,atol=1e-5)
    previous_assets=report.account.shift(1).fillna(cfg['initial_cash'])
    np.testing.assert_allclose(report.cost*previous_assets,fees,rtol=1e-10,atol=1e-5)
    first=dated.loc[dated.date.eq(report.index[0])]
    if first.direction.ne(1).any():
        raise ArithmeticError('Unexpected initial sell.')
    np.testing.assert_allclose(cfg['initial_cash']-report.account.iloc[0],
                               first.cost.sum()+first.slippage_loss.sum(),rtol=1e-9,atol=1e-5)
    expected_signals=pd.DatetimeIndex(['2015-12-31']).append(report.index[:-1])
    if (len(timing)!=len(report) or not pd.DatetimeIndex(pd.to_datetime(timing.trade_date)).equals(report.index)
        or not pd.DatetimeIndex(pd.to_datetime(timing.signal_date)).equals(expected_signals)):
        raise ValueError('Signal is not exactly one trading day before execution.')
    if report.cash.lt(-1e-6).any():
        raise ArithmeticError('Negative cash.')
    metrics=pd.DataFrame([dict(case='baseline',**bm),dict(case='slippage_3bp',**sm)])
    annual=pd.concat([ba.assign(case='baseline'),sa.assign(case='slippage_3bp')]).reset_index()
    curves=pd.DataFrame({'baseline_nav':bd.model_nav,'slippage_3bp_nav':sd.model_nav,
        'benchmark_nav':bd.benchmark_nav,'baseline_relative_nav':bd.relative_nav,
        'slippage_3bp_relative_nav':sd.relative_nav,'baseline_drawdown':bd.model_drawdown,
        'slippage_3bp_drawdown':sd.model_drawdown})
    destination.mkdir(parents=True,exist_ok=True)
    report.to_csv(destination/'slippage_daily_report.csv')
    orders.to_parquet(destination/'slippage_orders.parquet',index=False)
    timing.to_csv(destination/'signal_timing.csv',index=False)
    rejections.to_csv(destination/'price_rejections.csv',index=False)
    metrics.to_csv(destination/'metrics.csv',index=False)
    annual.to_csv(destination/'annual.csv',index=False)
    curves.to_csv(destination/'plot_series.csv',index_label='date')
    sd.to_csv(destination/'slippage_daily_returns.csv',index_label='date')
    metadata={'experiment':'fixed_directional_execution_price_slippage','slippage_bps':SLIPPAGE_BPS,
        'buy_execution':'next_close * 1.0003','sell_execution':'next_close * 0.9997',
        'commission':'max(actual_execution_value * original_fee_rate, 5 CNY)',
        'additional_slippage_cash_fee':False,'end_day_valuation':'unshifted adjusted close',
        'price_feasibility':'reject slipped prices outside observed daily low/high; no clipping',
        'capacity':'original 5% volume and amount caps; actual slipped notional also checked',
        'new_model_fit_or_parameter_search':False,'development_data_reused':True,
        'simplified_daily_slippage_model_not_order_book_replay':True,
        'frozen_model_sha256':selected['models_sha256'],
        'fixed_inputs_sha256':{n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in
            ['config.json','data/frozen_forecast.parquet','data/official_index_levels.csv','results/daily_report.csv']}}
    checks={'scope':'actual adverse execution prices in adjusted-price account',
        'trading_days':len(report),'order_records':len(orders),'positive_fills':len(fills),
        'price_range_rejections':len(rejections),'total_commission':float(fee.sum()),
        'execution_slippage_loss_attribution':float(orders.slippage_loss.sum()),
        'slippage_not_deducted_again_as_fee':True,'actual_prices_checked_against_reference':True,
        'actual_fill_prices_inside_daily_range':True,'actual_notional_and_volume_capacity_checked':True,
        'daily_cash_reconciled_from_slipped_trades':True,'first_day_NAV_loss_equals_commission_plus_slippage':True,
        'signal_timing_checked':True,'model_files_unchanged':True,'baseline_not_overwritten':True}
    for name,obj in [('experiment.json',metadata),('checks.json',checks)]:
        (destination/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    plot(curves,destination/'comparison.png')
    lines=['# 独立实验：买卖每边3bp成交价滑点','',
        '**原基线平均年度超额IR 1.532没有加入滑点。** 本实验直接修改执行价格，重新重放完整账户；不是在原价成交后加收费用。','',
        '| 情形 | 平均年度超额IR | 全期超额IR | 十年年化 | 最大绝对回撤 | 最大相对回撤 |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for label,row in [('基础费用、无滑点',bm),('每边3bp成交价滑点',sm)]:
        lines.append(f"| {label} | {row['mean_annual_net_active_ir']:.3f} | {row['pooled_net_active_ir']:.3f} | "
            f"{row['calendar_10year_cagr']:.2%} | {row['absolute_max_drawdown']:.2%} | {row['relative_max_drawdown']:.2%} |")
    lines+=['','![无滑点与每边3bp成交价滑点](comparison.png)','','## 固定规则','',
        '- 买入价=次日复权收盘价×1.0003；卖出价=次日复权收盘价×0.9997。',
        '- 原有买入0.05%、卖出0.15%、每笔最低5元，在滑点后的实际成交金额上计算。',
        '- 股票日终估值仍用原收盘价；滑点损失已经进入交易盈亏，不再作为额外费用扣一次。',
        '- 滑后价格若超出该日高低价区间，则拒绝成交；成交量和实际成交额均受原5%容量约束。',
        f'- 本轮有{len(rejections)}条价格越界拒单，成交路径随之改变；回撤较低不代表滑点能改善策略。',
        '- 模型、预测、选股与持仓规则不改；滑点引起的现金与成交数量变化会进入后续账户。',
        '- 2016—2025共2430个交易日，同首日投入的H00300全收益基准；这是开发期研究。',
        '- 固定日线滑点仍是简化执行模型，不等于盘口和收盘竞价重放，也没有按订单规模估计动态冲击。','',
        '## 重新模拟（需要股票行情）','','```bash',
        'python run.py slippage --data-dir /path/to/qlib_data --output runs/slippage3bp','```','',
        '## 重算保存实验（无需股票行情）','','```bash',
        'python run.py slippage-report --output runs/slippage-report','```','',
        '`slippage_orders.parquet`含参考收盘、实际成交价、基础费用和滑点损失归因；'
        '`price_rejections.csv`保留价格越界拒单，`experiment.json`和`checks.json`记录方案及核查。','']
    (destination/'README.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')
    return metrics


def plot(curves,filename):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,1,figsize=(12,9),sharex=True,gridspec_kw={'height_ratios':[2,1,1]})
    for prefix,label,color in [('baseline','Base fees, no slippage','#176a91'),
                                ('slippage_3bp','3bp adverse execution prices','#c77c2c')]:
        axes[0].plot(curves.index,curves[prefix+'_nav'],label=label,color=color)
        axes[1].plot(curves.index,curves[prefix+'_relative_nav'],color=color)
        axes[2].plot(curves.index,curves[prefix+'_drawdown']*100,color=color)
    axes[0].plot(curves.index,curves.benchmark_nav,label='CSI300 total return',color='#888',lw=1)
    axes[0].set_yscale('log');axes[0].legend(loc='upper left')
    axes[0].set_title('2016–2025 development | fixed model | directional execution-price slippage')
    axes[1].axhline(1,color='#aaa',lw=.7)
    for axis,label in zip(axes,['NAV (log scale)','NAV / CSI300 TR','Absolute drawdown (%)']):
        axis.set_ylabel(label);axis.grid(alpha=.2)
    fig.tight_layout();fig.savefig(filename,dpi=160);plt.close(fig)
