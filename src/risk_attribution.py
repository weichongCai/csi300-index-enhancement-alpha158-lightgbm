"""Realized risk and drawdown attribution of the frozen next-open account.

This module diagnoses saved trades. It does not change predictions or trading rules.
"""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
import pandas as pd

COMPONENTS = ['market', 'nonmarket_gross', 'commission', 'slippage']
LABELS = ['市场线性分量', '非市场毛分量', '基础费用', '滑点']


def episodes(series, initial_date='2015-12-31'):
    """Non-overlapping high-water-mark excursions, including an initial NAV of 1."""
    peak_date, peak = pd.Timestamp(initial_date), 1.
    active = None
    output = []
    for day, value in series.items():
        if value >= peak:
            if active:
                active.update(recovery=str(day.date()), underwater_trading_days=active['length'])
                output.append(active)
                active = None
            peak_date, peak = day, value
        else:
            if active is None:
                active = {'peak': str(peak_date.date()), 'trough': str(day.date()),
                          'depth': value / peak - 1, 'recovery': None, 'length': 0}
            active['length'] += 1
            if value / peak - 1 < active['depth']:
                active.update(trough=str(day.date()), depth=value / peak - 1)
    if active:
        active['underwater_trading_days'] = active['length']
        output.append(active)
    return pd.DataFrame(output).sort_values('depth').reset_index(drop=True)


def variance_attribution(components, total):
    """Euler covariance contributions to realized variance and volatility."""
    variance = total.var(ddof=1)
    covariance = components.apply(lambda x: x.cov(total))
    np.testing.assert_allclose(components.sum(axis=1), total, atol=1e-11)
    np.testing.assert_allclose(covariance.sum(), variance, atol=1e-13)
    return pd.DataFrame({'component': covariance.index,
                         'variance_share': covariance.to_numpy() / variance,
                         'annual_volatility_contribution': covariance.to_numpy() / np.sqrt(variance) * np.sqrt(252)})


def linked_contribution(components, daily, peak, trough, relative=False):
    """Exact peak-to-trough linking, not an arithmetic sum of daily returns."""
    nav = daily.relative_nav if relative else daily.model_nav
    selected = (daily.index > pd.Timestamp(peak)) & (daily.index <= pd.Timestamp(trough))
    base = nav.loc[pd.Timestamp(peak)] if pd.Timestamp(peak) in nav.index else 1.
    previous = nav.shift(1).fillna(1.)
    weight = previous / base
    c = components.copy()
    if relative:
        c['market'] -= daily.benchmark_return
        c = c.div(1 + daily.benchmark_return, axis=0)
    contribution = c.loc[selected].mul(weight.loc[selected], axis=0).sum()
    observed = nav.loc[pd.Timestamp(trough)] / base - 1
    np.testing.assert_allclose(contribution.sum(), observed, atol=2e-11)
    return contribution, selected, weight


def build(root, data, out):
    from .research_build import read_bin
    root, data, out = map(Path, [root, data, out])
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    source = root / 'results/next_open_3bp/open'
    report = pd.read_csv(source/'daily_report.csv', index_col=0, parse_dates=True, float_precision='round_trip')
    daily = pd.read_csv(root/'results/next_open_3bp/report/next_open_daily.csv', index_col=0,
                        parse_dates=True, float_precision='round_trip')
    orders = pd.read_parquet(source/'orders.parquet')
    orders['date'] = pd.to_datetime(orders.date)
    orders['signed_quantity'] = orders.filled_amount * np.where(orders.direction.eq(1), 1., -1.)
    quantity_change = orders.pivot_table(index='date', columns='instrument', values='signed_quantity', aggfunc='sum').reindex(report.index).fillna(0.)
    quantity = quantity_change.cumsum()
    previous_quantity = quantity.shift(1).fillna(0.)
    calendar = pd.DatetimeIndex(pd.read_csv(data/'calendars/day.txt', header=None)[0])
    calendar = calendar[calendar <= '2025-12-31']
    positions = calendar.get_indexer(report.index)
    if (positions < 0).any():
        raise ValueError('Missing account date in source calendar.')
    marks = pd.DataFrame(index=report.index, columns=quantity.columns, dtype=float)
    opening = marks.copy()
    for code in marks:
        close = read_bin(data, code, 'close', calendar).astype(float)
        op = read_bin(data, code, 'open', calendar).astype(float)
        # Match the frozen account's last-price mark on unavailable trading days.
        marked = pd.Series(np.where(np.isfinite(op) & (op > 0), close, np.nan)).ffill().to_numpy()
        marks[code] = marked[positions]
        opening[code] = op[positions]
    held = quantity.abs().gt(1e-6)
    if not (marks.notna() | ~held).all().all():
        raise ValueError('Unmarked held stock.')
    stock_value = (quantity*marks).where(held, 0.).sum(axis=1)
    np.testing.assert_allclose(stock_value + report.cash, report.account, atol=.02, rtol=1e-10)
    previous_mark = marks.shift(1)
    effective_open = opening.where(opening.gt(0) & np.isfinite(opening), previous_mark)
    # Overnight belongs to yesterday's inventory; intraday belongs to post-open inventory.
    overnight = (previous_quantity * (effective_open-previous_mark)).where(previous_quantity.abs().gt(1e-6), 0.).fillna(0.)
    intraday = (quantity * (marks-effective_open)).where(held, 0.).fillna(0.)
    orders['slippage_loss'] = orders.signed_quantity * (orders.price-orders.reference_price)
    def matrix(field):
        return orders.pivot_table(index='date', columns='instrument', values=field, aggfunc='sum').reindex(index=report.index, columns=quantity.columns).fillna(0.)
    fees, slip = matrix('cost'), matrix('slippage_loss')
    net_pnl = overnight + intraday - fees - slip
    actual_pnl = report.account - report.account.shift(1).fillna(1e8)
    error = net_pnl.sum(axis=1)-actual_pnl
    np.testing.assert_allclose(net_pnl.sum(axis=1), actual_pnl, atol=.03, rtol=1e-9)
    previous_assets = report.account.shift(1).fillna(1e8)
    daily['overnight_gross'] = overnight.sum(axis=1)/previous_assets
    daily['intraday_gross'] = intraday.sum(axis=1)/previous_assets
    daily['gross_return'] = daily.overnight_gross + daily.intraday_gross
    daily['commission'] = -fees.sum(axis=1)/previous_assets
    daily['slippage'] = -slip.sum(axis=1)/previous_assets
    daily['two_sided_turnover'] = orders.groupby('date').value.sum().reindex(daily.index, fill_value=0.)/previous_assets
    stock_weights = quantity*marks / report.account.to_numpy()[:, None]
    daily['stock_exposure'] = stock_weights.sum(axis=1)
    daily['largest_weight'] = stock_weights.max(axis=1)
    daily['effective_names'] = stock_weights.sum(axis=1)**2/(stock_weights**2).sum(axis=1)
    daily['held_names'] = held.sum(axis=1)
    np.testing.assert_allclose(daily.gross_return+daily.commission+daily.slippage, daily.model_return, atol=1e-10)
    daily.to_csv(out/'daily_inputs.csv', index_label='date')
    net_pnl.div(previous_assets, axis=0).to_parquet(out/'stock_net_return_contributions.parquet')
    # Previous-close holdings: no same-day closing weights in risk exposure estimates.
    proxy = pd.read_parquet(root/'results/research/factors/risk_proxy_panel.parquet')
    styles = ['volatility_60', 'momentum_252_21', 'log_mean_amount_20']
    for field in styles:
        x = proxy[field].replace([np.inf, -np.inf], np.nan)
        group = x.groupby(level='datetime')
        proxy[field] = (x-group.transform('mean'))/group.transform('std')
    holdings = pd.read_parquet(source/'holdings.parquet').rename(columns={'date':'datetime'})
    holdings['datetime'] = pd.to_datetime(holdings.datetime)
    h = holdings.merge(proxy.reset_index(), on=['datetime','instrument'], how='left', validate='one_to_one')
    next_date = dict(zip(daily.index[:-1], daily.index[1:]))
    h['trade_date'] = h.datetime.map(next_date)
    h = h.loc[h.trade_date.notna()].copy()
    exposure = pd.DataFrame(index=daily.index)
    total_weight = h.groupby('trade_date').weight.sum()
    for field in ['beta', *styles]:
        coverage = h.weight.where(h[field].notna(), 0.).groupby(h.trade_date).sum()/total_weight
        exposure[field+'_coverage'] = coverage
        exposure[field] = (h.weight*h[field]).groupby(h.trade_date).sum().where(coverage.ge(.98))
    exposure.to_csv(out/'prior_close_exposures.csv', index_label='date')
    risk_universe = sorted(quantity.columns)
    files = [source/'daily_report.csv', source/'orders.parquet', source/'holdings.parquet',
             root/'results/next_open_3bp/report/next_open_daily.csv',
             root/'results/research/factors/risk_proxy_panel.parquet',
             root/'results/research/factors/daily_rank_ic.csv', root/'data/official_index_levels.csv',
             root/'models/first_2016_model.txt',root/'models/frozen_2017_2025_model.txt',root/'data/frozen_forecast.parquet']
    provenance = {'case':'next_open_3bp', 'source_account_days':len(daily),
                  'maximum_daily_PnL_reconciliation_error_CNY':float(error.abs().max()),
                  'data_path_local_only':str(data.resolve()),
                  'source_sha256':{p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                  'source_calendar_sha256':hashlib.sha256((data/'calendars/day.txt').read_bytes()).hexdigest(),
                  'input_sha256':{name:hashlib.sha256((out/name).read_bytes()).hexdigest() for name in
                                  ['daily_inputs.csv','stock_net_return_contributions.parquet','prior_close_exposures.csv']},
                  'price_files_sha256':{f'{code}/{field}':hashlib.sha256((data/'features'/code.lower()/f'{field}.day.bin').read_bytes()).hexdigest()
                                        for code in risk_universe for field in ['open','close']},
                  'full_Barra':False, 'strategy_changed':False,
                  'paper_account_is_different_prefunded_execution':True}
    # Do not publish an author's local filesystem path.
    provenance.pop('data_path_local_only')
    (out/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n', encoding='utf-8')
    return generate(root, out, out)


def generate(root, inputs, output):
    root, inputs, out = map(Path, [root, inputs, output])
    provenance=json.loads((inputs/'provenance.json').read_text(encoding='utf-8'))
    for name,expected in provenance['source_sha256'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=expected:
            raise ValueError('Frozen attribution source changed: '+name)
    for name,expected in provenance['input_sha256'].items():
        if hashlib.sha256((inputs/name).read_bytes()).hexdigest()!=expected:
            raise ValueError('Saved attribution input changed: '+name)
    if out != inputs:
        if out.exists():raise FileExistsError(out)
        out.mkdir(parents=True)
    daily = pd.read_csv(inputs/'daily_inputs.csv', index_col=0, parse_dates=True, float_precision='round_trip')
    stock = pd.read_parquet(inputs/'stock_net_return_contributions.parquet')
    exposure = pd.read_csv(inputs/'prior_close_exposures.csv', index_col=0, parse_dates=True)
    # Boundary sensitivity: exclude the exceptional first funding / benchmark-proxy day.
    fit = daily.iloc[1:]
    beta = fit.gross_return.cov(fit.benchmark_return)/fit.benchmark_return.var(ddof=1)
    daily['market'] = beta*daily.benchmark_return
    daily['nonmarket_gross'] = daily.gross_return-daily.market
    components = daily[COMPONENTS]
    np.testing.assert_allclose(components.sum(axis=1), daily.model_return, atol=1e-10)
    risks = []
    for boundary, selected in [('exclude_initial_day',daily.index[1:]),('include_initial_day',daily.index)]:
        for target in ['absolute','active']:
            c = components.loc[selected].copy()
            y = daily.loc[selected,'model_return'].copy()
            if target == 'active':
                c['market'] -= daily.loc[selected,'benchmark_return']
                y -= daily.loc[selected,'benchmark_return']
            table = variance_attribution(c, y)
            table['target'],table['sample'] = target,boundary
            risks.append(table)
    risks = pd.concat(risks,ignore_index=True)
    risks.to_csv(out/'risk_variance_contributions.csv',index=False)
    covariance = stock.iloc[1:].apply(lambda x:x.cov(daily.model_return.iloc[1:]))
    total_variance = daily.model_return.iloc[1:].var(ddof=1)
    np.testing.assert_allclose(covariance.sum(), total_variance, atol=1e-12)
    sr = pd.DataFrame({'instrument':covariance.index,'variance_share':covariance.to_numpy()/total_variance,
                       'mean_daily_net_return_contribution':stock.iloc[1:].mean().to_numpy()})
    sr.sort_values('variance_share',ascending=False).to_csv(out/'stock_risk_contributions.csv',index=False)
    records, stock_rows = [], []
    for kind, field in [('absolute','model_nav'),('relative','relative_nav')]:
        eps = episodes(daily[field]);eps.to_csv(out/f'{kind}_drawdown_episodes.csv', index=False)
        for rank, row in eps.head(5).iterrows():
            contrib, selected, weight = linked_contribution(components,daily,row.peak,row.trough,kind=='relative')
            window = daily.loc[selected]
            benchmark = (1+window.benchmark_return).prod()-1
            portfolio = (1+window.model_return).prod()-1
            risk_slice = exposure.loc[selected]
            signal_rank = pd.read_csv(root/'results/research/factors/daily_rank_ic.csv',index_col=0,parse_dates=True).MODEL_LIGHTGBM
            # Evaluate only labels whose t+6 exit is inside the episode's trough date.
            indexer = daily.index.get_indexer(signal_rank.index)
            exit_pos = indexer+6
            allowed = (indexer>=0)&(exit_pos<len(daily))
            exits = pd.Series(pd.NaT,index=signal_rank.index,dtype='datetime64[ns]')
            exits.loc[allowed] = daily.index[exit_pos[allowed]].to_numpy()
            local_ic = signal_rank.loc[(signal_rank.index>=row.peak)&(exits<=pd.Timestamp(row.trough))]
            record = {'kind':kind,'rank':rank+1,**row.to_dict(),'portfolio_return':portfolio,
                      'benchmark_same_period_return':benchmark,'relative_same_period_return':(1+portfolio)/(1+benchmark)-1,
                      **{f'{name}_linked_contribution':float(contrib[name]) for name in COMPONENTS},
                      'mean_stock_exposure':window.stock_exposure.mean(),
                      'mean_largest_weight':window.largest_weight.mean(),
                      'mean_effective_names':window.effective_names.mean(),
                      'two_sided_turnover_sum':window.two_sided_turnover.sum(),
                      'mature_within_window_mean_RankIC':local_ic.mean(), 'mature_IC_days':len(local_ic)}
            for field2 in ['beta','volatility_60','momentum_252_21','log_mean_amount_20']:
                record['mean_'+field2] = risk_slice[field2].mean()
                record[field2+'_valid_days'] = risk_slice[field2].count()
            record['peak_to_trough_trading_days'] = len(window)
            record['window_fitted_gross_beta'] = window.gross_return.cov(window.benchmark_return)/window.benchmark_return.var()
            # Supplemental time-varying market proxy, only on fully covered episodes.
            # beta inputs use previous-close inventory and pre-existing historical beta.
            dynamic_ok = risk_slice.beta.notna().all()
            record['dynamic_proxy_full_coverage'] = bool(dynamic_ok)
            for component in COMPONENTS:
                record['dynamic_'+component+'_linked_contribution'] = np.nan
            if dynamic_ok:
                dynamic = components.copy()
                dynamic.loc[selected,'market'] = risk_slice.beta*window.benchmark_return
                dynamic.loc[selected,'nonmarket_gross'] = window.gross_return-dynamic.loc[selected,'market']
                dynamic_contribution,_,_ = linked_contribution(dynamic,daily,row.peak,row.trough,kind=='relative')
                for component in COMPONENTS:
                    record['dynamic_'+component+'_linked_contribution'] = dynamic_contribution[component]
            records.append(record)
            # Individual-name portfolio PnL, in percentage points of episode-start assets.
            # This is NOT per-stock active alpha relative to index constituent weights.
            contribution = stock.loc[selected].mul((daily.model_nav.shift(1).fillna(1.) /
                              (daily.model_nav.loc[pd.Timestamp(row.peak)] if pd.Timestamp(row.peak) in daily.index else 1.)).loc[selected],axis=0).sum()
            np.testing.assert_allclose(contribution.sum(),portfolio,atol=2e-10)
            for code, value in contribution.items():
                stock_rows.append({'kind':kind,'rank':rank+1,'peak':row.peak,'trough':row.trough,
                                   'instrument':code,'portfolio_PnL_contribution':value})
    periods = pd.DataFrame(records)
    periods.to_csv(out/'drawdown_attribution.csv',index=False)
    pd.DataFrame(stock_rows).to_csv(out/'drawdown_stock_contributions.csv',index=False)
    annual=[]
    for year,g in daily.groupby(daily.index.year):
        absbase=daily.model_nav.shift(1).fillna(1.).loc[g.index[0]]
        w=daily.model_nav.shift(1).fillna(1.).loc[g.index]/absbase
        linked=g[COMPONENTS].mul(w,axis=0).sum()
        net=(1+g.model_return).prod()-1
        np.testing.assert_allclose(linked.sum(),net,atol=1e-10)
        annual.append({'year':year,'net_return':net,'benchmark_return':(1+g.benchmark_return).prod()-1,
                       'gross_arithmetic_mean_times_252':g.gross_return.mean()*252,
                       'gross_active_mean_times_252':(g.gross_return-g.benchmark_return).mean()*252,
                       'net_active_mean_times_252':g.active_return.mean()*252,
                       'commission_mean_times_252':-g.commission.mean()*252,
                       'slippage_mean_times_252':-g.slippage.mean()*252,
                       'commission_daily_fraction_sum':-g.commission.sum(),'slippage_daily_fraction_sum':-g.slippage.sum(),
                       'two_sided_turnover':g.two_sided_turnover.sum(),
                       **{f'{c}_linked':linked[c] for c in COMPONENTS}})
    annual=pd.DataFrame(annual);annual.to_csv(out/'annual_attribution.csv',index=False)
    roll_beta = daily.model_return.rolling(252,min_periods=200).cov(daily.benchmark_return)/daily.benchmark_return.rolling(252,min_periods=200).var()
    rolling = pd.DataFrame({'rolling_252d_net_beta':roll_beta,
                           'rolling_252d_active_mean_annualized':daily.active_return.rolling(252).mean()*252,
                           'rolling_252d_tracking_error':daily.active_return.rolling(252).std(ddof=1)*np.sqrt(252)})
    rolling['rolling_252d_active_IR']=rolling.rolling_252d_active_mean_annualized/rolling.rolling_252d_tracking_error
    rolling.to_csv(out/'rolling_diagnostics.csv',index_label='date')
    daily.to_csv(out/'daily_attribution.csv',index_label='date')
    net_beta = fit.model_return.cov(fit.benchmark_return)/fit.benchmark_return.var()
    summary={'case':'next_open_3bp','start':str(daily.index[0].date()),'end':str(daily.index[-1].date()),
             'days':len(daily),'gross_beta_descriptive_excluding_initial_day':float(beta),
             'net_beta_descriptive_excluding_initial_day':float(net_beta),
             'net_volatility':float(fit.model_return.std(ddof=1)*np.sqrt(252)),
             'tracking_error':float(fit.active_return.std(ddof=1)*np.sqrt(252)),
             'market_correlation':float(fit.model_return.corr(fit.benchmark_return)),
             'mean_stock_exposure':float(daily.stock_exposure.mean()),
             'mean_effective_names':float(daily.effective_names.mean()),
             'mean_largest_weight':float(daily.largest_weight.mean()),
             'maximum_name_weight':float(daily.largest_weight.max()),
             'annual_average_commission_fraction_sum':float(-daily.commission.sum()/10),
             'annual_average_slippage_fraction_sum':float(-daily.slippage.sum()/10),
             'annual_average_two_sided_turnover':float(daily.two_sided_turnover.sum()/10),
             'daily_components_max_abs_return_error':float(abs(components.sum(axis=1)-daily.model_return).max()),
             'risk_variance_shares_sum_to_one':True,'episode_contributions_exactly_linked':True,
             'nonmarket_is_not_pure_alpha_or_specific_risk':True,
             'ex_post_descriptive_not_forecast':True,'full_Barra':False,
             'historical_industry_and_index_weights_available':False,
             'model_or_execution_parameters_changed':False,'reused_development_period':True}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    figures(daily,periods,risks,rolling,exposure,out)
    write_report(summary,periods,annual,risks,out)
    print(json.dumps(summary,ensure_ascii=False),flush=True)
    return summary


def figures(daily,periods,risks,rolling,exposure,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','DejaVu Sans'],'axes.unicode_minus':False})
    blue,red,green='#176a91','#bb514b','#228371'
    fig,axes=plt.subplots(3,1,figsize=(13,10),sharex=True)
    axes[0].plot(daily.index,daily.model_nav,label='策略净值',color=blue)
    axes[0].plot(daily.index,daily.benchmark_nav,label='沪深300全收益',color='#8c929b')
    axes[0].set_yscale('log');axes[0].set_ylabel('净值（对数）');axes[0].legend()
    axes[0].set_title('开盘研究账户 · 每边3bp滑点 · 2016—2025复用开发区间')
    axes[1].fill_between(daily.index,0,daily.model_drawdown*100,color=red,alpha=.45)
    axes[1].plot(daily.index,(daily.benchmark_nav/daily.benchmark_nav.cummax().clip(lower=1)-1)*100,color='#888',label='基准回撤')
    axes[1].set_ylabel('绝对回撤（%）');axes[1].legend()
    axes[2].fill_between(daily.index,0,daily.relative_drawdown*100,color=green,alpha=.4)
    axes[2].set_ylabel('相对净值回撤（%）')
    for kind,ax in [('absolute',axes[1]),('relative',axes[2])]:
        row=periods[(periods.kind==kind)&(periods['rank']==1)].iloc[0]
        ax.axvspan(pd.Timestamp(row.peak),pd.Timestamp(row.trough),color='#d99f32',alpha=.18)
        late=pd.Timestamp(row.trough)>daily.index[-1]-pd.Timedelta(days=500)
        ax.annotate(f'{row.trough}: {row.depth:.2%}',xy=(pd.Timestamp(row.trough),row.depth*100),
                    xytext=(-12 if late else 15,12),ha='right' if late else 'left',
                    textcoords='offset points',fontsize=10)
    for ax in axes:ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'drawdown_overview.png',dpi=150);plt.close(fig)
    chosen=pd.concat([periods[periods.kind=='absolute'].head(3),periods[periods.kind=='relative'].head(3)])
    fig,axes=plt.subplots(2,3,figsize=(15,8))
    for ax,(_,row) in zip(axes.ravel(),chosen.iterrows()):
        val=np.array([row[c+'_linked_contribution'] for c in COMPONENTS])*100
        ax.bar(range(4),val,color=[blue,green,red,'#ce9640'])
        ax.axhline(0,color='#888',lw=.7);ax.set_xticks(range(4),LABELS,rotation=20)
        for i,v in enumerate(val):ax.text(i,v,f'{v:+.2f}',ha='center',va='bottom' if v>=0 else 'top',fontsize=9)
        title='绝对回撤' if row.kind=='absolute' else '相对回撤'
        ax.set_title(f'{title} {row.depth:.2%}\n{row.peak} → {row.trough}',fontsize=11)
        ax.set_ylabel('贡献（百分点，精确链接）');ax.grid(axis='y',alpha=.15)
    fig.suptitle('高水位至谷底的分解：非市场毛分量包含遗漏风格与选股，不能称为纯Alpha',fontsize=12)
    fig.tight_layout();fig.savefig(out/'drawdown_components.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(13,10),sharex=True)
    axes[0].plot(rolling.index,rolling.rolling_252d_net_beta,color=blue,label='252日回看组合净beta')
    axes[0].axhline(1,color='#888',ls='--');axes[0].set_ylabel('市场beta');axes[0].legend()
    axes[1].plot(rolling.index,rolling.rolling_252d_active_IR,color=green,label='252日回看超额IR')
    axes[1].axhline(0,color='#888');axes[1].set_ylabel('超额IR');axes[1].legend()
    for field,label in [('volatility_60','60日波动'),('momentum_252_21','252−21日动量'),('log_mean_amount_20','成交额对数')]:
        axes[2].plot(exposure.index,exposure[field].rolling(60,min_periods=40).mean(),label=label)
    axes[2].axhline(0,color='#888',lw=.6);axes[2].set_ylabel('60日均暴露（横截面z）');axes[2].legend(ncol=3)
    axes[0].set_title('回看诊断与前收盘持仓风格代理（覆盖不足98%留空；不是Barra）')
    for ax in axes:ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'risk_and_decay.png',dpi=150);plt.close(fig)
    main=risks[risks['sample']=='exclude_initial_day']
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for target,ax in zip(['absolute','active'],axes):
        x=main[main.target==target].set_index('component').loc[COMPONENTS]
        v=x.variance_share.to_numpy()*100
        ax.bar(LABELS,v,color=[blue,green,red,'#ce9640'])
        for i,value in enumerate(v):ax.text(i,value,f'{value:.2f}%',ha='center',va='bottom' if value>=0 else 'top')
        ax.set_title('绝对收益方差贡献' if target=='absolute' else '日净超额收益方差贡献')
        ax.set_ylabel('协方差贡献 / 总方差（%）');ax.grid(axis='y',alpha=.2)
    fig.suptitle('事后单市场模型：贡献允许为负，合计100%；残余部分不是纯个股风险')
    fig.tight_layout();fig.savefig(out/'variance_attribution.png',dpi=150);plt.close(fig)


def write_report(s,periods,annual,risks,out):
    lines=['# 风险归因与回撤分析','','分析对象为冻结LightGBM的 **t+1开盘、每边3bp滑点研究账户**，2016-01-04至2025-12-31，2,430个交易日。原模型、预测、持仓路径与执行参数均未改动。模拟盘另有竞价前现金预算约束，不直接继承本报告结果。','',
           '## 分析结论','',
           '进一步的持仓、风格、换仓、逐月及尾部交易日解释见[详细归因报告](../deep_attribution/README.md)。下文保留原单市场口径；两种模型分配有所不同，原账户净值没有改变。','',
           f"- 平均股票敞口{s['mean_stock_exposure']:.2%}，净收益事后市场beta约{s['net_beta_descriptive_excluding_initial_day']:.3f}。市场残差预测目标并不使多头组合市场中性。",
           f"- 排除首个建仓/基准近似日，年化净波动{s['net_volatility']:.2%}，跟踪误差{s['tracking_error']:.2%}，组合与基准日收益相关系数{s['market_correlation']:.3f}。",
           f"- 年均双边换手{s['annual_average_two_sided_turnover']:.2f}倍；基础费用和滑点的日占比年均加总为{s['annual_average_commission_fraction_sum']:.2%}、{s['annual_average_slippage_fraction_sum']:.2%}。这是实际路径成本归因，不是扣减CAGR或无成本策略重跑。",
           f"- 平均有效持仓数{s['mean_effective_names']:.1f}，平均最大单股权重{s['mean_largest_weight']:.2%}，全期最大单股权重{s['maximum_name_weight']:.2%}。Top50并不等于每天严格2%等权。",'',
           '## 主要回撤区间','',
           '| 类型 | 高水位 | 谷底 | 回撤 | 同期策略收益 | 同期基准收益 | 恢复高水位 |',
           '| --- | --- | --- | ---: | ---: | ---: | --- |']
    for kind in ['absolute','relative']:
        for _,r in periods[periods.kind==kind].head(3).iterrows():
            recovery=r.recovery if pd.notna(r.recovery) else '截至2025-12-31未恢复'
            lines.append(f"| {'绝对净值' if kind=='absolute' else '相对净值'} | {r.peak} | {r.trough} | {r.depth:.2%} | {r.portfolio_return:.2%} | {r.benchmark_same_period_return:.2%} | {recovery} |")
    lines+=['','恢复日期是谷底之后首次达到该次高水位的日期；不是“谷底至恢复”的独立收益。初始资金净值1作为首个高水位。区间高水位当日不计入损益，计算从下一交易日开始。','',
            '![净值与两类回撤](drawdown_overview.png)','','## 事后风险归因','',
            '采用单市场事后描述性模型：`gross_t = beta_gross × benchmark_t + nonmarket_gross_t`；再扣基础费用和滑点，精确得到账户净收益。beta为全期排除首日的带截距OLS斜率，截距保留在非市场毛分量，不是事前预测。非市场毛分量同时包含行业/风格遗漏、个股选择、现金和交易路径，不能称为纯Alpha或Barra特质收益。','',
            '`variance_share_k = Cov(component_k, total_return) / Var(total_return)`；贡献可为负，合计100%。超额归因使用`(beta_gross−1)×benchmark_t`、同一非市场毛分量及成本。','',
            '| 分量 | 绝对收益方差占比 | 日净超额收益方差占比 |','| --- | ---: | ---: |']
    r=risks[risks['sample']=='exclude_initial_day'].pivot(index='component',columns='target',values='variance_share')
    for c,label in zip(COMPONENTS,LABELS):lines.append(f'| {label} | {r.loc[c,"absolute"]:.2%} | {r.loc[c,"active"]:.2%} |')
    lines+=['','方差表排除初始建仓日，以免首日基准开盘近似和建仓边界主导诊断；包含首日的敏感性结果也保存在CSV。它们不改变已有回测评价口径。','',
            '![方差归因](variance_attribution.png)','','## 回撤损益贡献','',
            '下表将每日分量按“前一日资产/回撤起点资产”链接，合计精确等于峰谷回撤。相对回撤先将每日分量转为相对净值收益：`(r_p−r_b)/(1+r_b)`，再按相对净值链接。百分比是回撤百分点，不是人民币亏损或简单日收益加总。','',
            '| 类型及谷底 | 市场线性分量 | 非市场毛分量 | 基础费用 | 滑点 | 合计 |',
            '| --- | ---: | ---: | ---: | ---: | ---: |']
    for kind in ['absolute','relative']:
        for _,row in periods[periods.kind==kind].head(3).iterrows():
            vals=[row[c+'_linked_contribution'] for c in COMPONENTS]
            lines.append(f"| {'绝对' if kind=='absolute' else '相对'} {row.trough} | "+' | '.join(f'{v:+.2%}' for v in vals)+f' | {sum(vals):.2%} |')
    absolute=periods[(periods.kind=='absolute')&(periods['rank']==1)].iloc[0]
    relative=periods[(periods.kind=='relative')&(periods['rank']==1)].iloc[0]
    lines+=['','![峰谷回撤分解](drawdown_components.png)','','## 重点区间：为什么会回撤','',
            f"**2022年绝对回撤：市场下跌叠加主动分量与成本拖累。** {absolute.peak}至{absolute.trough}，基准下跌{abs(absolute.benchmark_same_period_return):.2%}，策略下跌{abs(absolute.depth):.2%}。固定beta分解中，市场分量{absolute.market_linked_contribution*100:.2f}个百分点，非市场毛分量{absolute.nonmarket_gross_linked_contribution*100:.2f}个百分点，基础费用与滑点合计{(absolute.commission_linked_contribution+absolute.slippage_linked_contribution)*100:.2f}个百分点。该阶段既有系统性下跌，也有相对落后，不能只说“熊市拖累”。",
            '',f"**2025年相对回撤：组合上涨，但没有跟上基准。** {relative.peak}至{relative.trough}，策略上涨{relative.portfolio_return:.2%}，基准上涨{relative.benchmark_same_period_return:.2%}；相对净值回撤{relative.depth:.2%}，不是亏损{abs(relative.depth):.2%}。该阶段实际毛收益回归beta约{relative.window_fitted_gross_beta:.3f}，前收盘持仓beta代理均值{relative.mean_beta:.3f}。用十年固定beta解释短期会把beta变化混进非市场分量。",'']
    if relative.dynamic_proxy_full_coverage:
        vals=[relative['dynamic_'+c+'_linked_contribution'] for c in COMPONENTS]
        lines += [f"该区间{int(relative.peak_to_trough_trading_days)}天的前收盘beta代理全部达到98%覆盖要求。补充时变beta代理归因：低于基准的市场敏感度贡献{vals[0]*100:+.2f}个百分点，非市场毛分量{vals[1]*100:+.2f}个百分点，基础费用{vals[2]*100:+.2f}、滑点{vals[3]*100:+.2f}个百分点，合计{sum(vals)*100:.2f}个百分点。因此，低beta未跟上上涨只能解释部分落后，其余仍需检查个股/遗漏风格与成本。",'',
                  '此处beta原始估计来自SH000300价格指数，收益归因使用H00300全收益基准，属于基准近似的代理分解；前收盘持仓也不等于开盘换仓后的全天风险。不同归因模型的划分可以不同，不能将该残余当成纯选股原因。','']
    row2025=annual[annual.year==2025].iloc[0]
    lines += [f"**2025年全年：毛优势被成本覆盖。** 在相同已成交路径上，日毛超额均值×252为{row2025.gross_active_mean_times_252:.2%}，基础费和滑点日均×252合计{(row2025.commission_mean_times_252+row2025.slippage_mean_times_252):.2%}，净超额日均×252变为{row2025.net_active_mean_times_252:.2%}。这个恒等式说明成本是全年净超额转负的直接账面因素；不表示实际取消费用后仍能保持同一持仓路径。",'',
              '## 前收盘风险暴露与模型诊断','',
              '| 区间谷底 | beta代理均值 | 波动z | 动量z | 流动性z | 区间内已成熟RankIC | beta覆盖有效日/区间日数 |',
              '| --- | ---: | ---: | ---: | ---: | ---: | --- |']
    for _,row in pd.concat([periods[periods.kind=='absolute'].head(2),periods[periods.kind=='relative'].head(2)]).iterrows():
        form=lambda v:'缺失' if pd.isna(v) else f'{v:.3f}'
        lines.append(f"| {row.trough} | {form(row.mean_beta)} | {form(row.mean_volatility_60)} | {form(row.mean_momentum_252_21)} | {form(row.mean_log_mean_amount_20)} | {form(row.mature_within_window_mean_RankIC)} | {int(row.beta_valid_days)}/{int(row.peak_to_trough_trading_days)} |")
    lines+=['','z为相对当日等权历史成员的暴露，不是相对真实指数权重；均值只取覆盖达标日，其他风格各自覆盖天数见CSV。RankIC仅纳入t+6退出日不晚于谷底的已成熟标签，预测仍针对原五日收盘残差目标，不等同实际开盘交易收益。不同阶段IC差异是诊断线索，没有进行因果或显著性断言。','',
            '## 回撤原因与证据边界','',
            '1. **市场风险与主动风险不同。** 接近满仓的多头指增在系统性下跌中仍会亏损；同时跑赢下跌的基准不代表没有绝对回撤。峰谷期间的基准收益和市场线性分量用于量化这一点。',
            '2. **选股毛优势必须覆盖交易成本。** 年度表同时给出毛收益、净超额、费用和滑点。费用即使波动很小、对方差贡献接近0，仍可持续拖累收益和恢复速度。单市场残余变差不能自动解释为机器学习失效。',
            '3. **持仓并非行业/风格中性。** 前收盘持仓的波动、动量、流动性代理参考当日等权历史沪深300成员做z分数，覆盖不足98%留空；缺少真实历史基准权重，无法把偏离归为具体行业配置效应。',
            '4. **模型时效性需要持续观察。** 2017后模型被冻结；回看beta、超额IR及区间内成熟RankIC用于定位表现变化，但不能由时间先后断言“冻结导致回撤”。本次没有依据回撤区间重新调参。','',
            '![风险与衰减诊断](risk_and_decay.png)','','## 年度执行与收益诊断','',
            '| 年度 | 净收益 | 基准收益 | 日净超额均值×252 | 毛收益日均×252 | 基础费日占比加总 | 滑点日占比加总 | 双边换手 |',
            '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for _,row in annual.iterrows():lines.append(f"| {int(row.year)} | {row.net_return:.2%} | {row.benchmark_return:.2%} | {row.net_active_mean_times_252:.2%} | {row.gross_arithmetic_mean_times_252:.2%} | {row.commission_daily_fraction_sum:.2%} | {row.slippage_daily_fraction_sum:.2%} | {row.two_sided_turnover:.2f}倍 |")
    lines+=['','日均×252是算术年化，不能与复利年度收益直接相减。成本为同一已成交路径的实际费用与成交价差，不是无成本独立账户。','',
            '## 个股归因与信息完整性','',
            '`stock_net_return_contributions.parquet`保留每只股票每日对组合净收益的贡献；`stock_risk_contributions.csv`为单股与组合净收益的协方差风险贡献；`drawdown_stock_contributions.csv`保留主要峰谷期间全部股票的净损益贡献。个股排名是对组合资产的贡献，没有历史指数个股权重，不能解释为逐股主动Alpha。股票名称可能变更，因此结果以代码为准。','',
            '最大绝对回撤区间的主要个股负贡献如下，按区间起点组合资产计；是损益贡献，不是股票自身跌幅：','',
            '| 股票代码 | 对组合净收益的贡献 |','| --- | ---: |']
    stock=pd.read_csv(out/'drawdown_stock_contributions.csv')
    worst=stock[(stock.kind=='absolute')&(stock['rank']==1)].sort_values('portfolio_PnL_contribution').head(5)
    for _,row in worst.iterrows():lines.append(f'| {row.instrument} | {row.portfolio_PnL_contribution*100:.3f}个百分点 |')
    lines += ['',f"上述五只合计{worst.portfolio_PnL_contribution.sum()*100:.2f}个百分点，占该次净回撤幅度约{abs(worst.portfolio_PnL_contribution.sum()/absolute.depth):.1%}；不支持把整次回撤归结为某一只异常股票。该比例为净损失比较，正负贡献会相互抵消，不是风险方差占比。",'',
            '日损益通过“前收盘持仓的隔夜收益 + 开盘后持仓的日内收益 − 成交价滑点 − 基础费用”重建，并与原账户逐日核对。停牌估值沿用原账户的最近有效价格；没有删除异常股票或改写原始账户。归因保留已声明的复权单位及日线执行局限。','',
            '## 复现与核查','',
            '无需原始行情，从保存的充分统计量重算归因、表格和四张图：','',
            '```bash','python -m src.risk_attribution report --input results/risk_attribution --output runs/risk-report','```','',
            '使用原始Qlib缓存重建逐股损益及前收盘暴露：','',
            '```bash','python -m src.risk_attribution build --data-dir /path/to/qlib_data --output runs/risk-rebuild','python -m unittest tests.test_risk_attribution','```','',
            '新输出目录必须不存在。输入指纹见`provenance.json`；逐日恒等式、协方差加总和峰谷链接均有断言。2016—2025已反复开发使用，本报告提供事后解释，不证明新样本外盈利。尚无完整历史行业、规模/估值暴露、基准权重、风险因子收益、因子协方差和特质风险，因此不是完整Barra事前风险模型，也不是Brinson行业配置/选择归因。','',
            '方法参考：[MSCI因子风险数据结构](https://developer.msci.com/apis/factor-analytics)、[CFA多因子模型与主动风险](https://www.cfainstitute.org/insights/professional-learning/refresher-readings/2026/using-multifactor-models)、[Qlib评价实现](https://github.com/microsoft/qlib/blob/main/qlib/contrib/evaluate.py)。本报告具体数值全部来自本项目保存账户。','']
    (out/'README.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='action',required=True)
    b=sub.add_parser('build');b.add_argument('--data-dir',type=Path,required=True);b.add_argument('--output',type=Path,required=True)
    r=sub.add_parser('report');r.add_argument('--input',type=Path,required=True);r.add_argument('--output',type=Path,required=True)
    a=p.parse_args();root=Path(__file__).resolve().parents[1]
    if a.action=='build':build(root,a.data_dir,a.output)
    else:generate(root,a.input,a.output)
