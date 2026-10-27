"""Reproducible comparison with five published CSI300 enhancement fund NAVs."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .metrics import drawdown, ir


def fund_returns(published, calendar, codes):
    """Reinvest cash on all publication dates, then align; never fill a missing NAV."""
    required = {'code', 'date', 'unit_nav', 'cash_per_unit'}
    if not required.issubset(published.columns):
        raise ValueError('Missing published NAV columns.')
    result = pd.DataFrame(index=calendar)
    for code in codes:
        rows = published.loc[published.code.eq(code)].sort_values('date')
        dates = pd.DatetimeIndex(rows.date)
        if rows.empty or dates.has_duplicates:
            raise ValueError(f'Empty or duplicate published dates: {code}')
        nav = rows.unit_nav.to_numpy(dtype=float)
        cash = rows.cash_per_unit.to_numpy(dtype=float)
        if not (np.isfinite(nav).all() and (nav > 0).all()
                and np.isfinite(cash).all() and (cash >= 0).all()):
            raise ValueError(f'Invalid NAV or dividend: {code}')
        gross = np.r_[1., (nav[1:] + cash[1:]) / nav[:-1]]
        wealth = pd.Series(np.cumprod(gross), index=dates)
        if not calendar.isin(wealth.index).all():
            raise ValueError(f'Missing account-calendar NAV: {code}')
        aligned = wealth.loc[calendar]
        returns = aligned.pct_change(fill_method=None)
        returns.iloc[0] = 0.  # Initial purchase is at the same first close.
        result[code] = returns
    return result


def compare(root):
    root = Path(root)
    metadata = json.loads((root / 'data/peers/sources.json').read_text(encoding='utf-8'))
    published = pd.read_csv(root / 'data/peers/published_nav.csv',
                            dtype={'code': str}, parse_dates=['date'])
    model = pd.read_csv(root / 'results/daily.csv', index_col=0, parse_dates=True)
    if (len(model) != 2430 or model.index.has_duplicates
            or not model.index.is_monotonic_increasing
            or model.index[0] != pd.Timestamp('2016-01-04')
            or model.index[-1] != pd.Timestamp('2025-12-31')):
        raise ValueError('Expected the fixed 2016–2025 account calendar.')
    funds = metadata['funds']
    returns = fund_returns(published, model.index, funds)
    returns.insert(0, 'model', model.model_return)
    benchmark = model.benchmark_return
    if benchmark.iloc[0] != 0 or not np.isfinite(returns.to_numpy()).all():
        raise ValueError('Invalid funding match or non-finite returns.')
    navs = (1 + returns).cumprod()
    benchmark_nav = (1 + benchmark).cumprod()
    relative = navs.div(benchmark_nav, axis=0)
    absolute_dd = navs.apply(drawdown)
    relative_dd = relative.apply(drawdown)
    names = {'model': '冻结模型（复权回测）', **{c: f['name'] for c, f in funds.items()}}
    annual, metrics = [], []
    for code in returns:
        r = returns[code]
        active = r - benchmark
        ratios = []
        wins = 0
        for year, group in r.groupby(r.index.year):
            b = benchmark.loc[group.index]
            ratio = ir(group - b)
            ratios.append(ratio)
            model_return = float((1 + group).prod() - 1)
            benchmark_return = float((1 + b).prod() - 1)
            wins += int(model_return > benchmark_return)
            annual.append({'code': code, 'name': names[code], 'year': int(year),
                            'return': model_return, 'benchmark_return': benchmark_return,
                            'active_ir': ratio})
        metrics.append({'code': code, 'name': names[code],
            'series_type': 'development_backtest' if code == 'model' else 'published_fund_nav',
            'mean_annual_net_active_ir': float(np.mean(ratios)), 'pooled_net_active_ir': ir(active),
            'calendar_10year_cagr': float(navs[code].iloc[-1] ** .1 - 1),
            'relative_10year_cagr': float(relative[code].iloc[-1] ** .1 - 1),
            'absolute_max_drawdown': float(absolute_dd[code].min()),
            'relative_max_drawdown': float(relative_dd[code].min()),
            'annual_tracking_error': float(active.std(ddof=1) * np.sqrt(252)),
            'positive_active_years': wins, 'terminal_nav': float(navs[code].iloc[-1])})
    navs['H00300'] = benchmark_nav
    plot_data = pd.concat({'nav': navs, 'relative_nav': relative,
                           'absolute_drawdown': absolute_dd,
                           'relative_drawdown': relative_dd}, axis=1)
    return pd.DataFrame(metrics), pd.DataFrame(annual), returns, plot_data, metadata


def save_comparison(root, destination):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError('Choose a new comparison output directory.')
    metrics, annual, returns, plot_data, metadata = compare(root)
    destination.mkdir(parents=True)
    metrics.to_csv(destination / 'metrics.csv', index=False)
    annual.to_csv(destination / 'annual.csv', index=False)
    returns.to_csv(destination / 'daily_returns.csv', index_label='date')
    plot_data.to_csv(destination / 'plot_series.csv', index_label='date')
    plot(plot_data, annual, metadata, destination)
    text = ['# 与五只沪深300增强基金对照', '',
            '2016—2025，共2430个交易日；同首日收盘投入、统一H00300全收益基准。', '',
            '| 样本 | 平均年度超额IR | 全期超额IR | 十年年化 | 最大回撤 | 跟踪误差 |',
            '| --- | ---: | ---: | ---: | ---: | ---: |']
    for row in metrics.itertuples():
        text.append(f'| {row.name} | {row.mean_annual_net_active_ir:.3f} | '
                    f'{row.pooled_net_active_ir:.3f} | {row.calendar_10year_cagr:.2%} | '
                    f'{row.absolute_max_drawdown:.2%} | {row.annual_tracking_error:.2%} |')
    text += ['', '![净值、相对净值与相对回撤](comparison.png)', '',
             '![逐年超额IR](annual_ir.png)', '',
             '基金使用单位净值和每份现金分红重建再投资回报，保留所有公布日期后再对齐共同交易日；不把累计净值当复权净值，不填补缺失报价。模型保留首日费用，基金初始申购费未模拟。', '',
             '这五只长期存续基金是固定示例，不是全市场无偏排名。基金净值包含基金运营费用，模型仅扣声明交易费用；资产规模、集中度、行业风格约束与风险预算不同。模型的2016—2025为反复使用的开发期，基金为真实公布净值，不能据此认定实盘优于这些产品。', '',
             '统一基准是100%沪深300全收益指数，不是每只基金合同中各自的业绩比较基准。此次仅添加比较，没有重新训练、选参或改变交易规则。', '',
             '数据来源与分红口径见[data/peers/README.md](../../data/peers/README.md)。']
    (destination / 'README.md').write_text('\n'.join(text) + '\n', encoding='utf-8', newline='\n')
    return metrics


def plot(series, annual, metadata, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels = {'model': 'Model (development backtest)',
              **{c: f['chart_label'] for c, f in metadata['funds'].items()},
              'H00300': 'CSI300 total return'}
    colors = dict(zip(labels, ['#176a91', '#d18c31', '#8866aa', '#36856e',
                               '#ba6277', '#7d853b', '#555555']))
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True,
                              gridspec_kw={'height_ratios': [2, 1, 1]})
    for code in labels:
        width = 2.1 if code == 'model' else 1.0
        axes[0].plot(series.index, series['nav'][code], color=colors[code],
                      lw=width, label=labels[code], ls='--' if code == 'H00300' else '-')
        if code == 'H00300':
            continue
        axes[1].plot(series.index, series['relative_nav'][code], color=colors[code], lw=width)
        axes[2].plot(series.index, series['relative_drawdown'][code] * 100,
                      color=colors[code], lw=width)
    axes[0].set_yscale('log')
    axes[0].legend(loc='upper left', fontsize=8, ncol=2)
    axes[0].set_title('2016–2025 | frozen research backtest vs five published fund NAVs')
    axes[0].set_ylabel('NAV (log scale)')
    axes[1].set_ylabel('NAV / CSI300 TR')
    axes[1].axhline(1, color='#aaa', linewidth=.7)
    axes[2].set_ylabel('Relative drawdown (%)')
    for axis in axes:
        axis.grid(alpha=.2)
    fig.tight_layout(rect=[0, .025, 1, 1])
    fig.text(.08, .012, 'Same first-close funding. Different fees, scale and risk constraints; '
             'selected surviving examples, not a market-wide ranking.', fontsize=8)
    fig.savefig(destination / 'comparison.png', dpi=160)
    plt.close(fig)
    order = list(labels)[:-1]
    matrix = annual.pivot(index='code', columns='year', values='active_ir').loc[order]
    matrix['Mean'] = matrix.mean(axis=1)
    fig, axis = plt.subplots(figsize=(13, 4.6))
    image = axis.imshow(matrix, cmap='RdYlGn', vmin=-3, vmax=3, aspect='auto')
    axis.set_xticks(range(11), [str(x) for x in matrix.columns])
    axis.set_yticks(range(len(order)), [labels[c] for c in order])
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            axis.text(j, i, f'{matrix.iloc[i, j]:.2f}', ha='center', va='center', fontsize=8)
    axis.set_title('Annual daily active IR vs CSI300 total return | 252 days, sample standard deviation')
    fig.colorbar(image, ax=axis, shrink=.8)
    fig.tight_layout()
    fig.savefig(destination / 'annual_ir.png', dpi=160)
    plt.close(fig)
