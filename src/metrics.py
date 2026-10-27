"""Funding-matched total-return benchmark, active IR, NAV and drawdown."""
from pathlib import Path
import json
import numpy as np
import pandas as pd


def ir(active):
    std = active.std(ddof=1)
    return float(active.mean() / std * np.sqrt(252)) if std > 0 else float('nan')


def drawdown(nav):
    # Initial capital is a valid peak, even before the first day's trading fee.
    peak = np.maximum.accumulate(np.r_[1., np.asarray(nav, dtype=float)])[1:]
    return np.asarray(nav, dtype=float) / peak - 1


def evaluate(root, report):
    cfg = json.loads((Path(root) / 'config.json').read_text(encoding='utf-8'))
    benchmark_levels = pd.read_csv(Path(root) / 'data/official_index_levels.csv',
                                  index_col=0, parse_dates=True)['H00300']
    levels = benchmark_levels.reindex(report.index)
    if levels.isna().any() or not report.index.is_monotonic_increasing:
        raise ValueError('Benchmark and account dates do not match.')
    benchmark = levels.pct_change(fill_method=None)
    benchmark.iloc[0] = 0.  # Buy the benchmark at the same first execution close.
    net = report['return'] - report['cost']
    nav = report['account'] / cfg['initial_cash']
    np.testing.assert_allclose(nav, (1 + net).cumprod(), rtol=1e-8, atol=1e-10)
    active = net - benchmark
    benchmark_nav = (1 + benchmark).cumprod()
    daily = pd.DataFrame({'model_return': net, 'benchmark_return': benchmark,
                           'active_return': active, 'model_nav': nav,
                           'benchmark_nav': benchmark_nav, 'relative_nav': nav / benchmark_nav,
                           'model_drawdown': drawdown(nav),
                           'relative_drawdown': drawdown(nav / benchmark_nav)})
    annual_rows = []
    for year, group in daily.groupby(daily.index.year):
        annual_rows.append({'year': int(year),
            'model_return': float((1 + group.model_return).prod() - 1),
            'benchmark_return': float((1 + group.benchmark_return).prod() - 1),
            'net_active_ir': ir(group.active_return)})
    annual = pd.DataFrame(annual_rows).set_index('year')
    if len(annual) != 10 or len(report) != 2430:
        raise ValueError('This frozen evaluation requires 10 complete years / 2430 trading days.')
    metrics = {
        'accounting': cfg['accounting'], 'start': str(report.index[0].date()),
        'end': str(report.index[-1].date()), 'trading_days': len(report),
        'mean_annual_net_active_ir': float(annual.net_active_ir.mean()),
        'pooled_net_active_ir': ir(active), 'calendar_10year_cagr': float(nav.iloc[-1] ** .1 - 1),
        'absolute_max_drawdown': float(daily.model_drawdown.min()),
        'relative_max_drawdown': float(daily.relative_drawdown.min()),
        'active_tracking_error': float(active.std(ddof=1) * np.sqrt(252)),
        'terminal_nav': float(nav.iloc[-1]),
        'mean_stock_exposure': float((report['value'] / report['account']).mean()),
        'benchmark': 'H00300 total return; initial close funding matched',
        'development_data_reused': True, 'raw_share_ledger_equivalence_claimed': False,
    }
    return metrics, annual, daily


def save_results(root, report, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    metrics, annual, daily = evaluate(root, report)
    (destination / 'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + '\n',
                                             encoding='utf-8')
    annual.to_csv(destination / 'annual.csv')
    daily.to_csv(destination / 'daily.csv', index_label='date')
    plot(daily, destination / 'backtest.png')
    return metrics


def plot(daily, filename):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True,
                              gridspec_kw={'height_ratios': [2, 1, 1]})
    daily[['model_nav', 'benchmark_nav']].rename(
        columns={'model_nav': 'Frozen Alpha158 model', 'benchmark_nav': 'CSI300 total return'}
    ).plot(ax=axes[0], logy=True, color=['#176a91', '#979da5'])
    axes[0].set_ylabel('Net asset value (log scale)')
    axes[0].set_title('2016–2025 development backtest | adjusted-price accounting | after trading fees')
    daily.relative_nav.plot(ax=axes[1], color='#18775b', label='Model / benchmark')
    axes[1].axhline(1, color='#aaa', linewidth=.8)
    axes[1].set_ylabel('Relative NAV')
    (daily[['model_drawdown', 'relative_drawdown']] * 100).rename(
        columns={'model_drawdown': 'Absolute drawdown', 'relative_drawdown': 'Relative drawdown'}
    ).plot(ax=axes[2], color=['#b14a44', '#18775b'])
    axes[2].set_ylabel('Drawdown (%)')
    for axis in axes:
        axis.grid(alpha=.2)
        axis.legend(loc='upper left')
    fig.tight_layout()
    fig.savefig(filename, dpi=160)
    plt.close(fig)
