"""Replay the frozen forecast using the requested adjusted-price accounting."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from qlib.backtest import backtest
from qlib.backtest.executor import SimulatorExecutor
from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy
from qlib.data import D
from .exchange import AdjustedExchange
from .orders import decide_orders


class FrozenStrategy(TopkDropoutStrategy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.timing = []

    def generate_trade_decision(self, execute_result=None):
        return decide_orders(self)


class FillExecutor(SimulatorExecutor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fills = []

    def _collect_data(self, trade_decision, level=0):
        result, info = super()._collect_data(trade_decision, level)
        for order, value, cost, price in result:
            self.fills.append({
                'date': str(order.start_time.date()), 'instrument': order.stock_id,
                'direction': int(order.direction), 'requested_amount': float(order.amount),
                'filled_amount': float(order.deal_amount), 'price': float(price),
                'value': float(value), 'cost': float(cost),
            })
        return result, info


def run(root, data_dir, destination):
    root, data_dir, destination = map(Path, [root, data_dir, destination])
    if destination.exists():
        raise FileExistsError('Choose a new output directory; old runs are never overwritten.')
    cfg = json.loads((root / 'config.json').read_text(encoding='utf-8'))
    prediction = pd.read_parquet(root / 'data/frozen_forecast.parquet').iloc[:, 0]
    if (len(prediction) != 729016 or prediction.index.has_duplicates
            or prediction.index.get_level_values('datetime').min() != pd.Timestamp('2015-12-31')
            or prediction.index.get_level_values('datetime').max() != pd.Timestamp('2025-12-31')
            or not np.isfinite(prediction).all()):
        raise ValueError('Unexpected frozen prediction dates, shape or values.')
    symbols = list(D.list_instruments(D.instruments('csi300'),
                   start_time='2008-01-01', end_time=cfg['end'], as_list=True))
    exchange = AdjustedExchange(
        capacity_fraction=cfg['daily_capacity_fraction'], codes=symbols,
        start_time='2015-12-31', end_time=cfg['end'], freq='day', deal_price='close',
        limit_threshold=cfg['directional_limit_threshold'], trade_unit=cfg['trade_unit'],
        open_cost=cfg['open_cost'], close_cost=cfg['close_cost'], min_cost=cfg['min_cost'],
        subscribe_fields=['$high', '$low', '$amount', 'Ref($close,1)'])
    if exchange.trade_w_adj_price:
        raise ValueError('Qlib factor is missing; refusing to silently disable lot conversion.')
    strategy = FrozenStrategy(signal=prediction, topk=cfg['topk'], n_drop=cfg['n_drop'],
                               hold_thresh=cfg['hold_thresh'], risk_degree=cfg['buy_budget_fraction'])
    strategy.prior_closes = exchange.quote_df['$close'].unstack(level='instrument')
    executor = FillExecutor(time_per_step='day', generate_portfolio_metrics=True)
    portfolio, _ = backtest(start_time=cfg['start'], end_time=cfg['end'], strategy=strategy,
                            executor=executor, benchmark='SH000300', account=cfg['initial_cash'],
                            exchange_kwargs={'exchange': exchange})
    report, positions = portfolio['1day']
    net = report['return'] - report['cost']
    np.testing.assert_allclose((1 + net).cumprod(), report['account'] / cfg['initial_cash'],
                               rtol=1e-8, atol=1e-10)
    if report['cash'].lt(-1e-6).any():
        raise ArithmeticError('Negative cash.')
    calendar = pd.DatetimeIndex(pd.read_csv(data_dir / 'calendars/day.txt', header=None)[0])
    timing = pd.DataFrame(strategy.timing)
    signal_pos = calendar.get_indexer(pd.to_datetime(timing['signal_date']))
    trade_pos = calendar.get_indexer(pd.to_datetime(timing['trade_date']))
    if not np.all((signal_pos >= 0) & (trade_pos - signal_pos == 1)):
        raise ArithmeticError('Signal must precede execution by exactly one trading day.')
    destination.mkdir(parents=True)
    report.to_csv(destination / 'daily_report.csv')
    pd.DataFrame(executor.fills).to_csv(destination / 'orders.csv', index=False)
    timing.to_csv(destination / 'signal_timing.csv', index=False)
    holdings = [{'date': date, 'instrument': code, 'weight': weight}
                for date, position in positions.items()
                for code, weight in position.get_stock_weight_dict().items()]
    pd.DataFrame(holdings).to_csv(destination / 'holdings.csv', index=False)
    return report
