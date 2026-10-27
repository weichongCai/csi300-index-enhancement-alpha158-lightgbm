"""Plan orders using only previous-close scores, prices, cash and holdings."""
import numpy as np
import pandas as pd
from qlib.backtest.decision import Order, TradeDecisionWO


def decide_orders(strategy):
    step = strategy.trade_calendar.get_trade_step()
    trade_start, trade_end = strategy.trade_calendar.get_step_time(step)
    signal_start, signal_end = strategy.trade_calendar.get_step_time(step, shift=1)
    score = strategy.signal.get_signal(start_time=signal_start, end_time=signal_end)
    if score is None:
        return TradeDecisionWO([], strategy)
    if isinstance(score, pd.DataFrame):
        score = score.iloc[:, 0]
    score = score.dropna()
    position = strategy.trade_position
    held = position.get_stock_list()
    previous = strategy.prior_closes.loc[pd.Timestamp(signal_start).normalize()].reindex(
        score.index.union(pd.Index(held))).copy()
    for code in held:
        if not np.isfinite(previous[code]) or previous[code] <= 0:
            previous[code] = position.get_stock_price(code)
    prices = previous.to_dict()
    score = score.loc[[c for c in score.index if np.isfinite(prices[c]) and prices[c] > 0]]
    last = score.reindex(held).sort_values(ascending=False).index
    new = score.loc[~score.index.isin(last)].sort_values(ascending=False).index[
        :max(0, strategy.n_drop + strategy.topk - len(last))]
    combined = score.reindex(last.union(new)).sort_values(ascending=False).index
    sell = last[last.isin(combined[-strategy.n_drop:])] if strategy.n_drop else pd.Index([])
    sell = [c for c in sell if position.get_stock_count(
        c, bar=strategy.trade_calendar.get_freq()) >= strategy.hold_thresh]
    buy = new[:max(0, len(sell) + strategy.topk - len(last))]
    orders = []
    expected_cash = position.get_cash()
    for code in sell:
        amount = position.get_stock_amount(code=code)
        if amount <= 0:
            continue
        value = amount * prices[code]
        expected_cash += max(0, value - max(
            value * strategy.trade_exchange.close_cost, strategy.trade_exchange.min_cost))
        orders.append(Order(stock_id=code, amount=amount, start_time=trade_start,
                            end_time=trade_end, direction=Order.SELL))
    per_stock = max(0, expected_cash) * strategy.risk_degree / len(buy) if len(buy) else 0
    for code in buy:
        budget = max(0, per_stock - strategy.trade_exchange.min_cost) / (
            1 + strategy.trade_exchange.open_cost)
        amount = budget / prices[code]
        if amount > 0:
            orders.append(Order(stock_id=code, amount=amount, start_time=trade_start,
                                end_time=trade_end, direction=Order.BUY))
    strategy.timing.append({'trade_date': str(trade_start.date()),
                            'signal_date': str(signal_start.date())})
    return TradeDecisionWO(orders, strategy)
