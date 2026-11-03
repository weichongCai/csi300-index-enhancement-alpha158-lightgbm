"""Independent, fixed-rule risk-budget experiment; frozen alpha is never refit.

Joint stock/benchmark return covariance needs no fictitious index holdings.
All decision inputs are sliced at the previous completed close.
"""
from pathlib import Path
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from qlib.backtest.decision import Order, TradeDecisionWO
from .backtest import FrozenStrategy

POLICY = dict(window=252, minimum_observations=126, target_stock_weight=.99,
              maximum_stock_weight=.04, annual_te_limit=.10,
              beta_lower=.85, beta_upper=1.15, two_way_turnover_limit=.40,
              risk_penalty=.10, alpha_tracking_penalty=.50,
              TE_budget_type='soft variance budget with explicit overflow; full investment has priority',
              TE_overflow_penalty=10., candidate_pool='all current history-qualified members plus actual holdings',
              numerical_dust_target_weight=1e-4, unknown_specific_volatility_reserve=1.0,
              signal_budget='reference rank exposure if attainable; otherwise maximum attainable under hard constraints, explicit shortfall',
              solver_objective_scale=1000.,
              buy_cost_proxy=.0008, sell_cost_proxy=.0018,
              covariance='joint LedoitWolf on complete histories; single-market conditional extension for partial histories',
              review='first trading day of each ISO week',
              admission='new positions require 252 nonmissing consecutive return observations',
              missing='partial history: market model; fewer than 126 returns: 100% annual specific-vol reserve, no target holding',
              infeasible='keep actual holdings on hard-constraint failure; record TE soft-budget overflow')


def estimate_risk(stock_returns, benchmark_returns, asof, codes, *, window=252, minimum=126, allow_reserve=False):
    """No return imputation. Explicit PSD market extension prevents a stale-name deadlock.

    Complete names use joint shrinkage. A partial name is beta*market + independent
    residual; missing residual correlations are an assumption, never observed zeros.
    Unestimable names use beta=1 and a clearly marked 100% annual specific-vol reserve.
    This reserve is a scenario convention, NOT an upper bound on possible loss.
    """
    x = stock_returns.loc[:asof, codes].tail(window).join(benchmark_returns.rename('__benchmark'))
    x = x.replace([np.inf,-np.inf],np.nan)
    if not allow_reserve and len(x.dropna()) < minimum:
        raise ValueError(f'insufficient joint history: {len(x.dropna())}')
    x=x.loc[x.__benchmark.notna()]
    if len(x)<minimum:raise ValueError('insufficient benchmark history')
    full=[c for c in codes if x[c].notna().all()]
    # The inverse/precision matrix is unused; disabling it leaves covariance exactly unchanged.
    fitted=LedoitWolf(store_precision=False).fit(x[[*full,'__benchmark']].to_numpy(dtype=float))
    market_var=float(fitted.covariance_[-1,-1])
    if market_var<=1e-14:raise ValueError('benchmark has no variance')
    n=len(codes);matrix=np.zeros((n+1,n+1));beta=np.zeros(n);kinds={};reserve=[]
    slots=[codes.index(c) for c in full]+[n]
    matrix[np.ix_(slots,slots)]=fitted.covariance_
    for c in full:
        i=codes.index(c);beta[i]=matrix[i,-1]/market_var;kinds[c]='joint_shrinkage'
    partial=[];specific={}
    for c in codes:
        if c in full:continue
        i=codes.index(c);pair=x[[c,'__benchmark']].dropna();partial.append(i)
        if len(pair)>=minimum:
            a=pair.to_numpy();sample=np.cov(a,rowvar=False,ddof=1)
            beta[i]=sample[0,1]/sample[1,1]
            residual=a[:,0]-beta[i]*a[:,1]
            specific[i]=max(float(np.var(residual,ddof=1)),1e-12)
            kinds[c]='partial_history_market_extension'
        else:
            beta[i]=1.;specific[i]=POLICY['unknown_specific_volatility_reserve']**2/252
            kinds[c]='unestimable_100pct_specific_vol_reserve';reserve.append(c)
        matrix[i,-1]=matrix[-1,i]=beta[i]*market_var
        for j in slots[:-1]:matrix[i,j]=matrix[j,i]=beta[i]*matrix[-1,j]
    for i in partial:
        for j in partial:matrix[i,j]=beta[i]*beta[j]*market_var
        matrix[i,i]+=specific[i]
    # Unestimable-name residuals are assumed perfectly correlated, conservatively
    # aggregating their designated reserve rather than diversifying it away.
    for c in reserve:
        for d in reserve:
            if c!=d:matrix[codes.index(c),codes.index(d)]+=POLICY['unknown_specific_volatility_reserve']**2/252
    x.attrs['risk_kind']=kinds;x.attrs['reserve_codes']=reserve
    return matrix,beta,x,float(fitted.shrinkage_)


def risk_numbers(weights, covariance):
    """Cash earns zero; benchmark weight is -1 even if stocks total < 1."""
    v = np.r_[weights, -1.]
    cv = covariance @ v
    variance = float(v @ cv)
    if variance < -1e-12:
        raise ValueError('negative variance')
    return np.sqrt(max(0., variance) * 252), v * cv * 252


def optimize_weights(reference, previous, covariance, beta, *, bootstrap=False, policy=POLICY, upper_bounds=None, signal=None, return_details=False):
    import cvxpy as cp
    n = len(reference)
    w = cp.Variable(n)
    delta = w - previous
    v = cp.hstack([w, -1.])
    active_var = cp.quad_form(v, cp.psd_wrap(covariance * 252))
    cost = policy['buy_cost_proxy'] * cp.sum(cp.pos(delta)) + policy['sell_cost_proxy'] * cp.sum(cp.pos(-delta))
    overflow=cp.Variable(nonneg=True)
    loss = (policy['alpha_tracking_penalty'] * cp.sum_squares(w-reference)
            + policy['risk_penalty'] * active_var + cost + policy['TE_overflow_penalty']*overflow)
    turn = 2. if bootstrap else policy['two_way_turnover_limit']
    upper=np.full(n,policy['maximum_stock_weight']) if upper_bounds is None else np.asarray(upper_bounds)
    constraints = [w >= 0, w <= upper,
                   cp.sum(w) == policy['target_stock_weight'],
                   cp.norm1(delta) <= turn,
                   beta @ w >= policy['beta_lower'], beta @ w <= policy['beta_upper'],
                   active_var <= policy['annual_te_limit'] ** 2 + overflow]
    details={}
    if signal is not None:
        signal=np.asarray(signal)
        # First identify the attainable signal frontier under actual hard limits.
        # The TE constraint is soft (overflow is unrestricted); exclude it from this LP.
        frontier=cp.Problem(cp.Maximize(1000*signal@w),constraints[:-1])
        try:frontier.solve(solver='CLARABEL',tol_gap_abs=1e-8,tol_feas=1e-8,max_iter=200)
        except cp.error.SolverError as error:raise ValueError(f'signal frontier solver: {error}') from error
        if frontier.status!='optimal' or w.value is None:
            raise ValueError(f'signal frontier status: {frontier.status}')
        maximum=float(signal@w.value)
        floor=min(float(signal@reference),maximum-1e-7)
        details=dict(attainable_rank_exposure=maximum,required_rank_exposure=floor,
            rank_reference_feasibility_shortfall=max(0.,float(signal@reference)-maximum))
        constraints.append(signal@w >= floor)
    # Scaling the entire objective changes numerical conditioning, not its minimizer.
    problem = cp.Problem(cp.Minimize(policy['solver_objective_scale']*loss), constraints)
    try:problem.solve(solver='CLARABEL', tol_gap_abs=1e-8, tol_feas=1e-8, max_iter=200)
    except cp.error.SolverError as error:raise ValueError(f'solver error: {error}') from error
    if problem.status != 'optimal' or w.value is None:
        raise ValueError(f'optimizer status: {problem.status}')
    result = np.maximum(np.asarray(w.value), 0.)
    te, _ = risk_numbers(result, covariance)
    checks = [abs(result.sum()-policy['target_stock_weight']) < 2e-6,
              np.all(result <= upper+2e-6),
              abs(result-previous).sum() <= turn+2e-6,
              policy['beta_lower']-2e-6 <= beta@result <= policy['beta_upper']+2e-6,
              te**2 <= policy['annual_te_limit']**2+float(overflow.value)+2e-6]
    if not all(checks):
        raise ArithmeticError('post-solve constraint verification failed')
    if signal is not None and signal@result<details['required_rank_exposure']-2e-6:
        raise ArithmeticError('signal budget verification failed')
    answer=(result,float(problem.value/policy['solver_objective_scale']))
    return (*answer,details) if return_details else answer


class RiskBudgetStrategy(FrozenStrategy):
    def __init__(self, *, stock_returns, benchmark_returns, mode, **kwargs):
        super().__init__(**kwargs)
        self.stock_returns = stock_returns
        self.benchmark_returns = benchmark_returns
        self.mode = mode
        self.last_week = None
        self.decisions = []
        self.targets = []

    def generate_trade_decision(self, execute_result=None):
        step = self.trade_calendar.get_trade_step()
        start, end = self.trade_calendar.get_step_time(step)
        signal_start, _ = self.trade_calendar.get_step_time(step, shift=1)
        asof = pd.Timestamp(signal_start).normalize()
        week = (start.isocalendar().year, start.isocalendar().week)
        if week == self.last_week:
            return TradeDecisionWO([], self)
        self.last_week = week
        score = self.signal.get_signal(start_time=signal_start, end_time=signal_start)
        if score is None:
            return TradeDecisionWO([], self)
        if isinstance(score, pd.DataFrame): score = score.iloc[:, 0]
        score = score.dropna().sort_index().sort_values(ascending=False, kind='stable')
        position = self.trade_position
        held = position.get_stock_list()
        prices = self.prior_closes.loc[asof].copy()
        for c in held:
            if not np.isfinite(prices.get(c, np.nan)) or prices[c] <= 0:
                prices[c] = position.get_stock_price(c)
        history = self.stock_returns.loc[:asof].tail(POLICY['window'])
        good = history.notna().sum().eq(POLICY['window'])
        eligible = [c for c in score.index if good.get(c, False) and prices.get(c, 0)>0]
        top = eligible[:50]
        row = {'trade_date':str(start.date()), 'signal_date':str(asof.date()), 'mode':self.mode}
        self.timing.append({k:row[k] for k in ['trade_date','signal_date']})
        if len(top) < 50:
            self.decisions.append(dict(row, status='skip: fewer than 50 qualified stocks'))
            return TradeDecisionWO([], self)
        codes = sorted(set(eligible) | set(held))
        nav = position.calculate_value()
        previous = np.array([position.get_stock_amount(c)*prices[c]/nav if c in held else 0. for c in codes])
        reference = np.array([POLICY['target_stock_weight']/50 if c in top else 0. for c in codes])
        ranks=score.reindex(eligible).rank(pct=True)
        signal=np.array([ranks.get(c,0.) for c in codes])
        try:
            covariance, beta, hist, shrink = estimate_risk(self.stock_returns, self.benchmark_returns, asof, codes,allow_reserve=True)
            upper=np.array([0. if c in hist.attrs['reserve_codes'] else POLICY['maximum_stock_weight'] for c in codes])
            # Holdings outside the current admitted member pool can be reduced but not increased.
            upper=np.minimum(upper,np.array([POLICY['maximum_stock_weight'] if c in eligible else previous[i] for i,c in enumerate(codes)]))
            target, objective,details = (optimize_weights(reference, previous, covariance, beta, bootstrap=not held,upper_bounds=upper,signal=signal,return_details=True)
                                 if self.mode == 'optimized' else (reference,np.nan,{}))
            te, rc = risk_numbers(target, covariance)
            prev_te, _ = risk_numbers(previous, covariance)
            used=target>1e-8;scenario_history=hist.loc[:,np.r_[used,True]].dropna()
            scenario = scenario_history.iloc[:,:-1].to_numpy() @ target[used]-scenario_history.iloc[:,-1].to_numpy()
            cutoff = np.quantile(scenario, .05) if len(scenario) else np.nan
            self.decisions.append(dict(row, status='planned', n=len(codes), observations=len(hist),
                history_end=str(hist.index.max().date()), shrinkage=shrink, objective=objective,
                prior_predicted_te=prev_te, target_predicted_te=te,
                target_beta=float(beta@target), target_max_weight=float(target.max()),
                TE_budget_excess=max(0.,te-POLICY['annual_te_limit']),
                TE_variance_budget_excess=max(0.,te**2-POLICY['annual_te_limit']**2),
                target_two_way_turnover=float(abs(target-previous).sum()),
                prior_unestimable_weight=float(sum(previous[codes.index(c)] for c in hist.attrs['reserve_codes'])),
                reference_rank_exposure=float(signal@reference),target_rank_exposure=float(signal@target),
                historical_active_VaR95=-float(cutoff),
                historical_active_ES95=-float(scenario[scenario<=cutoff].mean()) if len(scenario) else np.nan,
                target_worst_past_day=float(scenario.min()) if len(scenario) else np.nan,
                target_expected_linear_cost=float((np.maximum(target-previous,0)*.0008+np.maximum(previous-target,0)*.0018).sum())))
            self.decisions[-1].update(details)
            for i,c in enumerate(codes):
                self.targets.append(dict(row, instrument=c, previous_weight=previous[i], reference_weight=reference[i],
                    target_weight=target[i],signal_rank=signal[i],annual_active_variance_contribution=rc[i]))
                self.targets[-1]['risk_kind']=hist.attrs['risk_kind'][c]
            self.targets.append(dict(row, instrument='__benchmark', previous_weight=-1., reference_weight=-1.,
                target_weight=-1., annual_active_variance_contribution=rc[-1]))
        except (ValueError, ArithmeticError) as error:
            self.decisions.append(dict(row, status=f'skip: {error}'))
            return TradeDecisionWO([], self)
        orders = []
        # Both experimental arms share the legacy sell-then-buy funding assumption.
        # Orders use previous-close fixed quantities. Actual fills can violate target weights.
        for direction in [Order.SELL, Order.BUY]:
            for i,c in enumerate(codes):
                quantity = max(0., target[i]*nav/prices[c])
                old = position.get_stock_amount(c) if c in held else 0.
                # No optimizer numerical dust: a target smaller than 1bp is compiled
                # to a full exit (or no new order). Target vs fill weights stay separate.
                if target[i]<POLICY['numerical_dust_target_weight']:quantity=0.
                difference = quantity-old
                if direction == Order.SELL and difference < -1e-8:
                    if position.get_stock_count(c, bar=self.trade_calendar.get_freq()) >= 1:
                        orders.append(Order(stock_id=c, amount=-difference, start_time=start, end_time=end, direction=direction))
                if direction == Order.BUY and difference > 1e-8:
                    orders.append(Order(stock_id=c, amount=difference, start_time=start, end_time=end, direction=direction))
        return TradeDecisionWO(orders, self)


def build_inputs(root, data, historical_benchmark):
    from .research_build import read_bin
    calendar = pd.DatetimeIndex(pd.read_csv(data/'calendars/day.txt', header=None)[0])
    calendar = calendar[(calendar>='2014-01-01') & (calendar<='2025-12-31')]
    # read_bin offsets refer to the original calendar, so read with the full prefix.
    full = pd.DatetimeIndex(pd.read_csv(data/'calendars/day.txt', header=None)[0])
    full = full[full<='2025-12-31']
    forecast = pd.read_parquet(root/'data/frozen_forecast.parquet').iloc[:,0]
    codes = sorted(forecast.index.get_level_values('instrument').unique())
    closes = pd.DataFrame({c:read_bin(data,c,'close',full).astype(float) for c in codes}, index=full).loc[calendar]
    r = closes.pct_change(fill_method=None)
    benchmark = pd.read_csv(root/'data/official_index_levels.csv',index_col=0,parse_dates=True).H00300
    raw=json.loads(Path(historical_benchmark).read_text(encoding='utf-8'))
    if not raw.get('success'):raise ValueError('official history request failed')
    old=pd.DataFrame(raw['data'])
    if not old.indexCode.eq('H00300').all():raise ValueError('wrong historical index')
    old=pd.Series(old.close.to_numpy(dtype=float),index=pd.to_datetime(old.tradeDate,format='%Y%m%d')).sort_index()
    overlap=old.index.intersection(benchmark.index)
    if len(overlap)<1:raise ValueError('history has no overlap for validation')
    np.testing.assert_allclose(old.loc[overlap],benchmark.loc[overlap],atol=.011,rtol=0)
    benchmark=pd.concat([old.loc[old.index<benchmark.index.min()],benchmark]).sort_index()
    b = benchmark.pct_change(fill_method=None).reindex(calendar)
    return r,b,forecast


def run(root,data,out,historical_benchmark,weekly_reference=None):
    from .model import init_data
    from .open_execution import CausalExecutionExchange,OpeningExecutor,PREVIOUS,LAG_VOLUME,LAG_AMOUNT,validate
    from qlib.data import D
    from qlib.backtest import backtest
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True)
    init_data(data)
    cfg = json.loads((root/'config.json').read_text(encoding='utf-8'))
    stocks,benchmark,forecast = build_inputs(root,data,historical_benchmark)
    (out/'official_benchmark_history.json').write_bytes(Path(historical_benchmark).read_bytes())
    stock_path=out/'risk_stock_returns.parquet'; stocks.to_parquet(stock_path)
    benchmark.to_frame('benchmark_return').to_csv(out/'risk_benchmark_returns.csv')
    plan=dict(POLICY, development_period_reused=True,no_parameter_grid_search=True,design_revised_using_development_feedback=True,
        frozen_forecast_sha256=hashlib.sha256((root/'data/frozen_forecast.parquet').read_bytes()).hexdigest(),
        first_day_benchmark='same opening proxy as published next-open experiment',
        comparison='weekly_equal vs optimized isolate construction; daily baseline includes frequency difference',
        index_holdings_available=False, official_Barra=False, funding='same legacy sequential sell-then-buy assumption',
        stock_price_basis='adjusted cached close; no separate corporate-action ledger',
        historical_benchmark_url='https://www.csindex.com.cn/csindex-home/perf/index-perf?indexCode=H00300&startDate=20131231&endDate=20160108',
        historical_benchmark_sha256=hashlib.sha256(Path(historical_benchmark).read_bytes()).hexdigest())
    (out/'policy.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    symbols=list(D.list_instruments(D.instruments('csi300'),start_time='2008-01-01',end_time=cfg['end'],as_list=True))
    for mode in ['weekly_equal','optimized']:
        if mode=='weekly_equal' and weekly_reference is not None:
            import shutil
            # An explicitly selected unchanged reference account can be reused, then audited again.
            shutil.copytree(weekly_reference,out/mode)
            print('UNCHANGED_WEEKLY_REFERENCE_REUSED',flush=True)
            continue
        market=CausalExecutionExchange(execution='open',slippage_bps=3.,capacity_fraction=.05,codes=symbols,
            start_time='2015-12-31',end_time=cfg['end'],freq='day',limit_threshold=.095,trade_unit=100,
            open_cost=.0005,close_cost=.0015,min_cost=5,subscribe_fields=['$open',PREVIOUS,LAG_VOLUME,LAG_AMOUNT])
        strategy=RiskBudgetStrategy(stock_returns=stocks,benchmark_returns=benchmark,mode=mode,
            signal=forecast,topk=50,n_drop=5,hold_thresh=1,risk_degree=.99)
        strategy.prior_closes=market.quote_df['$close'].unstack(level='instrument')
        executor=OpeningExecutor(time_per_step='day',generate_portfolio_metrics=True)
        portfolio,_=backtest(start_time=cfg['start'],end_time=cfg['end'],strategy=strategy,executor=executor,
            benchmark='SH000300',account=cfg['initial_cash'],exchange_kwargs={'exchange':market})
        report,positions=portfolio['1day']; orders=pd.DataFrame(executor.fills)
        dest=out/mode;dest.mkdir()
        report.to_csv(dest/'daily_report.csv');orders.to_parquet(dest/'orders.parquet',index=False)
        pd.DataFrame(strategy.timing).to_csv(dest/'signal_timing.csv',index=False)
        pd.DataFrame(strategy.decisions).to_csv(dest/'decisions.csv',index=False)
        pd.DataFrame(strategy.targets).to_parquet(dest/'targets.parquet',index=False)
        pd.DataFrame(market.rejections).to_csv(dest/'rejections.csv',index=False)
        pd.DataFrame([dict(date=d,instrument=c,weight=w) for d,p in positions.items()
            for c,w in p.get_stock_weight_dict().items()]).to_parquet(dest/'holdings.parquet',index=False)
        # Timing validation for weekly decisions uses the same one-trading-day invariant.
        validate(cfg,report,orders,pd.DataFrame(strategy.timing),data)
        print('RISK_ACCOUNT_VALIDATED',mode,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--historical-benchmark',type=Path,required=True)
    parser.add_argument('--weekly-reference',type=Path)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    run(Path(__file__).resolve().parents[1],args.data_dir,args.output,args.historical_benchmark,args.weekly_reference)
