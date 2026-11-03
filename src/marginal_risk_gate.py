"""Keep frozen daily Top50/Drop5; trade only for positive estimated net utility.

Rolling calibration uses only labels whose exit open has already occurred.
No score is treated as a return before calibration. This is a new development
experiment, not a pristine holdout and not a full Barra risk model.
"""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import pandas as pd
from scipy.signal import lfilter
from qlib.backtest.decision import Order,TradeDecisionWO
from .backtest import FrozenStrategy
from .orders import decide_orders

PLAN=dict(horizon=5,calibration_window=126,minimum_calibration_days=60,
          calibration_half_life=60,uncertainty_haircut_SE=1.,HAC_lags=5,
          risk_aversion=4.,risk_window=252,risk_half_life=60,
          beta_minimum_pairs=126,specific_minimum_pairs=20,
          unknown_annual_specific_vol=1.,new_model_training=False,
          comparison=['baseline_replay','net_edge','risk_net_edge','net_edge_mean','risk_net_edge_mean'],
          stock_pool_and_top50_drop5_unchanged=True,development_data_reused=True,
          calibration_target='5-day open-to-open stock return, cross-section demeaned',
          risk_model='conditional single benchmark factor plus diagonal residual variance; no benchmark shrinkage',
          fees_and_slippage='buy 5bp, sell 15bp, 3bp each-side price slippage, minimum CNY5')


def ew_sum(a,window=252,half=60):
    decay=2**(-1/half);z=lfilter([1.],[1.,-decay],a)
    if len(z)>window:z[window:]-=decay**window*z[:-window].copy()
    return z


def conditional_risk(stock,benchmark):
    good=np.isfinite(stock)&np.isfinite(benchmark)
    n=np.cumsum(good);n[252:]-=np.cumsum(good)[:-252]
    x=np.where(good,benchmark,0.);y=np.where(good,stock,0.)
    mass=ew_sum(good.astype(float));den=np.maximum(mass,1e-30)
    mx=ew_sum(x)/den;my=ew_sum(y)/den
    vx=np.maximum(ew_sum(x*x)/den-mx*mx,0.)
    vy=np.maximum(ew_sum(y*y)/den-my*my,0.)
    cov=ew_sum(x*y)/den-mx*my
    known=(n>=126)&(vx>1e-14)
    beta=np.ones(len(stock));beta[known]=cov[known]/vx[known]
    residual=np.maximum(vy+beta*beta*vx-2*beta*cov,1e-12)
    residual[n<20]=1/252
    return beta,residual,known


def conservative_slope(values):
    a=np.asarray(values,dtype=float);a=a[np.isfinite(a)]
    if len(a)<PLAN['minimum_calibration_days']:return np.nan,np.nan,np.nan
    a=a[-PLAN['calibration_window']:]
    w=2.**(np.arange(len(a))/PLAN['calibration_half_life']);w/=w.sum()
    mean=float(w@a);e=a-mean;variance=float((w*w)@(e*e))
    for lag in range(1,PLAN['HAC_lags']+1):
        variance+=2*(1-lag/(PLAN['HAC_lags']+1))*float((w[lag:]*w[:-lag])@(e[lag:]*e[:-lag]))
    se=np.sqrt(max(0.,variance)*len(a)/(len(a)-1))
    return mean,float(se),max(0.,mean-PLAN['uncertainty_haircut_SE']*se)


def calibration_schedule(slopes,signal_dates):
    rows=[]
    for day in signal_dates:
        usable=slopes.loc[slopes.label_exit.le(day)&slopes.slope.notna()].tail(PLAN['calibration_window'])
        mean,se,lower=conservative_slope(usable.slope.to_numpy())
        rows.append(dict(signal_date=day,observations=len(usable),slope=mean,SE=se,
            conservative_slope=lower,max_label_exit=usable.label_exit.max() if len(usable) else pd.NaT))
    result=pd.DataFrame(rows).set_index('signal_date')
    assert (result.max_label_exit.dropna()<=result.max_label_exit.dropna().index).all()
    return result


def variance_change(weights,beta,specific,market_var,old,new):
    old_weight=weights.get(old,0.)
    portfolio_beta=sum(w*beta.get(c,1.) for c,w in weights.items())
    change=old_weight*(beta.get(new,1.)-beta.get(old,1.))
    return ((2*(portfolio_beta-1)*change+change*change)*market_var
            +old_weight**2*(specific.get(new,1/252)-specific.get(old,1/252)))


class MarginalGateStrategy(FrozenStrategy):
    def __init__(self,*,mode,calibration,risk_beta,risk_specific,market_variance,**kwargs):
        super().__init__(**kwargs);self.mode=mode;self.calibration=calibration
        self.risk_beta=risk_beta;self.risk_specific=risk_specific;self.market_variance=market_variance
        self.gate_decisions=[]

    def generate_trade_decision(self,execute_result=None):
        step=self.trade_calendar.get_trade_step()
        start,end=self.trade_calendar.get_step_time(step)
        signal_start,_=self.trade_calendar.get_step_time(step,shift=1)
        day=pd.Timestamp(signal_start).normalize();fit=self.calibration.loc[day]
        if self.mode=='baseline_replay' or not np.isfinite(fit.conservative_slope):
            return decide_orders(self)
        score=self.signal.get_signal(start_time=signal_start,end_time=signal_start)
        if score is None:return TradeDecisionWO([],self)
        if isinstance(score,pd.DataFrame):score=score.iloc[:,0]
        score=score.dropna()
        # Calibration and decisions must standardize the SAME contemporaneous
        # historical-member score universe, before price/tradability filtering.
        sd=score.std(ddof=1);z=(score-score.mean())/sd if sd>0 else score*0
        position=self.trade_position;held=position.get_stock_list()
        previous=self.prior_closes.loc[day].reindex(score.index.union(pd.Index(held))).copy()
        for c in held:
            if not np.isfinite(previous[c]) or previous[c]<=0:previous[c]=position.get_stock_price(c)
        prices=previous.to_dict()
        score=score.loc[[c for c in score.index if np.isfinite(prices[c]) and prices[c]>0]]
        coefficient=max(0.,float(fit.slope)) if self.mode.endswith('_mean') else float(fit.conservative_slope)
        last=score.reindex(held).sort_values(ascending=False).index
        new=score.loc[~score.index.isin(last)].sort_values(ascending=False).index[:max(0,self.n_drop+self.topk-len(last))]
        combined=score.reindex(last.union(new)).sort_values(ascending=False).index
        sell=last[last.isin(combined[-self.n_drop:])] if self.n_drop else pd.Index([])
        sell=[c for c in sell if position.get_stock_count(c,bar=self.trade_calendar.get_freq())>=self.hold_thresh]
        buy=list(new[:max(0,len(sell)+self.topk-len(last))])
        sell=sorted(sell,key=lambda c:score.get(c,-np.inf))
        nav=position.calculate_value()
        weights={c:position.get_stock_amount(c)*prices[c]/nav for c in held}
        beta=self.risk_beta.loc[day].to_dict();specific=self.risk_specific.loc[day].to_dict()
        approved_sell=[];approved_buy=[];vacant=max(0,self.topk-len(last))
        # Fill existing empty slots independently of optional replacements.
        approved_buy.extend(buy[:vacant]);available=buy[vacant:]
        for j,old in enumerate(sell):
            incoming=available[j] if j<len(available) else None
            value=position.get_stock_amount(old)*prices[old]
            forced=old not in score.index or len(held)>self.topk
            edge=float(coefficient*(z.get(incoming,0.)-z.get(old,0.))) if incoming else -np.inf
            cost=max(.0015,5/max(value,1e-20))+max(.0005,5/max(value,1e-20))+.0006
            change=variance_change(weights,beta,specific,float(self.market_variance.loc[day]),old,incoming) if incoming else 0.
            penalty=(PLAN['risk_aversion']/2*PLAN['horizon']*change/max(weights.get(old,0.),1e-20)
                     if self.mode.startswith('risk_net_edge') else 0.)
            utility=edge-cost-penalty
            accepted=forced or (incoming is not None and utility>0.)
            self.gate_decisions.append(dict(trade_date=str(start.date()),signal_date=str(day.date()),old=old,new=incoming,
                slope=float(fit.slope),slope_SE=float(fit.SE),conservative_slope=float(fit.conservative_slope),
                applied_slope=coefficient,
                max_label_exit=str(fit.max_label_exit.date()),z_old=z.get(old,np.nan),z_new=z.get(incoming,np.nan),
                expected_edge=edge,cost_fraction=cost,risk_variance_change=change,risk_penalty=penalty,
                marginal_utility=utility,accepted=accepted,forced=forced))
            if accepted:
                approved_sell.append(old)
                if incoming is not None:approved_buy.append(incoming)
                moved=weights.pop(old,0.)
                if incoming is not None:weights[incoming]=moved
        orders=[];expected_cash=position.get_cash()
        for c in approved_sell:
            amount=position.get_stock_amount(c)
            if amount<=0:continue
            value=amount*prices[c]
            expected_cash+=max(0,value-max(value*self.trade_exchange.close_cost,self.trade_exchange.min_cost))
            orders.append(Order(stock_id=c,amount=amount,start_time=start,end_time=end,direction=Order.SELL))
        per_stock=max(0,expected_cash)*self.risk_degree/len(approved_buy) if approved_buy else 0.
        for c in approved_buy:
            budget=max(0,per_stock-self.trade_exchange.min_cost)/(1+self.trade_exchange.open_cost)
            amount=budget/prices[c]
            if amount>0:orders.append(Order(stock_id=c,amount=amount,start_time=start,end_time=end,direction=Order.BUY))
        self.timing.append(dict(trade_date=str(start.date()),signal_date=str(day.date())))
        return TradeDecisionWO(orders,self)


def inputs(root,data,out):
    from .research_build import read_bin
    forecast=pd.read_parquet(root/'data/frozen_forecast.parquet').iloc[:,0]
    cal=pd.DatetimeIndex(pd.read_csv(data/'calendars/day.txt',header=None)[0]);cal=cal[cal<='2025-12-31']
    codes=sorted(forecast.index.get_level_values('instrument').unique())
    slots=cal.get_indexer(pd.DatetimeIndex(sorted(forecast.index.get_level_values('datetime').unique())))
    op=pd.DataFrame({c:read_bin(data,c,'open',cal).astype(float) for c in codes},index=cal)
    closes=pd.DataFrame({c:read_bin(data,c,'close',cal).astype(float) for c in codes},index=cal)
    signal_dates=cal[slots];rows=[]
    for i,day in enumerate(signal_dates):
        p=slots[i]
        if p+6>=len(cal):continue
        s=forecast.xs(day,level='datetime');sd=s.std(ddof=1)
        if not sd>0:continue
        z=(s-s.mean())/sd;y=op.iloc[p+6].reindex(s.index)/op.iloc[p+1].reindex(s.index)-1
        valid=z.notna()&np.isfinite(y)&op.iloc[p+1].reindex(s.index).gt(0)&op.iloc[p+6].reindex(s.index).gt(0)
        if valid.sum()<100:continue
        x=z.loc[valid];target=y.loc[valid];x=x-x.mean();target=target-target.mean()
        rows.append(dict(signal_date=day,label_exit=cal[p+6],pairs=int(valid.sum()),slope=float(x@target/(x@x))))
    slopes=pd.DataFrame(rows).set_index('signal_date');schedule=calibration_schedule(slopes,signal_dates)
    schedule.to_csv(out/'calibration_schedule.csv');slopes.to_csv(out/'mature_label_slopes.csv')
    official=pd.read_csv(root/'results/risk_management/risk_benchmark_returns.csv',index_col=0,parse_dates=True).benchmark_return
    market=official.reindex(cal).to_numpy();r=closes.pct_change(fill_method=None)
    rb={};rv={};known={}
    for c in codes:rb[c],rv[c],known[c]=conditional_risk(r[c].to_numpy(),market)
    beta=pd.DataFrame(rb,index=cal);specific=pd.DataFrame(rv,index=cal)
    beta.to_parquet(out/'risk_beta.parquet');specific.to_parquet(out/'risk_specific.parquet')
    m=np.where(np.isfinite(market),market,0.);w=ew_sum(np.isfinite(market).astype(float))
    variance=pd.Series(np.maximum(ew_sum(m*m)/np.maximum(w,1e-30)-(ew_sum(m)/np.maximum(w,1e-30))**2,1e-12),index=cal)
    variance.to_csv(out/'market_variance.csv',header=['variance'])
    return forecast,schedule,beta,specific,variance


def run(root,data,out,cases=None):
    from .model import init_data
    from .open_execution import CausalExecutionExchange,OpeningExecutor,PREVIOUS,LAG_VOLUME,LAG_AMOUNT,validate
    from qlib.data import D
    from qlib.backtest import backtest
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);init_data(data)
    cfg=json.loads((root/'config.json').read_text(encoding='utf-8'))
    forecast,fit,beta,specific,variance=inputs(root,data,out)
    cases=PLAN['comparison'] if cases is None else cases
    if not set(cases).issubset(PLAN['comparison']):raise ValueError('Unknown case')
    plan=dict(PLAN,executed_cases=cases,forecast_sha256=hashlib.sha256((root/'data/frozen_forecast.parquet').read_bytes()).hexdigest(),
        no_signal_future_exit_used=True,initial_warmup='original unmodified strategy until 60 mature daily slopes',
        strategy_model_unchanged=True,fixed_small_ablation=True,design_after_prior_development_feedback=True,
        independent_holdout_claimed=False,alpha_haircut_cases=['one_SE','nonnegative_mean'],risk_penalty_cases=[0,4])
    (out/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    symbols=list(D.list_instruments(D.instruments('csi300'),start_time='2008-01-01',end_time=cfg['end'],as_list=True))
    market=CausalExecutionExchange(execution='open',slippage_bps=3.,capacity_fraction=.05,codes=symbols,
        start_time='2015-12-31',end_time=cfg['end'],freq='day',limit_threshold=.095,trade_unit=100,
        open_cost=.0005,close_cost=.0015,min_cost=5,subscribe_fields=['$open',PREVIOUS,LAG_VOLUME,LAG_AMOUNT])
    for mode in cases:
        market.rejections=[]
        strategy=MarginalGateStrategy(mode=mode,calibration=fit,risk_beta=beta,risk_specific=specific,market_variance=variance,
            signal=forecast,topk=50,n_drop=5,hold_thresh=1,risk_degree=.95)
        strategy.prior_closes=market.quote_df['$close'].unstack(level='instrument')
        executor=OpeningExecutor(time_per_step='day',generate_portfolio_metrics=True)
        portfolio,_=backtest(start_time=cfg['start'],end_time=cfg['end'],strategy=strategy,executor=executor,
            benchmark='SH000300',account=cfg['initial_cash'],exchange_kwargs={'exchange':market})
        report,positions=portfolio['1day'];orders=pd.DataFrame(executor.fills)
        dest=out/mode;dest.mkdir();report.to_csv(dest/'daily_report.csv');orders.to_parquet(dest/'orders.parquet',index=False)
        pd.DataFrame(strategy.timing).to_csv(dest/'signal_timing.csv',index=False)
        pd.DataFrame(strategy.gate_decisions).to_parquet(dest/'gate_decisions.parquet',index=False)
        pd.DataFrame(market.rejections).to_csv(dest/'rejections.csv',index=False)
        pd.DataFrame([dict(date=d,instrument=c,weight=w) for d,p in positions.items() for c,w in p.get_stock_weight_dict().items()]).to_parquet(dest/'holdings.parquet',index=False)
        validate(cfg,report,orders,pd.DataFrame(strategy.timing),data)
        if mode=='baseline_replay':
            saved=pd.read_csv(root/'results/next_open_3bp/open/daily_report.csv',index_col=0,parse_dates=True,float_precision='round_trip')
            np.testing.assert_allclose(report.account,saved.account,atol=1e-5,rtol=1e-12)
        print('MARGINAL_GATE_ACCOUNT_VALIDATED',mode,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cases',nargs='+',choices=PLAN['comparison'])
    args=p.parse_args();run(Path(__file__).resolve().parents[1],args.data_dir,args.output,args.cases)
