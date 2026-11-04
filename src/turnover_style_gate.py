"""Fixed 2x2 development experiment: industry/style risk and planned turnover cap.

Frozen alpha, no stock/cash exposure target change, no parameter sweep.
"""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from qlib.backtest.decision import Order,TradeDecisionWO
from .orders import decide_orders
from .marginal_risk_gate import MarginalGateStrategy,PLAN,variance_change
from .industry_style_risk import RiskArrays,factor_variance_change

CASES=['current_replay','turnover_6pct','industry_style','industry_style_turnover_6pct']


def planned_turnover(orders,prices,nav):
    return .5*sum(o.amount*prices[o.stock_id] for o in orders)/nav


def cap_exemption(records,held_count):
    if held_count==0:return 'initial_investment'
    if any(r['mandatory_exit'] or (r['forced'] and r['new'] is None) for r in records):return 'mandatory_exit_or_pure_trim'
    return 'none'


class ControlledGate(MarginalGateStrategy):
    def __init__(self,*,factor_risk=None,turnover_limit=None,**kwargs):
        super().__init__(**kwargs);self.factor_risk=factor_risk;self.turnover_limit=turnover_limit;self.turnover_plans=[]

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
        records_start=len(self.gate_decisions)
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
            if self.factor_risk is not None:
                fx,fc,fd,ready=self.factor_risk.at(day)
                if ready and incoming is not None:
                    change=factor_variance_change(weights,fx,fc,fd,old,incoming)
            penalty=(PLAN['risk_aversion']/2*PLAN['horizon']*change/max(weights.get(old,0.),1e-20)
                     if self.mode.startswith('risk_net_edge') else 0.)
            utility=edge-cost-penalty
            accepted=forced or (incoming is not None and utility>0.)
            self.gate_decisions.append(dict(trade_date=str(start.date()),signal_date=str(day.date()),old=old,new=incoming,
                slope=float(fit.slope),slope_SE=float(fit.SE),conservative_slope=float(fit.conservative_slope),
                applied_slope=coefficient,
                max_label_exit=str(fit.max_label_exit.date()),z_old=z.get(old,np.nan),z_new=z.get(incoming,np.nan),
                expected_edge=edge,cost_fraction=cost,risk_variance_change=change,risk_penalty=penalty,
                marginal_utility=utility,accepted=accepted,forced=forced,mandatory_exit=old not in score.index))
            if accepted:
                approved_sell.append(old)
                if incoming is not None:approved_buy.append(incoming)
                moved=weights.pop(old,0.)
                if incoming is not None:weights[incoming]=moved
        # Apply budget to whole replacement pairs, removing the last admitted pair.
        # This leaves preceding sequential marginal-risk calculations unchanged.
        cap_exempt=cap_exemption(self.gate_decisions[records_start:],len(held))
        orders=make_orders(self,approved_sell,approved_buy,prices,start,end)
        planned=planned_turnover(orders,prices,nav)
        if self.turnover_limit is not None and cap_exempt=='none':
            candidates=[r for r in self.gate_decisions[records_start:] if r['accepted'] and not r['mandatory_exit'] and r['new'] is not None]
            while planned>self.turnover_limit+1e-12 and candidates:
                row=candidates.pop();row['accepted']=False;row['turnover_rejected']=True
                approved_sell.remove(row['old']);approved_buy.remove(row['new'])
                orders=make_orders(self,approved_sell,approved_buy,prices,start,end)
                planned=planned_turnover(orders,prices,nav)
            if planned>self.turnover_limit+1e-12:cap_exempt='vacancy_fill'
        self.turnover_plans.append(dict(trade_date=str(start.date()),signal_date=str(day.date()),
            planned_one_way_turnover=planned,limit=self.turnover_limit,exception=cap_exempt))
        self.timing.append(dict(trade_date=str(start.date()),signal_date=str(day.date())))
        return TradeDecisionWO(orders,self)


def make_orders(self,approved_sell,approved_buy,prices,start,end):
        position=self.trade_position
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
        return orders


def run(root,data,cache,risk,out,cases=None):
    from .model import init_data
    from .open_execution import CausalExecutionExchange,OpeningExecutor,PREVIOUS,LAG_VOLUME,LAG_AMOUNT,validate
    from qlib.data import D
    from qlib.backtest import backtest
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);init_data(data)
    cfg=json.loads((root/'config.json').read_text(encoding='utf-8'))
    forecast=pd.read_parquet(root/'data/frozen_forecast.parquet').iloc[:,0]
    fit=pd.read_csv(cache/'calibration_schedule.csv',index_col=0,parse_dates=True);fit.max_label_exit=pd.to_datetime(fit.max_label_exit)
    beta=pd.read_parquet(cache/'risk_beta.parquet');specific=pd.read_parquet(cache/'risk_specific.parquet')
    variance=pd.read_csv(cache/'market_variance.csv',index_col=0,parse_dates=True).iloc[:,0]
    factors=RiskArrays(risk)
    cases=CASES if cases is None else cases
    (out/'plan.json').write_text(json.dumps(dict(cases=CASES,executed_cases=cases,planned_one_way_cap=.06,
        cap_definition='half planned absolute order notional / prior NAV, using prior close',
        cap_exceptions=['warmup before60 mature calibration days','initial investment','mandatory index exit or pure selling trim','vacancy fill if no optional pairs remain'],
        actual_open_turnover_not_hard_capped=True,risk_model=json.loads((risk/'metadata.json').read_text(encoding='utf-8')),
        unchanged=['frozen forecast','Top50 Drop5','5-day calibration','gamma4','3bp each side','fees','capacity','stock exposure rule'],
        development_reused=True,no_pristine_holdout=True,paper_changed=False),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    symbols=list(D.list_instruments(D.instruments('csi300'),start_time='2008-01-01',end_time=cfg['end'],as_list=True))
    market=CausalExecutionExchange(execution='open',slippage_bps=3.,capacity_fraction=.05,codes=symbols,
        start_time='2015-12-31',end_time=cfg['end'],freq='day',limit_threshold=.095,trade_unit=100,
        open_cost=.0005,close_cost=.0015,min_cost=5,subscribe_fields=['$open',PREVIOUS,LAG_VOLUME,LAG_AMOUNT])
    for case in cases:
        market.rejections=[]
        strategy=ControlledGate(mode='risk_net_edge_mean',calibration=fit,risk_beta=beta,risk_specific=specific,market_variance=variance,
            factor_risk=factors if case.startswith('industry_style') else None,
            turnover_limit=.06 if '6pct' in case else None,signal=forecast,topk=50,n_drop=5,hold_thresh=1,risk_degree=.95)
        strategy.prior_closes=market.quote_df['$close'].unstack(level='instrument')
        executor=OpeningExecutor(time_per_step='day',generate_portfolio_metrics=True)
        portfolio,_=backtest(start_time=cfg['start'],end_time=cfg['end'],strategy=strategy,executor=executor,
            benchmark='SH000300',account=cfg['initial_cash'],exchange_kwargs={'exchange':market})
        report,positions=portfolio['1day'];orders=pd.DataFrame(executor.fills);dest=out/case;dest.mkdir()
        report.to_csv(dest/'daily_report.csv');orders.to_parquet(dest/'orders.parquet',index=False)
        pd.DataFrame(strategy.timing).to_csv(dest/'signal_timing.csv',index=False)
        pd.DataFrame(strategy.gate_decisions).to_parquet(dest/'gate_decisions.parquet',index=False)
        pd.DataFrame(strategy.turnover_plans).to_csv(dest/'turnover_plans.csv',index=False)
        pd.DataFrame(market.rejections).to_csv(dest/'rejections.csv',index=False)
        pd.DataFrame([dict(date=d,instrument=c,weight=w) for d,p in positions.items() for c,w in p.get_stock_weight_dict().items()]).to_parquet(dest/'holdings.parquet',index=False)
        validate(cfg,report,orders,pd.DataFrame(strategy.timing),data)
        if case=='current_replay':
            saved=pd.read_csv(root/'results/net_edge_improvement/account/daily_report.csv',index_col=0,parse_dates=True,float_precision='round_trip')
            np.testing.assert_allclose(report.account,saved.account,atol=1e-5,rtol=1e-12)
        print('CONTROLLED_GATE_VALIDATED',case,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data-dir',type=Path,required=True);p.add_argument('--calibration-cache',type=Path,required=True)
    p.add_argument('--risk-cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cases',nargs='+',choices=CASES)
    a=p.parse_args();run(Path(__file__).resolve().parents[1],a.data_dir,a.calibration_cache,a.risk_cache,a.output,a.cases)
