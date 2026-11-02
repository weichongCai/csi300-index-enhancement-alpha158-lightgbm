"""Fixed next-open experiment; no same-day high/low/volume in execution gates.

Run from repository root: python -m src.open_execution --data-dir PATH --output PATH
Existing model, configuration, release manifest and published accounts are untouched.
"""
from pathlib import Path
import os,json,argparse,hashlib
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('MLFLOW_DISABLE_AGENT_HINT','1')
import numpy as np
import pandas as pd
from qlib.backtest.exchange import Exchange
from qlib.backtest.decision import Order
from qlib.backtest.executor import SimulatorExecutor
from .exchange import AdjustedExchange
from .backtest import FrozenStrategy

PREVIOUS='Ref($close,1)'
LAG_VOLUME='Ref(Mean($volume,20),1)'
LAG_AMOUNT='Ref(Mean($amount,20),1)'


class CausalExecutionExchange(AdjustedExchange):
    def __init__(self,*,execution='open',slippage_bps=3.,**kwargs):
        self.execution=execution;self.slippage_bps=slippage_bps;self.rejections=[]
        super().__init__(deal_price=execution,**kwargs)

    def _update_limit(self,limit_threshold):
        p=self.quote_df['$'+self.execution];previous=self.quote_df[PREVIOUS]
        change=p/previous-1
        invalid=p.isna()|previous.isna()|p.le(0)|previous.le(0)
        self.quote_df['limit_buy']=invalid|change.ge(limit_threshold)
        self.quote_df['limit_sell']=invalid|change.le(-limit_threshold)

    def check_stock_suspended(self,stock_id,start_time,end_time):
        if stock_id not in self.quote.get_all_stock():return True
        p=self.quote.get_data(stock_id,start_time,end_time,field='$'+self.execution,method='ts_data_last')
        return p is None or not np.isfinite(float(p)) or p<=0

    def check_order(self,order):
        okay=Exchange.check_order(self,order)
        if not okay:self.rejections.append({'date':str(order.start_time.date()),'instrument':order.stock_id,
            'direction':int(order.direction),'reason':'missing execution reference or directional threshold'})
        return okay

    def get_deal_price(self,stock_id,start_time,end_time,direction,method='ts_data_last'):
        p=self.quote.get_data(stock_id,start_time,end_time,field='$'+self.execution,method=method)
        if p is None or not np.isfinite(float(p)) or p<=0:return np.nan
        return float(p)*(1+self.slippage_bps/10000*(1 if direction==Order.BUY else -1))

    def capacities(self,order):
        return (self._field(order,LAG_VOLUME)*100*self.capacity_fraction,
                self._field(order,LAG_AMOUNT)*1000*self.capacity_fraction)

    def _calc_trade_info_by_order(self,order,position,dealt_order_amount):
        # Deliberately do not call the old method: it reads current-day volume.
        price=self.get_deal_price(order.stock_id,order.start_time,order.end_time,order.direction)
        factor=self.get_factor(order.stock_id,order.start_time,order.end_time);order.factor=factor
        volume,amount=self.capacities(order)
        valid=all(np.isfinite(v) and v>0 for v in [price,factor,volume,amount])
        already=float(dealt_order_amount.get(order.stock_id,0.))
        quantity=min(order.amount,max(0,min(volume,amount/price)-already)) if valid else 0.
        rate=self.open_cost if order.direction==Order.BUY else self.close_cost
        liquidate_full=False
        if position is not None:
            if order.direction==Order.SELL:
                held=position.get_stock_amount(order.stock_id) if position.check_stock(order.stock_id) else 0.
                quantity=min(quantity,held)
                liquidate_full=np.isclose(quantity,held)
            else:
                cash=position.get_cash()
                if quantity*price+max(quantity*price*rate,self.min_cost)>cash:
                    quantity=min(quantity,max(0,self._get_buy_amount_by_cash_limit(price,cash,rate)))
        if valid and not liquidate_full:quantity=np.floor(quantity*factor/100.)*100./factor
        if position is not None and order.direction==Order.SELL:quantity=min(quantity,held)
        if order.direction==Order.BUY and order.stock_id.startswith('SH688') and quantity*factor<199.9:quantity=0.
        value=quantity*price if valid else 0.
        cost=max(value*rate,self.min_cost) if value>1e-5 else 0.
        if position is not None and order.direction==Order.BUY:
            while quantity>0 and value+cost>position.get_cash()+1e-7:
                quantity=max(0,quantity-100/factor);value=quantity*price
                cost=max(value*rate,self.min_cost) if value>1e-5 else 0.
        order.deal_amount=quantity
        return price,value,cost


class OpeningExecutor(SimulatorExecutor):
    def __init__(self,**kwargs):super().__init__(**kwargs);self.fills=[]
    def _collect_data(self,trade_decision,level=0):
        result,info=super()._collect_data(trade_decision,level)
        market=self.trade_exchange
        for order,value,cost,price in result:
            ref=market._field(order,'$'+market.execution);vc,ac=market.capacities(order)
            self.fills.append({'date':str(order.start_time.date()),'instrument':order.stock_id,
                'direction':int(order.direction),'requested_amount':float(order.amount),
                'filled_amount':float(order.deal_amount),'price':float(price),'reference_price':ref,
                'value':float(value),'cost':float(cost),'lag_volume_cap':vc,'lag_amount_cap':ac})
        return result,info


def simulate(root,data,dest,execution):
    from qlib.data import D
    from qlib.backtest import backtest
    cfg=json.loads((root/'config.json').read_text(encoding='utf-8'))
    forecast=pd.read_parquet(root/'data/frozen_forecast.parquet').iloc[:,0]
    symbols=list(D.list_instruments(D.instruments('csi300'),start_time='2008-01-01',end_time=cfg['end'],as_list=True))
    market=CausalExecutionExchange(execution=execution,slippage_bps=3.,capacity_fraction=.05,codes=symbols,
        start_time='2015-12-31',end_time=cfg['end'],freq='day',
        limit_threshold=.095,trade_unit=100,open_cost=.0005,close_cost=.0015,min_cost=5,
        subscribe_fields=['$open',PREVIOUS,LAG_VOLUME,LAG_AMOUNT])
    assert not market.trade_w_adj_price
    strategy=FrozenStrategy(signal=forecast,topk=50,n_drop=5,hold_thresh=1,risk_degree=.95)
    strategy.prior_closes=market.quote_df['$close'].unstack(level='instrument')
    executor=OpeningExecutor(time_per_step='day',generate_portfolio_metrics=True)
    portfolio,_=backtest(start_time=cfg['start'],end_time=cfg['end'],strategy=strategy,executor=executor,
        benchmark='SH000300',account=cfg['initial_cash'],exchange_kwargs={'exchange':market})
    report,positions=portfolio['1day'];orders=pd.DataFrame(executor.fills)
    dest.mkdir();report.to_csv(dest/'daily_report.csv');orders.to_parquet(dest/'orders.parquet',index=False)
    pd.DataFrame(strategy.timing).to_csv(dest/'signal_timing.csv',index=False)
    pd.DataFrame(market.rejections).to_csv(dest/'rejections.csv',index=False)
    pd.DataFrame([{'date':d,'instrument':s,'weight':w} for d,p in positions.items()
        for s,w in p.get_stock_weight_dict().items()]).to_parquet(dest/'holdings.parquet',index=False)
    validate(cfg,report,orders,pd.DataFrame(strategy.timing),data)
    return report


def validate(cfg,report,orders,timing,data):
    positive=orders.value.gt(1e-5);fills=orders.loc[positive]
    np.testing.assert_allclose(fills.price,fills.reference_price*np.where(fills.direction.eq(1),1.0003,.9997),rtol=1e-12)
    np.testing.assert_allclose(fills.value,fills.price*fills.filled_amount,rtol=1e-12,atol=1e-6)
    np.testing.assert_allclose(orders.cost,np.where(positive,np.maximum(orders.value*np.where(orders.direction.eq(1),.0005,.0015),5),0),rtol=1e-12,atol=1e-6)
    o=orders.assign(date=pd.to_datetime(orders.date));fees=o.groupby('date').cost.sum().reindex(report.index,fill_value=0)
    buys=o[o.direction.eq(1)].groupby('date').value.sum().reindex(report.index,fill_value=0)
    sells=o[o.direction.eq(0)].groupby('date').value.sum().reindex(report.index,fill_value=0)
    prior=report.account.shift(1).fillna(cfg['initial_cash'])
    np.testing.assert_allclose(report.cash,report.cash.shift(1).fillna(cfg['initial_cash'])+sells-buys-fees,atol=1e-5,rtol=1e-10)
    np.testing.assert_allclose(report.cost*prior,fees,atol=1e-5,rtol=1e-10)
    np.testing.assert_allclose((1+report['return']-report.cost).cumprod(),report.account/cfg['initial_cash'],rtol=1e-8)
    assert report.cash.ge(-1e-6).all() and len(report)==2430
    cap=fills.groupby(['date','instrument']).agg(v=('value','sum'),a=('lag_amount_cap','first'),q=('filled_amount','sum'),c=('lag_volume_cap','first'))
    assert cap.v.le(cap.a+1e-5).all() and cap.q.le(cap.c+1e-6).all()
    calendar=pd.DatetimeIndex(pd.read_csv(Path(data)/'calendars/day.txt',header=None)[0])
    assert np.all(calendar.get_indexer(pd.to_datetime(timing.trade_date))-calendar.get_indexer(pd.to_datetime(timing.signal_date))==1)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data-dir',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    config={'experiment':'next_open_3bp_fixed_forecast','labels_and_model_unchanged':True,
        'cases':['open','close'],'reference_slippage_bps':3,'capacity':'5% of previous completed 20-day average volume and amount',
        'directional_threshold':.095,'use_current_day_high_low_for_fill_gate':False,
        'use_current_day_close_or_volume_for_open_fill_gate':False,'position_rule':'original daily Top50/Drop5',
        'funding_assumption':'sell then buy sequentially at reference price; simultaneous auction funding not proven',
        'development_period_reused':True,
        'forecast_sha256':hashlib.sha256((root/'data/frozen_forecast.parquet').read_bytes()).hexdigest()}
    (args.output/'experiment.json').write_text(json.dumps(config,indent=2)+'\n',encoding='utf-8')
    from .model import init_data
    init_data(args.data_dir)
    for case in ['open','close']:
        simulate(root,args.data_dir,args.output/case,case)
        print('ACCOUNT_CHECKED',case,flush=True)


if __name__=='__main__':main()
