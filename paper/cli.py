"""Persistent local paper account. No broker API and no real-money order routing."""
from pathlib import Path
from types import SimpleNamespace
from collections import defaultdict
import argparse,copy,hashlib,json,sqlite3,sys,os
os.environ.setdefault('MLFLOW_DISABLE_AGENT_HINT','1')
os.environ.setdefault('OMP_NUM_THREADS','4')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def encode(value):
    return json.dumps(value,ensure_ascii=False,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else str(v))


class Ledger:
    def __init__(self,path):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        self.db=sqlite3.connect(self.path,timeout=10)
        self.db.execute('CREATE TABLE IF NOT EXISTS days(asof TEXT PRIMARY KEY,payload TEXT NOT NULL)')
    def latest(self):
        row=self.db.execute('SELECT payload FROM days ORDER BY asof DESC LIMIT 1').fetchone()
        return json.loads(row[0]) if row else None
    def save(self,asof,payload):
        self.db.execute('INSERT INTO days VALUES(?,?)',(asof,encode(payload)));self.db.commit()


class Calendar:
    def __init__(self,signal,trade):self.signal,self.trade=map(pd.Timestamp,[signal,trade])
    def get_trade_step(self):return 0
    def get_step_time(self,step=0,shift=0):
        day=self.signal if shift else self.trade
        return day,day+pd.Timedelta(hours=23,minutes=59)
    def get_freq(self):return 'day'


def run(feed,account,asof=None,historical=False):
    from src.model import init_data,predict
    from src.orders import decide_orders
    from src.slippage_exchange import SlippageExchange
    from qlib.data import D
    from qlib.backtest.position import Position
    from qlib.backtest.decision import Order
    feed,account=Path(feed).resolve(),Path(account).resolve()
    if not (feed/'verified.json').exists() and not historical:
        raise ValueError('Forward paper account requires a hash-verified feed.')
    calendar=pd.DatetimeIndex(pd.read_csv(feed/'calendars/day.txt',header=None)[0])
    day=pd.Timestamp(asof) if asof else calendar.max()
    today=pd.Timestamp.now(tz='Asia/Shanghai').tz_localize(None).normalize()
    if day not in calendar or day>calendar.max():raise ValueError('Requested close unavailable.')
    if not historical and (day!=today or pd.Timestamp.now(tz='Asia/Shanghai').hour<16):
        raise ValueError('Forward run requires today’s completed close, after 16:00 Shanghai time.')
    future=pd.DatetimeIndex(pd.read_csv(feed/'calendars/day_future.txt',header=None)[0])
    next_days=future[future>day]
    if not len(next_days):raise ValueError('Verified next trading date unavailable.')
    next_day=next_days[0]
    policy={name:digest(ROOT/name) for name in ['config.json','models/frozen_2017_2025_model.txt',
        'src/model.py','src/orders.py','src/exchange.py','src/slippage_exchange.py','paper/cli.py']}
    cfg=json.loads((ROOT/'config.json').read_text(encoding='utf-8'))
    ledger=Ledger(account/'ledger.sqlite');ledger.db.execute('BEGIN IMMEDIATE')
    previous=ledger.latest()
    if previous and previous['asof']==str(day.date()):
        ledger.db.rollback()
        (account/'latest.json').write_text(encode(previous)+'\n',encoding='utf-8')
        export(ledger,account)
        print('ALREADY_RECORDED',day.date());return previous
    if previous and (previous['asof']>str(day.date()) or previous['policy']!=policy or previous['historical']!=historical):
        raise ValueError('Account chronology, frozen policy or mode changed; inspect before continuing.')
    init_data(feed)
    if previous:
        pos=Position(cash=previous['cash'],position_dict=copy.deepcopy(previous['holdings']))
    else:pos=Position(cash=cfg['initial_cash'])
    signal_day=pd.Timestamp(previous['asof']) if previous else day
    settle_day=pd.Timestamp(previous['next_trade_date']) if previous else None
    scores=predict(ROOT,str(day.date()),str(day.date())).iloc[:,0]
    if len(scores)<250 or not np.isfinite(scores).all():raise ValueError('Insufficient signal coverage.')
    codes=sorted(set(scores.index.get_level_values('instrument'))|set(pos.get_stock_list())|
                 {p['instrument'] for p in previous['plan']} if previous else set(scores.index.get_level_values('instrument')))
    quotes=D.features(codes,['$close','$factor'],start_time=str(signal_day.date()),end_time=str(day.date()))
    def quote(code,date,field):
        try:return float(quotes.loc[(code,pd.Timestamp(date)),field])
        except KeyError:return np.nan
    scales={}
    if previous:
        for code,old in previous['basis'].items():
            current=quote(code,signal_day,'$factor');old_factor=old['factor']
            if not np.isfinite(current) or current<=0:raise ValueError('Previous unit basis unavailable: '+code)
            scale=current/old_factor;new_close=quote(code,signal_day,'$close')
            if not np.isfinite(new_close) or not np.isclose(new_close,old['close']*scale,rtol=2e-5):
                raise ValueError('Historical price revision inconsistent with factor: '+code)
            scales[code]=scale
            if code in pos.get_stock_list():
                pos.position[code]['amount']/=scale;pos.position[code]['price']*=scale
    fills=[]
    if previous and settle_day<=day:
        exchange=SlippageExchange(slippage_bps=3.,capacity_fraction=cfg['daily_capacity_fraction'],
            codes=codes,start_time=str(signal_day.date()),end_time=str(day.date()),freq='day',deal_price='close',
            limit_threshold=cfg['directional_limit_threshold'],trade_unit=cfg['trade_unit'],
            open_cost=cfg['open_cost'],close_cost=cfg['close_cost'],min_cost=cfg['min_cost'],
            subscribe_fields=['$high','$low','$amount','Ref($close,1)'])
        dealt=defaultdict(float)
        for planned in previous['plan']:
            code=planned['instrument'];amount=planned['adjusted_amount']/scales[code]
            order=Order(stock_id=code,amount=amount,start_time=settle_day,
                end_time=settle_day+pd.Timedelta(hours=23,minutes=59),direction=planned['direction'])
            value,cost,price=exchange.deal_order(order,position=pos,dealt_order_amount=dealt)
            dealt[code]+=order.deal_amount
            fills.append({'order_id':planned['order_id'],'trade_date':str(settle_day.date()),'instrument':code,
                'direction':order.direction,'filled_adjusted_amount':float(order.deal_amount),'value':float(value),
                'cost':float(cost),'price':float(price) if np.isfinite(price) else None,
                'status':('filled' if np.isclose(order.deal_amount,amount) else 'partial') if order.deal_amount>0 else 'rejected_or_zero',
                'requested_adjusted_amount':float(amount),'settlement_recorded_on':str(day.date())})
    stale=[]
    for code in pos.get_stock_list():
        price=quote(code,day,'$close')
        if np.isfinite(price) and price>0:pos.update_stock_price(code,price)
        else:stale.append(code)
    if previous:
        elapsed=int(((calendar>signal_day)&(calendar<=day)).sum())
        for _ in range(elapsed):pos.add_count_all('day')
    if pos.get_cash() < -1e-6:raise ArithmeticError('Negative paper cash.')
    starting_cash=previous['cash'] if previous else cfg['initial_cash']
    cash_from_fills=starting_cash+sum(f['value']*(1 if f['direction']==0 else -1)-f['cost'] for f in fills)
    if not np.isclose(pos.get_cash(),cash_from_fills,rtol=1e-11,atol=1e-5):
        raise ArithmeticError('Cash does not reconcile to fills and fees.')
    score=scores.xs(day,level='datetime');prior={c:quote(c,day,'$close') for c in codes}
    class Signal:
        def get_signal(self,**kwargs):return score
    strategy=SimpleNamespace(trade_calendar=Calendar(day,next_day),signal=Signal(),trade_position=pos,
        prior_closes=pd.DataFrame([prior],index=[day]),n_drop=cfg['n_drop'],topk=cfg['topk'],
        hold_thresh=cfg['hold_thresh'],risk_degree=cfg['buy_budget_fraction'],timing=[],
        trade_exchange=SimpleNamespace(open_cost=cfg['open_cost'],close_cost=cfg['close_cost'],min_cost=cfg['min_cost']))
    decision=decide_orders(strategy);plan=[]
    for order in decision.get_decision():
        plan.append({'order_id':f'PAPER-{day.date()}-{order.stock_id}-{int(order.direction)}',
            'instrument':order.stock_id,'direction':int(order.direction),'adjusted_amount':float(order.amount)})
    basis={}
    for code in set(pos.get_stock_list())|{p['instrument'] for p in plan}:
        factor=quote(code,day,'$factor');close=quote(code,day,'$close')
        if not (np.isfinite(factor) and factor>0 and np.isfinite(close) and close>0):
            raise ValueError('Cannot freeze unit basis: '+code)
        basis[code]={'factor':factor,'close':close}
    holdings={c:pos.position[c] for c in pos.get_stock_list()}
    result={'asof':str(day.date()),'next_trade_date':str(next_day.date()),'historical':historical,
        'recorded_at':str(pd.Timestamp.now(tz='Asia/Shanghai')),'policy':policy,'feed':str(feed),
        'cash':float(pos.get_cash()),'holdings':holdings,'nav':float(pos.calculate_value()/cfg['initial_cash']),
        'basis':basis,'fills':fills,'plan':plan,'stale_valuations':stale,'score_count':len(scores),
        'benchmark_total_return_available':False,'broker_connected':False,'slippage_bps':3.,
        'accounting':cfg['accounting']}
    ledger.save(str(day.date()),result)
    (account/'latest.json').write_text(encode(result)+'\n',encoding='utf-8')
    export(ledger,account)
    print('PAPER_DAY_RECORDED',day.date(),'NAV',result['nav'],'holdings',len(holdings),'pending',len(plan),flush=True)
    return result


def export(ledger,account):
    days=[json.loads(r[0]) for r in ledger.db.execute('SELECT payload FROM days ORDER BY asof')]
    pd.DataFrame([{'date':d['asof'],'nav':d['nav'],'cash':d['cash'],'holdings':len(d['holdings']),
        'pending_orders':len(d['plan']),'stale_valuations':len(d['stale_valuations'])} for d in days]).to_csv(account/'daily.csv',index=False)
    latest=days[-1]
    pd.DataFrame(latest['plan']).to_csv(account/'pending_orders.csv',index=False)
    pd.DataFrame([f for d in days for f in d['fills']]).to_csv(account/'fills.csv',index=False)
    lines=['# 沪深300本地模拟盘','',f"截至：{latest['asof']}；下一交易日：{latest['next_trade_date']}",'',
        f"净值：{latest['nav']:.6f}；现金：{latest['cash']:,.2f}元；持仓：{len(latest['holdings'])}只；待执行：{len(latest['plan'])}笔。",'',
        '这是本地日线模拟账户，不连接券商。前一日固定计划在下一交易日行情产生后按3bp滑点结算。',
        '账户沿用复权单位口径，不代表完整真实现金股份账簿。尚未接入同期H00300全收益行情，不计算或宣传模拟盘超额IR。',
        '历史演练标记：'+str(latest['historical']),'','[每日账户](daily.csv) · [待执行计划](pending_orders.csv) · [模拟成交](fills.csv)','']
    (account/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    if len(days)>1:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(9,4));ax.plot(pd.to_datetime([d['asof'] for d in days]),[d['nav'] for d in days],marker='o')
        ax.set_title('Local paper NAV | historical rehearsal' if latest['historical'] else 'Forward local paper NAV')
        ax.grid(alpha=.3);fig.autofmt_xdate();fig.tight_layout();fig.savefig(account/'nav.png',dpi=130);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--feed',type=Path,required=True);p.add_argument('--account',type=Path,required=True)
    p.add_argument('--asof');p.add_argument('--historical-rehearsal',action='store_true');a=p.parse_args()
    run(a.feed,a.account,a.asof,a.historical_rehearsal)
