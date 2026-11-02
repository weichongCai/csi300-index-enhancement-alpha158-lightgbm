"""Rebuild saved execution accounts from signed fills and source closing marks."""
from pathlib import Path
import argparse,json
import pandas as pd
import numpy as np
from .research_build import read_bin
parser=argparse.ArgumentParser()
parser.add_argument('--data-dir',type=Path,required=True)
args=parser.parse_args()
REPO=Path(__file__).resolve().parents[1]
source=REPO/'results/next_open_3bp';data=args.data_dir
cal=pd.DatetimeIndex(pd.read_csv(data/'calendars/day.txt',header=None)[0]);cal=cal[cal<='2025-12-31']
result={}
for case in ['open','close']:
    report=pd.read_csv(source/case/'daily_report.csv',index_col=0,parse_dates=True,float_precision='round_trip')
    orders=pd.read_parquet(source/case/'orders.parquet');orders['date']=pd.to_datetime(orders.date)
    orders['flow']=orders.filled_amount*np.where(orders.direction.eq(1),1.,-1.)
    flows=orders.pivot_table(index='date',columns='instrument',values='flow',aggfunc='sum').reindex(report.index).fillna(0.)
    quantity=flows.cumsum();assert quantity.to_numpy().min()>-1e-6
    marks=pd.DataFrame(index=report.index,columns=quantity.columns,dtype=float)
    positions=cal.get_indexer(report.index)
    for code in marks:
        close=read_bin(data,code,'close',cal).astype(float)
        reference=read_bin(data,code,case,cal)
        marked=pd.Series(np.where(np.isfinite(reference)&(reference>0),close,np.nan)).ffill().to_numpy()
        marks[code]=marked[positions]
    active=quantity.abs().gt(1e-6)
    assert (marks.notna()|~active).all().all()
    stock=(quantity*marks).where(active,0.).sum(axis=1)
    np.testing.assert_allclose(stock,report.value,rtol=1e-9,atol=.02)
    np.testing.assert_allclose(stock+report.cash,report.account,rtol=1e-9,atol=.02)
    first=orders[orders.date.eq(report.index[0])]
    close0=marks.loc[report.index[0]].reindex(first.instrument).to_numpy()
    pnl=(first.filled_amount.to_numpy()*(close0-first.price.to_numpy())).sum()-first.cost.sum()
    np.testing.assert_allclose(pnl,report.account.iloc[0]-1e8,atol=1e-5)
    result[case]={'independent_quantity_ledger_rebuilt_days':len(report),'short_positions':False,
        'cash_plus_marked_stock_matches_account':True,'first_day_intraday_PnL_included':True,
        'maximum_absolute_account_difference_CNY':float(abs(stock+report.cash-report.account).max())}
(source/'report/independent_account_audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result),flush=True)
