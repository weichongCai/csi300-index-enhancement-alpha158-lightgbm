"""Offline, inspectable paper-account dashboard generated from committed ledger rows."""
from pathlib import Path
import json,sqlite3,csv

def render(account,names=None):
    account=Path(account);names=names or {}
    with sqlite3.connect(account/'ledger.sqlite') as db:
        days=[json.loads(r[0]) for r in db.execute('SELECT payload FROM days ORDER BY asof')]
    if not days:raise ValueError('No paper account day yet.')
    for d in days:
        ranked=[]
        for order in d['plan']:
            p=order.copy();basis=d['basis'][p['instrument']]
            p['name']=names.get(p['instrument'],p['instrument'])
            p['reference_price']=basis['close']/basis['factor']
            p['estimated_shares']=int(p['adjusted_amount']*basis['factor']//100)*100
            p['reference_value']=p['estimated_shares']*p['reference_price']
            if p['direction']==1 and p['instrument'].startswith('SH688') and p['estimated_shares']<200:
                p['estimated_shares']=0;p['reference_value']=0
            ranked.append(p)
        d['display_plan']=ranked
        d['display_holdings']=[{'instrument':c,'name':names.get(c,c),
            'value':h['amount']*h['price'],'weight':h['amount']*h['price']/(d['nav']*100000000)}
            for c,h in d['holdings'].items()]
    last=days[-1]
    fields=['instrument','name','direction','estimated_shares','reference_price','reference_value']
    for side,label in [(1,'next_buys'),(0,'next_sells')]:
        with (account/f'{label}.csv').open('w',encoding='utf-8-sig',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader()
            writer.writerows(p for p in last['display_plan'] if p['direction']==side)
    template=Path(__file__).with_name('dashboard.html').read_text(encoding='utf-8')
    payload=json.dumps(days,ensure_ascii=False,allow_nan=False).replace('<',chr(92)+'u003c')
    (account/'dashboard.html').write_text(template.replace('__ACCOUNT_DATA__',payload),encoding='utf-8')

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--account',type=Path,required=True);p.add_argument('--names',type=Path)
    a=p.parse_args();render(a.account,json.loads(a.names.read_text(encoding='utf-8')) if a.names else None)
