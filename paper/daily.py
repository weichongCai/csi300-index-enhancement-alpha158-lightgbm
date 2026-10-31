"""One locked daily invocation; the Codex heartbeat invokes this entry point."""
from pathlib import Path
import argparse,datetime,json,os,subprocess,sys
import pandas as pd

def main(home):
    home=Path(home).resolve();home.mkdir(parents=True,exist_ok=True)
    lock=home/'daily.lock'
    try:handle=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:raise RuntimeError('Another update is active; inspect lock before retrying.')
    os.write(handle,str(os.getpid()).encode());os.close(handle)
    status={'started_at':str(pd.Timestamp.now(tz='Asia/Shanghai')),'mode':'local_paper_only'}
    try:
        now=pd.Timestamp.now(tz='Asia/Shanghai')
        if now.weekday()>=5:
            status.update(state='market_weekend',message='休市，保留上一个交易日计划，不生成重复订单。');return
        if now.hour<17:raise RuntimeError('Daily job runs after 17:00 Shanghai time.')
        env=os.environ.copy();env['CSI300_FEED_DIR']=str(home/'feeds');env['MPLCONFIGDIR']=str(home/'matplotlib')
        env['MLFLOW_DISABLE_AGENT_HINT']='1';env['OMP_NUM_THREADS']='4'
        app=home/'app';logs=home/'logs';logs.mkdir(exist_ok=True)
        def invoke(name,args):
            with (logs/f'{now.date()}-{name}.log').open('a',encoding='utf-8') as f:
                subprocess.run([sys.executable,'-B','-X','utf8',str(app/'paper'/name),*args],env=env,
                    stdout=f,stderr=subprocess.STDOUT,check=True,timeout=2400)
        invoke('update_feed.py',[])
        feed=Path(json.loads((home/'feeds/current.json').read_text(encoding='utf-8-sig'))['path'])
        latest=(feed/'calendars/day.txt').read_text().splitlines()[-1]
        if latest!=str(now.date()):
            status.update(state='waiting_for_current_close',latest_market_date=latest,
                message='尚无当日完整行情，暂停生成新计划。');return
        invoke('cli.py',['--feed',str(feed),'--account',str(home/'account')])
        args=['--account',str(home/'account')]
        if (home/'names.json').exists():args+=['--names',str(home/'names.json')]
        invoke('dashboard.py',args)
        last=json.loads((home/'account/latest.json').read_text(encoding='utf-8'))
        status.update(state='ready',asof=last['asof'],next_trade_date=last['next_trade_date'],
            pending_orders=len(last['plan']),nav=last['nav'],dashboard=str(home/'account/dashboard.html'))
    except Exception as e:
        status.update(state='blocked',message=f'{type(e).__name__}: {e}')
        raise
    finally:
        status['finished_at']=str(pd.Timestamp.now(tz='Asia/Shanghai'))
        (home/'status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        lock.unlink()
        print(json.dumps(status,ensure_ascii=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--home',type=Path,required=True);a=p.parse_args();main(a.home)
