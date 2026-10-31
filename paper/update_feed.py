"""Verified public Qlib release; stream archive and retain only recent daily inputs."""
from pathlib import Path,PurePosixPath
import urllib.request,json,hashlib,tarfile,sys,datetime,os,uuid
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
BASE=Path(os.environ.get('CSI300_FEED_DIR',str(ROOT/'work/paper-feed')));BASE.mkdir(parents=True,exist_ok=True)
def fetch(url):
    proxy=os.environ.get('CSI300_HTTP_PROXY')
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({'https':proxy}) if proxy else urllib.request.ProxyHandler())
    return opener.open(urllib.request.Request(url,headers={'User-Agent':'csi300-paper'}),timeout=60)
class Reader:
    def __init__(self,raw):self.raw=raw;self.sha=hashlib.sha256();self.count=0
    def read(self,n=-1):
        v=self.raw.read(n);self.sha.update(v);self.count+=len(v);return v
def main():
    tag=str(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).date())
    final=BASE/tag
    if (final/'verified.json').exists():print('FEED_ALREADY_VERIFIED',tag);return
    url='https://github.com/chenditc/investment_data/releases/download/'+tag+'/'
    assets={name:{'browser_download_url':url+name} for name in ['qlib_bin.manifest.json','qlib_bin.tar.gz']}
    with fetch(assets['qlib_bin.manifest.json']['browser_download_url']) as r:manifest=json.load(r)
    if final.exists():raise FileExistsError('Incomplete final feed requires inspection before retry.')
    out=BASE/('.staging-'+tag+'-'+uuid.uuid4().hex)
    out.mkdir()
    prior=BASE/'current.json'
    prior_path=Path(json.loads(prior.read_text(encoding='utf-8-sig'))['path']) if prior.exists() else ROOT/'work/cn_data'
    calendar=(prior_path/'calendars/day.txt').read_text().splitlines()
    earliest=max(0,len(calendar)-400)
    fields={'open','high','low','close','vwap','volume','amount','factor'};saved=0
    universe=(prior_path/'instruments/csi300.txt').read_text().splitlines()
    allowed={line.split()[0].lower() for line in universe};allowed.add('sh000300')
    with fetch(assets['qlib_bin.tar.gz']['browser_download_url']) as response:
        reader=Reader(response)
        with tarfile.open(fileobj=reader,mode='r|gz') as archive:
            for member in archive:
                if not member.isfile():continue
                parts=PurePosixPath(member.name).parts
                if 'features' in parts:
                    i=parts.index('features');rel=parts[i:]
                    if len(rel)!=3 or rel[-1].split('.')[0] not in fields or rel[1].lower() not in allowed:continue
                elif 'calendars' in parts:
                    i=parts.index('calendars');rel=parts[i:]
                    if len(rel)!=2 or rel[-1] not in {'day.txt','day_future.txt'}:continue
                elif 'instruments' in parts:
                    i=parts.index('instruments');rel=parts[i:]
                    if len(rel)!=2 or rel[-1]!='csi300.txt':continue
                else:continue
                if any(p in {'..','.'} or ':' in p for p in rel):raise ValueError('Unsafe archive path.')
                target=out.joinpath(*rel);target.parent.mkdir(parents=True,exist_ok=True)
                data=archive.extractfile(member).read()
                if rel[0]=='features':
                    a=np.frombuffer(data,dtype='<f4');first=int(a[0]);skip=max(0,earliest-first)
                    values=a[1+skip:]
                    if not len(values):continue
                    data=np.concatenate([np.array([first+skip],dtype='<f4'),values]).astype('<f4').tobytes()
                target.write_bytes(data);saved+=1
        while reader.read(1024*1024):pass
    expected=manifest['archive_sha256'].removeprefix('sha256:')
    if reader.sha.hexdigest()!=expected or reader.count!=manifest['archive_size_bytes']:
        raise ValueError('Public archive digest/length mismatch; feed stays unverified.')
    actual=(out/'calendars/day.txt').read_text().splitlines()[-1]
    new_members={line.split()[0].lower() for line in (out/'instruments/csi300.txt').read_text().splitlines()}
    missing=new_members-allowed
    if missing:
        # Rare membership expansion: second verified stream only for new symbols.
        with fetch(assets['qlib_bin.tar.gz']['browser_download_url']) as response:
            extra=Reader(response)
            with tarfile.open(fileobj=extra,mode='r|gz') as archive:
                for member in archive:
                    parts=PurePosixPath(member.name).parts
                    if not member.isfile() or 'features' not in parts:continue
                    rel=parts[parts.index('features'):]
                    if len(rel)!=3 or rel[1].lower() not in missing or rel[-1].split('.')[0] not in fields:continue
                    if any(p in {'..','.'} or ':' in p for p in rel):raise ValueError('Unsafe archive path.')
                    a=np.frombuffer(archive.extractfile(member).read(),dtype='<f4')
                    first=int(a[0]);skip=max(0,earliest-first);values=a[1+skip:]
                    if not len(values):continue
                    target=out.joinpath(*rel);target.parent.mkdir(parents=True,exist_ok=True)
                    target.write_bytes(np.concatenate([np.array([first+skip],dtype='<f4'),values]).astype('<f4').tobytes());saved+=1
            while extra.read(1024*1024):pass
            if extra.sha.hexdigest()!=expected:raise ValueError('Second stream digest mismatch.')
    proof={'release':tag,'latest_market_date':actual,'source_manifest':manifest,
        'verified_archive_sha256':expected,'archive_bytes':reader.count,'retained_files':saved,
        'retained_from_calendar_index':earliest,'archive_streamed_not_saved':True}
    (out/'verified.json').write_text(json.dumps(proof,indent=2)+'\n',encoding='utf-8')
    out.rename(final)
    pointer=BASE/'current.json.tmp'
    pointer.write_text(json.dumps({'path':str(final),'release':tag})+'\n',encoding='utf-8')
    pointer.replace(BASE/'current.json')
    print('PAPER_FEED_VERIFIED',json.dumps(proof),flush=True)
if __name__=='__main__':main()
