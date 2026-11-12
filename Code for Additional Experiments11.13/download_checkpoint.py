"""Download the public EEGPT checkpoint and verify the fixed SHA-256.
No machine-specific proxy is required; standard environment proxy settings apply.
"""
from pathlib import Path
import argparse
from experiment_paths import WEIGHTS_ROOT
import hashlib,json,time,requests

HERE=WEIGHTS_ROOT
DEST=HERE/'eegpt_mcae_58chs_4s_large4E.ckpt'
EXPECTED='9d63ebc80bba1ad5e3277df4eecfa05d2745e6243a12ff743c4c0a93930dfac0'
SIZE=1020602524
ORIGIN='https://figshare.com/ndownloader/files/46452745?private_link=e37df4f8a907a866df4b'
MIRROR='https://hf-mirror.com/Zachary1150/ear-eeg-fm-benchmark-model/resolve/501c7fa6a368c74f77de916a03b91a4f3178f712/eegpt/eegpt_mcae_58chs_4s_large4E.ckpt?download=true'

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    HERE.mkdir(parents=True,exist_ok=True)
    if DEST.exists():
     assert DEST.stat().st_size==SIZE and sha(DEST)==EXPECTED
     print('EXISTING CHECKPOINT VERIFIED',flush=True)
     raise SystemExit

    partial=DEST.with_suffix('.ckpt.part')
    for url,proxy in [(ORIGIN,None),(MIRROR,None)]:
     for attempt in range(3):
      offset=partial.stat().st_size if partial.exists() else 0
      if offset==SIZE:break
      try:
       print('CONNECT',url.split('/')[2],'proxy',bool(proxy),'offset',offset,flush=True)
       r=requests.get(url,proxies=proxy,headers={'Range':f'bytes={offset}-'} if offset else {},stream=True,timeout=(12,40))
       r.raise_for_status()
       assert r.status_code in [200,206],f'Unexpected status {r.status_code}'
       assert 'text/html' not in r.headers.get('Content-Type',''),'Not a checkpoint response'
       if offset and r.status_code==200:offset=0
       mode='ab' if offset else 'wb'; last=time.time()
       with partial.open(mode) as f:
        for b in r.iter_content(2*1024*1024):
         f.write(b)
         if time.time()-last>20:print('DOWNLOADED_MB',round(f.tell()/1e6,1),flush=True);last=time.time()
       r.close()
       if partial.stat().st_size==SIZE:break
      except Exception as e:print('DOWNLOAD ATTEMPT',type(e).__name__,str(e)[:160],flush=True)
     if partial.exists() and partial.stat().st_size==SIZE:
      digest=sha(partial);assert digest==EXPECTED,(digest,EXPECTED)
      partial.replace(DEST)
      (HERE/'checkpoint_provenance.json').write_text(json.dumps({'official_source':ORIGIN,'download_source':url,'expected_sha256':EXPECTED,'actual_sha256':digest,'bytes':SIZE,'official_repository':'https://github.com/BINE022/EEGPT','mirror_reference':'501c7fa6a368c74f77de916a03b91a4f3178f712'},indent=2),encoding='utf-8')
      print('CHECKPOINT VERIFIED',digest,flush=True)
      break
    else:raise RuntimeError('Could not download a complete verified checkpoint')


if __name__=='__main__':
    main()
