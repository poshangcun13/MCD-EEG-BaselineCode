"""Match the existing resting benchmark, separately for each N-back load."""
from pathlib import Path
import sys, json, hashlib, argparse
import numpy as np
import pandas as pd
import scipy.io as sio
import torch
from sklearn.model_selection import StratifiedKFold

HERE=Path(__file__).resolve().parent
REST=HERE
from experiment_paths import NB_DATA, OUTPUT_ROOT
sys.path.insert(0,str(REST))
import run_resting_benchmark as R
ROOT=OUTPUT_ROOT/'nback'

def load(condition,out):
    arrays=[]; hashes=[]; sources={}
    for session in (1,2):
        path=NB_DATA/f'{condition}-BACK-{session}.mat'
        mat=sio.loadmat(path); a=mat['X']
        assert a.shape==(10,60,800,120) and np.isfinite(a).all()
        x=a.transpose(0,3,1,2).reshape(1200,60,800).copy()
        y=np.repeat(np.arange(10),120)
        h=[hashlib.sha256(v.tobytes()).hexdigest() for v in x]
        assert len(set(h))==len(h)
        pd.DataFrame({'sample_index':np.arange(1200),'subject_id':y+1,
                      'trial_in_subject':np.tile(np.arange(1,121),10),'sha256':h}).to_csv(out/f'session{session}_samples.csv',index=False)
        sources[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
        arrays.append(x);hashes.append(h)
    assert not set(hashes[0]) & set(hashes[1]), 'Cross-session duplicate epochs'
    R.dump(out/'data_audit.json',{'input_shape':[10,60,800,120],'model_shape':[1200,60,800],
        'classes':10,'samples_per_subject_per_session':120,'source_sha256':sources,
        'cross_session_exact_duplicates':0,'finite':True,
        'assumptions':['X axis 0 is subject identity and has matching subject order in sessions 1 and 2',
                       '60-channel order matches across sessions','fs=400 Hz follows adjacent project protocol; MAT has no fs metadata'],
        'labels':'1-based subject axis positions, not externally verified person identifiers'})
    return arrays[0],y,arrays[1],y.copy()

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--epochs',type=int,default=80)
    parser.add_argument('--conditions',type=int,nargs='+',default=[0,1,2]);args=parser.parse_args()
    ROOT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    R.dump(ROOT/'protocol.json',{'train':'session 1 only, separately for each N-back load',
        'validation':'StratifiedKFold(5, shuffle=True, random_state=42) within session 1; 960 train / 240 val',
        'test':'all 1200 session-2 epochs evaluated by each of 5 models; never used for selection or normalization fitting',
        'summary':'mean +/- sample SD (ddof=1) across five trained models; same full test set',
        'metrics':'Acc; macro Recall/F1; macro subject one-vs-rest interpolated EER',
        'training':'Same local eight model implementations and hyperparameters as resting; 80 max epochs, min 20, patience 15, best val CE',
        'seeds':[4201,4202,4203,4204,4205], 'fs_assumption_hz':400,
        'architecture_notice':'Local recap variants trained from scratch, not official pretrained implementations',
        'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),
        'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [REST/'run_resting_benchmark.py',REST/'resting_models.py',Path(__file__)]}})
    for condition in args.conditions:
        R.OUT=ROOT/f'{condition}-back';R.OUT.mkdir(exist_ok=True)
        x,y,z,t=load(condition,R.OUT)
        folds=list(StratifiedKFold(5,shuffle=True,random_state=42).split(x,y))
        membership=[]
        for f,(tr,va) in enumerate(folds,1):
            assert set(tr).isdisjoint(va)
            for role,ix in [('train',tr),('validation',va)]:
                membership.extend({'fold':f,'role':role,'sample_index':int(i),'subject_id':int(y[i]+1)} for i in ix)
        pd.DataFrame(membership).to_csv(R.OUT/'fold_membership.csv',index=False)
        print('CONDITION',condition,flush=True)
        for name in R.NAMES:R.run(name,x,y,z,t,folds,args)
        del x,y,z,t
    print('ALL THREE CONDITIONS COMPLETE',flush=True)

if __name__=='__main__':main()
