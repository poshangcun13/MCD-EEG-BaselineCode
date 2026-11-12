"""Compact EEGNet-style temporal/spatial CNN, unseen-subject evaluation.

Outer-test labels never control epoch selection. Each outer train set is split
by subjects for validation; two fixed learning rates are tried, then the selected
epoch count/configuration is refit on the full outer train set.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
from pathlib import Path
import argparse
import random
import json
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import butter,sosfiltfilt,resample_poly
import torch
from torch import nn
from sklearn.model_selection import StratifiedShuffleSplit
from stage_classification import splits,metrics,dump
from experiment_paths import DATA_ROOT, OUTPUT_ROOT

HERE=Path(__file__).resolve().parent
OUT=OUTPUT_ROOT/'cognitive_stage'/'neural'


def preprocess_trials(x):
    """Trials x 60 channels x 1600 samples, 400 Hz -> standardized 100 Hz."""
    x=np.asarray(x,dtype=np.float64)
    if x.ndim!=3 or x.shape[1:]!=(60,1600) or not np.isfinite(x).all():
        raise ValueError('Expected finite EEG with shape (trials,60,1600) at 400 Hz')
    x=sosfiltfilt(butter(3,[1,40],btype='bandpass',fs=400,output='sos'),x,axis=-1)
    x=resample_poly(x,1,4,axis=-1)
    x=x-x.mean(axis=1,keepdims=True)
    x=x-x.mean(axis=-1,keepdims=True)
    x=x/np.maximum(x.std(axis=(1,2),keepdims=True),1e-20)
    return x.astype(np.float32)


class StageNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv2d(1,8,(1,33),padding=(0,16),bias=False),nn.BatchNorm2d(8),
            nn.Conv2d(8,16,(60,1),groups=8,bias=False),nn.BatchNorm2d(16),nn.ELU(),
            nn.AvgPool2d((1,4)),nn.Dropout(.4),
            nn.Conv2d(16,16,(1,17),padding=(0,8),groups=16,bias=False),
            nn.Conv2d(16,16,1,bias=False),nn.BatchNorm2d(16),nn.ELU(),
            nn.AvgPool2d((1,8)),nn.Dropout(.4),nn.Flatten(),nn.Linear(16*12,2))

    def forward(self,x):
        return self.net(x)


def seed(n):
    random.seed(n);np.random.seed(n);torch.manual_seed(n);torch.cuda.manual_seed_all(n)


def extract():
    arrays=[];records=[]
    sos=butter(3,[1,40],btype='bandpass',fs=400,output='sos')
    sources=[]
    for pop in ['deaf','hearing']:
        for folder,suffix in [('阅读阶段',''),('记忆阶段','_memory')]:
            p=DATA_ROOT/folder/f'EEG_{pop}{suffix}.mat'
            sources.append({'path':str(p),'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns})
            raw=loadmat(p,variable_names=[f'EEG_{pop}'])[f'EEG_{pop}']
            print('RAW',p.name,flush=True)
            for s in range(raw.shape[0]):
                x=np.ascontiguousarray(raw[s].transpose(2,0,1))
                arrays.append(preprocess_trials(x))
                stage=0 if folder=='阅读阶段' else 1
                records.extend({'population':pop,'stage':stage,'subject':f'{pop}_{s+1:02d}',
                                'trial':trial+1,'source':p.name} for trial in range(x.shape[0]))
            del raw
    X=np.concatenate(arrays)
    assert X.shape==(5760,60,400) and np.isfinite(X).all()
    np.save(OUT/'signals.npy',X)
    pd.DataFrame(records).to_csv(OUT/'sample_index.csv',index=False)
    dump(OUT/'sources.json',sources)
    return X


def infer(model,x,batch=128):
    model.eval();result=[]
    with torch.no_grad():
        for part in x.split(batch):
            logits=model(part)
            result.append((logits[:,1]-logits[:,0]).cpu().numpy())
    return np.concatenate(result)


def train(x,y,tr,va,lr,epochs,run_seed,track=True):
    seed(run_seed)
    model=StageNet().cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.01)
    best_acc=-1;best_epoch=0;history=[]
    tr_tensor=torch.tensor(tr,device='cuda')
    lossfn=nn.CrossEntropyLoss()
    for epoch in range(1,epochs+1):
        model.train()
        shuffled=tr_tensor[torch.randperm(len(tr_tensor),device='cuda')]
        total_loss=0
        for batch in shuffled.split(128):
            optimizer.zero_grad(set_to_none=True)
            loss=lossfn(model(x[batch]),y[batch]);loss.backward();optimizer.step()
            total_loss+=loss.item()*len(batch)
        if track:
            scores=infer(model,x[va])
            acc=float(np.mean((scores>0)==y[va].cpu().numpy()))
            history.append({'epoch':epoch,'train_loss':total_loss/len(tr),'validation_accuracy':acc})
            if acc>best_acc:
                best_acc=acc;best_epoch=epoch
            if epoch%10==0:
                print('  epoch',epoch,'validation',round(acc,4),'best',round(best_acc,4),flush=True)
    return model,best_acc,best_epoch,history


def run(X,meta,name,epochs):
    dest=OUT/name;dest.mkdir(exist_ok=True)
    x=torch.tensor(X[:,None],device='cuda')
    y_np=meta.stage.to_numpy();y=torch.tensor(y_np,dtype=torch.long,device='cuda')
    pred=np.full(len(meta),-1);score=np.full(len(meta),np.nan);foldids=np.zeros(len(meta),int)
    rows=[];members=[]
    for fold,(tr,te) in enumerate(splits(meta,5,42),1):
        subjects=meta.iloc[tr][['subject','population']].drop_duplicates().reset_index(drop=True)
        ss=StratifiedShuffleSplit(n_splits=1,test_size=.25,random_state=100+fold)
        a,b=next(ss.split(subjects.subject,subjects.population))
        innertr=tr[meta.subject.iloc[tr].isin(subjects.subject.iloc[a])]
        val=tr[meta.subject.iloc[tr].isin(subjects.subject.iloc[b])]
        assert set(meta.subject.iloc[innertr]).isdisjoint(meta.subject.iloc[val])
        assert set(meta.subject.iloc[tr]).isdisjoint(meta.subject.iloc[te])
        choices=[]
        print('CNN',name,'fold',fold,'train/val/test subjects',len(a),len(b),meta.subject.iloc[te].nunique(),flush=True)
        for lr in [.001,.0003]:
            model,acc,ep,hist=train(x,y,innertr,val,lr,epochs,200+fold)
            choices.append((acc,ep,lr))
            pd.DataFrame(hist).to_csv(dest/f'fold{fold}_lr{lr}_learning.csv',index=False)
            del model
        acc,ep,lr=max(choices,key=lambda z:z[0])
        # Refit only after validation chooses lr and epoch count.
        model,_,_,_=train(x,y,tr,None,lr,ep,300+fold,False)
        s=infer(model,x[te]);p=(s>0).astype(int)
        pred[te],score[te],foldids[te]=p,s,fold
        row={'fold':fold,'validation_accuracy':acc,'selected_epochs':ep,'learning_rate':lr,
             **metrics(y_np[te],p,s)}
        rows.append(row)
        torch.save({'state_dict':model.cpu().state_dict(),'model':'StageNet','epochs':ep,
                    'train_subjects':sorted(meta.subject.iloc[tr].unique()),'fs_out':100},dest/f'fold{fold}.pt')
        for role,idx in [('train',tr),('test',te),('inner_train',innertr),('validation',val)]:
            members.extend({'fold':fold,'role':role,'subject':sub} for sub in sorted(meta.subject.iloc[idx].unique()))
        dump(dest/'progress.json',rows)
        print('CNN',name,'FOLD',fold,'selected',ep,lr,'test',round(row['accuracy'],4),flush=True)
        del model
    assert (foldids>0).all()
    output=meta.assign(fold=foldids,prediction=pred,decision_score=score)
    output.to_csv(dest/'oof_predictions.csv',index=False)
    pd.DataFrame(members).to_csv(dest/'fold_subjects.csv',index=False)
    pd.DataFrame(rows).to_csv(dest/'fold_metrics.csv',index=False)
    result={'dataset':name,'samples':len(meta),'subjects':meta.subject.nunique(),
            **metrics(y_np,pred,score),'folds':rows,'target_met':bool(np.mean(y_np==pred)>=.65),
            'note':'Exploratory 5-fold subject-disjoint OOF; validation picks lr/epochs; no external test.'}
    dump(dest/'metrics.json',result)
    print('CNN RESULT',name,result['accuracy'],flush=True)


def check_sources():
    for source in json.loads((OUT/'sources.json').read_text(encoding='utf-8')):
        stat=Path(source['path']).stat()
        assert stat.st_size==source['bytes'] and stat.st_mtime_ns==source['mtime_ns'], 'Source changed'


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--reuse-signals',action='store_true')
    parser.add_argument('--epochs',type=int,default=50)
    parser.add_argument('--datasets',nargs='+',default=['combined'])
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    assert torch.cuda.is_available()
    dump(OUT/'plan.json',{'learning_rates':[.001,.0003],'max_epochs':args.epochs,
                         'validation_fraction':.25,'validation_unit':'subject','outer_folds':5,
                         'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,
                         'normalization':'CAR, trial-wise scalar std; no cohort/test adaptation',
                         'filter_hz':[1,40],'fs_in':400,'fs_out':100})
    if args.reuse_signals:
        for src in json.loads((OUT/'sources.json').read_text(encoding='utf-8')):
            st=Path(src['path']).stat();assert st.st_size==src['bytes'] and st.st_mtime_ns==src['mtime_ns']
        X=np.load(OUT/'signals.npy')
    else:
        X=extract()
    meta=pd.read_csv(OUT/'sample_index.csv')
    for name in args.datasets:
        mask=np.ones(len(meta),bool) if name=='combined' else (meta.population==name).to_numpy()
        run(X[mask],meta.loc[mask].reset_index(drop=True),name,args.epochs)


if __name__=='__main__':
    main()
