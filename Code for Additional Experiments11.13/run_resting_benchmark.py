"""Five train-session folds, each evaluated on untouched full rewear session."""
from pathlib import Path
import argparse,copy,hashlib,json,random,time
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.signal import butter,sosfilt
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score,recall_score,f1_score,roc_curve
import torch
from torch import nn
import resting_models as M
from experiment_paths import REST_DATA, OUTPUT_ROOT

HERE=Path(__file__).resolve().parent
OUT=OUTPUT_ROOT/'resting_state'
NAMES=['LRMD','FBCNet','EEGNet','EEGViT','EEGConformer','EEGPT','BrainprintNet','ATCNet']
def dump(p,obj):p.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
def seed(s):
    random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)
def build(name):
    return {'LRMD':lambda:M.LRMD_EEGNet(10,60,800),'FBCNet':lambda:M.FBCNet(10,5,60),
            'EEGNet':lambda:M.SingleBandEEGNet(10,60),'EEGViT':lambda:M.EEGViT(10),
            'EEGConformer':lambda:M.EEGConformer(10,60,800),
            'EEGPT':lambda:M.EEGPT(10,60,800),'BrainprintNet':lambda:M.BrainprintNet(10,60),
            'ATCNet':lambda:M.ATCNet(10,60,800)}[name]()
def representation(x,name):
    if name not in ['FBCNet','BrainprintNet']:return x[:,None]
    bands=[(1,4),(4,8),(8,13),(13,30),(30,45)] if name=='FBCNet' else [(8+2*i,10+2*i) for i in range(12)]
    # Stable SOS realization, physical fs from MAT metadata. No session mixing.
    a=np.stack([sosfilt(butter(4 if name=='FBCNet' else 5,[lo,hi],btype='bandpass',fs=400,output='sos'),x,axis=-1) for lo,hi in bands],axis=1)
    if name=='BrainprintNet':a=a[...,::2]
    return a.astype('float32')
def scores(model,x,bs):
    model.eval();out=[]
    with torch.no_grad():
        for b in x.split(bs):
            with torch.autocast('cuda',dtype=torch.bfloat16):z=model(b)
            out.append(z.float().softmax(-1).cpu().numpy())
    return np.concatenate(out)
def metrics(y,p):
    pred=p.argmax(1);eers=[]
    for c in range(10):
        fpr,tpr,_=roc_curve(y==c,p[:,c]);f=interp1d(fpr,tpr)
        eers.append(float(brentq(lambda z:1-z-f(z),0,1)))
    return {'Acc':accuracy_score(y,pred),'EER':float(np.mean(eers)),
            'Recall':recall_score(y,pred,average='macro',zero_division=0),
            'F1':f1_score(y,pred,average='macro',zero_division=0)}
def load():
    a=sio.loadmat(REST_DATA/'resting_first_10subjects.mat');b=sio.loadmat(REST_DATA/'resting_rewear_10subjects.mat')
    assert np.array_equal(a['channel_names'],b['channel_names'])
    assert np.array_equal(a['included_subjects'],b['included_subjects'])
    assert int(a['fs'].item())==int(b['fs'].item())==400
    x=a['EEG'].transpose(2,0,1).copy();z=b['EEG'].transpose(2,0,1).copy()
    y=a['subject_id'].ravel().astype(int)-1;t=b['subject_id'].ravel().astype(int)-1
    assert x.shape==z.shape==(645,60,800) and np.isfinite(x).all() and np.isfinite(z).all()
    h=lambda q:[hashlib.sha256(v.tobytes()).hexdigest() for v in q]
    hx,hz=h(x),h(z);assert not set(hx)&set(hz),'Exact cross-session duplicates'
    assert len(set(hx))==len(x) and len(set(hz))==len(z)
    audit={'shape':list(x.shape),'fs':400,'seconds':2,'classes':10,
           'first_counts':np.bincount(y).tolist(),'rewear_counts':np.bincount(t).tolist(),
           'cross_session_exact_duplicates':0,'channel_order_match':True,
           'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in REST_DATA.glob('*.mat')}}
    dump(OUT/'data_audit.json',audit)
    for label,mat,ids,hashes in [('first',a,y,hx),('rewear',b,t,hz)]:
        pd.DataFrame({'sample_index':np.arange(len(ids)),'subject_id':ids+1,
                      'trial_in_subject':mat['trial_in_subject'].ravel(),
                      'start_seconds':mat['trial_start_seconds'].ravel(),'sha256':hashes}).to_csv(OUT/f'{label}_samples.csv',index=False)
    return x,y,z,t
def run(name,raw,y,test,yt,folds,args):
    dest=OUT/name;dest.mkdir(exist_ok=True)
    if all((dest/f'fold{f}_metrics.json').exists() for f in range(1,6)):return
    print('PREPROCESS',name,flush=True);a=representation(raw,name);b=representation(test,name)
    bs=8 if name=='BrainprintNet' else 32
    for fold,(tr,va) in enumerate(folds,1):
        if (dest/f'fold{fold}_metrics.json').exists():continue
        start=time.time();seed(4200+fold)
        mean=a[tr].mean(axis=(0,3),keepdims=True,dtype=np.float64).astype('float32')
        std=a[tr].std(axis=(0,3),keepdims=True,dtype=np.float64).astype('float32').clip(1e-6)
        x=torch.as_tensor((a-mean)/std,device='cuda');t=torch.as_tensor((b-mean)/std,device='cuda')
        labels=torch.as_tensor(y,device='cuda');model=build(name).cuda()
        # Initialize lazy classifier BEFORE optimizer construction.
        model.eval()
        with torch.no_grad():model(x[:2])
        model.train()
        lr={'EEGViT':3e-4,'EEGPT':5e-4}.get(name,1e-3)
        opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.01)
        best=float('inf');best_epoch=0;state=None;history=[]
        idx=torch.as_tensor(tr,device='cuda')
        print('START',name,fold,'parameters',sum(p.numel() for p in model.parameters()),flush=True)
        for epoch in range(1,args.epochs+1):
            model.train();total=0
            for ix in idx[torch.randperm(len(idx),device='cuda')].split(bs):
                opt.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):loss=nn.functional.cross_entropy(model(x[ix]),labels[ix],label_smoothing=.1)
                assert torch.isfinite(loss),'Nonfinite loss'
                loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();total+=loss.item()*len(ix)
            p=scores(model,x[va],bs);vl=float(-np.log(p[np.arange(len(va)),y[va]].clip(1e-12)).mean())
            history.append({'epoch':epoch,'train_loss':total/len(tr),'val_loss':vl,'val_acc':float((p.argmax(1)==y[va]).mean())})
            if vl<best-1e-5:best=vl;best_epoch=epoch;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            if epoch%10==0:print('EPOCH',name,fold,epoch,'val_loss',round(vl,4),'best',best_epoch,'seconds',round(time.time()-start),flush=True)
            if epoch>=20 and epoch-best_epoch>=15:break
        model.load_state_dict(state);p=scores(model,t,bs);m=metrics(yt,p)
        pd.DataFrame(history).to_csv(dest/f'fold{fold}_history.csv',index=False)
        pred=pd.DataFrame({'sample_index':np.arange(len(yt)),'subject_id':yt+1,'predicted_subject_id':p.argmax(1)+1})
        for c in range(10):pred[f'p_subject_{c+1}']=p[:,c]
        pred.to_csv(dest/f'fold{fold}_predictions.csv',index=False)
        np.savez(dest/f'fold{fold}_normalization.npz',mean=mean,std=std)
        torch.save({'model':name,'state_dict':state,'best_epoch':best_epoch,'seed':4200+fold},dest/f'fold{fold}_model.pt')
        dump(dest/f'fold{fold}_metrics.json',dict(m,best_epoch=best_epoch,best_val_loss=best,epochs_run=epoch,seconds=time.time()-start))
        print('DONE',name,fold,m,'seconds',round(time.time()-start),flush=True)
        del model,x,t,opt,state;torch.cuda.empty_cache()
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--epochs',type=int,default=80);parser.add_argument('--models',nargs='+',default=NAMES);args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    dump(OUT/'protocol.json',{'task':'10-class closed-set subject identification','train':'first session only','validation':'StratifiedKFold(5,shuffle=True,random_state=42) within first session','test':'entire rewear session for each of five models; no test-driven selection','report':'mean +/- sample SD (ddof=1) over five repeated full-test evaluations; not independent test folds','epochs_max':args.epochs,'minimum_epochs':20,'early_stop_patience':15,'selection':'lowest first-session validation cross entropy','normalization':'per band/channel mean/std over training-fold samples and time only','batch_size':'32; BrainprintNet 8','learning_rate':'0.001; EEGViT 0.0003; EEGPT 0.0005','optimizer':'AdamW weight_decay=.01, label_smoothing=.1, clip_grad_norm=5','precision':'bfloat16 autocast','EER':'macro mean of 10 one-vs-rest ROC EERs from softmax scores','recall_f1':'macro','architectures':'local recap variants, trained from scratch; not official pretrained benchmarks','changes':'800 time points; actual fs=400; SOS filter; independent EEGPT blocks; lazy BrainprintNet head initialized before optimizer','torch':torch.__version__,'gpu':torch.cuda.get_device_name()})
    x,y,z,t=load();folds=list(StratifiedKFold(5,shuffle=True,random_state=42).split(x,y))
    membership=[]
    for f,(tr,va) in enumerate(folds,1):
        assert set(tr).isdisjoint(va) and len(tr)+len(va)==len(x)
        for role,ix in [('train',tr),('validation',va)]:membership.extend({'fold':f,'role':role,'sample_index':int(i),'subject_id':int(y[i]+1)} for i in ix)
    pd.DataFrame(membership).to_csv(OUT/'fold_membership.csv',index=False)
    for name in args.models:run(name,x,y,z,t,folds,args)
    print('ALL COMPLETE',flush=True)
if __name__=='__main__':main()
