"""Eight documented models, identical subject folds; no test-driven stopping."""
from pathlib import Path
import argparse,json,random,time
import numpy as np
import pandas as pd
import torch
from torch import nn
from scipy.special import expit
from sklearn.model_selection import StratifiedShuffleSplit
from benchmark_models import FACTORIES
from neural_stage import seed
from stage_classification import splits,metrics,dump
from experiment_paths import OUTPUT_ROOT

HERE=Path(__file__).resolve().parent
OUT=OUTPUT_ROOT/'cognitive_stage'/'eight_model_results'
NEURAL=OUTPUT_ROOT/'cognitive_stage'/'neural'


def inference(model,x):
    model.eval();scores=[]
    with torch.no_grad():
        for batch in x.split(256):
            with torch.autocast('cuda',dtype=torch.bfloat16):logits=model(batch)
            scores.append((logits[:,1]-logits[:,0]).float().cpu().numpy())
    return np.concatenate(scores)


def fit(name,x,y,tr,va,lr,epochs,run_seed):
    seed(run_seed);model=FACTORIES[name]().cuda()
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.01)
    idx=torch.tensor(tr,device='cuda');best=-1;best_ep=0;hist=[]
    for epoch in range(1,epochs+1):
        model.train();loss_sum=0.
        for batch in idx[torch.randperm(len(idx),device='cuda')].split(256):
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                loss=nn.functional.cross_entropy(model(x[batch]),y[batch],label_smoothing=.05)
            if not torch.isfinite(loss):raise ValueError(f'Nonfinite loss {name}')
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step()
            loss_sum+=float(loss.detach())*len(batch)
        if va is not None:
            s=inference(model,x[va]);acc=float(np.mean((s>=0)==y[va].cpu().numpy()))
            hist.append({'epoch':epoch,'train_loss':loss_sum/len(tr),'validation_accuracy':acc})
            if acc>best:best=acc;best_ep=epoch
            if epoch%10==0:print(' epoch',epoch,'val',round(acc,4),'best',round(best,4),flush=True)
            if epoch-best_ep>=18 and epoch>=25:break
    return model,best,best_ep,hist


def run(name,x,y,meta,args):
    dest=OUT/name;dest.mkdir(parents=True,exist_ok=True)
    for fold,(tr,te) in enumerate(splits(meta,5,42),1):
        if (dest/f'fold{fold}_predictions.csv').exists():
            print('RESUME',name,fold,flush=True);continue
        subs=meta.iloc[tr][['subject','population']].drop_duplicates().reset_index(drop=True)
        a,b=next(StratifiedShuffleSplit(n_splits=1,test_size=.25,random_state=100+fold).split(subs.subject,subs.population))
        it=tr[meta.subject.iloc[tr].isin(subs.subject.iloc[a])]
        va=tr[meta.subject.iloc[tr].isin(subs.subject.iloc[b])]
        assert set(meta.subject.iloc[it]).isdisjoint(meta.subject.iloc[va])
        assert set(meta.subject.iloc[tr]).isdisjoint(meta.subject.iloc[te])
        choices=[]
        print('START',name,'fold',fold,flush=True)
        for lr in [.001,.0003]:
            model,acc,ep,hist=fit(name,x,y,it,va,lr,args.epochs,700+fold)
            choices.append((acc,ep,lr));pd.DataFrame(hist).to_csv(dest/f'fold{fold}_lr{lr}_history.csv',index=False)
            del model
        acc,ep,lr=max(choices,key=lambda z:z[0]);probs=[]
        for offset in [800]:
            model,_,_,_=fit(name,x,y,tr,None,lr,ep,offset+fold)
            probs.append(expit(inference(model,x[te])))
            torch.save({'state_dict':model.cpu().state_dict(),'name':name,'seed':offset+fold,'epochs':ep,'lr':lr,
                        'train_subjects':sorted(meta.subject.iloc[tr].unique())},dest/f'fold{fold}_seed{offset+fold}.pt')
            del model
        prob=np.mean(probs,axis=0)
        out=meta.iloc[te].copy();out['fold']=fold;out['prediction']=(prob>=.5).astype(int);out['decision_score']=prob
        # Preserve the source row index for audit and exact OOF ordering.
        out['sample_index']=te;out.to_csv(dest/f'fold{fold}_predictions.csv',index=False)
        dump(dest/f'fold{fold}_selection.json',{'validation_accuracy':acc,'epochs':ep,'lr':lr,'seeds':[800+fold]})
        members=[]
        for role,indices in [('train',tr),('test',te),('inner_train',it),('validation',va)]:
            members.extend({'fold':fold,'role':role,'subject':s} for s in sorted(meta.subject.iloc[indices].unique()))
        pd.DataFrame(members).to_csv(dest/f'fold{fold}_subjects.csv',index=False)
        print('DONE',name,fold,metrics(out.stage,out.prediction,out.decision_score),flush=True)
    output=pd.concat([pd.read_csv(dest/f'fold{f}_predictions.csv') for f in range(1,6)]).sort_values('sample_index').reset_index(drop=True)
    pd.testing.assert_frame_equal(output[meta.columns],meta)
    assert np.array_equal(output.sample_index,np.arange(len(meta)))
    output.to_csv(dest/'oof_predictions.csv',index=False)
    pd.concat([pd.read_csv(dest/f'fold{f}_subjects.csv') for f in range(1,6)]).to_csv(dest/'fold_subjects.csv',index=False)
    dump(dest/'metrics.json',metrics(output.stage,output.prediction,output.decision_score))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--epochs',type=int,default=50)
    parser.add_argument('--models',nargs='+',default=[n for n in FACTORIES if n not in ['EEGNet-lite','MS-EEGNet']])
    args=parser.parse_args();torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=True)
    dump(OUT/'plan.json',{'models':[n for n in FACTORIES if n!='MS-EEGNet']+['EEGNet-lite-Ens2'],'train_population':'combined; report hearing/deaf test subsets',
                         'outer_folds':5,'outer_seed':42,'max_epochs':args.epochs,'learning_rates':[.001,.0003],
                         'validation_subject_fraction':.25,'early_stopping_patience':18,
                         'model_selection':'train-side validation only; no outer test tuning',
                         'new_models':'single models; one extra two-member ensemble, total ensemble rows <=2',
                         'normalization':'cached trial-only CAR, temporal demean, overall trial std',
                         'metrics':'fold mean +/- sample standard deviation; macro recall and F1; binary EER',
                         'torch':torch.__version__,'gpu':torch.cuda.get_device_name()})
    meta=pd.read_csv(NEURAL/'sample_index.csv')
    X=np.load(NEURAL/'signals.npy')
    assert X.shape==(len(meta),60,400)
    x=torch.tensor(X[:,None],device='cuda');y=torch.tensor(meta.stage.to_numpy(),dtype=torch.long,device='cuda')
    for name in args.models:run(name,x,y,meta,args)
    print('ALL REQUESTED MODELS COMPLETE',flush=True)


if __name__=='__main__':main()
