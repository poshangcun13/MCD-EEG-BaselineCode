"""Uniform binary metrics and independent artifact checks for all table rows."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,recall_score,f1_score,roc_curve,roc_auc_score

from experiment_paths import OUTPUT_ROOT
from stage_classification import splits
HERE=Path(__file__).resolve().parent
ROOT=OUTPUT_ROOT/'cognitive_stage'/'eight_model_results'
NEURAL=OUTPUT_ROOT/'cognitive_stage'/'neural'
MODEL_ORDER=['EEGNet-lite','EEGNet-SE','MS-EEGNet-SF','ShallowLogVar','EEG-TCN','EEGConformer-lite','FBCNet-lite','EEGNet-lite-Ens2']


def eer(y,score):
    fpr,tpr,_=roc_curve(y,score,pos_label=1)
    delta=fpr+tpr-1
    i=int(np.flatnonzero(delta>=0)[0])
    if i==0:return float(fpr[0])
    t=-delta[i-1]/(delta[i]-delta[i-1])
    return float(fpr[i-1]+t*(fpr[i]-fpr[i-1]))


def measure(y,p,s):
    return {'Acc':accuracy_score(y,p),'EER':eer(y,s),'Recall':recall_score(y,p,average='macro'),
            'F1':f1_score(y,p,average='macro'),'AUC':roc_auc_score(y,s)}


def collect():
    expected=pd.read_csv(NEURAL/'sample_index.csv')
    folds=[];pooled=[]
    for name in MODEL_ORDER:
        p=ROOT/name
        if not (p/'oof_predictions.csv').exists():continue
        data=pd.read_csv(p/'oof_predictions.csv')
        pd.testing.assert_frame_equal(data[expected.columns],expected)
        assert len(data)==5760 and not data.duplicated(['subject','stage','trial']).any()
        assert (data.groupby('subject')['fold'].nunique()==1).all()
        assert (data.groupby(['subject','stage']).size()==64).all()
        assert np.isfinite(data.decision_score).all() and data.prediction.isin([0,1]).all()
        members=pd.read_csv(p/'fold_subjects.csv')
        reference=np.zeros(len(expected),dtype=int)
        for fold,(_,test) in enumerate(splits(expected,5,42),1):reference[test]=fold
        np.testing.assert_array_equal(data.fold,reference)
        for f in range(1,6):
            member=members[members.fold==f]
            tr=set(member.loc[member.role=='train','subject']);te=set(member.loc[member.role=='test','subject'])
            assert tr.isdisjoint(te) and tr|te==set(expected.subject)
            assert set(data.loc[data.fold==f,'subject'])==te
            if 'validation' in set(member.role):
                va=set(member.loc[member.role=='validation','subject']);it=set(member.loc[member.role=='inner_train','subject'])
                assert va.isdisjoint(it) and va|it==tr
            for pop in ['hearing','deaf']:
                g=data.loc[(data.fold==f)&(data.population==pop)]
                folds.append({'model':name,'population':pop,'fold':f,'samples':len(g),**measure(g.stage,g.prediction,g.decision_score)})
        for pop in ['hearing','deaf','combined']:
            g=data if pop=='combined' else data.loc[data.population==pop]
            pooled.append({'model':name,'population':pop,'samples':len(g),**measure(g.stage,g.prediction,g.decision_score)})
    f=pd.DataFrame(folds);p=pd.DataFrame(pooled)
    if len(f):
        f.to_csv(ROOT/'fold_metrics_all.csv',index=False,encoding='utf-8-sig')
        p.to_csv(ROOT/'pooled_metrics_all.csv',index=False,encoding='utf-8-sig')
        summary=[]
        for (name,pop),g in f.groupby(['model','population'],sort=False):
            row={'model':name,'population':pop}
            for metric in ['Acc','EER','Recall','F1','AUC']:
                row[metric+'_mean']=float(g[metric].mean());row[metric+'_sd']=float(g[metric].std(ddof=1))
            summary.append(row)
        s=pd.DataFrame(summary)
        s.to_csv(ROOT/'table_numeric.csv',index=False,encoding='utf-8-sig')
        print(s[['model','population','Acc_mean','Acc_sd']].to_string(index=False))
    return f,p


if __name__=='__main__':collect()
