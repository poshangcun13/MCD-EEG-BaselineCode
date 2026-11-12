"""Official pretrained EEGPT encoder with a source-trained linear probe.

Table 5: Memory -> IQ/Re-cap, hearing/deaf separately.
Table 6: first -> second recording, rest and each N-back load separately.
"""
from pathlib import Path
from functools import partial
import argparse,hashlib,json,random,time
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.signal import resample_poly
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score,recall_score,f1_score,roc_curve
import torch
from torch import nn

from experiment_paths import DATA_ROOT, NB_DATA, REST_DATA, OUTPUT_ROOT, WEIGHTS_ROOT
from eegpt_channels import PRETRAIN_CHANNELS
import EEGPT_mcae as M

HERE=Path(__file__).resolve().parent
NB=NB_DATA
BASE=DATA_ROOT
REST=REST_DATA
OUT=OUTPUT_ROOT/'pretrained_eegpt'
WEIGHTS=WEIGHTS_ROOT/'encoder_weights.pt'
TASKS=['table5_hearing','table5_deaf','table6_rest','table6_0back','table6_1back','table6_2back']

def dump(p,obj):p.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def seed(s):
 random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)

def channels():
 mat=sio.loadmat(REST/'resting_first_10subjects.mat',variable_names=['channel_names'])
 names=[str(v.item()).upper() for v in mat['channel_names'].ravel()]
 used=list(PRETRAIN_CHANNELS)
 assert len(names)==60 and len(used)==58 and set(used).issubset(names)
 return names,used,np.array([names.index(c) for c in used])

def load_task(task):
 arrays=[];labels=[];paths=[]
 if task.startswith('table5'):
  group=task.split('_')[1];k=23 if group=='hearing' else 22
  # The hearing result uses the author-validated IQ NPZ export when available.
  backup=BASE/'IQ'/'EEG_hearing_IQ.npz'
  if group=='hearing' and backup.exists():
   source=BASE/'记忆阶段'/'EEG_hearing_memory.mat'
   raw=sio.loadmat(source,variable_names=['EEG_hearing'])['EEG_hearing'].astype('float32')
   assert raw.shape==(23,60,1600,64)
   x=raw.transpose(0,3,1,2).reshape(1472,60,1600).copy()
   y=np.repeat(np.arange(23),64)
   with np.load(backup,allow_pickle=False) as f:
    z=f['EEG'].astype('float32');t=f['subject_labels'].astype(int)
   assert z.shape==x.shape and np.array_equal(t,y)
   assert np.isfinite(x).all() and np.isfinite(z).all()
   return x,y,z,t,[source,backup]
  for directory,name in [('记忆阶段',f'EEG_{group}_memory.mat'),('IQ',f'EEG_{group}_IQ.mat')]:
   path=BASE/directory/name;m=sio.loadmat(path,variable_names=[f'EEG_{group}'])
   raw=m[f'EEG_{group}'].astype('float32');del m
   assert raw.shape==(k,60,1600,64)
   arrays.append(raw.transpose(0,3,1,2).reshape(k*64,60,1600).copy())
   labels.append(np.repeat(np.arange(k),64));paths.append(path)
   del raw
 elif task=='table6_rest':
  for name in ['resting_first_10subjects.mat','resting_rewear_10subjects.mat']:
   path=REST/name;m=sio.loadmat(path)
   assert int(m['fs'].item())==400
   arrays.append(m['EEG'].transpose(2,0,1).astype('float32').copy())
   labels.append(m['subject_id'].ravel().astype(int)-1);paths.append(path)
 elif task.startswith('table6_'):
  load=int(task[-5])
  for session in [1,2]:
   path=NB/f'{load}-BACK-{session}.mat';m=sio.loadmat(path)
   assert m['X'].shape==(10,60,800,120)
   arrays.append(m['X'].transpose(0,3,1,2).reshape(1200,60,800).astype('float32').copy())
   labels.append(np.repeat(np.arange(10),120));paths.append(path)
 else:raise ValueError(task)
 assert arrays[0].shape==arrays[1].shape
 assert np.array_equal(np.unique(labels[0]),np.unique(labels[1]))
 assert all(np.isfinite(x).all() for x in arrays)
 return arrays[0],labels[0],arrays[1],labels[1],paths

def sample_audit(task,x,y,z,t,paths,dest):
 hashes=[]
 for role,a,ids in [('source',x,y),('target',z,t)]:
  hh=[hashlib.sha256(v.tobytes()).hexdigest() for v in a]
  assert len(set(hh))==len(hh),'Within-recording exact duplicates'
  pd.DataFrame({'sample_index':np.arange(len(a)),'subject_id':ids+1,'sha256_float32':hh}).to_csv(dest/f'{role}_samples.csv',index=False)
  hashes.append(set(hh))
 assert not hashes[0]&hashes[1],'Cross-recording exact duplicates'
 dump(dest/'data_audit.json',{'task':task,'source_shape':list(x.shape),'target_shape':list(z.shape),'source_counts':np.bincount(y).tolist(),'target_counts':np.bincount(t).tolist(),'cross_recording_exact_duplicates':0,'source_files':{str(p):sha(p) for p in paths},'source':str(paths[0]),'target':str(paths[1]),'sampling_rate_hz':400,'channel_order':'Rest metadata; author confirmed same 60-channel order for all tasks','subject_order':'Author-confirmed corresponding subject axes across Memory/IQ and N-back sessions; resting labels from MAT'})

def make_encoder(points):
 model=M.EEGTransformer(img_size=[58,points],patch_size=64,embed_num=4,embed_dim=512,depth=8,num_heads=8,mlp_ratio=4.,drop_rate=0.,attn_drop_rate=0.,drop_path_rate=0.,init_std=.02,qkv_bias=True,norm_layer=partial(nn.LayerNorm,eps=1e-6))
 weights=torch.load(WEIGHTS,map_location='cpu',weights_only=True)
 result=model.load_state_dict(weights,strict=True)
 assert not result.missing_keys and not result.unexpected_keys
 model.requires_grad_(False);model.eval()
 return model.cuda()

def features(raw,dest,role,indices,used,batch=8):
 p=dest/f'{role}_features.npy';meta=dest/f'{role}_features_meta.json'
 if p.exists() and meta.exists():return np.load(p)
 start=time.time();points=raw.shape[-1]*16//25
 assert points in [512,1024]
 model=make_encoder(points);ids=model.prepare_chan_ids(used).cuda()
 output=np.lib.format.open_memmap(dest/f'{role}_features.partial.npy',mode='w+',dtype='float32',shape=(len(raw),points//64,2048))
 for start_i in range(0,len(raw),batch):
  a=raw[start_i:start_i+batch,indices].copy()
  # Fixed per-segment operations: no fitted recording-wide target statistics.
  a=resample_poly(a,up=16,down=25,axis=-1).astype('float32')
  a-=a.mean(axis=1,keepdims=True) # average reference on selected channels
  a-=a.mean(axis=2,keepdims=True) # each channel's segment temporal mean
  rms=np.sqrt(np.mean(a*a,axis=(1,2),keepdims=True)).clip(1e-6)
  a/=rms
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
   f=model(torch.as_tensor(a,device='cuda'),ids).flatten(2)
  output[start_i:start_i+len(a)]=f.float().cpu().numpy()
  if start_i%(batch*10)==0:print('FEATURES',dest.name,role,start_i,'/',len(raw),'seconds',round(time.time()-start),flush=True)
 output.flush();del output
 (dest/f'{role}_features.partial.npy').replace(p)
 assert all(not v.requires_grad for v in model.parameters())
 # Confirm frozen backbone tensors remained identical to the extracted checkpoint.
 weights=torch.load(WEIGHTS,map_location='cpu',weights_only=True)
 assert all(torch.equal(v.detach().cpu(),weights[k]) for k,v in model.state_dict().items())
 dump(meta,{'shape':[len(raw),points//64,2048],'seconds':time.time()-start,'encoder_strict_load':True,'encoder_frozen_and_unchanged':True,'input_channels':used,'resampling':'400 -> 256 Hz, scipy resample_poly(up=16,down=25), duration preserved','input_normalization':'fixed per-segment selected-channel average reference, per-channel temporal demeaning, global segment RMS; no dataset-level fitted statistics','adapter':'official target_encoder, 58 channel IDs; flatten four summary tokens per temporal patch'})
 del model;torch.cuda.empty_cache()
 return np.load(p)

class LinearProbe(nn.Module):
 def __init__(self,patches,k):
  super().__init__()
  self.drop=nn.Dropout(.5)
  self.probe1=nn.Linear(2048,16)
  self.probe2=nn.Linear(patches*16,k)
 def forward(self,x):
  # Same two linear readout stages as official BCIC2A example, adapted to K.
  with torch.no_grad():
   self.probe1.weight.copy_(torch.renorm(self.probe1.weight,p=2,dim=0,maxnorm=1.))
   self.probe2.weight.copy_(torch.renorm(self.probe2.weight,p=2,dim=0,maxnorm=.25))
  return self.probe2(self.probe1(self.drop(x)).flatten(1))

def predict(model,x):
 model.eval()
 with torch.inference_mode():return torch.cat([model(b).softmax(1).cpu() for b in x.split(128)]).numpy()

def eer(y,p,k):
 fpr,tpr,_=roc_curve(y,p)
 return float(brentq(lambda v:1-v-interp1d(fpr,tpr)(v),0,1))

def metrics(y,p):
 k=p.shape[1];pred=p.argmax(1)
 macro=float(np.mean([eer(y==c,p[:,c],k) for c in range(k)]))
 pooled=eer(np.eye(k)[y].ravel(),p.ravel(),k)
 return {'Acc':float(accuracy_score(y,pred)),'EER_macro':macro,'EER_pooled':pooled,'Recall':float(recall_score(y,pred,average='macro',zero_division=0)),'F1':float(f1_score(y,pred,average='macro',zero_division=0))}

def train_task(task,src,y,tgt,yt,dest,epochs):
 folds=list(StratifiedKFold(5,shuffle=True,random_state=42).split(src,y))
 membership=[]
 for fold,(tr,va) in enumerate(folds,1):
  assert set(tr).isdisjoint(va)
  for role,ix in [('train',tr),('validation',va)]:membership.extend({'fold':fold,'role':role,'sample_index':int(i),'subject_id':int(y[i]+1)} for i in ix)
 pd.DataFrame(membership).to_csv(dest/'fold_membership.csv',index=False)
 for fold,(tr,va) in enumerate(folds,1):
  if (dest/f'fold{fold}_metrics.json').exists():continue
  start=time.time();seed(4200+fold)
  mean=src[tr].mean(axis=(0,1),keepdims=True,dtype=np.float64).astype('float32')
  std=src[tr].std(axis=(0,1),keepdims=True,dtype=np.float64).astype('float32').clip(1e-6)
  x=torch.as_tensor((src-mean)/std,device='cuda');test=torch.as_tensor((tgt-mean)/std,device='cuda');labels=torch.as_tensor(y,device='cuda')
  model=LinearProbe(src.shape[1],len(np.unique(y))).cuda()
  opt=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=.01)
  best=float('inf');state=None;best_epoch=0;history=[];idx=torch.as_tensor(tr,device='cuda')
  print('START',task,'fold',fold,'train',len(tr),'val',len(va),flush=True)
  for epoch in range(1,epochs+1):
   model.train();total=0
   for b in idx[torch.randperm(len(idx),device='cuda')].split(32):
    opt.zero_grad(set_to_none=True)
    loss=nn.functional.cross_entropy(model(x[b]),labels[b],label_smoothing=.1)
    assert torch.isfinite(loss)
    loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();total+=loss.item()*len(b)
   p=predict(model,x[va]);vl=float(-np.log(p[np.arange(len(va)),y[va]].clip(1e-12)).mean())
   history.append({'epoch':epoch,'train_loss':total/len(tr),'validation_loss':vl,'validation_acc':float((p.argmax(1)==y[va]).mean())})
   if vl<best-1e-5:best=vl;best_epoch=epoch;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
   if epoch%10==0:print('EPOCH',task,fold,epoch,'val_loss',round(vl,5),'best',best_epoch,flush=True)
   if epoch>=20 and epoch-best_epoch>=15:break
  model.load_state_dict(state)
  p=predict(model,test);m=metrics(yt,p)
  m['EER']=m['EER_pooled'] if task.startswith('table5') else m['EER_macro']
  pred=pd.DataFrame({'sample_index':np.arange(len(yt)),'subject_id':yt+1,'predicted_subject_id':p.argmax(1)+1})
  for c in range(p.shape[1]):pred[f'p_subject_{c+1}']=p[:,c]
  pred.to_csv(dest/f'fold{fold}_predictions.csv',index=False)
  pd.DataFrame(history).to_csv(dest/f'fold{fold}_history.csv',index=False)
  np.savez(dest/f'fold{fold}_feature_normalization.npz',mean=mean,std=std)
  torch.save({'probe':state,'seed':4200+fold,'best_epoch':best_epoch,'classes':p.shape[1],'patches':src.shape[1],'encoder':'../encoder_weights.pt'},dest/f'fold{fold}_probe.pt')
  dump(dest/f'fold{fold}_metrics.json',dict(m,best_epoch=best_epoch,best_validation_loss=best,epochs_run=epoch,seconds=time.time()-start,train_samples=len(tr),validation_samples=len(va),test_samples=len(yt)))
  print('DONE',task,fold,m,'seconds',round(time.time()-start),flush=True)
  del x,test,labels,model,opt,state;torch.cuda.empty_cache()

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--tasks',nargs='+',default=TASKS);parser.add_argument('--epochs',type=int,default=80);args=parser.parse_args()
 assert args.epochs==80,'This locked experiment uses 80 max epochs.'
 OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
 names,used,indices=channels()
 config={'method':'EEGPT official pretrained target encoder + frozen-backbone linear probe','checkpoint_sha256':'9d63ebc80bba1ad5e3277df4eecfa05d2745e6243a12ff743c4c0a93930dfac0','weights_extracted_sha256':sha(WEIGHTS),'official_commit':'ea7282f698c406cd423ccbedbe9960853761144c','model':'EEGTransformer 25,287,168 encoder parameters, embed_dim512, embed_num4, depth8, heads8, patch64','raw_channel_order':names,'selected_58_channels':used,'dropped_channels':[c for c in names if c not in used],'channel_order_confirmation':'Author confirmed identical 60-channel order across all tasks','preprocessing':'400->256 Hz polyphase resampling, duration preserved; selected-channel average reference; segment temporal demeaning; segment-global RMS normalization, all fixed per sample','feature_normalization':'Training-fold mean/std per embedding coordinate over source samples and temporal patches; target transform only','encoder':'strict load of all target_encoder tensors; requires_grad=False; eval mode; tensors verified unchanged','readout':'Two linear stages as in official BCIC2A example: 2048->16 per patch, flatten, patches*16->K, dropout0.5, maxnorm1/0.25; no trainable channel projection','training':'AdamW lr5e-4 weight_decay.01 batch32 label_smoothing.1 clip_grad_norm5, FP32 readout','epochs_max':80,'minimum_epochs':20,'early_stop_patience':15,'selection':'lowest source-validation cross entropy, improvement1e-5','folds':'StratifiedKFold(5,shuffle=True,random_state=42) within source only; same full target recording for every fold','training_seeds':[4201,4202,4203,4204,4205],'hyperparameter_search':'none; one configuration fixed before target evaluation','EER':'both saved; Table5 primary pooled flattened one-vs-rest to match old scripts; Table6 primary macro subject-wise to match existing controls','torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'script_sha256':sha(Path(__file__))}
 protocol=OUT/'protocol.json'
 if protocol.exists():assert json.loads(protocol.read_text(encoding='utf-8'))==config,'Protocol changed; use a new results directory'
 else:dump(protocol,config)
 for task in args.tasks:
  dest=OUT/task;dest.mkdir(exist_ok=True)
  if all((dest/f'fold{f}_metrics.json').exists() for f in range(1,6)):
   print('SKIP COMPLETE',task,flush=True);continue
  print('LOAD',task,flush=True)
  x,y,z,t,paths=load_task(task);sample_audit(task,x,y,z,t,paths,dest)
  a=features(x,dest,'source',indices,used);del x
  b=features(z,dest,'target',indices,used);del z
  train_task(task,a,y,b,t,dest,args.epochs)
  del a,b,y,t
 print('ALL 30 FOLDS COMPLETE',flush=True)

if __name__=='__main__':main()
