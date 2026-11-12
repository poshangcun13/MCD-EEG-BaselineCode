"""Extract tensors from a hash-verified official-format training checkpoint.

The source pickle contains optimizer/scheduler objects. Use an explicit class
allowlist rather than unrestricted pickle execution; export tensors thereafter.
"""
from pathlib import Path
import argparse
from experiment_paths import WEIGHTS_ROOT
import collections,hashlib,json,pickle,pickletools,types,zipfile
import torch

HERE=WEIGHTS_ROOT
CHECKPOINT=HERE/'eegpt_mcae_58chs_4s_large4E.ckpt'
EXPECTED='9d63ebc80bba1ad5e3277df4eecfa05d2745e6243a12ff743c4c0a93930dfac0'
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

class TrainingStatePlaceholder:
 def __setstate__(self,state):self.__dict__.update(state)

def safe_getattr(obj,name):
 if name not in ('_annealing_cos','_annealing_linear'):
  raise pickle.UnpicklingError(f'Unexpected scheduler attribute {name}')
 return lambda *args,**kwargs:None

ALLOWED={
 ('__builtin__','dict'):dict,('__builtin__','getattr'):safe_getattr,
 ('builtins','dict'):dict,('builtins','getattr'):safe_getattr,
 ('collections','OrderedDict'):collections.OrderedDict,
 ('collections','defaultdict'):collections.defaultdict,
 ('torch','FloatStorage'):torch.FloatStorage,
 ('torch._utils','_rebuild_tensor_v2'):torch._utils._rebuild_tensor_v2,
 ('torch._utils','_rebuild_parameter'):torch._utils._rebuild_parameter,
 ('torch.optim.adamw','AdamW'):TrainingStatePlaceholder,
 ('torch.optim.lr_scheduler','OneCycleLR'):TrainingStatePlaceholder}

class RestrictedUnpickler(pickle.Unpickler):
 def find_class(self,module,name):
  if (module,name) not in ALLOWED:raise pickle.UnpicklingError(f'Forbidden global {module}.{name}')
  return ALLOWED[module,name]

if __name__=='__main__':
 argparse.ArgumentParser(description=__doc__).parse_args()
 assert digest(CHECKPOINT)==EXPECTED
 with zipfile.ZipFile(CHECKPOINT) as z:
  data=z.read(next(k for k in z.namelist() if k.endswith('data.pkl')))
 globals_used=sorted({a for op,a,p in pickletools.genops(data) if op.name=='GLOBAL'})
 for g in globals_used:assert tuple(g.split(' ',1)) in ALLOWED,g
 module=types.SimpleNamespace(__name__='restricted_checkpoint_pickle',Unpickler=RestrictedUnpickler)
 checkpoint=torch.load(CHECKPOINT,map_location='cpu',pickle_module=module,weights_only=False)
 weights={k.removeprefix('target_encoder.'):v for k,v in checkpoint['state_dict'].items() if k.startswith('target_encoder.')}
 assert weights and all(isinstance(v,torch.Tensor) for v in weights.values())
 torch.save(weights,HERE/'encoder_weights.pt')
 report={'source_sha256':EXPECTED,'tensor_keys':len(weights),'tensor_parameters':sum(v.numel() for v in weights.values()),'globals_used':globals_used,'extracted_weights_sha256':digest(HERE/'encoder_weights.pt'),'loading':'restricted class allowlist; optimizer/scheduler discarded; extracted tensors load with weights_only=True'}
 (HERE/'weights_extraction.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
 print(json.dumps(report,indent=2))
