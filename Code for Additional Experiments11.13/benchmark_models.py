"""Explicit, lightweight architecture variants for cognitive-stage classification.

Names ending in -lite/-SE/-MS identify local implementations, not official
checkpoint reproductions. All consume the same 60 x 400 preprocessed signal.
"""
import torch
from torch import nn
from neural_stage import StageNet


class SE(nn.Module):
    def __init__(self,c):
        super().__init__();self.layers=nn.Sequential(nn.Linear(c,max(2,c//4)),nn.ELU(),nn.Linear(max(2,c//4),c),nn.Sigmoid())
    def forward(self,x):
        return x*self.layers(x.mean(dim=(2,3)))[:,:,None,None]


def stem(kernel=33,width=8):
    return nn.Sequential(nn.Conv2d(1,width,(1,kernel),padding=(0,kernel//2),bias=False),nn.BatchNorm2d(width),
                         nn.Conv2d(width,width*2,(60,1),groups=width,bias=False),nn.BatchNorm2d(width*2),nn.ELU(),
                         nn.AvgPool2d((1,4)),nn.Dropout(.4))


class EEGNetSE(nn.Module):
    def __init__(self):
        super().__init__();self.features=nn.Sequential(stem(),
            nn.Conv2d(16,16,(1,17),padding=(0,8),groups=16,bias=False),nn.Conv2d(16,16,1,bias=False),
            nn.BatchNorm2d(16),nn.ELU(),SE(16),nn.AvgPool2d((1,8)),nn.Dropout(.4),nn.Flatten())
        self.head=nn.Linear(192,2)
    def forward(self,x):return self.head(self.features(x))




class ShallowLogVar(nn.Module):
    def __init__(self):
        super().__init__();self.features=nn.Sequential(nn.Conv2d(1,16,(1,25),padding=(0,12),bias=False),
            nn.Conv2d(16,24,(60,1),bias=False),nn.BatchNorm2d(24))
        self.drop=nn.Dropout(.5);self.head=nn.Linear(24*8,2)
    def forward(self,x):
        x=self.features(x).squeeze(2).reshape(-1,24,8,50)
        return self.head(self.drop(torch.log(x.var(dim=-1,unbiased=False).clamp(min=1e-6)).flatten(1)))


class SpatialFirstMS(nn.Module):
    """Low-rank spatial compression followed by multiscale temporal branches."""
    def __init__(self):
        super().__init__()
        self.spatial=nn.Sequential(nn.Conv2d(1,16,(60,1),bias=False),nn.BatchNorm2d(16))
        self.branches=nn.ModuleList([nn.Sequential(nn.Conv1d(16,16,k,padding=k//2,groups=16,bias=False),
                   nn.Conv1d(16,8,1,bias=False),nn.BatchNorm1d(8),nn.ELU()) for k in [17,33,65]])
        self.tail=nn.Sequential(nn.AvgPool1d(4),nn.Dropout(.4),
                   nn.Conv1d(24,24,17,padding=8,groups=24,bias=False),nn.Conv1d(24,24,1,bias=False),
                   nn.BatchNorm1d(24),nn.ELU(),nn.AvgPool1d(8),nn.Dropout(.4),nn.Flatten(),nn.Linear(288,2))
    def forward(self,x):
        x=self.spatial(x).squeeze(2)
        return self.tail(torch.cat([b(x) for b in self.branches],dim=1))


class EEGTCN(nn.Module):
    def __init__(self):
        super().__init__();self.stem=stem()
        self.layers=nn.ModuleList([nn.Sequential(nn.Conv1d(16,16,5,padding=2*d,dilation=d),
                 nn.BatchNorm1d(16),nn.ELU(),nn.Dropout(.3)) for d in [1,2,4]])
        self.head=nn.Sequential(nn.AvgPool1d(8),nn.Flatten(),nn.Linear(192,2))
    def forward(self,x):
        x=self.stem(x).squeeze(2)
        for layer in self.layers:x=x+layer(x)
        return self.head(x)


class EEGConformerLite(nn.Module):
    def __init__(self):
        super().__init__();self.stem=stem()
        self.patch=nn.Conv1d(16,24,4,stride=4)
        self.position=nn.Parameter(torch.randn(1,25,24)*.02)
        layer=nn.TransformerEncoderLayer(24,4,48,dropout=.3,activation='gelu',batch_first=True)
        self.transformer=nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
        self.head=nn.Sequential(nn.Dropout(.4),nn.Flatten(),nn.Linear(25*24,2))
    def forward(self,x):
        x=self.patch(self.stem(x).squeeze(2)).transpose(1,2)+self.position
        return self.head(self.transformer(x))


class FBCNetLite(nn.Module):
    """Fixed differentiable FFT filter bank + grouped spatial log-variance."""
    def __init__(self):
        super().__init__();freq=torch.fft.rfftfreq(400,d=1/100)
        masks=torch.stack([((freq>=lo)&(freq<hi)).float() for lo,hi in [(1,4),(4,8),(8,14),(14,30),(30,40.01)]])
        self.register_buffer('masks',masks)
        self.spatial=nn.Sequential(nn.Conv2d(5,40,(60,1),groups=5,bias=False),nn.BatchNorm2d(40),nn.ELU())
        self.head=nn.Sequential(nn.Dropout(.4),nn.Linear(40*8,2))
    def forward(self,x):
        # float32 FFT avoids unsupported half-precision non-power-of-two transforms.
        with torch.autocast(device_type=x.device.type,enabled=False):
            spectrum=torch.fft.rfft(x[:,0].float(),dim=-1)
            banded=torch.fft.irfft(spectrum[:,None]*self.masks[None,:,None],n=400,dim=-1)
        h=self.spatial(banded).reshape(-1,40,8,50)
        return self.head(torch.log(h.var(dim=-1,unbiased=False).clamp(min=1e-6)).flatten(1))


FACTORIES={'EEGNet-lite':StageNet,'EEGNet-SE':EEGNetSE,
           'ShallowLogVar':ShallowLogVar,'EEG-TCN':EEGTCN,'EEGConformer-lite':EEGConformerLite,
           'FBCNet-lite':FBCNetLite}
FACTORIES['MS-EEGNet-SF']=SpatialFirstMS
