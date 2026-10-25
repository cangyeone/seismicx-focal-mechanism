"""Per-velocity-model differentiable first-P MLP, with optional Fourier features."""
from pathlib import Path
import math
import torch
from torch import nn

class TravelTimeMLP(nn.Module):
    def __init__(self,interfaces=(),width=128,depth=5,fourier=True,radius_km=6371.):
        super().__init__();self.radius=radius_km
        self.config=dict(interfaces=list(interfaces),width=width,depth=depth,fourier=fourier,radius_km=radius_km)
        self.register_buffer('interfaces',torch.tensor(interfaces,dtype=torch.float32))
        self.register_buffer('frequencies',torch.tensor([1.,2.,4.,8.]) if fourier else torch.empty(0))
        n=6+len(interfaces)+4*len(self.frequencies)
        modules=[nn.Linear(n,width),nn.SiLU()]
        for _ in range(depth-1):modules.extend([nn.Linear(width,width),nn.SiLU()])
        modules.append(nn.Linear(width,1));self.network=nn.Sequential(*modules)
        nn.init.zeros_(self.network[-1].weight);nn.init.zeros_(self.network[-1].bias)
    def features(self,x):
        h=x[:,0:1];d=x[:,1:2];R=self.radius
        chord2=h.square()+4*R*(R-h)*torch.sin(d/(2*R)).square()
        length=torch.sqrt(chord2+1e-12)
        even_d=(torch.sqrt(d.square()+1)-1)/3000
        logh=torch.log1p(h/5)/math.log(41)
        logd=.5*torch.log1p((d/25).square())/(.5*math.log1p(120**2))
        base=torch.cat((h/200,even_d,logh,logd,h/(length+1),d.square()/(length+1).square()),-1)
        values=[base]
        if self.interfaces.numel():values.append(torch.relu(h-self.interfaces[None,:])/200)
        if self.frequencies.numel():
            phase=2*torch.pi*torch.cat((logh,logd),-1)[...,None]*self.frequencies
            values.extend((phase.sin().flatten(1),phase.cos().flatten(1)))
        return torch.cat(values,-1),length[:,0]
    def forward(self,x):
        features,length=self.features(x)
        slowness=.125+.05*self.network(features)[:,0]
        return (length-1e-6)*slowness
    def time_gradient(self,x,create_graph=False):
        if not x.requires_grad:x=x.detach().requires_grad_(True)
        time=self(x)
        gradient=torch.autograd.grad(time.sum(),x,create_graph=create_graph)[0]
        return time,gradient
    def predict(self,depth_km,distance_km,device=None):
        device=next(self.parameters()).device if device is None else torch.device(device)
        dtype=next(self.parameters()).dtype
        h,d=torch.broadcast_tensors(torch.as_tensor(depth_km,device=device,dtype=dtype),torch.as_tensor(distance_km,device=device,dtype=dtype))
        shape=h.shape;x=torch.stack((h.reshape(-1),d.reshape(-1)),-1).requires_grad_(True)
        if not bool(torch.isfinite(x).all()) or bool((x<0).any()) or bool((x[:,0]>200).any()) or bool((x[:,1]>3000).any()):raise ValueError('domain: depth 0–200 km, epicentral distance 0–3000 km')
        with torch.enable_grad():time,g=self.time_gradient(x)
        angle=torch.rad2deg(torch.atan2(g[:,1].abs()*self.radius/(self.radius-x[:,0]),-g[:,0]))
        singular=(x[:,0]==0)&(x[:,1]==0)
        boundary=(x[:,0,None]-self.interfaces).abs().amin(-1)<1e-4 if self.interfaces.numel() else torch.zeros(len(x),dtype=torch.bool,device=device)
        valid=torch.isfinite(time)&(time>=0)&torch.isfinite(angle)&(angle>=0)&(angle<=180)&(g[:,1]>=-1e-6)&~singular&~boundary
        return dict(time=time.detach().reshape(shape),gradient=g.detach().reshape(shape+(2,)),takeoff_angle=torch.where(valid,angle,float('nan')).detach().reshape(shape),gradient_valid=valid.reshape(shape),on_interface=boundary.reshape(shape))
    @classmethod
    def load(cls,path,device='cpu',trainable=False):
        checkpoint=torch.load(Path(path),map_location=device,weights_only=True)
        model=cls(**checkpoint['config']).to(device);model.load_state_dict(checkpoint['state_dict']);model.eval();model.requires_grad_(trainable);model.metadata=checkpoint['metadata'];return model


def load_model(name='iasp91',device='cpu'):
    """Load a packaged model; predict uses km and returns seconds/degrees."""
    if name not in ('iasp91','ak135','prem','iasp91-plain'):
        raise ValueError('unknown packaged model')
    return TravelTimeMLP.load(Path(__file__).parent/'models'/(name+'.pt'),device=device)
