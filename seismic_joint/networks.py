"""Smooth time fields; physical gradients are obtained by differentiating time."""
import math
import torch
from torch import nn
from .schema import BODY_PHASES, EARTH_MODELS


def mlp(nin, nout, width, depth):
    modules = [nn.Linear(nin, width), nn.SiLU()]
    for _ in range(depth-1):
        modules.extend([nn.Linear(width, width), nn.SiLU()])
    modules.append(nn.Linear(width, nout))
    return nn.Sequential(*modules)


def geometry_features(x):
    h, d = x[:, :1], x[:, 1:2]
    length = torch.sqrt(h.square()+4*6371*(6371-h)*torch.sin(d/(2*6371)).square()+1e-12)
    logh = torch.log1p(h/5)/math.log(41)
    logd = .5*torch.log1p((d/25).square())/(.5*math.log1p(120**2))
    values = [h/200, (torch.sqrt(d.square()+1)-1)/3000, logh, logd,
              h/(length+1), d.square()/(length+1).square()]
    for z in (15., 20., 24.4, 35.):
        values.append(torch.relu(h-z)/200)
    frequency = x.new_tensor([1.,2.,4.,8.])
    phase = 2*torch.pi*torch.cat([logh,logd], dim=1)[...,None]*frequency
    values.extend([phase.sin().flatten(1), phase.cos().flatten(1)])
    return torch.cat(values, dim=1), length[:,0]


class BodyTravelNet(nn.Module):
    def __init__(self, width=256, depth=5):
        super().__init__()
        self.config = dict(width=width, depth=depth)
        self.network = mlp(26+len(EARTH_MODELS)+len(BODY_PHASES), 1, width, depth)
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, x, earth, phase):
        features, length = geometry_features(x)
        categories = torch.cat([torch.nn.functional.one_hot(earth, len(EARTH_MODELS)),
            torch.nn.functional.one_hot(phase, len(BODY_PHASES))], dim=1).to(x.dtype)
        correction = self.network(torch.cat([features,categories], dim=1))[:,0]
        is_s = (phase == 0)|(phase == 5)|(phase == 6)|(phase == 7)|(phase == 8)|(phase == 10)|(phase == 12)
        base = torch.where(is_s, .23, .13)
        return (length-1e-6)*(base+.05*correction)


class PhaseAvailabilityNet(nn.Module):
    def __init__(self, width=128):
        super().__init__()
        self.config = dict(width=width)
        self.network = mlp(26+len(EARTH_MODELS), len(BODY_PHASES), width, 4)

    def forward(self, x, earth):
        features, _ = geometry_features(x)
        category = torch.nn.functional.one_hot(earth, len(EARTH_MODELS)).to(x.dtype)
        return self.network(torch.cat([features,category], dim=1))


class SurfaceVelocityNet(nn.Module):
    def __init__(self, width=128, depth=4):
        super().__init__()
        self.config = dict(width=width, depth=depth)
        self.network = mlp(10, 2, width, depth)
        self.register_buffer('target_mean', torch.ones(2)*3.5)
        self.register_buffer('target_std', torch.ones(2))

    def forward(self, period, earth, wave):
        logt = torch.log(period/2)/math.log(40)
        phase = 2*torch.pi*logt[:,None]*period.new_tensor([1.,2.])
        features = torch.cat([logt[:,None], phase.sin(), phase.cos(),
            torch.nn.functional.one_hot(earth, 3).to(period.dtype),
            torch.nn.functional.one_hot(wave, 2).to(period.dtype)], dim=1)
        return self.target_mean + self.target_std*self.network(features)


class ExpertBodyTravelNet(nn.Module):
    """Independent phase/Earth experts, warm-started from the shared conditional net."""
    def __init__(self,width=256,depth=5,architecture='phase_experts'):
        super().__init__()
        self.config=dict(width=width,depth=depth,architecture=architecture)
        self.groups=len(BODY_PHASES)*len(EARTH_MODELS)
        shapes=[(26,width)]+[(width,width)]*(depth-1)+[(width,1)]
        self.weights=nn.ParameterList([nn.Parameter(torch.zeros(self.groups,i,o)) for i,o in shapes])
        self.biases=nn.ParameterList([nn.Parameter(torch.zeros(self.groups,1,o)) for _,o in shapes])
        phases=torch.arange(self.groups)%len(BODY_PHASES)
        is_s=(phases==0)|(phases==5)|(phases==6)|(phases==7)|(phases==8)|(phases==10)|(phases==12)
        self.register_buffer('base',torch.where(is_s,.23,.13))

    def initialize_shared(self,shared):
        with torch.no_grad():
            category=[]
            for earth in range(len(EARTH_MODELS)):
                for phase in range(len(BODY_PHASES)):
                    value=torch.zeros(len(EARTH_MODELS)+len(BODY_PHASES),device=self.base.device)
                    value[earth]=1;value[len(EARTH_MODELS)+phase]=1;category.append(value)
            category=torch.stack(category)
            layers=[m for m in shared.network if isinstance(m,nn.Linear)]
            for i,layer in enumerate(layers):
                weight=layer.weight[:,:26] if i==0 else layer.weight
                self.weights[i].copy_(weight.T[None].expand_as(self.weights[i]))
                bias=layer.bias[None].expand(self.groups,-1)
                if i==0:bias=bias+category@layer.weight[:,26:].T
                self.biases[i].copy_(bias[:,None])

    def forward_grouped(self,x):
        # x [39,B,2], every expert gets an equally sized independent sample batch.
        g,b,_=x.shape
        h,length=geometry_features(x.reshape(-1,2));h=h.reshape(g,b,-1)
        for i,(weight,bias) in enumerate(zip(self.weights,self.biases)):
            h=torch.bmm(h,weight)+bias
            if i+1<len(self.weights):h=torch.nn.functional.silu(h)
        return (length.reshape(g,b)-1e-6)*(self.base[:,None]+.05*h[:,:,0])

    def forward(self,x,earth,phase):
        features,length=geometry_features(x)
        groups=earth*len(BODY_PHASES)+phase
        output=torch.zeros_like(length)
        for group in torch.unique(groups).tolist():
            use=groups==group;h=features[use]
            for i,(weight,bias) in enumerate(zip(self.weights,self.biases)):
                h=h@weight[group]+bias[group]
                if i+1<len(self.weights):h=torch.nn.functional.silu(h)
            output[use]=(length[use]-1e-6)*(self.base[group]+.05*h[:,0])
        return output


def body_network(config):
    return ExpertBodyTravelNet(**config) if config.get('architecture')=='phase_experts' else BodyTravelNet(**config)
