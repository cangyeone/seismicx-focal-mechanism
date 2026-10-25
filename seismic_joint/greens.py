"""GRTM physical Green bank, with exact linear source derivatives."""
import hashlib
import json
from pathlib import Path
from contextlib import nullcontext
import numpy as np
import torch


class GRTMGreens:
    def __init__(self,assets=None,device='cpu',cache=None):
        self.assets=Path(assets) if assets else Path(__file__).resolve().parents[1]/'assets'
        self.device=torch.device(device)
        self.cache=Path(cache) if cache else self.assets.parent/'green_cache'
        self.cache.mkdir(parents=True,exist_ok=True)

    def definition(self,name):
        return json.loads((self.assets/'earth'/f'{name}.json').read_text())

    def configuration(self,earth,depth_km,distances_km,dt,n_samples,duration_s):
        if n_samples<64 or n_samples&(n_samples-1):raise ValueError('GRTM n_samples must be a power of two >=64')
        if duration_s<=0 or dt<=0:raise ValueError('Positive dt and source_duration_s required')
        definition=self.definition(earth)
        layers=np.asarray(definition['layers'],dtype=float).copy()
        layers[:,:4]*=1000
        distances=np.asarray(distances_km,dtype=float)
        if (distances<=0).any() or (distances>500).any():
            raise ValueError('This CAP configuration is validated only for positive regional distances <=500 km')
        config=dict(layers=layers.tolist(),source_depth=float(depth_km)*1000,receiver_depth=0.,
            distances=(distances*1000).tolist(),dt=float(dt),log2_samples=int(np.log2(n_samples)),
            t0=[0.]*len(distances),taper=.3,ptam=0,source_type=0)
        t=np.arange(n_samples)*dt
        # Dimensionless moment history; GRTM convolves impulse kernels with dt.
        stf=.5-.5*np.cos(np.pi*np.clip(t/duration_s,0,1))
        return config,layers,stf

    def basis(self,earth,depth_km,distances_km,azimuth_deg,dt,n_samples,duration_s=1.,grid=None):
        import grtm
        from grtm.sensitivity import ForwardOperator
        config,layers,stf=self.configuration(earth,depth_km,distances_km,dt,n_samples,duration_s)
        identity=dict(config=config,azimuth=np.asarray(azimuth_deg).tolist(),source_duration_s=duration_s,
            grtm_version=grtm.__version__,units='si',coordinates='ned',grid=grid,base_moment_Nm=1e15)
        key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
        path=self.cache/f'{key}.npz'
        if path.exists():
            with np.load(path) as data:bank=data['basis']
        else:
            context=torch.cuda.device(self.device) if self.device.type=='cuda' else nullcontext()
            with context:
                options=dict(threads=4)
                if self.device.type=='cuda':options['device']=self.device.index or 0
                operator=ForwardOperator(config,backend='cuda' if self.device.type=='cuda' else 'cpu',
                    units='si',coordinates='ned',azimuth=azimuth_deg,field='displacement',
                    stf=stf,parameters=(),grid=grid,**options)
                linear=operator.linearize(layers,np.zeros(6))
                bank=np.stack([linear.jvp(np.zeros_like(layers),np.eye(6)[i]*1e15) for i in range(6)],axis=-1)
            if not np.isfinite(bank).all():raise RuntimeError('GRTM produced a nonfinite Green bank')
            tmp=path.with_suffix('.tmp.npz')
            np.savez_compressed(tmp,basis=bank)
            tmp.replace(path)
            (self.cache/f'{key}.json').write_text(json.dumps(identity,indent=2)+'\n')
        return torch.as_tensor(bank,dtype=torch.float64,device=self.device)

    def differentiable_forward(self,earth,depth_km,distances_km,azimuth_deg,dt,n_samples,duration_s=1.,parameters=()):
        """Return native first-order layer/MT operator; source geometry is fixed.

        This is for structural experiments. Source-only fitting uses the exact
        linear bank above, avoiding repeated physical propagation each iteration.
        """
        from grtm.torch import Forward
        config,layers,stf=self.configuration(earth,depth_km,distances_km,dt,n_samples,duration_s)
        options=dict(units='si',coordinates='ned',azimuth=azimuth_deg,field='displacement',
            stf=stf,parameters=parameters,threads=4)
        if self.device.type=='cuda':options['device']=self.device.index or 0
        op=Forward(config,backend='cuda' if self.device.type=='cuda' else 'cpu',vjp_method='wavefield',**options)
        return op,torch.as_tensor(layers,dtype=torch.float64,device=self.device)


def interpolate_depth(depth,grid,banks):
    """Piecewise-linear approximation only; not a native GRTM geometry derivative."""
    if grid.ndim!=1 or len(grid)<2 or banks.shape[0]!=len(grid) or (grid[1:]<=grid[:-1]).any():
        raise ValueError('Need increasing depth grid and matching banks')
    if depth<grid[0] or depth>grid[-1]:raise ValueError('Depth outside Green-bank interpolation grid')
    index=torch.searchsorted(grid,depth.detach()).clamp(1,len(grid)-1)
    fraction=(depth-grid[index-1])/(grid[index]-grid[index-1])
    return banks[index-1]+fraction*(banks[index]-banks[index-1])
