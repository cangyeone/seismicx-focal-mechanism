"""Differentiable neural body arrivals and dispersive surface-wave group times."""
from pathlib import Path
import torch
from .legacy_first_p import TravelTimeMLP
from .networks import body_network, PhaseAvailabilityNet, SurfaceVelocityNet
from .schema import BODY_PHASES, EARTH_MODELS


class NeuralTravelTimes:
    def __init__(self, assets=None, device='cpu', earth_model='iasp91'):
        self.assets=Path(assets) if assets else Path(__file__).resolve().parents[1]/'assets'
        if earth_model not in EARTH_MODELS:raise ValueError('Unknown Earth model')
        self.earth_model=earth_model;self.earth_index=EARTH_MODELS.index(earth_model)
        self.device=torch.device(device)
        self.first_p=TravelTimeMLP.load(self.assets/'first_p'/f'{earth_model}.pt',device=device)
        self.body=None;self.availability=None;self.surface=None
        path=self.assets/'body_travel_expert.pt'
        if not path.exists():path=self.assets/'body_travel.pt'
        if path.exists():
            state=torch.load(path,map_location=device,weights_only=True)
            self.body=body_network(state['config']).to(device).eval().requires_grad_(False)
            self.body.load_state_dict(state['state_dict'])
            self.availability=PhaseAvailabilityNet(**state['availability_config']).to(device).eval().requires_grad_(False)
            self.availability.load_state_dict(state['availability_state_dict'])
            self.body_metadata=state['metadata']
        path=self.assets/'surface_velocity.pt'
        if path.exists():
            state=torch.load(path,map_location=device,weights_only=True)
            self.surface=SurfaceVelocityNet(**state['config']).to(device).eval().requires_grad_(False)
            self.surface.load_state_dict(state['state_dict'])

    def _points(self,depth_km,distance_km):
        h,d=torch.broadcast_tensors(torch.as_tensor(depth_km,dtype=torch.float32,device=self.device),
            torch.as_tensor(distance_km,dtype=torch.float32,device=self.device))
        x=torch.stack([h.reshape(-1),d.reshape(-1)],dim=1)
        if not torch.isfinite(x).all() or (x<0).any() or (x[:,0]>200).any() or (x[:,1]>3000).any():
            raise ValueError('Neural body domain: depth 0–200 km; distance 0–3000 km')
        if not x.requires_grad:x=x.requires_grad_(True)
        return x,h.shape

    def body_arrival(self,depth_km,distance_km,phase='P_first',create_graph=False):
        """Keep create_graph=True to differentiate the source-ray direction again."""
        if phase in ('Pg_regional','Sg_regional'):
            # Regional catalog Pg/Sg includes the upgoing direct p/s branch;
            # TauP's literal Pg/Sg names contain downgoing crustal rays only.
            names=('p','Pg') if phase=='Pg_regional' else ('s','Sg')
            a=self.body_arrival(depth_km,distance_km,names[0],create_graph)
            b=self.body_arrival(depth_km,distance_km,names[1],create_graph)
            choose=a['arrival_valid']&(~b['arrival_valid']|(a['raw_time']<=b['raw_time']))
            result={}
            for key in ['time','raw_time','takeoff_angle','gradient_valid','arrival_valid','availability_probability','on_interface']:
                result[key]=torch.where(choose,a[key],b[key])
            result['gradient']=torch.where(choose[...,None],a['gradient'],b['gradient'])
            moho=float(self.first_p.interfaces.max())
            crust=torch.as_tensor(depth_km,device=self.device)<moho
            result['arrival_valid']=result['arrival_valid']&crust
            near=a['arrival_valid']&b['arrival_valid']&((a['raw_time']-b['raw_time']).abs()<.05)
            result['gradient_valid']=result['gradient_valid']&crust&~near
            result['time']=torch.where(result['arrival_valid'],result['time'],float('nan'))
            result['takeoff_angle']=torch.where(result['gradient_valid'],result['takeoff_angle'],float('nan'))
            result.update(phase=phase,earth_model=self.earth_model,near_branch_switch=near)
            return result
        x,shape=self._points(depth_km,distance_km)
        with torch.enable_grad():
            if phase=='P_first':
                time=self.first_p(x)
                confidence=torch.ones_like(time)
                interfaces=self.first_p.interfaces
            else:
                if self.body is None:raise FileNotFoundError('body_travel.pt is not trained yet')
                if phase not in BODY_PHASES:raise ValueError(f'Unsupported phase {phase}')
                ei=torch.full((len(x),),self.earth_index,dtype=torch.long,device=self.device)
                pi=torch.full_like(ei,BODY_PHASES.index(phase))
                time=self.body(x,ei,pi)
                confidence=self.availability(x,ei).sigmoid()[:,BODY_PHASES.index(phase)]
                interfaces=x.new_tensor(self.body_metadata['models'][self.earth_model]['interfaces'])
            gradient=torch.autograd.grad(time.sum(),x,create_graph=create_graph,retain_graph=True)[0]
            angle=torch.rad2deg(torch.atan2(gradient[:,1].abs()*6371/(6371-x[:,0]),-gradient[:,0]))
        interface=(x[:,0,None]-interfaces).abs().amin(1)<.002
        finite=torch.isfinite(time)&torch.isfinite(gradient).all(1)
        arrival_valid=(confidence>.99)&finite&(time>=0)
        gradient_valid=arrival_valid&~interface&~((x[:,0]==0)&(x[:,1]==0))&(gradient[:,1]>=-1e-6)
        # Internal numerical consistency is a quality flag, not an error certificate.
        return dict(time=torch.where(arrival_valid,time,float('nan')).reshape(shape),
            raw_time=time.reshape(shape),gradient=gradient.reshape(shape+(2,)),
            takeoff_angle=torch.where(gradient_valid,angle,float('nan')).reshape(shape),
            gradient_valid=gradient_valid.reshape(shape),arrival_valid=arrival_valid.reshape(shape),
            availability_probability=confidence.reshape(shape),on_interface=interface.reshape(shape),
            phase=phase,earth_model=self.earth_model)

    def surface_arrival(self,distance_km,period_s,wave='rayleigh'):
        if self.surface is None:raise FileNotFoundError('surface_velocity.pt is not trained yet')
        if wave not in ('rayleigh','love'):raise ValueError('wave must be rayleigh or love')
        distance,period=torch.broadcast_tensors(torch.as_tensor(distance_km,dtype=torch.float32,device=self.device),
            torch.as_tensor(period_s,dtype=torch.float32,device=self.device))
        if not torch.isfinite(distance).all() or not torch.isfinite(period).all() or (distance<0).any() or (distance>3000).any():
            raise ValueError('Surface distance must be within 0–3000 km')
        if (period<2).any() or (period>80).any():raise ValueError('Surface neural period domain: 2–80 seconds')
        shape=period.shape;flat=period.reshape(-1)
        ei=torch.full((flat.numel(),),self.earth_index,device=self.device,dtype=torch.long)
        wi=torch.full_like(ei,0 if wave=='rayleigh' else 1)
        velocity=self.surface(flat,ei,wi).reshape(shape+(2,))
        valid=torch.isfinite(velocity).all(-1)&(velocity>0).all(-1)
        time=distance/velocity[...,1]
        return dict(time=torch.where(valid,time,float('nan')),phase_velocity=velocity[...,0],
            group_velocity=velocity[...,1],valid=valid,wave=wave,mode=0,period=period)
