"""Amplitude-preserving five-segment CAP plus probabilistic first motions."""
from dataclasses import dataclass
import math
import torch
from .mechanism import moment_sdr,nodal_planes,polarity_loss,radiation


@dataclass
class CAPSettings:
    body_band_hz: tuple=(.05,.5)
    surface_band_hz: tuple=(.02,.1)
    window_samples: int=128
    body_before_p_s: float=2.
    body_after_s_s: float=2.
    surface_padding_s: float=15.
    body_shift_s: float=2.
    surface_shift_s: float=5.
    polarity_weight: float=.2
    shift_penalty: float=.001
    starts: int=64
    steps: int=1000


def bandpass(wave,dt,band,order=4):
    """Identical zero-phase FFT filter for data/synthetics; padded to limit wrap."""
    low,high=band
    if not 0<low<high<.5/dt:raise ValueError('Filter band must be below Nyquist')
    n=wave.shape[-2]
    frequency=torch.fft.rfftfreq(2*n,dt,device=wave.device).to(wave.dtype)
    safe=frequency.clamp_min(1e-12)
    response=(1+(low/safe)**(2*order)).rsqrt()*(1+(safe/high)**(2*order)).rsqrt()
    response[0]=0
    spectrum=torch.fft.rfft(wave,n=2*n,dim=-2)
    return torch.fft.irfft(spectrum*response[:,None],n=2*n,dim=-2)[...,:n,:]


def to_rtd(ned,azimuth_deg):
    angle=torch.deg2rad(azimuth_deg)
    n,e,d=ned.unbind(-1)
    shape=(1,)*(ned.ndim-3)+(len(angle),1)
    ca,sa=angle.cos().reshape(shape),angle.sin().reshape(shape)
    return torch.stack([n*ca+e*sa,-n*sa+e*ca,d],dim=-1)


def sample_components(waves,times,components,dt,t0):
    """Linear sampling; exact piecewise time derivative through interpolation.

    waves [B,S,T,3], times [B,S,G,K], t0 [S]. Returns values and sample coverage.
    """
    b,s,t,c=waves.shape
    sample=(times-t0[None,:,None,None])/dt
    left=torch.floor(sample).long()
    fraction=sample-left.to(sample.dtype)
    valid=(sample>=0)&(sample<=t-1)
    left=left.clamp(0,t-2)
    components=components[None,None,:,None].expand_as(left)
    flat=waves.reshape(b,s,t*c)
    a=torch.gather(flat,2,(left*c+components).reshape(b,s,-1)).reshape_as(sample)
    z=torch.gather(flat,2,((left+1)*c+components).reshape(b,s,-1)).reshape_as(sample)
    # Clamp endpoints explicitly so an exact final sample does not use fraction=0.
    fraction=(sample-left.to(sample.dtype)).clamp(0,1)
    return a+(z-a)*fraction,valid


class CAPObjective:
    def __init__(self,observed_ned,basis,distances,azimuth,dt,t0,travel,depth_km,rays,
                 probability_u,polarity_valid,settings=None):
        self.settings=settings or CAPSettings();self.dt=dt;self.rays=rays.detach()
        self.probability_u=probability_u.detach();self.polarity_valid=polarity_valid
        self.basis=basis;self.azimuth=azimuth;self.t0=t0
        self.observed=to_rtd(observed_ned,azimuth)
        self.body_observed=bandpass(self.observed,dt,self.settings.body_band_hz)
        self.surface_observed=bandpass(self.observed,dt,self.settings.surface_band_hz)
        self.components=torch.tensor([2,0,2,0,1],device=basis.device)
        p=travel.body_arrival(depth_km,distances,'P_first')
        s=travel.body_arrival(depth_km,distances,'S_first')
        if not p['arrival_valid'].all() or not s['arrival_valid'].all():
            raise ValueError('No reliable P/S arrival for one or more CAP stations')
        ptime=p['time'].detach().to(basis.dtype);stime=s['time'].detach().to(basis.dtype)
        start_body=ptime-self.settings.body_before_p_s
        duration_body=(stime-ptime+self.settings.body_before_p_s+self.settings.body_after_s_s).clamp(8,80)
        periods=torch.linspace(1/self.settings.surface_band_hz[1],1/self.settings.surface_band_hz[0],16,device=basis.device)
        starts,durations=[start_body,start_body],[duration_body,duration_body]
        for wave in ['rayleigh','rayleigh','love']:
            group=travel.surface_arrival(distances[:,None],periods[None],wave)
            if not group['valid'].all():raise ValueError('Invalid neural surface group velocity')
            time=group['time'].detach().to(basis.dtype)
            lower=time.amin(1)-self.settings.surface_padding_s
            upper=time.amax(1)+self.settings.surface_padding_s
            starts.append(lower);durations.append(upper-lower)
        start=torch.stack(starts,dim=1);duration=torch.stack(durations,dim=1)
        self.duration=duration
        u=torch.linspace(0,1,self.settings.window_samples,device=basis.device,dtype=basis.dtype)
        self.times=start[...,None]+duration[...,None]*u
        self.group_index=torch.tensor([0,0,1,1,2],device=basis.device)
        self.shift_limit=basis.new_tensor([self.settings.body_shift_s,self.settings.surface_shift_s,self.settings.surface_shift_s])
        maxshift=self.shift_limit[self.group_index][None,:]
        # One fixed coverage mask; the optimizer cannot improve by sliding out of data.
        obs_lower=t0[:,None]+maxshift
        obs_upper=t0[:,None]+(observed_ned.shape[1]-1)*dt-maxshift
        syn_lower=maxshift;syn_upper=(basis.shape[1]-1)*dt-maxshift
        self.segment_valid=(start>=torch.maximum(obs_lower,syn_lower))&((start+duration)<=torch.minimum(obs_upper,syn_upper))
        exponent=basis.new_tensor([1.13,1.13,.74,.74,.55])
        distance_weight=(distances[:,None].to(basis.dtype)/100).pow(exponent)
        self.weight=distance_weight.square()*duration/self.settings.window_samples*self.segment_valid
        taper=torch.sin(torch.linspace(0,torch.pi,self.settings.window_samples,device=basis.device,dtype=basis.dtype)).square()
        self.sample_weight=self.weight[...,None]*taper
        data_body,_=sample_components(self.body_observed[None],self.times[None],self.components,dt,t0)
        data_surface,_=sample_components(self.surface_observed[None],self.times[None],self.components,dt,t0)
        self.target=torch.cat([data_body[:,:,:2],data_surface[:,:,2:]],dim=2)
        self.energy=(self.target.square()*self.sample_weight[None]).sum()
        if not torch.isfinite(self.energy) or self.energy<=0:raise ValueError('No complete nonzero CAP windows; provide longer response-corrected records')
        # Filter each fixed source basis once; optimization remains exactly linear in MT.
        self.filtered_basis=[]
        for band in [self.settings.body_band_hz,self.settings.surface_band_hz]:
            rows=[]
            for i in range(6):rows.append(bandpass(to_rtd(basis[...,i],azimuth),dt,band))
            self.filtered_basis.append(torch.stack(rows,dim=-1))

    def __call__(self,angles,log_amplitude,raw_shift):
        moment=moment_sdr(angles[:,0],angles[:,1],angles[:,2])
        amplitude=log_amplitude.exp()
        shifts=raw_shift.tanh()*self.shift_limit
        times=self.times[None]+shifts[:,:,self.group_index,None]
        outputs=[]
        zero_t0=torch.zeros_like(self.t0)
        for bank in self.filtered_basis:
            wave=torch.einsum('stck,bk->bstc',bank,moment)*amplitude[:,None,None,None]
            values,_=sample_components(wave,times,self.components,self.dt,zero_t0)
            outputs.append(values)
        predicted=torch.cat([outputs[0][:,:,:2],outputs[1][:,:,2:]],dim=2)
        residual=((predicted-self.target).square()*self.sample_weight[None]).sum(dim=(1,2,3))/self.energy
        if self.polarity_valid.any() and self.settings.polarity_weight>0:
            first=polarity_loss(moment,self.rays,self.probability_u,self.polarity_valid)
        else:first=torch.zeros_like(residual)
        shift_regularization=(shifts/self.shift_limit).square().mean(dim=(1,2))
        loss=residual+self.settings.polarity_weight*first+self.settings.shift_penalty*shift_regularization
        return loss,dict(waveform=residual,polarity=first,shifts=shifts,predicted=predicted,moment=moment)


def fit_joint(objective,seed=20261025):
    config=objective.settings;device=objective.basis.device;dtype=objective.basis.dtype
    generator=torch.Generator(device=device).manual_seed(seed)
    angles=torch.nn.Parameter(torch.rand((config.starts,3),device=device,dtype=dtype,generator=generator))
    with torch.no_grad():
        angles[:,0]*=360;angles[:,1]=5+80*angles[:,1];angles[:,2]=angles[:,2]*360-180
    log_amplitude=torch.nn.Parameter(torch.zeros(config.starts,device=device,dtype=dtype))
    shift=torch.nn.Parameter(torch.zeros(config.starts,len(objective.azimuth),3,device=device,dtype=dtype))
    optimizer=torch.optim.Adam([{'params':[angles],'lr':.8},
        {'params':[log_amplitude],'lr':.03},{'params':[shift],'lr':.03}])
    history=[]
    for step in range(config.steps):
        loss,parts=objective(angles,log_amplitude,shift)
        optimizer.zero_grad();loss.sum().backward();optimizer.step()
        with torch.no_grad():angles[:,1].clamp_(.05,89.95);log_amplitude.clamp_(-12,12)
        if step%100==0 or step+1==config.steps:
            history.append(dict(step=step+1,best_objective=float(loss.min().detach())))
    with torch.no_grad():
        loss,parts=objective(angles,log_amplitude,shift);best=int(loss.argmin())
        m=parts['moment'][best];scalar=1e15*log_amplitude[best].exp()
        tensor=(m*scalar).cpu().numpy()
        used=objective.polarity_valid
        sign=radiation(m,objective.rays)>0
        mismatch=int(((sign!=(objective.probability_u>.5))&used).sum())
        variance_reduction=1-float(parts['waveform'][best])
    report=dict(mode='cap_polarity',moment_tensor_Nm=tensor.tolist(),nodal_planes=nodal_planes(tensor),
        scalar_moment_Nm=float(scalar),moment_magnitude=float((torch.log10(scalar)-9.1)/1.5),
        objective=float(loss[best]),waveform_normalized_squared_residual=float(parts['waveform'][best]),
        variance_reduction=variance_reduction,polarity_loss=float(parts['polarity'][best]),
        polarity_mismatches=mismatch,polarity_stations=int(used.sum()),
        segment_order=['Pnl_D','Pnl_R','Rayleigh_D','Rayleigh_R','Love_T'],
        valid_segments=objective.segment_valid.cpu().tolist(),shifts_s=parts['shifts'][best].cpu().tolist(),
        shift_sign='Synthetic is sampled at t+shift to match observed(t)',
        multistart=config.starts,optimization_history=history,
        coordinates='NED',moment_order=['NN','EE','DD','NE','ND','ED'],
        depth_method='fixed physical GRTM bank at each configured depth; optional outer depth grid')
    traces=dict(observed=objective.target[0].cpu().numpy(),predicted=parts['predicted'][best].cpu().numpy(),
        relative_times_s=objective.times.cpu().numpy(),valid=objective.segment_valid.cpu().numpy())
    return report,traces
