"""One JSON event -> first-motion or CAP+first-motion focal mechanism."""
from dataclasses import fields
import json
from pathlib import Path
import numpy as np
import torch
from .cap import CAPObjective,CAPSettings,fit_joint
from .greens import GRTMGreens
from .mechanism import fit_polarity,ray_direction
from .travel import NeuralTravelTimes


def geographic_geometry(event_lat,event_lon,station_lat,station_lon):
    """Spherical epicentral arc km and source-to-station clockwise azimuth."""
    lat1,lon1,lat2,lon2=map(torch.deg2rad,torch.broadcast_tensors(event_lat,event_lon,station_lat,station_lon))
    dl=lon2-lon1
    a=((lat2-lat1)/2).sin().square()+lat1.cos()*lat2.cos()*(dl/2).sin().square()
    distance=2*6371*torch.atan2(a.clamp(0,1).sqrt(),(1-a).clamp(0,1).sqrt())
    azimuth=torch.rad2deg(torch.atan2(dl.sin()*lat2.cos(),lat1.cos()*lat2.sin()-lat1.sin()*lat2.cos()*dl.cos()))%360
    return distance,azimuth


def predict_first_motion(raw_z,assets=None,device=None):
    """Preserves the differentiable probability path; input is raw 100 Hz Z up."""
    assets=Path(assets) if assets else Path(__file__).resolve().parents[1]/'assets'
    device=raw_z.device if device is None else device
    model=torch.jit.load(str(assets/'polarity_raw.jit'),map_location=device).eval()
    probability,direction,valid=model(raw_z.to(device))
    return dict(probability_u=probability[:,0],probabilities=probability,direction=direction,valid=valid)


def window_raw_z(raw_z,p_time_s,t0_s=0.,dt_s=.01):
    """Differentiable linear 1024-point crop; nominal P at sample 512."""
    if raw_z.ndim!=1 or raw_z.numel()<1024 or abs(float(dt_s)-.01)>1e-7:
        raise ValueError('Continuous Z must be one 100 Hz trace with at least 1024 samples')
    center=(torch.as_tensor(p_time_s,device=raw_z.device,dtype=raw_z.dtype)-t0_s)/dt_s
    sample=center+torch.arange(1024,device=raw_z.device,dtype=raw_z.dtype)-512
    if not torch.isfinite(sample).all() or sample[0]<0 or sample[-1]>raw_z.numel()-1:
        raise ValueError('Predicted/picked P has insufficient 1024-point Z context')
    left=sample.floor().long().clamp(0,raw_z.numel()-2)
    fraction=(sample-left.to(sample.dtype)).clamp(0,1)
    return raw_z[left]+fraction*(raw_z[left+1]-raw_z[left])


def read_event(path,device,assets=None):
    path=Path(path).resolve();config=json.loads(path.read_text());base=path.parent
    stations=config['stations'];event=config['event']
    if not stations:raise ValueError('No stations')
    ids=[item['id'] for item in stations]
    if len(set(ids))!=len(ids):raise ValueError('Duplicate station IDs')
    geometry=[]
    for station in stations:
        if 'distance_km' in station and 'azimuth_deg' in station:
            geometry.append([float(station['distance_km']),float(station['azimuth_deg'])])
        else:
            values=[event['latitude'],event['longitude'],station['latitude'],station['longitude']]
            distance,azimuth=geographic_geometry(*[torch.tensor(x,dtype=torch.float64) for x in values])
            geometry.append([float(distance),float(azimuth)])
    array=torch.tensor(geometry,dtype=torch.float64,device=device)
    if not torch.isfinite(array).all() or (array[:,0]<=0).any():raise ValueError('Invalid source/station geometry')
    probabilities=torch.full((len(stations),),.5,dtype=torch.float64,device=device)
    valid=torch.zeros(len(stations),dtype=torch.bool,device=device)
    raw_rows=[];raw_ids=[];crop_travel=None
    for i,station in enumerate(stations):
        if 'probability_u' in station:
            q=float(station['probability_u'])
            if not 0<=q<=1:raise ValueError('probability_u must be in [0,1]')
            probabilities[i]=q;valid[i]=True
        elif 'polarity' in station:
            label=station['polarity'].upper()
            if label not in ('U','D'):raise ValueError('polarity must be U or D')
            probabilities[i]=.995 if label=='U' else .005;valid[i]=True
        elif 'raw_z_file' in station:
            raw=np.load(base/station['raw_z_file'],allow_pickle=False)
            if raw.shape!=(1024,):raise ValueError('raw_z_file must be a 1024-point, 100 Hz, Z-up waveform')
            raw_rows.append(torch.tensor(raw.astype(np.float64),device=device));raw_ids.append(i)
        elif 'raw_z_continuous_file' in station:
            raw=torch.tensor(np.load(base/station['raw_z_continuous_file'],allow_pickle=False),dtype=torch.float64,device=device)
            if 'p_time_s' in station:ptime=float(station['p_time_s'])
            else:
                if crop_travel is None:crop_travel=NeuralTravelTimes(assets=assets,device=device,earth_model=config.get('earth_model','iasp91'))
                arrival=crop_travel.body_arrival(event['depth_km'],array[i,0],station.get('phase','P_first'))
                if not bool(arrival['arrival_valid']):raise ValueError(f'No reliable neural P time for {station["id"]}')
                ptime=arrival['time']
            cut=window_raw_z(raw,ptime,station['raw_z_t0_s'],station.get('raw_z_dt_s',.01))
            raw_rows.append(cut);raw_ids.append(i)
    if raw_rows:
        with torch.no_grad():
            result=predict_first_motion(torch.stack(raw_rows),assets=assets)
        index=torch.tensor(raw_ids,device=device)
        probabilities[index]=result['probability_u'].to(probabilities.dtype);valid[index]=result['valid']
    result=dict(config=config,station_ids=ids,distances=array[:,0],azimuth=array[:,1],
        probability_u=probabilities,polarity_valid=valid)
    if 'waveform' in config:
        description=config['waveform']
        if description.get('quantity')!='displacement_m':
            raise ValueError('CAP requires instrument-response-corrected displacement in metres; ADC counts cannot be fitted as physical amplitudes')
        wavefile=base/description['file']
        if wavefile.suffix=='.npz':
            with np.load(wavefile,allow_pickle=False) as data:wave=data['waveforms'].copy()
        else:wave=np.load(wavefile,allow_pickle=False)
        if wave.ndim!=3 or wave.shape[0]!=len(stations) or wave.shape[-1]!=3 or not np.isfinite(wave).all():
            raise ValueError('Waveforms must be finite [stations,samples,3] aligned to stations list')
        wave=torch.tensor(wave,dtype=torch.float64,device=device)
        coordinates=description.get('coordinates','NEZ')
        if coordinates=='NEZ':wave=torch.stack([wave[...,0],wave[...,1],-wave[...,2]],dim=-1)
        elif coordinates!='NED':raise ValueError('coordinates must be NEZ (Z up) or NED (D down)')
        dt=float(description['dt_s'])
        if dt<=0:raise ValueError('dt_s must be positive')
        t0=torch.as_tensor(description.get('t0_s',0),device=device,dtype=wave.dtype)
        if t0.ndim==0:t0=t0.repeat(len(stations))
        if t0.shape!=(len(stations),) or not torch.isfinite(t0).all():raise ValueError('t0_s must be scalar or one finite value per station')
        result.update(observed=wave,dt=dt,t0=t0)
    return result


def invert_event(event_path,output,mode='joint',device='cpu',assets=None):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    event=read_event(event_path,device,assets)
    config=event['config'];earth=config.get('earth_model','iasp91')
    travel=NeuralTravelTimes(assets=assets,device=device,earth_model=earth)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    depth_grid=config.get('depth_grid_km',[config['event']['depth_km']])
    if mode=='polarity':depth_grid=[config['event']['depth_km']]
    reports=[];traces={}
    for depth in depth_grid:
        phases=[station.get('phase','P_first') for station in config['stations']]
        direct_p={'P_first','Pg_regional','P','p','Pn','Pg'}
        if any(phase not in direct_p for phase in phases):
            raise ValueError('First-motion constraints require direct P-family phases; converted/reflected and S-family times are supported by arrivals, not by the direct-P radiation likelihood')
        # Match each observed first motion to its requested ray family.
        angle=torch.full_like(event['distances'],float('nan'))
        gradient_valid=torch.zeros_like(event['polarity_valid'])
        for phase in sorted(set(phases)):
            mask=torch.tensor([p==phase for p in phases],device=device)
            picked=travel.body_arrival(depth,event['distances'][mask],phase)
            angle[mask]=picked['takeoff_angle'].to(angle.dtype)
            gradient_valid[mask]=picked['gradient_valid']
        arrival=dict(takeoff_angle=angle,gradient_valid=gradient_valid)
        valid=event['polarity_valid']&arrival['gradient_valid']
        takeoff=torch.where(valid,arrival['takeoff_angle'].double(),torch.zeros_like(event['distances']))
        rays=ray_direction(event['azimuth'],takeoff)
        if mode=='polarity':
            report=fit_polarity(rays,event['probability_u'],valid,
                starts=config.get('polarity_starts',128),steps=config.get('polarity_steps',1000))
        else:
            if 'observed' not in event:raise ValueError('Joint CAP mode requires response-corrected waveforms')
            supplied=config.get('cap',{});known={field.name for field in fields(CAPSettings)}
            unknown=set(supplied)-known
            if unknown:raise ValueError(f'Unknown CAP options: {sorted(unknown)}')
            settings=CAPSettings(**supplied)
            length=config.get('synthetic_samples',1 << (event['observed'].shape[1]-1).bit_length())
            greens=GRTMGreens(assets=assets,device=device,cache=config.get('green_cache'))
            print(json.dumps(dict(stage='green_bank',depth_km=depth,device=device)),flush=True)
            basis=greens.basis(earth,depth,event['distances'].cpu().numpy(),event['azimuth'].cpu().numpy(),
                event['dt'],length,config.get('source_duration_s',1.))
            # The data windows/weights must stay identical across depth candidates.
            objective=CAPObjective(event['observed'],basis,event['distances'],event['azimuth'],
                event['dt'],event['t0'],travel,config['event']['depth_km'],rays,event['probability_u'],valid,settings)
            report,traces[depth]=fit_joint(objective)
            report['window_reference_depth_km']=float(config['event']['depth_km'])
        report.update(event_id=config['event'].get('id','event'),earth_model=earth,depth_km=float(depth),
            station_ids=event['station_ids'],travel_time_engine='DNN; TauP not imported at runtime',
            polarity_ray_phases=phases,
            azimuth_deg=event['azimuth'].cpu().tolist(),distance_km=event['distances'].cpu().tolist(),
            takeoff_angle_deg=[float(x) if torch.isfinite(x) else None for x in arrival['takeoff_angle'].detach().cpu()],
            takeoff_gradient_valid=arrival['gradient_valid'].detach().cpu().tolist(),
            probability_u=event['probability_u'].detach().cpu().tolist(),
            polarity_input_valid=event['polarity_valid'].cpu().tolist(),
            first_p_angle_caveat='Smooth first-arrival fields can have large angle errors at ray-branch switches; valid is not an accuracy certificate.')
        reports.append(report)
        print(json.dumps(dict(stage='depth_done',depth_km=depth,objective=report['objective'])),flush=True)
    best=min(reports,key=lambda x:x['objective'])
    best['polarity_mismatch_fraction']=best['polarity_mismatches']/max(1,best.get('polarity_stations',best.get('stations_used',0)))
    used_az=sorted(a for a,v in zip(best['azimuth_deg'],best['takeoff_gradient_valid']) if v)
    if len(used_az)>1:
        best['azimuthal_gap_deg']=max(b-a for a,b in zip(used_az,used_az[1:]+[used_az[0]+360]))
    manual=[(station.get('manual_label_for_audit_only'),q,v) for station,q,v in
        zip(config['stations'],best['probability_u'],best['polarity_input_valid'])]
    manual=[(label,q) for label,q,v in manual if label in ('U','D') and v]
    if manual:best['cnn_manual_audit']=dict(n=len(manual),agreement=sum((q>.5)==(label=='U') for label,q in manual))
    if traces:
        np.savez_compressed(output/'waveform_fit.npz',**traces[best['depth_km']])
    best['depth_candidates']=[dict(depth_km=r['depth_km'],objective=r['objective']) for r in reports]
    (output/'result.json').write_text(json.dumps(best,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    (output/'all_depth_results.json').write_text(json.dumps(reports,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    from .plots import plot_result
    plot_result(output)
    return best
