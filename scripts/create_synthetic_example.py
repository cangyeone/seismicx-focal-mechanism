"""Physical GRTM recovery example, explicitly synthetic (not a real-event result)."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from seismic_joint.greens import GRTMGreens
from seismic_joint.mechanism import moment_sdr,ray_direction,radiation
from seismic_joint.travel import NeuralTravelTimes


def main():
    p=argparse.ArgumentParser();p.add_argument('--device',default='cuda:0');args=p.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    folder=ROOT/'examples/synthetic';folder.mkdir(parents=True,exist_ok=True)
    distances=np.array([50.,70.,90.,120.,150.,180.,220.,280.])
    azimuth=np.array([10.,55.,100.,145.,190.,235.,280.,325.])
    truth=dict(strike=35.,dip=50.,rake=-70.,moment_magnitude=4.,depth_km=10.)
    greens=GRTMGreens(device=args.device)
    n=1024;dt=.25
    print('Computing one physical GRTM source basis',flush=True)
    bank=greens.basis('iasp91',truth['depth_km'],distances,azimuth,dt,n,1.)
    angles=bank.new_tensor([truth['strike'],truth['dip'],truth['rake']])
    moment=moment_sdr(*angles)
    scale=10**(1.5*truth['moment_magnitude']+9.1)/1e15
    clean=torch.einsum('stck,k->stc',bank,moment)*scale
    rng=np.random.default_rng(20261025)
    rms=float(clean.square().mean().sqrt())
    observed=clean.cpu().numpy()+rng.normal(size=clean.shape)*rms*.005
    # External data convention is N,E,Z-up; GRTM is N,E,D-down.
    observed[:,:,2]*=-1
    np.savez_compressed(folder/'observations.npz',waveforms=observed)
    travel=NeuralTravelTimes(device=args.device)
    arrival=travel.body_arrival(truth['depth_km'],distances,'P_first')
    rays=ray_direction(torch.tensor(azimuth,device=args.device),arrival['takeoff_angle'].double())
    q=torch.sigmoid(radiation(moment,rays)/.1).detach().cpu().numpy()
    stations=[dict(id=f'SYN{i+1:02d}',distance_km=float(d),azimuth_deg=float(a),probability_u=float(u))
        for i,(d,a,u) in enumerate(zip(distances,azimuth,q))]
    config=dict(event=dict(id='synthetic_GRTM_recovery',depth_km=10.),earth_model='iasp91',
        waveform=dict(file='observations.npz',quantity='displacement_m',coordinates='NEZ',dt_s=dt,t0_s=0),
        stations=stations,source_duration_s=1.,synthetic_samples=n,
        depth_grid_km=[8.,10.,12.],cap=dict(starts=48,steps=1000,surface_padding_s=8.))
    (folder/'event.json').write_text(json.dumps(config,indent=2)+'\n')
    truth.update(description='Synthetic physical GRTM waveforms; 0.5% global RMS Gaussian noise; neural-ray soft polarity targets, not CNN predictions.',
        moment_tensor_Nm=(moment*scale*1e15).cpu().tolist(),noise_global_rms_fraction=.005)
    (folder/'truth.json').write_text(json.dumps(truth,indent=2)+'\n')
    print(json.dumps(dict(event=str(folder/'event.json'),truth=truth)),flush=True)


if __name__=='__main__':main()
