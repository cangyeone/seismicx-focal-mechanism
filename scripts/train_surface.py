"""Generate GRTM fundamental-mode labels and train continuous period networks."""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
os.environ.setdefault('OMP_NUM_THREADS','2')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import grtm
import numpy as np
import torch
from seismic_joint.networks import SurfaceVelocityNet
from seismic_joint.schema import EARTH_MODELS


def generate(device):
    out = ROOT/'data/surface.npz'
    if out.exists():
        return dict(np.load(out))
    data = {key: [] for key in ['period','earth','wave','velocity','split','valid']}
    for mi,name in enumerate(EARTH_MODELS):
        definition = json.loads((ROOT/'assets/earth'/f'{name}.json').read_text())
        sw = grtm.SurfaceWave(np.asarray(definition['layers']),device=device,units='native',missing='nan')
        for wi,wave in enumerate(['rayleigh','love']):
            for si,(n,split) in enumerate([(512,'train'),(128,'validation'),(128,'test')]):
                rng = np.random.default_rng(202610250+mi*10+si)
                # Stratified log-periods; independently jittered across splits.
                periods = 2*np.exp((np.arange(n)+rng.uniform(.05,.95,n))/n*np.log(40))
                result = sw.dispersion(periods,wave=wave,mode=0,kind='group')
                array = lambda x:x.detach().cpu().numpy() if hasattr(x,'detach') else np.asarray(x)
                velocity = np.stack([array(result['phase_velocity']), array(result['velocity'])], axis=1)
                valid = array(result['valid']) & np.isfinite(velocity).all(1) & (velocity>0).all(1)
                data['period'].append(periods);data['earth'].append(np.full(n,mi))
                data['wave'].append(np.full(n,wi));data['velocity'].append(velocity)
                data['split'].append(np.full(n,si));data['valid'].append(valid)
                print(json.dumps(dict(stage='surface_teacher',model=name,wave=wave,split=split,
                    valid=int(valid.sum()),total=n)),flush=True)
    data = {key:np.concatenate(value) for key,value in data.items()}
    np.savez_compressed(out,**data)
    return data


def main():
    p=argparse.ArgumentParser();p.add_argument('--device',default='cuda:1');p.add_argument('--steps',type=int,default=6000)
    args=p.parse_args();torch.set_num_threads(2);torch.manual_seed(20261025)
    torch.backends.cuda.matmul.allow_tf32=False
    data=generate(args.device)
    device=torch.device(args.device)
    tt=lambda x:torch.as_tensor(x,device=device)
    period=tt(data['period']).float();earth=tt(data['earth']).long();wave=tt(data['wave']).long()
    y=tt(data['velocity']).float();valid=tt(data['valid'])
    train=torch.where((tt(data['split'])==0)&valid)[0]
    validation=torch.where((tt(data['split'])==1)&valid)[0]
    test=torch.where((tt(data['split'])==2)&valid)[0]
    model=SurfaceVelocityNet().to(device)
    model.target_mean.copy_(y[train].mean(0));model.target_std.copy_(y[train].std(0))
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-7)
    best=float('inf');history=[];start=time.time()
    for step in range(args.steps):
        ids=train[torch.randint(len(train),(512,),device=device)]
        prediction=model(period[ids],earth[ids],wave[ids])
        loss=((prediction-y[ids])/model.target_std).square().mean()
        optimizer.zero_grad();loss.backward();optimizer.step()
        lr=1e-3*(.02+.98*.5*(1+math.cos(math.pi*step/args.steps)))
        for group in optimizer.param_groups:group['lr']=lr
        if step%250==0 or step+1==args.steps:
            with torch.no_grad():
                error=(model(period[validation],earth[validation],wave[validation])-y[validation]).abs()
            record=dict(step=step+1,validation_mae_km_s=error.mean(0).tolist(),
                validation_max_km_s=error.amax(0).tolist(),seconds=time.time()-start)
            history.append(record);print(json.dumps(record),flush=True)
            score=error.mean().item()
            if score<best:
                best=score;torch.save(dict(config=model.config,state_dict=model.state_dict(),
                    metadata=dict(earth_models=EARTH_MODELS,waves=['rayleigh','love'],mode=0,
                        output_order=['phase_velocity_km_s','group_velocity_km_s'],period_range_s=[2,80],
                        teacher='GRTM elastic flat layered solver',grtm_version=grtm.__version__,
                        fixed_earth_models=True,train=len(train),validation=len(validation))),ROOT/'assets/surface_velocity.pt')
    state=torch.load(ROOT/'assets/surface_velocity.pt',map_location=device,weights_only=True)
    model.load_state_dict(state['state_dict'])
    report=dict(status='complete',train=int(len(train)),validation=int(len(validation)),test=int(len(test)),
        invalid_labels=int((~valid).sum()),history=history,holdout=[])
    with torch.no_grad():
        for mi,name in enumerate(EARTH_MODELS):
            for wi,wavename in enumerate(['rayleigh','love']):
                idx=test[(earth[test]==mi)&(wave[test]==wi)]
                error=(model(period[idx],earth[idx],wave[idx])-y[idx]).abs()
                report['holdout'].append(dict(model=name,wave=wavename,n=len(idx),
                    mae_km_s=error.mean(0).tolist(),p99_km_s=torch.quantile(error,.99,dim=0).tolist(),
                    maximum_km_s=error.amax(0).tolist()))
    (ROOT/'results/surface_training.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(stage='surface_done',holdout=report['holdout'])),flush=True)


if __name__=='__main__':main()
