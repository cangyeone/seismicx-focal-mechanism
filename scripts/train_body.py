"""Train time + input-gradient supervision; independent phase existence classifier."""
import argparse
import json
import math
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from seismic_joint.networks import BodyTravelNet,PhaseAvailabilityNet
from seismic_joint.schema import BODY_PHASES,EARTH_MODELS


def load(split,device,metadata):
    flat={key:[] for key in ['x','earth','phase','time','gradient','velocity','angle','gradient_good']}
    all_x,all_earth,available=[],[],[]
    for mi,name in enumerate(EARTH_MODELS):
        a=dict(np.load(ROOT/'data'/f'{name}-{split}.npz'))
        x=a['inputs'];y=a['labels'];good=np.isfinite(y[:,:,0])
        rows,phases=np.where(good)
        on_interface=np.min(abs(x[:,0,None]-np.array(metadata['models'][name]['interfaces'])[None]),axis=1)<.002
        flat['x'].append(x[rows]);flat['earth'].append(np.full(len(rows),mi))
        flat['phase'].append(phases);flat['time'].append(y[rows,phases,0])
        flat['gradient'].append(y[rows,phases,1:3]);flat['angle'].append(y[rows,phases,3])
        flat['velocity'].append(y[rows,phases,4])
        flat['gradient_good'].append(~on_interface[rows] & (a['branch_gap'][rows,phases]>.05))
        all_x.append(x);all_earth.append(np.full(len(x),mi));available.append(good)
    out={key:torch.as_tensor(np.concatenate(value),device=device) for key,value in flat.items()}
    for key in ['x','time','gradient','velocity','angle']:out[key]=out[key].float()
    out['all_x']=torch.as_tensor(np.concatenate(all_x),device=device,dtype=torch.float32)
    out['all_earth']=torch.as_tensor(np.concatenate(all_earth),device=device,dtype=torch.long)
    out['available']=torch.as_tensor(np.concatenate(available),device=device,dtype=torch.float32)
    return out


def evaluate(model,data,all_rows=False):
    size=len(data['x']);ids=torch.arange(size,device=data['x'].device)
    if not all_rows:
        # Fixed validation cohort, distributed over every earth/phase group.
        groups=[ids[(data['earth']==m)&(data['phase']==p)][:256]
            for m in range(3) for p in range(len(BODY_PHASES))]
        ids=torch.cat(groups)
    times,angles,positions=[],[],[]
    for first in range(0,len(ids),4096):
        ix=ids[first:first+4096];x=data['x'][ix].detach().requires_grad_(True)
        prediction=model(x,data['earth'][ix],data['phase'][ix])
        gradient=torch.autograd.grad(prediction.sum(),x)[0]
        angle=torch.rad2deg(torch.atan2(gradient[:,1].abs()*6371/(6371-x[:,0]),-gradient[:,0]))
        times.append((prediction-data['time'][ix]).abs().detach())
        angles.append((angle-data['angle'][ix]).abs().detach());positions.append(ix)
    return torch.cat(times),torch.cat(angles),torch.cat(positions)


def main():
    p=argparse.ArgumentParser();p.add_argument('--device',default='cuda:0');p.add_argument('--steps',type=int,default=18000)
    p.add_argument('--batch',type=int,default=4096);p.add_argument('--wait',action='store_true');p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    while not (ROOT/'data/body_manifest.json').exists():
        if not args.wait:raise FileNotFoundError('Run generate_body_labels.py first')
        time.sleep(15)
    torch.set_num_threads(2);torch.manual_seed(20261025);torch.backends.cuda.matmul.allow_tf32=False
    device=torch.device(args.device);metadata=json.loads((ROOT/'data/body_manifest.json').read_text())
    train=load('train',device,metadata);valid=load('validation',device,metadata)
    model=BodyTravelNet().to(device);availability=PhaseAvailabilityNet().to(device)
    if args.resume and (ROOT/'assets/body_travel.pt').exists():
        state=torch.load(ROOT/'assets/body_travel.pt',map_location=device,weights_only=True)
        model.load_state_dict(state['state_dict'])
        availability.load_state_dict(state['availability_state_dict'])
    optimizer=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=1e-7)
    classifier_optimizer=torch.optim.AdamW(availability.parameters(),lr=1e-3,weight_decay=1e-6)
    best=float('inf');history=[];start=time.time()
    for step in range(args.steps):
        ids=torch.randint(len(train['x']),(args.batch,),device=device)
        x=train['x'][ids].detach().requires_grad_(True)
        prediction=model(x,train['earth'][ids],train['phase'][ids])
        gradient=torch.autograd.grad(prediction.sum(),x,create_graph=True)[0]
        time_loss=torch.nn.functional.smooth_l1_loss((prediction-train['time'][ids])/.1,
            torch.zeros_like(prediction),beta=1)
        dg=(gradient-train['gradient'][ids])*train['velocity'][ids,None]
        dg=dg*torch.stack([torch.ones_like(x[:,0]),6371/(6371-x[:,0])],axis=1)
        mask=train['gradient_good'][ids]
        gradient_loss=dg[mask].square().mean()
        loss=time_loss+1000*gradient_loss
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10);optimizer.step()
        ci=torch.randint(len(train['all_x']),(args.batch,),device=device)
        logits=availability(train['all_x'][ci],train['all_earth'][ci])
        closs=torch.nn.functional.binary_cross_entropy_with_logits(logits,train['available'][ci])
        classifier_optimizer.zero_grad();closs.backward();classifier_optimizer.step()
        scale=.02+.98*.5*(1+math.cos(math.pi*step/args.steps))
        for group in optimizer.param_groups:group['lr']=5e-4*scale
        for group in classifier_optimizer.param_groups:group['lr']=1e-3*scale
        if step%500==0 or step+1==args.steps:
            te,ae,vi=evaluate(model,valid)
            # Equal phase contribution prevents common phases hiding poor rare phases.
            scores=[te[valid['phase'][vi]==j].mean()+.05*ae[valid['phase'][vi]==j].mean()
                for j in range(len(BODY_PHASES)) if (valid['phase'][vi]==j).any()]
            score=float(torch.stack(scores).mean())
            record=dict(step=step+1,loss=float(loss.detach()),validation_time_mae_s=float(te.mean()),
                validation_time_p99_s=float(torch.quantile(te,.99)),validation_angle_mae_deg=float(ae.mean()),
                validation_angle_p99_deg=float(torch.quantile(ae,.99)),score=score,seconds=time.time()-start)
            history.append(record);print(json.dumps(record),flush=True)
            if score<best:
                best=score;torch.save(dict(config=model.config,state_dict=model.state_dict(),
                    availability_config=availability.config,availability_state_dict=availability.state_dict(),
                    metadata=dict(metadata,selected_on='internal validation equal-phase time+angle score',
                        training_step=step+1,validation=record,gradient_supervision=True,
                        angle_convention='0 down,90 horizontal,180 up',
                        first_P_uses_existing_checkpoint=True)),ROOT/'assets/body_travel.pt')
            (ROOT/'results/body_training_progress.json').write_text(json.dumps(record,indent=2)+'\n')
    state=torch.load(ROOT/'assets/body_travel.pt',map_location=device,weights_only=True)
    model.load_state_dict(state['state_dict']);availability.load_state_dict(state['availability_state_dict'])
    test=load('test',device,metadata);te,ae,vi=evaluate(model,test,all_rows=True)
    with torch.no_grad():
        probabilities=availability(test['all_x'],test['all_earth']).sigmoid()
        prediction=probabilities>.99;truth=test['available'].bool()
    reports=[]
    for mi,name in enumerate(EARTH_MODELS):
        for pi,phase in enumerate(BODY_PHASES):
            mask=(test['earth'][vi]==mi)&(test['phase'][vi]==pi)
            group=test['all_earth']==mi;chosen=prediction[group,pi];actual=truth[group,pi]
            record=dict(model=name,phase=phase,n=int(mask.sum()),
                availability_precision=float((chosen&actual).sum()/chosen.sum().clamp_min(1)),
                availability_recall=float((chosen&actual).sum()/actual.sum().clamp_min(1)))
            if mask.any():
                record.update(time_mae_s=float(te[mask].mean()),time_p95_s=float(torch.quantile(te[mask],.95)),
                    time_max_s=float(te[mask].amax()),angle_mae_deg=float(ae[mask].mean()),
                    angle_p95_deg=float(torch.quantile(ae[mask],.95)),angle_max_deg=float(ae[mask].amax()))
            reports.append(record)
    report=dict(status='complete',history=history,holdout=reports,
        selected_training_step=state['metadata']['training_step'],
        all_holdout_rays_included_in_error=True,first_P_unchanged=True)
    (ROOT/'results/body_training.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(stage='body_done',holdout=reports)),flush=True)


if __name__=='__main__':main()
