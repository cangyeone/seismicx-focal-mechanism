"""Phase-balanced, independent expert refinement from shared DNN checkpoint."""
import argparse
import json
import math
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from scripts.train_body import load,evaluate
from seismic_joint.networks import BodyTravelNet,ExpertBodyTravelNet
from seismic_joint.schema import BODY_PHASES,EARTH_MODELS


def main():
    p=argparse.ArgumentParser();p.add_argument('--device',default='cuda:1');p.add_argument('--steps',type=int,default=6000)
    p.add_argument('--batch-per-phase',type=int,default=128);args=p.parse_args()
    torch.set_num_threads(2);torch.manual_seed(20261026);torch.backends.cuda.matmul.allow_tf32=False
    device=torch.device(args.device)
    state=torch.load(ROOT/'assets/body_travel.pt',map_location=device,weights_only=True)
    train=load('train',device,state['metadata']);valid=load('validation',device,state['metadata'])
    shared=BodyTravelNet(**state['config']).to(device);shared.load_state_dict(state['state_dict'])
    model=ExpertBodyTravelNet(**state['config']).to(device);model.initialize_shared(shared)
    groups=[]
    for earth in range(3):
        for phase in range(len(BODY_PHASES)):
            ids=torch.where((train['earth']==earth)&(train['phase']==phase))[0]
            if not len(ids):raise RuntimeError(f'No training labels for {earth}:{phase}')
            groups.append(ids)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-8)
    best=float('inf');history=[];start=time.time()
    # Confirm expert warm-start algebra before changing any parameters.
    testids=torch.cat([g[:3] for g in groups])
    torch.testing.assert_close(model(train['x'][testids],train['earth'][testids],train['phase'][testids]),
        shared(train['x'][testids],train['earth'][testids],train['phase'][testids]),atol=3e-4,rtol=1e-5)
    for step in range(args.steps):
        ids=torch.stack([g[torch.randint(len(g),(args.batch_per_phase,),device=device)] for g in groups])
        x=train['x'][ids].detach().requires_grad_(True)
        prediction=model.forward_grouped(x)
        gradient=torch.autograd.grad(prediction.sum(),x,create_graph=True)[0]
        dt=(prediction-train['time'][ids])/.05
        lt=torch.nn.functional.smooth_l1_loss(dt,torch.zeros_like(dt),beta=1)
        dg=(gradient-train['gradient'][ids])*train['velocity'][ids][...,None]
        dg=dg*torch.stack([torch.ones_like(x[:,:,0]),6371/(6371-x[:,:,0])],dim=-1)
        good=train['gradient_good'][ids]
        loss=lt+1500*dg[good].square().mean()
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10);optimizer.step()
        scale=.03+.97*.5*(1+math.cos(math.pi*step/args.steps))
        for group in optimizer.param_groups:group['lr']=1e-4*scale
        if step%500==0 or step+1==args.steps:
            te,ae,vi=evaluate(model,valid)
            score=float(torch.stack([te[valid['phase'][vi]==j].mean()+.05*ae[valid['phase'][vi]==j].mean()
                for j in range(len(BODY_PHASES))]).mean())
            record=dict(step=step+1,loss=float(loss.detach()),validation_time_mae_s=float(te.mean()),
                validation_angle_mae_deg=float(ae.mean()),validation_time_p99_s=float(torch.quantile(te,.99)),
                score=score,seconds=time.time()-start)
            history.append(record);print(json.dumps(record),flush=True)
            if score<best:
                best=score
                payload=dict(state);payload['config']=model.config;payload['state_dict']=model.state_dict()
                payload['metadata']=dict(state['metadata'],expert_refinement_step=step+1,validation=record,
                    independent_expert_groups=39,phase_balanced_refinement=True)
                temp=ROOT/'assets/body_travel_expert.tmp.pt';torch.save(payload,temp);temp.replace(ROOT/'assets/body_travel_expert.pt')
            (ROOT/'results/body_expert_progress.json').write_text(json.dumps(record,indent=2)+'\n')
    state=torch.load(ROOT/'assets/body_travel_expert.pt',map_location=device,weights_only=True)
    model.load_state_dict(state['state_dict']);test=load('test',device,state['metadata'])
    te,ae,vi=evaluate(model,test,all_rows=True)
    reports=[]
    for earth,name in enumerate(EARTH_MODELS):
        for phase,label in enumerate(BODY_PHASES):
            use=(test['earth'][vi]==earth)&(test['phase'][vi]==phase)
            record=dict(model=name,phase=label,n=int(use.sum()))
            if use.any():record.update(time_mae_s=float(te[use].mean()),time_p95_s=float(torch.quantile(te[use],.95)),
                time_max_s=float(te[use].amax()),angle_mae_deg=float(ae[use].mean()),
                angle_p95_deg=float(torch.quantile(ae[use],.95)),angle_max_deg=float(ae[use].amax()))
            reports.append(record)
    report=dict(status='complete',history=history,holdout=reports,selected_step=state['metadata']['expert_refinement_step'])
    (ROOT/'results/body_expert_training.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(stage='body_experts_done',holdout=reports)),flush=True)


if __name__=='__main__':main()
