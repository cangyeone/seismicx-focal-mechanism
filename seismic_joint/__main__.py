import argparse
import json
from pathlib import Path
import torch
from .api import invert_event
from .travel import NeuralTravelTimes


def clean(value):
    if isinstance(value,torch.Tensor):return clean(value.detach().cpu().tolist())
    if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
    if isinstance(value,list):return [clean(v) for v in value]
    if isinstance(value,float) and not __import__('math').isfinite(value):return None
    return value


def main():
    p=argparse.ArgumentParser(prog='seismic-joint',description='DNN travel times + first motions + differentiable CAP')
    commands=p.add_subparsers(dest='command',required=True)
    inv=commands.add_parser('invert');inv.add_argument('event');inv.add_argument('--mode',choices=['polarity','joint'],default='joint')
    inv.add_argument('--device',default='cpu');inv.add_argument('--output',default='results/event');inv.add_argument('--assets')
    arr=commands.add_parser('arrivals');arr.add_argument('--depth',type=float,required=True)
    arr.add_argument('--distance',type=float,nargs='+',required=True);arr.add_argument('--phases',nargs='+',default=['P_first','S_first'])
    arr.add_argument('--earth-model',default='iasp91');arr.add_argument('--device',default='cpu');arr.add_argument('--assets')
    arr.add_argument('--period',type=float,nargs='+',default=[10,20,40])
    args=p.parse_args()
    if args.command=='invert':
        result=invert_event(args.event,args.output,args.mode,args.device,args.assets)
    else:
        travel=NeuralTravelTimes(args.assets,args.device,args.earth_model)
        result={phase:travel.body_arrival(args.depth,args.distance,phase) for phase in args.phases}
        if travel.surface is not None:
            distance=torch.tensor(args.distance,device=args.device)[:,None]
            period=torch.tensor(args.period,device=args.device)[None]
            for wave in ['rayleigh','love']:result[wave]=travel.surface_arrival(distance,period,wave)
    print(json.dumps(clean(result),ensure_ascii=False,indent=2,allow_nan=False))


if __name__=='__main__':main()
