"""Audit the complete CAP+polarity loss derivative, not only component kernels."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from seismic_joint.api import read_event
from seismic_joint.cap import CAPObjective,CAPSettings
from seismic_joint.greens import GRTMGreens
from seismic_joint.mechanism import ray_direction
from seismic_joint.travel import NeuralTravelTimes


def main():
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    device='cuda:0';data=read_event(ROOT/'examples/synthetic/event.json',device)
    travel=NeuralTravelTimes(device=device)
    arrival=travel.body_arrival(10.,data['distances'],'P_first')
    rays=ray_direction(data['azimuth'],arrival['takeoff_angle'].double())
    bank=GRTMGreens(device=device).basis('iasp91',10.,data['distances'].cpu().numpy(),data['azimuth'].cpu().numpy(),.25,1024,1.)
    objective=CAPObjective(data['observed'],bank,data['distances'],data['azimuth'],.25,data['t0'],travel,10.,
        rays,data['probability_u'],data['polarity_valid'],CAPSettings(surface_padding_s=8.))
    angles=bank.new_tensor([[33.,52.,-67.]],requires_grad=True)
    magnitude=bank.new_tensor([.1],requires_grad=True)
    shifts=torch.full((1,8,3),.12,dtype=bank.dtype,device=device,requires_grad=True)
    loss,_=objective(angles,magnitude,shifts)
    derivatives=torch.autograd.grad(loss.sum(),(angles,magnitude,shifts))
    records=[]
    for name,parameter,gradient,index,epsilon in [
        ('strike',angles,derivatives[0],(0,0),1e-3),('dip',angles,derivatives[0],(0,1),1e-3),
        ('rake',angles,derivatives[0],(0,2),1e-3),('log_amplitude',magnitude,derivatives[1],(0,),1e-5),
        ('Pnl_shift',shifts,derivatives[2],(0,2,0),1e-6),('Rayleigh_shift',shifts,derivatives[2],(0,2,1),1e-6),
        ('Love_shift',shifts,derivatives[2],(0,2,2),1e-6)]:
        with torch.no_grad():
            parameter[index]+=epsilon;plus=objective(angles,magnitude,shifts)[0].sum()
            parameter[index]-=2*epsilon;minus=objective(angles,magnitude,shifts)[0].sum()
            parameter[index]+=epsilon
        actual=float(gradient[index]);expected=float((plus-minus)/(2*epsilon))
        relative=abs(actual-expected)/max(abs(expected),abs(actual),1e-8)
        assert relative<2e-3,(name,relative,actual,expected)
        records.append(dict(parameter=name,autograd=actual,central_difference=expected,relative_error=relative))
    report=dict(status='pass',complete_CAP_plus_polarity_objective=True,records=records,
        geometry='fixed source depth and station locations; source-depth interpolation not claimed exact')
    (ROOT/'results/joint_gradient_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
