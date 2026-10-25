"""One matched regional case: refine spacing and high-wavenumber cutoff separately."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from grtm.sensitivity import ForwardOperator
from seismic_joint.greens import GRTMGreens
from seismic_joint.mechanism import moment_sdr
from seismic_joint.cap import bandpass


def main():
    torch.set_num_threads(4)
    device='cuda:0';green=GRTMGreens(device=device)
    distances=np.array([50.,70.,90.,120.,150.,180.,220.,280.]);az=np.array([10.,55.,100.,145.,190.,235.,280.,325.])
    config,layers,stf=green.configuration('iasp91',10.,distances,.25,1024,1.)
    source=moment_sdr(torch.tensor(35.,dtype=torch.float64),torch.tensor(50.,dtype=torch.float64),torch.tensor(-70.,dtype=torch.float64)).numpy()*1e15
    with torch.cuda.device(device):
        op=ForwardOperator(config,backend='cuda',units='si',coordinates='ned',azimuth=az,field='displacement',stf=stf,parameters=(),device=0,threads=4)
        linear=op.linearize(layers,source)
        original=linear.value.copy();grid=linear.grid
        records=[]
        for kind in ['spacing','cutoff']:
            refined=dict(grid);refined['steps']=(np.asarray(grid['steps'])*2).tolist()
            if kind=='spacing':refined['period_length']=float(grid['period_length'])*2
            actual=op(layers,source,grid=refined)
            record=dict(refinement=kind)
            for label,band in [('body',(.05,.5)),('surface',(.02,.1))]:
                a=bandpass(torch.tensor(actual),.25,band);b=bandpass(torch.tensor(original),.25,band)
                record[label+'_relative_l2']=float(torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(b))
            records.append(record);print(json.dumps(record),flush=True)
    report=dict(case='iasp91 flat layers; 10 km depth; 50–280 km; 1024 x .25 s; all NED components',
        original_grid=dict(period_length=float(grid['period_length']),minimum_steps=int(np.min(grid['steps'])),maximum_steps=int(np.max(grid['steps']))),
        refinements=records,acceptance_relative_l2=.02)
    report['status']='pass' if all(r[k]<.02 for r in records for k in ['body_relative_l2','surface_relative_l2']) else 'needs_refinement'
    (ROOT/'results/green_quadrature.json').write_text(json.dumps(report,indent=2)+'\n')
    if report['status']!='pass':raise RuntimeError('Default quadrature not converged for the synthetic case')


if __name__=='__main__':main()
