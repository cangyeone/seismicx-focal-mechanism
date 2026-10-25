"""Physical source gradients, DNN angle derivatives, and TauP-free runtime audit."""
import importlib.abc
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from seismic_joint.greens import GRTMGreens
from seismic_joint.mechanism import moment_sdr
from seismic_joint.travel import NeuralTravelTimes


class NoObsPy(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname=='obspy' or fullname.startswith('obspy.'):
            raise RuntimeError('Runtime must not import ObsPy or TauP: '+fullname)


def main():
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    sys.meta_path.insert(0,NoObsPy())
    travel=NeuralTravelTimes(device='cuda:1')
    depth=torch.tensor([10.,15.,12.],device='cuda:1',requires_grad=True)
    distance=torch.tensor([30.,70.,150.],device='cuda:1',requires_grad=True)
    phase_checks={}
    for phase in ['P_first','S_first','Pg_regional','Sg_regional','pP','sP']:
        result=travel.body_arrival(depth,distance,phase,create_graph=True)
        use=result['gradient_valid']
        if use.any():
            angle=result['takeoff_angle'][use]
            derivative=torch.autograd.grad(angle.sum(),(depth,distance),retain_graph=True)
            assert all(torch.isfinite(v).all() for v in derivative)
        phase_checks[phase]=dict(valid=int(use.sum()),total=len(use),second_derivatives_finite=True)
    period=torch.tensor([10.,20.,40.],device='cuda:1',requires_grad=True)
    group=travel.surface_arrival(distance,period)
    d=torch.autograd.grad(group['time'].sum(),(distance,period))
    torch.testing.assert_close(d[0],1/group['group_velocity'],atol=1e-6,rtol=1e-5)
    assert torch.isfinite(d[1]).all()
    greens=GRTMGreens(device='cuda:1')
    distances=np.array([50.,70.,90.,120.,150.,180.,220.,280.]);azimuth=np.array([10.,55.,100.,145.,190.,235.,280.,325.])
    bank=greens.basis('iasp91',10.,distances,azimuth,.25,1024,1.)
    op,layers=greens.differentiable_forward('iasp91',10.,distances,azimuth,.25,1024,1.)
    angle=torch.tensor([35.,50.,-70.],dtype=torch.float64,device='cuda:1',requires_grad=True)
    moment=moment_sdr(*angle)
    native=op(layers,moment*1e15)
    analytic=torch.einsum('stck,k->stc',bank,moment)
    waveform_error=float(torch.linalg.vector_norm(native-analytic)/torch.linalg.vector_norm(analytic))
    assert waveform_error<1e-5,waveform_error
    generator=torch.Generator(device='cuda:1').manual_seed(15)
    cotangent=torch.randn(native.shape,device='cuda:1',dtype=torch.float64,generator=generator)
    native_gradient=torch.autograd.grad((native*cotangent).sum(),angle,retain_graph=True)[0]
    bank_gradient=torch.autograd.grad((analytic*cotangent).sum(),angle)[0]
    gradient_error=float(torch.linalg.vector_norm(native_gradient-bank_gradient)/torch.linalg.vector_norm(bank_gradient))
    assert gradient_error<1e-4,gradient_error
    report=dict(status='pass',runtime_taup_and_obspy_blocked=True,phase_checks=phase_checks,
        neural_surface_distance_derivative_matches_inverse_group_velocity=True,
        native_vs_linear_bank_waveform_relative_error=waveform_error,
        native_vs_linear_bank_source_angle_gradient_relative_error=gradient_error,
        geometry_derivative_scope='DNN depth/distance second derivatives verified; native Green source-depth geometry remains fixed')
    (ROOT/'results/pipeline_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
