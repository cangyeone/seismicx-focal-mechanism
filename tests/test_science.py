"""Numerical contracts that can invalidate a joint mechanism solution if broken."""
import unittest
import numpy as np
import torch
torch.set_num_threads(2)
from seismic_joint.api import geographic_geometry,predict_first_motion,window_raw_z
from seismic_joint.cap import sample_components,bandpass,to_rtd
from seismic_joint.greens import interpolate_depth
from seismic_joint.mechanism import moment_sdr,moment_matrix,nodal_planes,ray_direction,radiation,fit_polarity,tensor_angle_deg
from seismic_joint.travel import NeuralTravelTimes


class ScienceTests(unittest.TestCase):
    def test_packaged_cnn_input_gradient(self):
        wave=torch.randn(1,1024,dtype=torch.float64,requires_grad=True)
        result=predict_first_motion(wave)
        gradient=torch.autograd.grad(result['probability_u'].sum(),wave)[0]
        self.assertTrue(bool(torch.isfinite(gradient).all()))
        self.assertGreater(float(gradient.abs().sum()),0)

    def test_continuous_crop_keeps_arrival_gradient(self):
        wave=torch.arange(4096,dtype=torch.float64)
        arrival=torch.tensor(15.005,dtype=torch.float64,requires_grad=True)
        crop=window_raw_z(wave,arrival)
        self.assertEqual(crop.numel(),1024)
        self.assertAlmostEqual(float(crop[512]),1500.5)
        self.assertAlmostEqual(float(torch.autograd.grad(crop.sum(),arrival)[0]),1024/.01)

    def test_ned_strike_slip_sign_and_scalar_moment(self):
        moment=moment_sdr(torch.tensor(0.,dtype=torch.float64),torch.tensor(90.,dtype=torch.float64),torch.tensor(0.,dtype=torch.float64))
        torch.testing.assert_close(moment,torch.tensor([0.,0.,0.,1.,0.,0.],dtype=torch.float64),atol=1e-12,rtol=0)
        rays=ray_direction(torch.tensor([45.,135.]),torch.tensor([90.,90.]))
        values=radiation(moment.float(),rays)
        self.assertGreater(values[0],.99);self.assertLess(values[1],-.99)
        self.assertAlmostEqual(float(torch.trace(moment_matrix(moment))),0)
        self.assertAlmostEqual(float(torch.linalg.vector_norm(moment_matrix(moment))),np.sqrt(2))

    def test_auxiliary_planes_reproduce_tensor(self):
        target=moment_sdr(torch.tensor(35.,dtype=torch.float64),torch.tensor(50.,dtype=torch.float64),torch.tensor(-70.,dtype=torch.float64))
        for plane in nodal_planes(target.numpy()):
            value=moment_sdr(*[torch.tensor(plane[k],dtype=torch.float64) for k in ['strike','dip','rake']])
            torch.testing.assert_close(value,target,atol=1e-12,rtol=1e-12)

    def test_first_p_gradient_and_spherical_angle(self):
        model=NeuralTravelTimes().first_p.double()
        x=torch.tensor([[10.,50.],[12.,100.],[75.,600.]],dtype=torch.float64,requires_grad=True)
        time=model(x);gradient=torch.autograd.grad(time.sum(),x,create_graph=True)[0]
        for column in range(2):
            delta=torch.zeros_like(x);delta[:,column]=1e-4
            difference=(model(x+delta)-model(x-delta))/(2e-4)
            torch.testing.assert_close(gradient[:,column],difference,atol=1e-8,rtol=1e-6)
        # Differentiating ray directions requires second derivatives of time.
        angle=torch.atan2(gradient[:,1]*6371/(6371-x[:,0]),-gradient[:,0])
        hessian_contraction=torch.autograd.grad(angle.sum(),x)[0]
        self.assertTrue(bool(torch.isfinite(hessian_contraction).all()))
        self.assertGreater(float(hessian_contraction.abs().sum()),0)

    def test_sampling_shift_derivative_and_last_sample(self):
        values=torch.arange(10,dtype=torch.float64)[None,None,:,None].expand(1,1,10,3)
        shift=torch.tensor(.2,dtype=torch.float64,requires_grad=True)
        times=torch.tensor([[[[1.,2.,3.]]]],dtype=torch.float64)+shift
        samples,valid=sample_components(values,times,torch.tensor([0]),1.,torch.tensor([0.]))
        self.assertTrue(bool(valid.all()))
        self.assertAlmostEqual(float(torch.autograd.grad(samples.sum(),shift)[0]),3)
        edge,valid=sample_components(values,torch.tensor([[[[9.]]]]),torch.tensor([0]),1.,torch.tensor([0.]))
        self.assertEqual(float(edge),9);self.assertTrue(bool(valid.all()))

    def test_filter_and_rotation_preserve_relative_amplitude(self):
        generator=torch.Generator().manual_seed(2)
        wave=torch.randn((2,512,3),dtype=torch.float64,generator=generator)
        a=bandpass(wave,.25,(.05,.5));b=bandpass(wave*3,.25,(.05,.5))
        torch.testing.assert_close(b,a*3,atol=1e-12,rtol=1e-12)
        rotated=to_rtd(wave,torch.tensor([0.,90.]))
        torch.testing.assert_close(rotated.square().sum(),wave.square().sum(),atol=1e-5,rtol=1e-8)

    def test_geometry_and_depth_interpolation_gradients(self):
        zero=torch.tensor(0.,dtype=torch.float64)
        distance,azimuth=geographic_geometry(zero,zero,zero,torch.tensor(1.,dtype=torch.float64))
        self.assertAlmostEqual(float(distance),6371*np.pi/180,places=10)
        self.assertAlmostEqual(float(azimuth),90,places=10)
        depth=torch.tensor(6.,requires_grad=True)
        bank=interpolate_depth(depth,torch.tensor([5.,10.,15.]),torch.tensor([[2.],[12.],[22.]]))
        self.assertAlmostEqual(float(torch.autograd.grad(bank.sum(),depth)[0]),2)

    def test_dense_synthetic_polarity_recovery(self):
        rng=np.random.default_rng(45)
        rays=ray_direction(torch.tensor(rng.uniform(0,360,256)),torch.tensor(rng.uniform(20,160,256)))
        moment=moment_sdr(torch.tensor(35.,dtype=torch.float64),torch.tensor(50.,dtype=torch.float64),torch.tensor(-70.,dtype=torch.float64))
        q=torch.sigmoid(radiation(moment,rays)/.1)
        result=fit_polarity(rays,q,starts=32,steps=700)
        self.assertLess(tensor_angle_deg(result['moment_tensor_normalized'],moment),.2)


if __name__=='__main__':unittest.main()
