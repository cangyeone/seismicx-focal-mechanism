"""NED double couples and first-motion likelihood, fully Torch differentiable."""
import math
import numpy as np
import torch


def moment_matrix(vector):
    nn,ee,dd,ne,nd,ed=vector.unbind(-1)
    return torch.stack([nn,ne,nd,ne,ee,ed,nd,ed,dd],dim=-1).reshape(vector.shape[:-1]+(3,3))


def moment_sdr(strike,dip,rake,scalar_moment=1.):
    # Use the same supported convention as the GRTM waveform engine.
    from grtm.torch import strike_dip_rake
    return strike_dip_rake(strike,dip,rake,scalar_moment)


def ray_direction(azimuth_deg,takeoff_deg):
    angle=torch.deg2rad(takeoff_deg);azimuth=torch.deg2rad(azimuth_deg)
    return torch.stack([angle.sin()*azimuth.cos(),angle.sin()*azimuth.sin(),angle.cos()],dim=-1)


def radiation(moment,rays):
    return torch.einsum('si,...ij,sj->...s',rays,moment_matrix(moment),rays)


def polarity_loss(moment,rays,probability_u,valid=None,softness=.1):
    """Soft U/D targets; equal-class CNN scores are not assumed calibrated."""
    amplitude=radiation(moment,rays)
    # Normalize by M0 so polarity evidence cannot be evaded by changing magnitude.
    scalar=torch.linalg.vector_norm(moment_matrix(moment),dim=(-2,-1))/math.sqrt(2)
    logits=amplitude/scalar.clamp_min(1e-20)[...,None]/softness
    q=probability_u.expand_as(logits)
    weight=(2*probability_u-1).abs()
    if valid is not None:weight=weight*valid.to(weight.dtype)
    losses=torch.nn.functional.binary_cross_entropy_with_logits(logits,q,reduction='none')
    if weight.sum()<=0:raise ValueError('No informative valid first-motion observations')
    return (losses*weight).sum(-1)/weight.sum()


def nodal_planes(moment):
    """Return equivalent planes from eigenvectors, without ObsPy/TauP."""
    m=np.asarray(moment,dtype=float)
    matrix=np.array([[m[0],m[3],m[4]],[m[3],m[1],m[5]],[m[4],m[5],m[2]]])
    _,v=np.linalg.eigh(matrix)
    normal=(v[:,2]+v[:,0])/np.sqrt(2);slip=(v[:,2]-v[:,0])/np.sqrt(2)
    def plane(n,s):
        if n[2]>0:n,s=-n,-s
        dip=np.arccos(np.clip(-n[2],-1,1))
        strike=np.arctan2(-n[0],n[1])
        along=np.array([np.cos(strike),np.sin(strike),0.])
        up=np.array([np.cos(dip)*np.sin(strike),-np.cos(dip)*np.cos(strike),-np.sin(dip)])
        rake=np.arctan2(np.dot(s,up),np.dot(s,along))
        return dict(strike=float(np.rad2deg(strike)%360),dip=float(np.rad2deg(dip)),rake=float(np.rad2deg(rake)))
    return [plane(normal,slip),plane(slip,normal)]


def tensor_angle_deg(a,b):
    """Angle between normalized moment tensors; auxiliary planes give zero."""
    aa=moment_matrix(torch.as_tensor(a,dtype=torch.float64));bb=moment_matrix(torch.as_tensor(b,dtype=torch.float64))
    cosine=(aa*bb).sum()/torch.sqrt(aa.square().sum()*bb.square().sum())
    return float(torch.rad2deg(torch.acos(cosine.clamp(-1,1))))


def fit_polarity(rays,probability_u,valid=None,starts=128,steps=1000,seed=20261025):
    # This fitter optimizes mechanisms at fixed geometry/observations. The public
    # polarity_loss remains differentiable in rays and probability targets.
    rays=rays.detach();probability_u=probability_u.detach()
    device=rays.device;dtype=rays.dtype
    generator=torch.Generator(device=device).manual_seed(seed)
    angles=torch.nn.Parameter(torch.rand((starts,3),device=device,dtype=dtype,generator=generator))
    with torch.no_grad():
        angles[:,0]*=360;angles[:,1]=5+80*angles[:,1];angles[:,2]=angles[:,2]*360-180
    optimizer=torch.optim.Adam([angles],lr=.8)
    for step in range(steps):
        moment=moment_sdr(angles[:,0],angles[:,1],angles[:,2])
        loss=polarity_loss(moment,rays,probability_u,valid)
        optimizer.zero_grad();loss.sum().backward();optimizer.step()
        with torch.no_grad():angles[:,1].clamp_(.05,89.95)
    with torch.no_grad():
        moment=moment_sdr(angles[:,0],angles[:,1],angles[:,2])
        loss=polarity_loss(moment,rays,probability_u,valid)
        index=int(loss.argmin());best=moment[index].cpu().numpy()
        predicted=radiation(moment[index],rays)>0
        use=(torch.ones_like(predicted) if valid is None else valid)
        mismatch=int(((predicted!=(probability_u>.5))&use).sum())
    return dict(mode='polarity',moment_tensor_normalized=best.tolist(),nodal_planes=nodal_planes(best),
        objective=float(loss[index]),stations_used=int(use.sum()),polarity_mismatches=mismatch,
        multistart=starts,relative_losses=sorted((loss-loss.min()).cpu().tolist())[:16],
        uncertainty_note='Multiple mechanisms may fit sparse first motions; objective spread is not a calibrated confidence interval.')
