"""Offline conversion of standard radial models to fixed flat GRTM layer stacks."""
import json
from pathlib import Path
import numpy as np
from obspy.taup import TauPyModel

ROOT = Path(__file__).resolve().parents[1]
folder = ROOT/'assets/earth'
folder.mkdir(parents=True, exist_ok=True)
for name in ['iasp91','ak135','prem']:
    model = TauPyModel(name)
    vmod = model.model.s_mod.v_mod
    tops = vmod.layers['top_depth']
    grid = sorted(set([*np.arange(0., 60., 5), *np.arange(60., 410.1, 10),
                       *tops[(tops>=0)&(tops<=410)]]))
    layers = []
    for i, top in enumerate(grid):
        sample = (top+grid[i+1])/2 if i+1 < len(grid) else top+.001
        vp = float(vmod.evaluate_below(sample, 'P').item())
        vs = float(vmod.evaluate_below(sample, 'S').item())
        rho = float(vmod.evaluate_below(sample, 'D').item())
        layers.append([float(top),vp,vs,rho,1000.,600.])
    output = dict(name=name, units='native', layers=layers,
        description='Flat isotropic GRTM approximation to the named radial model; midpoint layer velocities.',
        half_space_depth_km=410, fixed_qp=1000, fixed_qs=600,
        provenance='ObsPy TauPyModel standard velocity model; not a regional calibrated crust model',
        recommended_cap_distance_km=[10,500])
    (folder/f'{name}.json').write_text(json.dumps(output, indent=2)+'\n')
    print(name,len(layers),flush=True)
