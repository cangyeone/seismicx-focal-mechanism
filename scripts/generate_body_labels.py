"""OFFLINE teacher only. Resume chunked multiphase labels from original ObsPy."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scipy.stats import qmc
from seismic_joint.schema import BODY_PHASES, EARTH_MODELS, TAUP_PHASES

MODELS = {}


def model(name):
    if name not in MODELS:
        from obspy.taup import TauPyModel
        MODELS[name] = TauPyModel(name)
    return MODELS[name]


def compute(task):
    name, split, chunk, x = task
    m = model(name)
    y = np.full((len(x), len(BODY_PHASES), 5), np.nan, dtype=np.float64)
    gap = np.full((len(x), len(BODY_PHASES)), np.nan)
    for i, (h, d) in enumerate(x):
        rays = m.get_travel_times(float(h), float(d)/m.model.radius_of_planet*180/np.pi,
            phase_list=list(TAUP_PHASES), ray_param_tol=1e-6)
        by_name = {phase: [a for a in rays if a.name == phase] for phase in TAUP_PHASES}
        by_name['S_first'] = [a for a in rays if a.name in ('S', 's', 'Sn')]
        for j, phase in enumerate(BODY_PHASES):
            candidates = by_name[phase]
            if not candidates:
                continue
            a = min(candidates, key=lambda item: item.time)
            source_wave = 'S' if a.name[0].lower() == 's' else 'P'
            vmod = m.model.s_mod.v_mod
            down = a.phase.down_going[0]
            velocity = float((vmod.evaluate_below if down else vmod.evaluate_above)(h, source_wave).item())
            angle = np.deg2rad(a.takeoff_angle)
            y[i, j] = [a.time, -np.cos(angle)/velocity, a.ray_param/m.model.radius_of_planet,
                        a.takeoff_angle, velocity]
            gap[i, j] = min((b.time-a.time for b in candidates
                if abs(b.takeoff_angle-a.takeoff_angle) > 2), default=np.inf)
    return name, split, chunk, x, y, gap


def sample(n, seed):
    u = qmc.Sobol(4, scramble=True, seed=seed).random_base2(int(np.ceil(np.log2(n))))[:n]
    x = np.stack((u[:, 0]*200, u[:, 1]*3000), axis=1)
    regional = u[:, 2] < .5
    x[regional] = np.stack((u[regional, 0]*60, u[regional, 1]*600), axis=1)
    near = (u[:, 2] >= .5) & (u[:, 2] < .65)
    x[near, 1] = np.exp(np.log(.01)+u[near, 1]*np.log(3000/.01))
    x[:, 0] = np.maximum(x[:, 0], .001)
    x[:, 1] = np.maximum(x[:, 1], .001)
    return x


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train', type=int, default=32768)
    p.add_argument('--validation', type=int, default=2048)
    p.add_argument('--test', type=int, default=2048)
    p.add_argument('--workers', type=int, default=24)
    p.add_argument('--chunk', type=int, default=256)
    args = p.parse_args()
    folder = ROOT/'data/body_chunks'
    folder.mkdir(parents=True, exist_ok=True)
    (ROOT/'results').mkdir(exist_ok=True)
    tasks = []
    metadata = {}
    sizes = {'train': args.train, 'validation': args.validation, 'test': args.test}
    for mi, name in enumerate(EARTH_MODELS):
        m = model(name)
        layers = m.model.s_mod.v_mod.layers
        discontinuity = ((abs(layers[1:]['top_p_velocity']-layers[:-1]['bot_p_velocity']) > 1e-6)
                      | (abs(layers[1:]['top_s_velocity']-layers[:-1]['bot_s_velocity']) > 1e-6))
        interfaces = layers[1:]['top_depth'][discontinuity]
        metadata[name] = dict(radius_km=m.model.radius_of_planet,
            interfaces=interfaces[(interfaces > 0)&(interfaces < 200)].tolist())
        for si, (split, n) in enumerate(sizes.items()):
            x = sample(n, 20261025 + mi*100 + si)
            for first in range(0, n, args.chunk):
                chunk = first//args.chunk
                path = folder/f'{name}-{split}-{chunk:04d}.npz'
                if not path.exists():
                    tasks.append((name, split, chunk, x[first:first+args.chunk]))
    start = time.time()
    print(json.dumps(dict(stage='teacher_start', chunks=len(tasks), sizes=sizes, workers=args.workers)), flush=True)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn')) as pool:
        futures = [pool.submit(compute, task) for task in tasks]
        for done, future in enumerate(as_completed(futures), 1):
            name, split, chunk, x, y, gap = future.result()
            path = folder/f'{name}-{split}-{chunk:04d}.npz'
            np.savez_compressed(path, inputs=x, labels=y, branch_gap=gap)
            progress = dict(stage='body_teacher', chunks_done=done, new_chunks=len(tasks),
                elapsed_seconds=time.time()-start, last_model=name, last_split=split)
            if done%6 == 0 or done == len(tasks):
                (ROOT/'results/body_preparation_progress.json').write_text(json.dumps(progress, indent=2)+'\n')
                print(json.dumps(progress), flush=True)
    import obspy
    manifest = dict(teacher='original ObsPy TauPyModel', obspy_version=obspy.__version__,
        ray_param_tol=1e-6, phases=BODY_PHASES, models=metadata, sizes=sizes,
        label_columns=['time_s','dT_ddepth_s_km','dT_ddistance_s_km','takeoff_deg','source_velocity_km_s'],
        domain_depth_km=[0,200], domain_distance_km=[0,3000],
        missing='NaN and false mask; never imputed', separate_split_scrambles=True,
        group_definition='Earliest ray within each named phase; S_first=min(S,s,Sn)',
        runtime_uses_taup=False, seconds=time.time()-start)
    for name in EARTH_MODELS:
        for split in sizes:
            parts = [dict(np.load(path)) for path in sorted(folder.glob(f'{name}-{split}-*.npz'))]
            data = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
            np.savez_compressed(ROOT/'data'/f'{name}-{split}.npz', **data)
    (ROOT/'data/body_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(dict(stage='body_teacher_done', seconds=time.time()-start)), flush=True)


if __name__ == '__main__':
    main()
