"""Read one real CSNDATA event; raw Z stays unnormalized until packaged CNN."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import numpy as np
import h5py

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--year',type=int,default=2022)
    p.add_argument('--cache',type=Path,default=Path('/home/yuzy/csndata_polarity_cnn_20261024/cache'))
    p.add_argument('--source',type=Path,default=Path('/mnt/d/csndata'));args=p.parse_args()
    records=[json.loads(line) for line in (args.cache/str(args.year)/'records.jsonl').open()]
    validity=np.load(args.cache/str(args.year)/'valid_records.npy')
    groups=defaultdict(list)
    for row in records:
        if validity[row['cache_index']] and not row['event_id'].startswith('CB.'):
            groups[row['event_id']].append(row)
    candidates={key for key,_ in sorted(((key,len(v)) for key,v in groups.items()),key=lambda a:-a[1])[:40]}
    metadata={};azimuth={}
    with (args.source/f'{args.year}.pha').open() as f:
        for line in f:
            a=line.split()
            if len(a)<2 or a[1] not in candidates:continue
            if a[0]=='#EVENT':
                k=a.index('LOC');j=a.index('DEP')
                metadata[a[1]]=dict(id=a[1],longitude=float(a[k+1]),latitude=float(a[k+2]),depth_km=float(a[j+1]))
            elif a[0]=='PHASE' and len(a)>8 and a[3]=='POLARITY':
                k=a.index('DISTAZ')
                azimuth[(a[1],'.'.join(a[4:7]))]=float(a[k+2])
    selected=[key for key in candidates if key in metadata and 0<metadata[key]['depth_km']<200
        and min(abs(metadata[key]['depth_km']-x) for x in [20,35])>.01]
    event_id=max(selected,key=lambda key:len(groups[key]))
    folder=ROOT/'examples/csndata_polarity';folder.mkdir(parents=True,exist_ok=True)
    stations=[]
    with h5py.File(args.source/f'{args.year}.h5','r') as source:
        for i,row in enumerate(groups[event_id]):
            if (event_id,row['station']) not in azimuth:continue
            sample=int(row['p_sample'])
            raw=np.asarray(source[event_id][row['station']][row['channel']][sample-512:sample+512],dtype=np.float64)
            if len(raw)!=1024 or not np.isfinite(raw).all():continue
            name=f'raw_Z_{i:03d}.npy';np.save(folder/name,raw)
            stations.append(dict(id=row['station'],distance_km=row['distance_km'],
                azimuth_deg=azimuth[(event_id,row['station'])],raw_z_file=name,
                manual_label_for_audit_only='U' if row['label']==0 else 'D',
                phase='Pg_regional' if row['phase']=='Pg' else row['phase'],catalog_phase=row['phase']))
    config=dict(event=metadata[event_id],earth_model='iasp91',stations=stations,
        polarity_starts=256,polarity_steps=1200,
        provenance=dict(source='CSNDATA real event; raw 100 Hz Z; P at sample 512',
            waveform_usage='initial motion only; not instrument-response-corrected CAP displacement',
            manual_labels_used_as_inputs=False,source_files_opened_read_only=True))
    (folder/'event.json').write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(event=event_id,stations=len(stations),depth_km=metadata[event_id]['depth_km'],
        manual_counts=dict(Counter(item['manual_label_for_audit_only'] for item in stations))),ensure_ascii=False))


if __name__=='__main__':main()
