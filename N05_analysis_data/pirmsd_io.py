"""Read N04 observations without inventing a continuous ratchet history."""
import csv
import hashlib
import json
import math
import re
from pathlib import Path

NORMALIZED_METRICS={'grid_material_domains_v2','rienv_core_domains_v4','core_voronoi_radial_v1'}
FIELDS=['record','segment','iteration','md_step','pirmsd','metric','unit','valid','epoch',
        'reason','weight','xi_min','bias_energy','comparable','committed','source','multidomain_pirmsd','cluster_pirmsd_A']


def segment_dirs(root):
    root=Path(root)
    if re.fullmatch(r'PIRMSD_\d+',root.name):return [root]
    def segments(parent):
        return sorted((p for p in parent.glob('PIRMSD_*')
                       if re.fullmatch(r'PIRMSD_\d+', p.name) and p.is_dir()),
                      key=lambda p: int(p.name.split('_')[1]))

    direct = segments(root)
    cycles = sorted((p for p in root.glob('cycle_*') if re.fullmatch(r'cycle_\d+', p.name) and p.is_dir()),
                    key=lambda p: int(p.name.split('_')[1]))
    nested = [segment for cycle in cycles for segment in segments(cycle)]
    if direct and nested:
        raise ValueError('Select a single trajectory layout or segment: ' + str(root))
    return direct or nested


def segment_name(directory):
    if re.fullmatch(r'cycle_\d+', directory.parent.name):
        return f'{directory.parent.name}/{directory.name}'
    return directory.name


def committed_state(directory):
    directory=Path(directory);pointer=directory/'latest.json'
    if not pointer.exists():return None
    manifest=json.loads(pointer.read_text());path=(directory/manifest['state']).resolve()
    if path.parent!=directory.resolve():raise ValueError('Checkpoint state must be inside segment')
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=manifest['sha256']:raise ValueError('Checkpoint state checksum mismatch: '+str(path))
    state=json.loads(raw)
    if state.get('schema')!=1:raise ValueError('Unsupported checkpoint schema')
    return state


def _number(value):
    if value is None or value.lower() in ('none','null'):return None
    result=float(value)
    return result if math.isfinite(result) else None


def read_history(path, state=None):
    path=Path(path)
    if not path.exists():return []
    if state is None:state=committed_state(path.parent)
    if state is None:raise ValueError('Current N04 history needs latest.json: '+str(path))
    size=state['history_bytes']
    if not isinstance(size,int) or size<0:raise ValueError('Invalid committed history size')
    with path.open('rb') as stream:raw=stream.read(size)
    if len(raw)!=size:raise ValueError('Committed history missing/truncated: '+str(path))  # N04 discards precisely this uncommitted tail on restart.
    rows=[]
    segment = segment_name(path.parent)
    source = str(path.resolve())
    for line in raw.decode().splitlines():
        if not line.strip():continue
        match=re.fullmatch(r'The (\d+) is finished, the RMSD is ([^,\s]+), .*',line)
        if not match:raise ValueError('Unsupported N04 history format: '+str(path))
        values=dict(re.findall(r'(\w+)=([^,\s]+)',line))
        required={'metric','valid','epoch','reason','md_step','bias_energy'}
        if not required<=values.keys():raise ValueError('Incomplete N04 history fields: '+str(path))
        metric=values['metric']
        if metric not in NORMALIZED_METRICS and metric != 'cluster_pirmsd_v1':raise ValueError('Unsupported N04 metric: '+metric)
        score=_number(match[2]);valid=values.get('valid')
        if valid not in ('True','False'):raise ValueError('Invalid history validity flag')
        valid=valid=='True'
        if score is None or score<0:valid=False
        if metric in NORMALIZED_METRICS and score is not None and score>1+1e-8:valid=False
        weight=re.search(r'the total weighted atoms is ([^,\s]+)',line,re.I)
        minimum=re.search(r'the xi_min is ([^,\s]+)',line,re.I)
        reason=values.get('reason','unknown')
        if not weight or not minimum:raise ValueError('Missing history weight/minimum')
        epoch=int(values['epoch'])
        row=dict(segment=segment,iteration=int(match[1]),md_step=int(values['md_step']),
                 pirmsd=score,metric=metric,unit='Å' if metric=='cluster_pirmsd_v1' else 'dimensionless',
                 cluster_pirmsd_A=(_number(values['cluster_pirmsd_A']) if 'cluster_pirmsd_A' in values
                                   else score if metric == 'cluster_pirmsd_v1' else None),
                 multidomain_pirmsd=(_number(values['multidomain_pirmsd']) if 'multidomain_pirmsd' in values
                                     else score if metric in NORMALIZED_METRICS else None),
                 valid=valid,epoch=epoch,reason=reason,weight=_number(weight[1]) if weight else None,
                 xi_min=_number(minimum[1]) if minimum else None,bias_energy=_number(values.get('bias_energy')),
                 comparable=valid is True and epoch is not None and reason not in ('reference_transition','invalid_metric'),
                 committed=True,source=source)
        if row['iteration']>=state['next_iteration']:raise ValueError('History extends beyond committed iteration')
        if rows and row['iteration']<=rows[-1]['iteration']:raise ValueError('Duplicate/backward history iteration: '+str(path))
        rows.append(row)
    return rows


def read_pirmsd_history(root):
    rows=[]
    for directory in segment_dirs(root):rows.extend(read_history(directory/'history.log'))
    for index,row in enumerate(rows):row['record']=index
    return rows


def write_history_csv(rows,path):
    with Path(path).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=FIELDS);writer.writeheader();writer.writerows(rows)


def dump_parts(directory):
    directory=Path(directory);parts=[]
    for path in directory.glob('lmp_pirmsd*.lammpstrj'):
        match=re.fullmatch(r'lmp_pirmsd_(\d+)_([A-Za-z0-9]+)\.lammpstrj',path.name)
        if match:parts.append((int(match[1]),path))
    return sorted(parts,key=lambda item:(item[0],item[1].name))

def write_cycle_metrics(root, output):
    """Export saved stage observations from completed cycle summaries."""
    files = [p for p in (Path(root) / 'cycle_metrics').glob('*.json') if p.stem.isdecimal()]
    if not files:
        return 0
    fields = ['cycle', 'stage', 'cluster_pirmsd_A', 'multidomain_pirmsd']
    rows = []
    for path in sorted(files, key=lambda p: int(p.stem)):
        data = json.loads(path.read_text())
        if data.get('cycle') != int(path.stem):
            raise ValueError(f'Cycle number mismatch in {path}')
        cluster = data.get('cluster_pirmsd_A', {})
        domains = data.get('multidomain_pirmsd', {})
        for stage in ('start', 'after_md', 'after_bias', 'after_relax'):
            if stage in cluster or stage in domains:
                rows.append((data['cycle'], stage, cluster.get(stage), domains.get(stage)))
    if rows:
        with Path(output).open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(fields)
            writer.writerows(rows)
    return len(rows)
