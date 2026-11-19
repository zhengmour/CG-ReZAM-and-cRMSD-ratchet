"""ID-sorted orthogonal LAMMPS dumps, including N04 restart fragments."""
from pathlib import Path
import numpy as np
from pirmsd_io import segment_dirs,dump_parts,read_history,committed_state,segment_name

TYPE_MAP={'Si':1,'OB':2,'C':4,'N':5,'Na':6}


def dump_frames(path,type_map=None,max_step=None):
    type_map=type_map or TYPE_MAP
    with Path(path).open() as stream:
        while True:
            line=stream.readline()
            if not line:return
            if line.strip()!='ITEM: TIMESTEP':raise ValueError('Malformed/truncated dump: '+str(path))
            step=int(stream.readline())
            if max_step is not None and step>max_step:return
            if stream.readline().strip()!='ITEM: NUMBER OF ATOMS':raise ValueError('Missing atom count')
            n=int(stream.readline());header=stream.readline().split()
            if header[:3]!=['ITEM:','BOX','BOUNDS'] or any(s in header for s in ('xy','xz','yz')):
                raise ValueError('Only orthogonal LAMMPS boxes are supported')
            if header[3:]!=['pp','pp','pp']:raise ValueError('Topology analysis requires a fully periodic box')
            bounds=np.array([list(map(float,stream.readline().split())) for _ in range(3)])
            origin=bounds[:,0];cell=bounds[:,1]-origin
            if np.any(cell<=0):raise ValueError('Invalid box')
            columns=stream.readline().split()
            if columns[:2]!=['ITEM:','ATOMS']:raise ValueError('Missing atom columns')
            columns=columns[2:];data=np.array([list(map(float,stream.readline().split())) for _ in range(n)])
            if data.shape!=(n,len(columns)) or not np.isfinite(data).all():raise ValueError('Truncated/nonfinite dump')
            col={name:i for i,name in enumerate(columns)}
            if not {'id','type'}<=set(col):raise ValueError('Dump needs id and type columns')
            order=np.argsort(data[:,col['id']]);data=data[order];ids=data[:,col['id']].astype(int)
            if len(set(ids))!=n or not np.array_equal(ids,data[:,col['id']]):raise ValueError('Duplicate/noninteger atom IDs')
            types=data[:,col['type']].astype(int)
            xyz=None
            for names,scaled in [(('xu','yu','zu'),False),(('x','y','z'),False),(('xs','ys','zs'),True),(('xsu','ysu','zsu'),True)]:
                if set(names)<=set(col):
                    xyz=data[:,[col[v] for v in names]]
                    xyz=xyz*cell if scaled else xyz-origin
                    break
            if xyz is None:raise ValueError('Unsupported dump coordinates')
            # N04 writes unwrapped coordinates. All topology searches need a
            # common primary box, including nonzero box origins.
            xyz=np.mod(xyz,cell)
            static={f'idx_{key}':np.flatnonzero(np.isin(types,value)) for key,value in type_map.items()}
            if not len(static['idx_Si']):raise ValueError('No Si atoms in dump')
            if not len(static['idx_OB']):raise ValueError('No SP/OB sites: check --type-map against the simulation input')
            mol=data[:,col['mol']].astype(int) if 'mol' in col else np.arange(n)
            static.update(ids=ids,types=types,molnums=mol,resids=mol)
            frame=dict(frame=0,time=float(step),md_step=step,pos=xyz,cell=cell,source=str(Path(path).resolve()),axis_unit='step')
            yield frame,static


def read_segment(directory,type_map=None):
    directory=Path(directory);parts=dump_parts(directory);state=committed_state(directory)
    history=read_history(directory/'history.log',state)
    steps={r['iteration']:r['md_step'] for r in history if r['md_step'] is not None}
    if state is None:
        raise ValueError('New N04 fragments need latest.json to exclude uncommitted frames: '+str(directory))
    max_step=max(steps.values()) if steps else None
    if state['next_iteration'] and max_step is None:
        raise ValueError('Committed N04 history has no md_step')
    if not state['next_iteration']:return []
    selected={}
    starts=sorted(set(start for start,_ in parts if start<state['next_iteration']))
    for start,path in parts:
        if start>=state['next_iteration']:continue
        # Later restart fragments supersede abandoned suffixes of older files,
        # even when those suffixes have no exact duplicate timestep.
        next_starts=[s for s in starts if s>start]
        end=steps.get(next_starts[0]-1) if next_starts else max_step
        if next_starts and end is None:raise ValueError('Cannot locate restart boundary in history')
        lower=steps.get(start-1) if start>0 else None
        for frame,static in dump_frames(path,type_map,max_step=end):
            step=frame['md_step']
            if (max_step is not None and step>max_step) or (end is not None and step>end):continue
            if lower is not None and step<lower:continue
            if step in selected:
                old_start,old_frame,old_static=selected[step]
                if start==old_start and not (np.array_equal(static['ids'],old_static['ids']) and np.array_equal(static['types'],old_static['types']) and np.allclose(frame['cell'],old_frame['cell'],rtol=0,atol=1e-7) and np.allclose(frame['pos'],old_frame['pos'],rtol=0,atol=1e-6)):
                    raise ValueError('Conflicting restart fragments at same start/step; cannot order UUIDs: '+str(directory))
                if start<old_start:continue
            frame['segment']=segment_name(directory)
            selected[step]=(start,frame,static)
    frames=[(selected[s][1],selected[s][2]) for s in sorted(selected)]
    for previous,current in zip(frames,frames[1:]):
        if not np.array_equal(previous[1]['ids'],current[1]['ids']) or not np.array_equal(previous[1]['types'],current[1]['types']):
            raise ValueError('Atom identities/types changed within a PIRMSD segment')
    return frames


def read_pirmsd_dumps(root,type_map=None,dt_fs=None):
    frames=[];directories=segment_dirs(root)
    for directory in directories:frames.extend(read_segment(directory,type_map))
    if not frames:raise FileNotFoundError('No committed PIRMSD dump frames found')
    # Segment start times are not stored in current N04 output. Never turn a
    # concatenated observation index into physical time across different runs.
    physical=len({f['segment'] for f,_ in frames})==1 and dt_fs is not None
    for index,(frame,_) in enumerate(frames):
        frame['frame']=index
        frame['time']=frame['md_step']*dt_fs/1000 if physical else index
        frame['axis_unit']='ps' if physical else 'frame'
    return frames
