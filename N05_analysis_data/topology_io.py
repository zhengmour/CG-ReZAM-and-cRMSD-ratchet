import re
from multiprocessing import Pool
from pathlib import Path

def read_gro(name):
    """
    Read a .gro file and return a unified format:
    List[(frame_data, static_dict)]
    Since .gro contains only one frame, this returns a list of length 1.
    """
    import MDAnalysis as mda
    u = mda.Universe(str(name))

    # Follow the same selection rules as for xtc: name Si, DO*, C*, N*, Na*
    u_Si = u.select_atoms('name Si')
    u_OB = u.select_atoms('name DO*')
    u_C  = u.select_atoms('name C*')
    u_N  = u.select_atoms('name N* and not name Na* NA*')
    u_Na = u.select_atoms('name Na* NA*')

    ts = u.trajectory.ts  # only one frame

    try:
        molnums_all = u.atoms.molnums.copy()
    except:
        molnums_all = None   # No molnum info; later we will group by resid

    static = dict(
        idx_Si = u_Si.indices,
        idx_OB = u_OB.indices,
        idx_C  = u_C.indices,
        idx_N  = u_N.indices,
        idx_Na = u_Na.indices,

        molnums = molnums_all,   # shape = (N_atoms,)
        resids  = u.atoms.resids.copy(),    # shape = (N_atoms,)
        resnames = u.atoms.resnames.copy(),   # e.g. 'TPA', 'RSi'
        names   = u.atoms.names.copy(),       # atom names, e.g. 'Si', 'C1'
    )
    frame_data = dict(
        frame = ts.frame,
        time  = ts.time,
        pos   = u.atoms.positions.copy(),
        cell  = ts.dimensions[:3].copy(),
    )

    return [(frame_data, static)]

def read_gro_files(traj_path: Path, n_loop: int):
    """
    Search for all .gro files under traj_path,
    read them in parallel using read_gro, and finally flatten into one big list.
    Return format: List[(frame_data, static_dict)]
    """
    gro_files = sorted(
        [f for f in traj_path.iterdir() if f.is_file() and f.suffix == ".gro"]
    )
    if not gro_files:
        raise FileNotFoundError("No .gro files found")

    with Pool(n_loop) as pool:
        frames_lists = pool.map(read_gro, gro_files)

    # flatten: List[List[frame]] -> List[frame]
    all_frames = []
    iframe = 0
    for sub in frames_lists:
        for frame_data, static in sub:
            frame_data["frame"] = iframe
            frame_data["time"] = iframe
            frame_data["axis_unit"] = "frame"
            all_frames.append((frame_data, static))
            iframe += 1
    return all_frames

#######################################################################################################

def read_xtc_files(traj_path: Path, n_loop: int):
    import MDAnalysis as mda
    # Read xtc files in the order of eq_cg, NPT, NVT
    topfile = traj_path/"init.gro"

    eq2file = traj_path/"eq2_cg.xtc"
    NPTfile = traj_path/"NPT.xtc"
    NVTfile = traj_path/"NVT.xtc"

    if NVTfile.exists():
        u = mda.Universe(topfile, eq2file, NPTfile, NVTfile)
    elif NPTfile.exists():
        u = mda.Universe(topfile, eq2file, NPTfile)
    elif eq2file.exists():
        u = mda.Universe(topfile, eq2file)

    u_Si = u.select_atoms('name Si')
    u_OB = u.select_atoms('name DO*')
    u_C = u.select_atoms('name C*')
    u_N = u.select_atoms('name N* and not name Na* NA*')
    u_Na = u.select_atoms('name Na* NA*')

    try:
        molnums_all = u.atoms.molnums.copy()
    except:
        molnums_all = None   # No molnum info; later we will group by resid

    static = dict(
        idx_Si = u_Si.indices,
        idx_OB = u_OB.indices,
        idx_C  = u_C.indices,
        idx_N  = u_N.indices,
        idx_Na = u_Na.indices,

        molnums = molnums_all,   # shape = (N_atoms,)
        resids  = u.atoms.resids.copy(),    # shape = (N_atoms,)
        resnames = u.atoms.resnames.copy(),   # e.g. 'TPA', 'RSi'
        names   = u.atoms.names.copy(),       # atom names, e.g. 'Si', 'C1'
    )

    time_offset = 0.0
    previous_time = None
    all_frames = []
    for ts in u.trajectory:
        raw_time = ts.time   # Time from the trajectory itself (each segment restarts from 0)
    
        # --- Detect whether "time goes backwards" (i.e., start of a new segment)
        if previous_time is not None and raw_time < previous_time:
            time_offset += previous_time

        continuous_time = raw_time + time_offset
        previous_time = raw_time
        
        frames_data = dict(
            frame = ts.frame,
            time  = continuous_time,
            pos   = u.atoms.positions.copy(),
            cell  = u.dimensions[:3].copy(),
        )
        all_frames.append((frames_data, static))

    return all_frames

#######################################################################################################

def read_data(name, type_map=None):
    import numpy as np
    import MDAnalysis as mda
    from lammps_io import TYPE_MAP
    type_map=type_map or TYPE_MAP
    u=mda.Universe(str(name),topology_format="DATA")
    static={f'idx_{key}':u.select_atoms('type '+' '.join(map(str,value if isinstance(value,list) else [value]))).indices for key,value in type_map.items()}
    static.update(ids=u.atoms.ids.copy(),types=u.atoms.types.copy(),molnums=None,resids=u.atoms.resids.copy())
    if not np.allclose(u.dimensions[3:],90):raise ValueError('Only orthogonal boxes supported')
    cell=u.dimensions[:3].copy()
    return [(dict(frame=0,time=0,pos=np.mod(u.atoms.positions.copy(),cell),cell=cell,axis_unit='frame',source=str(name)),static)]


def read_data_files(traj_path, n_loop, type_map=None):
    files=sorted((p for p in Path(traj_path).iterdir() if p.suffix in ('.lmp','.data')),key=lambda p:[int(t) if t.isdigit() else t for t in re.split(r'(\d+)',p.name)])
    if not files:raise FileNotFoundError('No .lmp/.data files')
    frames=[]
    for path in files:
        frame,static=read_data(path,type_map)[0];frame['frame']=frame['time']=len(frames);frames.append((frame,static))
    return frames


def read_dump_files(traj_path, n_loop, type_map=None, dt_fs=None):
    from lammps_io import dump_frames
    files=sorted((p for p in Path(traj_path).iterdir() if p.suffix in ('.dump','.lammpstrj')),key=lambda p:[int(t) if t.isdigit() else t for t in re.split(r'(\d+)',p.name)])
    if not files:raise FileNotFoundError('No dump/lammpstrj files')
    frames=[]
    for path in files:frames.extend(dump_frames(path,type_map))
    for index,(frame,_) in enumerate(frames):
        frame['frame']=index
        frame['time']=index;frame['axis_unit']='frame'
    return frames


def read_rmsd_dump_files(traj_path, n_loop, type_map=None, dt_fs=None):
    from lammps_io import read_pirmsd_dumps
    return read_pirmsd_dumps(traj_path,type_map,dt_fs)
