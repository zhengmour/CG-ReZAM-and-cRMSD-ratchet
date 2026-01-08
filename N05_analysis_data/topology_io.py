import re
from multiprocessing import Pool
import MDAnalysis as mda
from pathlib import Path

def read_gro(name):
    """
    Read a .gro file and return a unified format:
    List[(frame_data, static_dict)]
    Since .gro contains only one frame, this returns a list of length 1.
    """
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
            frame_data["time"] = iframe  # .gro file has no time; temporarily use frame index instead
            all_frames.append((frame_data, static))
            iframe += 1
    return all_frames

#######################################################################################################

def read_xtc_files(traj_path: Path, n_loop: int):
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

def read_data(name):
    try:
        u = mda.Universe(name, topology_format="DATA")
        ts = u.trajectory.ts  # only one frame

        u_Si = u.select_atoms("type 1")
        u_OB = u.select_atoms("type 3")
        u_C  = u.select_atoms("type 6")
        u_N  = u.select_atoms("type 7")
        u_Na = u.select_atoms("type 8")

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
    except Exception as e:
        print(e)
        return []
    
def read_data_files(traj_path: Path, n_loop: int):
    traj_files = sorted(
        [f for f in traj_path.iterdir() if f.is_file() and f.suffix in ['.lmp', '.data']],
        key=lambda x: int(x.stem)
    )
    if not traj_files:
        raise FileNotFoundError("No .lmp or .data files found")

    # Each data file returns a list[frame]; here we flatten all frames into a single list
    with Pool(n_loop) as pool:
        frames_lists = pool.map(read_data, traj_files)

    all_frames = []
    iframe = 0
    for sub in frames_lists:
        for frame_data, static in sub:
            frame_data["frame"] = iframe
            frame_data["time"] = iframe
            all_frames.append((frame_data, static))
            iframe += 1
    return all_frames

#######################################################################################################

def read_dump_files(traj_path: Path, n_loop: int):
    # Load topology
    topfile = traj_path / "lmp.data"
    if not topfile.exists():
        raise FileNotFoundError(f"No lmp.data topology file found in {traj_path}")

    # Collect all dump / lammpstrj files
    traj_files = sorted(
        [f for f in traj_path.iterdir()
         if f.is_file() and f.stem.startswith("lmp_")
         and f.suffix in ['.dump', '.lammpstrj']],
        key=lambda x: float(x.stem.split('_')[1])
    )
    times = [float(x.stem.split('_')[1]) for x in traj_files]
    if not traj_files:
        raise FileNotFoundError("No .dump or .lammpstrj files found")

    # Use LAMMPSDUMP + Data to read
    u = mda.Universe(
        str(topfile),
        [str(f) for f in traj_files],
        format="LAMMPSDUMP",
        # atom_style="id type xu yu zu"   # If it's x y z, change to "id type x y z" or remove
    )
    ts = u.trajectory.ts

    # LAMMPS systems: select by type
    u_Si = u.select_atoms("type 1")
    u_OB = u.select_atoms("type 3")
    u_C  = u.select_atoms("type 6")
    u_N  = u.select_atoms("type 7")
    u_Na = u.select_atoms("type 8")

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
    )

    all_frames = []
    iframe = 0
    for time, ts in zip(times, u.trajectory):
        frame_data = dict(
            frame = iframe,
            time  = time,
            pos   = u.atoms.positions.copy(),
            cell  = ts.dimensions[:3].copy(),
        )
        all_frames.append((frame_data, static))
        iframe += 1

    return all_frames

#######################################################################################################

def read_rmsd_dump(name):
    """
    name: e.g., /path/to/PIRMSD_1/lmp_pirmsd.lammpstrj
    return: list[(frame_data, static_dict)]
    """
    traj_root = name.parent.parent          # Go back to the directory containing lmp.data
    topfile = traj_root / "lmp.data"
    if not topfile.exists():
        raise FileNotFoundError(f"No lmp.data found in {traj_root}")

    try:
        u = mda.Universe(
            str(topfile),
            str(name),
            format="LAMMPSDUMP")
    except:
        print(f"Failed to read RMSD dump file: {name}")
        return None

    u_Si = u.select_atoms("type 1")
    u_OB = u.select_atoms("type 3")
    u_C  = u.select_atoms("type 6")
    u_N  = u.select_atoms("type 7")
    u_Na = u.select_atoms("type 8")

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
    )

    frames = []
    for ts in u.trajectory:
        frame_data = dict(
            frame = ts.frame,
            time  = ts.time,
            pos   = u.atoms.positions.copy(),
            cell  = ts.dimensions[:3].copy(),
        )
        frames.append((frame_data, static))

    return frames

def read_rmsd_dump_files(traj_path: Path, n_loop: int):
    traj_dirs = sorted(
        [f for f in traj_path.iterdir() if f.is_dir() and f.stem.startswith("PIRMSD_")],
        key=lambda x: float(x.stem.split('_')[1])
    )
    traj_files = [d / "lmp_pirmsd.lammpstrj" for d in traj_dirs]
    traj_files = [f for f in traj_files if f.exists()]
    if not traj_files:
        raise FileNotFoundError("No PIRMSD_*/lmp_pirmsd.lammpstrj files found")
    print(f"Found {len(traj_files)} PIRMSD directories")

    # Each file returns list[frame]; flatten into a single list
    with Pool(n_loop) as pool:
        frames_lists = pool.map(read_rmsd_dump, traj_files)

    all_frames = []
    iframe = 0
    for sub in frames_lists:
        if sub is None:
            iframe += 1
        else:
            for frame_data, static in sub:
                frame_data["frame"] = iframe
                frame_data["time"] = iframe
                all_frames.append((frame_data, static))
                iframe += 1
    return all_frames

#######################################################################################################

def read_env(name):
    try:
        with open(name) as f:
            lines = f.readlines()
        times, means, morethans = [], [], []
        titles = lines[0].split()[2:]
        for line in lines:
            if line.startswith('#!'): continue
            words = list(map(float, line.split()))
            for i, t in enumerate(titles):
                if t.endswith('time'):
                    times.append(words[i])
                elif t.endswith('.mean'):
                    means.append(words[i])
                elif t.endswith('.morethan'):
                    morethans.append(words[i])
        return (times, means, morethans)
    except:
        return None

def read_env_files(traj_path: Path, n_loop: int):
    files = sorted([f for f in traj_path.iterdir() if f.is_file() and f.suffix == '.COLVAR'],
                   key=lambda x: int(x.stem))
    if not files:
        return None

    with Pool(n_loop) as pool:
        results = pool.map(read_env, files)
    results = [r for r in results if r is not None]

    times, means, morethans = [], [], []
    last_time = 0
    for t, m, mt in results:
        times.extend([last_time + x for x in t])
        means.extend(m)
        morethans.extend(mt)
        last_time = times[-1]

    return [times, means, morethans]

#######################################################################################################

def read_rmsd(name):
    try:
        with open(name) as f:
            lines = f.readlines()
        frames, rmsds = [], []
        for line in lines:
            match = re.search(r"The (\d+) is finished, the RMSD is ([0-9.eE+-]+)", line)    
            if match:
                frames.append(int(match.group(1)))
                rmsds.append(float(match.group(2)))
        return (frames, rmsds)
    except:
        return None

def read_rmsd_files(traj_path: Path, n_loop: int):
    dirs = sorted([d for d in traj_path.iterdir() if d.is_dir() and d.stem.startswith("PIRMSD_")],
                   key=lambda x: int(x.stem.split('_')[1]))
    if not dirs:
        return None

    files = [d / "history.log" for d in dirs]
    with Pool(n_loop) as pool:
        results = pool.map(read_rmsd, files)
    results = [r for r in results if r is not None]

    frames, rmsds = [], []
    last_frame = 0.0
    for f, r in results:
        frames.extend([t + last_frame for t in f])
        rmsds.extend(r)
        last_frame = frames[-1]

    return [frames, rmsds]
