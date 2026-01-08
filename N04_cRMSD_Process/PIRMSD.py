import numpy as np
from pathlib import Path
from sklearn.cluster import DBSCAN
from collections import defaultdict
import math
import sys, json, re
from lammps import lammps
from mpi4py import MPI
import logging

try:
    import C_PIRMSD
except ImportError as e:
    print(f"✗ Import failed: {e}")
    print("Please run: python setup.py build_ext --inplace first")
    exit(1)

""""
PIRMSD_NUM structure:
├── cluster_<ITER>.json    # clusters info (used for restart)
├── lmp_es.lammpstrj       # final traj
├── lmp_pirmsd.lammpstrj   # periodic dump
└── history.log            # detailed logs including bias info
"""

Stype = 1 
LOOSE_ITERTION_SETTING = [100, 20, 0.0001]
STRICT_ITERTION_SETTING = [200, 50, 0.00001]
GRID_SIZES = np.array([3.0, 3.0, 3.0])
# Constant kB (kcal/mol/K)
kB = 0.0019872041

from time import perf_counter
from contextlib import contextmanager

@contextmanager
def timer(name="elapsed", logger=None):
    start = perf_counter()
    yield
    end = perf_counter()
    if logger:
        logger.info(f"{name}: {end - start:.6f} s")

class PIRMSD:
    """
    PIRMSD class:
      - finds clusters (optional)
      - computes RMSD per-cluster via C_PIRMSD interface
      - maintains per-cluster metadata including xi_min for per-cluster bias bookkeeping
    """
    def __init__(self, comm, settings, logger):
        self.comm = comm
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.logger = logger

        self.fixed_rotate = settings.get('fixed_rotate', False)
        self.critical_rmsd = settings.get('critical_rmsd', 9.0) if self.fixed_rotate else 0.0
        self.iteration_setting = LOOSE_ITERTION_SETTING

        # Read reference structure
        refer_structure_path = Path(settings.get('refer_structure', 'zsm.xyz'))
        self.logger.info(f"Initializing PIRMSD and reading reference structure from {refer_structure_path}")
        refer_structure = self._read_xyz(refer_structure_path )   # [id, type, x, y, z]
        self.refer_structure = C_PIRMSD._Sort2center(refer_structure)
        self.refer_center = self._center(self.refer_structure)

        center_settings = settings.get('center', {})
        self.centerRMSD = center_settings.get('status', False)
        self.alpha = center_settings.get('alpha', 0.25)    
        self.scale_factor = settings.get('scale_factor', 0.9)   

        self.search_clusters = settings.get('search_clusters', False)
        self.beta = settings.get('beta', 5.0)

        # ratchet / bias defaults
        self.bias_k = settings.get('bias_k', settings.get('k', 25.0)) 
        self.temperature = settings.get('temperature', None)

        # per-cluster data structure (list of dicts)
        # each cluster dict contains: center, radius, indices, previous_rmsd, move, fixed_rotate, iteration_setting, xi_min, bias_k
        self.clusters = [] 
        self.frame = 0
        self.start_frame = 0

        # historical global total RMSD for ratchet decision (not per-cluster)
        self.previous_total_rmsd = None
        self.global_xi_min = None  # historical min of the global CV (total_rmsd)

    # logging file handle path
    def touch_logfile(self, base_dir):
        self.logfile = base_dir / 'history.log'
        self.logger.info(f"History log will be written to {self.logfile}")

    # try to find last cluster file and last log entry to resume
    def get_begin_iteration(self, previous_dir):
        start_frame = 0
        cluster_files = []

        if self.logfile.exists():
            with open(self.logfile, 'r') as rf:
                lines = rf.readlines()
                if lines:
                    line = lines[-1].strip()
                    start_frame = int(re.search(r'The (\d+) is finished', line).group(1))
                    rmsd = re.search(r'RMSD is ([\d\.eE+-]+)', line)
                    if rmsd:    self.previous_total_rmsd = float(rmsd.group(1))
                    global_xi_min = re.search(r'xi_min is ([\d\.eE+-]+)', line)
                    if global_xi_min:  self.global_xi_min = float(global_xi_min.group(1))
            log_dir = self.logfile.parent

            cluster_files = list(log_dir.glob('cluster_*.json'))
        elif previous_dir:
            # Find all clusters_*.json files in the same directory as logfile
            cluster_files = list(previous_dir.glob('clusters_*.json'))
            previous_logfile = previous_dir / 'history.log'
            if previous_logfile.exists():
                with open(previous_logfile, 'r') as rf:
                    lines = rf.readlines()
                    if lines:
                        line = lines[-1].strip()
                        rmsd = re.search(r'RMSD is ([\d\.eE+-]+)', line)
                        if rmsd:    self.previous_total_rmsd = float(rmsd.group(1))
                        global_xi_min = re.search(r'xi_min is ([\d\.eE+-]+)', line)
                        if global_xi_min: self.global_xi_min = float(global_xi_min.group(1))
            
        if len(cluster_files) > 0:
            cluster_files = sorted(cluster_files, key=lambda p: int(re.search(r'cluster_(\d+)\.json', p.name).group(1)))
            self.read_clusters_infos(cluster_files[-1])
            self.logger.info(f"Loaded cluster file for restart: {cluster_files[-1]}")   

        self.logger.info(f"Starting from iteration {start_frame}")
        return start_frame

    # ----------------------
    # IO helpers
    # ----------------------
    def _read_xyz(self, file):
        structures = []
        count = 1.0
        with open(file, 'r') as rf:
            lines = rf.readlines()
            for line in lines[2:]:
                if line:
                    words = line.split()
                    structure = [0]*5
                    structure[0] = count
                    structure[1] = float(words[0])
                    structure[2] = float(words[1])
                    structure[3] = float(words[2])
                    structure[4] = float(words[3])
                    structures.append(structure)
                    count += 1.0
        return structures 
    
    def array_to_list(self, obj):
        if isinstance(obj, (np.ndarray, np.generic)):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self.array_to_list(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.array_to_list(x) for x in obj]
        return obj

    def list_to_array(self, obj):
        if isinstance(obj, list):
            return np.array(obj)
        elif isinstance(obj, dict):
            return {k: self.list_to_array(v) for k, v in obj.items()}
        return obj    

    def write_clusters_infos(self, filename):
        """
        Save clusters so restart inherits ratchet state.
        """
        clusters = self.array_to_list(self.clusters)
        for i, cluster in enumerate(clusters):
            clusters[i]['center'] = [x for x in cluster['center']]
            clusters[i]['indices'] = [int(i) for i in cluster['indices']]
        if self.rank == 0:
            with open(filename, 'w') as f:
                json.dump(clusters, f)

    def read_clusters_infos(self, filename):
        with open(filename, 'r') as f:
            clusters = json.load(f)
        # convert centers back to numpy arrays and indices to numpy arrays
        for c in clusters:
            if isinstance(c.get('center'), list):
                c['center'] = np.array(c['center'], dtype=float)
            if isinstance(c.get('indices'), list):
                c['indices'] = np.array(c['indices'], dtype=int)
        self.clusters = self.list_to_array(clusters)

    # ----------------------
    # cluster detection & processing
    # ----------------------
    def _compute_periodic_distance_matrix(self, positions, cell):
        """Compute distance matrix with periodic boundary conditions"""
        delta = positions[:, None, :] - positions[None, :, :]
        delta -= np.round(delta / cell) * cell
        distance_matrix = np.sqrt((delta ** 2).sum(axis=-1))
        return distance_matrix

    def _calculate_periodic_centroid(self, positions, cell):
        """Compute centroid with periodic boundary conditions"""
        # Use the first atom as reference point
        ref_pos = positions[0]
        
        # Move all atoms to the nearest image relative to the reference point
        adjusted_positions = []
        for pos in positions:
            diff = pos - ref_pos 
            diff = diff - cell * np.round(diff / cell)
            adjusted_positions.append(ref_pos + diff)
        adjusted_positions = np.array(adjusted_positions)
        centroid = np.mean(adjusted_positions, axis=0) % cell 
        return centroid

    def _dbscan_all_clusters(self, positions, cell, eps=5, min_samples=5):
        """
        Use DBSCAN to find all valid clusters and compute their centers and radii.
        Returns a list where each element is (center, radius, indices)
        """
        distance_matrix = self._compute_periodic_distance_matrix(positions, cell)

        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed').fit(distance_matrix)
        labels = clustering.labels_

        unique_labels = np.unique(labels)
        cluster_info = []

        for label in unique_labels:
            if label == -1:
                continue  
            
            indices = np.where(labels == label)[0]
            cluster_positions = positions[indices]
            center = self._calculate_periodic_centroid(cluster_positions, cell)

            distances = []
            for atom in cluster_positions:
                diff = atom - center
                diff = diff - cell * np.round(diff / cell)
                distances.append(np.sqrt(np.sum(diff ** 2)))

            radius = np.percentile(distances, 80)
            cluster_info.append((center, radius, indices))

        if not cluster_info:
            return None  # No valid cluster found

        return cluster_info

    def _match_clusters(self, prev_clusters, curr_clusters, min_overlap_union=0.8):
        """
        Support 1→many and many→1 cluster matching analysis.
        Returns {
            "merged_to": {curr_idx: [prev_idx, ...]},
            "split_from": {prev_idx: [curr_idx, ...]},
            "unmatched_prev": [...],
            "unmatched_curr": [...],
            "direct_matches": {curr_idx: prev_idx}
        }
        """
        n_prev = len(prev_clusters) 
        n_curr = len(curr_clusters) 

        if n_prev == 0:
            return {
                "merged_to": {},
                "split_from": {},
                "unmatched_prev": [],
                "unmatched_curr": list(range(n_curr)),
                "direct_matches": {}
            }

        prev_sets = [set(c["indices"]) for c in prev_clusters]
        curr_sets = [set(c["indices"]) for c in curr_clusters]

        overlaps_union = np.zeros((n_prev, n_curr))
        for i in range(n_prev):
            for j in range(n_curr):
                intersection = prev_sets[i] & curr_sets[j]
                union = prev_sets[i] | curr_sets[j]
                overlaps_union[i, j] = len(intersection) / len(union)

        unmatched_prev = list(range(len(prev_clusters)))
        unmatched_curr = list(range(len(curr_clusters)))

        merged_to = defaultdict(list)
        split_from = defaultdict(list)
        direct_matches = {}
        matched_prev = set()
        matched_curr = set()

        # Step 1: collect one-to-one strong matches
        for i in range(n_prev):
            for j in range(n_curr):
                if (overlaps_union[i, j] >= min_overlap_union):
                    # Max match in both directions
                    if np.argmax(overlaps_union[i]) == j and np.argmax(overlaps_union[:, j]) == i:
                        direct_matches[j] = i
                        matched_prev.add(i)
                        matched_curr.add(j)
                        self.logger.info(f"Direct match found: Previous cluster {i} <-> Current cluster {j}")

        # Step 2: collect many→1 (merge)
        for j in range(n_curr):
            if j in matched_curr:
                continue
            prev_matches = [ i for i in range(n_prev) if i not in matched_prev and overlaps_union[i, j] > 0.0 ]
            if len(prev_matches) >= 2:
                merged_to[j] = prev_matches
                matched_curr.add(j)
                matched_prev.update(prev_matches)
                self.logger.info(f"Merge detected: Current cluster {j} merged from Previous clusters {prev_matches}")

        # Step 3: collect 1→many (split)
        for i in range(n_prev):
            if i in matched_prev:
                continue
            curr_matches = [j for j in range(n_curr) if j not in matched_curr and overlaps_union[i, j] > 0.0]
            if len(curr_matches) >= 2:
                split_from[i] = curr_matches
                matched_prev.add(i)
                matched_curr.update(curr_matches)
                self.logger.info(f"Split detected: Previous cluster {i} split into Current clusters {curr_matches}")

        unmatched_prev = [i for i in range(n_prev) if i not in matched_prev]
        unmatched_curr = [j for j in range(n_curr) if j not in matched_curr]
        self.logger.info(f"New Clusters is {unmatched_curr}")

        return {
            "merged_to": dict(merged_to),
            "split_from": dict(split_from),
            "unmatched_prev": unmatched_prev,
            "unmatched_curr": unmatched_curr,
            "direct_matches": direct_matches
        }

    # ----------------------
    # Helper functions for cluster detection & processing
    # ----------------------

    def _dbscan_method_center(self, positions, cell, eps=5, min_samples=5):
        """
        Use DBSCAN to find the largest cluster
        """
        clusters = self._dbscan_all_clusters(positions, cell, eps, min_samples)

        if clusters:
            # Find the largest cluster
            max_cluster = max(clusters, key=lambda c: len(c[2]))
            max_cluster_center, max_cluster_radius, _ = max_cluster
            self.logger.info(f"Cluster center (dbscan)  : {max_cluster_center}; Cluster radius (distance): {max_cluster_radius:.4f}")

            return (max_cluster_center, max_cluster_radius)

        return None, np.mean(cell)
    
    def _sigmoid(self, current, total, steepness=5):
        if total == 0:
            return 0.0
        x = current / total
        return 1 / (1 + math.exp(-steepness * (x - 0.5)))

    def is_inside_cluster(self, dbscan_method_center, gird_density_center, cell, min_radius):
        delta = dbscan_method_center - gird_density_center
        delta -= np.round(delta / cell) * cell  # PBC correction
        return np.linalg.norm(delta) < min_radius

    def _grid_density_center(self, positions, cell, density_threshold=0.5):
        # For large datasets, FFT-based convolution is more efficient

        target_grids_per_dim = np.mean(cell / GRID_SIZES)
        target_grids_per_dim = 2 ** math.ceil(math.log2(target_grids_per_dim)) # Use power of 2 for FFT
        n_grids = np.array([target_grids_per_dim] * 3, dtype=int)
        
        # Initialize 3D grid counts
        grid_counts = np.zeros(n_grids)
        
        # Count atoms in each grid
        for coord in positions:
            normalized_coord = coord % cell
            grid_indices = (normalized_coord / GRID_SIZES).astype(int)
            grid_indices = np.minimum(grid_indices, n_grids - 1)
            grid_counts[grid_indices[0], grid_indices[1], grid_indices[2]] += 1
        
        # Create Gaussian kernel
        cluster_radius = min(cell) * 0.05
        sigma_grids = cluster_radius / GRID_SIZES
        
        # Create Gaussian kernel in frequency domain
        kernel = self._create_3d_gaussian_kernel_fft(n_grids, sigma_grids)
        
        # Convolution via FFT (automatically handles periodic boundaries)
        grid_fft = np.fft.fftn(grid_counts)
        kernel_fft = np.fft.fftn(kernel)
        convolved_fft = grid_fft * kernel_fft
        density = np.real(np.fft.ifftn(convolved_fft))

        # Find position of maximum density - use centroid of high density instead of simple max
        max_density = np.max(density)
        high_density_mask = density >= density_threshold * max_density

        # Use the maximum density grid as cluster center
        max_density_idx = np.unravel_index(np.argmax(density), density.shape)
        center_grid = np.array(max_density_idx, dtype=float)

        # Convert to real-space center
        center = (center_grid + 0.5) * GRID_SIZES
        center = center % cell

        cluster_volume_grids = np.sum(high_density_mask)
        grid_volume = np.prod(GRID_SIZES)
        cluster_volume = cluster_volume_grids * grid_volume
        cluster_radius_volume = (3 * cluster_volume / (4 * np.pi)) ** (1/3)

        self.logger.info(f"Max density: {max_density:.4f}")
        self.logger.info(f"Min density: {np.min(density):.4f}")
        self.logger.info(f"High density grids: {np.sum(high_density_mask)}")
        self.logger.info(f"Cluster center (grid)  : {center}")
        # self.logger.info(f"Cluster radius (grid)  : {cluster_radius_est:.4f}")
        self.logger.info(f"Cluster radius (volume): {cluster_radius_volume:.4f}")

        return (center, cluster_radius_volume)

    def _create_3d_gaussian_kernel_fft(self, n_grids, sigma_grids):
        """
        Create a 3D Gaussian kernel in frequency space (for FFT convolution)
        """
        nx, ny, nz = n_grids
        sx, sy, sz = sigma_grids
        
        # Create frequency domain coordinates
        kx = np.fft.fftfreq(nx, d=1.0)
        ky = np.fft.fftfreq(ny, d=1.0)
        kz = np.fft.fftfreq(nz, d=1.0)
        
        KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
        
        # Gaussian function remains Gaussian in frequency space
        kernel_fft = np.exp(-2 * np.pi**2 * (
            (KX * sx)**2 + (KY * sy)**2 + (KZ * sz)**2
        ))
        
        return kernel_fft

    # ----------------------
    # functions for centering and compressing atoms
    # ----------------------

    # Compute geometric center of the selected conformation
    def _center(self, atoms): 
        center = [0]*3
        count = 0
        for atom in atoms:
            if atom[1] == Stype:
                center[0] = center[0] + atom[2]
                center[1] = center[1] + atom[3] 
                center[2] = center[2] + atom[4] 
                count += 1
        center[0] = center[0] / count
        center[1] = center[1] / count
        center[2] = center[2] / count

        return np.array(center)

    # Translate current structure so that geometric center coincides with reference
    def _coincide(self, atoms): 
        center = self._center(atoms)
        displacement = self.refer_center - center
        for atom in atoms:
            atom[2] += displacement[0]
            atom[3] += displacement[1]
            atom[4] += displacement[2]
        return atoms

    # Process atomic spatial conformation    
    def _compress_atom(self, atoms, cell, search_clusters):
        # Only extract Si atoms on the root rank
        if self.rank == 0:
            select_atom = []
            for atom in atoms:
                if atom[1] == Stype:
                    select_atom.append(atom)
            if not select_atom:
                positions = np.array([]).reshape(0, 3)
                indices = []
            else:
                positions = np.array([[atom[2], atom[3], atom[4]] for atom in select_atom])
                positions = positions % cell  # Ensure positions are inside the periodic box
                indices = [atom[0] for atom in select_atom]  
                types = [atom[1] for atom in select_atom]
        else:
            positions = None
            indices = None
            types = None
        
        # Broadcast Si atom data to all ranks
        positions = self.comm.bcast(positions, root=0)
        indices = self.comm.bcast(indices, root=0)
        types = self.comm.bcast(types, root=0)
        
        if positions.size == 0:
            return [[]]

        if search_clusters:
            with timer("finding all clusters", self.logger):
                clusters_infos = self._dbscan_all_clusters(positions, cell)

            current_clusters = []
            for center, radius, cluster_indices in clusters_infos:
                current_clusters.append({
                    "center": center, 
                    "radius": radius, 
                    "indices": cluster_indices, 
                    "previous_rmsd": None, 
                    "move": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
                    "fixed_rotate": False, 
                    "iteration_setting": STRICT_ITERTION_SETTING,
                })

            if len(self.clusters) > 0:                
                # Compare current clusters with previous ones to decide whether to inherit metadata
                mapping_infos = self._match_clusters(self.clusters, current_clusters)

                # For unmatched_curr we add new clusters; for matched pairs we inherit from previous clusters,
                # keeping the centroid at the more populated region.
                new_clusters = []
                # Fast index for previous clusters
                prev_clusters_dict = {i: cluster for i, cluster in enumerate(self.clusters)}
                
                # === 1. Handle direct matches ===
                for curr_j, prev_i in mapping_infos["direct_matches"].items():
                    curr_cluster = current_clusters[curr_j]
                    prev_cluster = prev_clusters_dict[prev_i]

                    # Inherit existing metadata
                    curr_cluster["previous_rmsd"] = prev_cluster.get("previous_rmsd")
                    curr_cluster["move"] = prev_cluster.get("move")
                    curr_cluster["fixed_rotate"] = prev_cluster.get("fixed_rotate")
                    curr_cluster["iteration_setting"] = prev_cluster.get("iteration_setting")

                    new_clusters.append(curr_cluster)

                # === 2. Handle unmatched_curr: add as new clusters ===
                for curr_j in mapping_infos["unmatched_curr"]:
                    curr_cluster = current_clusters[curr_j]
                    # Keep default initialization
                    new_clusters.append(curr_cluster)

                # === 3. Handle merged_to: multiple old clusters merged into one ===
                for curr_j, _ in mapping_infos["merged_to"].items():
                    curr_cluster = current_clusters[curr_j]
                    
                    # Reset properties for merged cluster
                    curr_cluster["previous_rmsd"] = None
                    curr_cluster["move"] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                    curr_cluster["fixed_rotate"] = False
                    curr_cluster["iteration_setting"] = STRICT_ITERTION_SETTING

                    new_clusters.append(curr_cluster)

                # === 4. Handle split_from: one old cluster splits into many ===
                for prev_i, curr_list in mapping_infos["split_from"].items():
                    for curr_j in curr_list:
                        curr_cluster = current_clusters[curr_j]

                        # Reset properties for split clusters
                        curr_cluster["previous_rmsd"] = None
                        curr_cluster["move"] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                        curr_cluster["fixed_rotate"] = False
                        curr_cluster["iteration_setting"] = STRICT_ITERTION_SETTING

                        new_clusters.append(curr_cluster)

                # === Update self.clusters ===
                self.clusters = new_clusters
            else:
                self.clusters = current_clusters
                    
        else:
            # Try to keep center as stable as possible on the same grid
            gird_density_center, gird_density_radius = self._grid_density_center(positions, cell)
            dbscan_method_center, dbscan_method_radius = self._dbscan_method_center(positions, cell)

            # Use smaller radius from two methods
            cluster_radius = min(gird_density_radius, dbscan_method_radius) 
 
            # If DBSCAN center is not inside cluster, use density-based center
            if not self.is_inside_cluster(dbscan_method_center, dbscan_method_center, cell, 0.8*cluster_radius):
                cluster_center = gird_density_center
            else:
                cluster_center = dbscan_method_center
            
            # cluster_center, cluster_radius = self._dbscan_method_center(positions, cell)
            if len(self.clusters) > 0:
                # If centroid changes significantly, reset optimization status
                if np.linalg.norm(np.array(self.clusters[0]["center"])-cluster_center) > np.mean(GRID_SIZES):
                    self.clusters[0]["previous_rmsd"] = None
                    self.clusters[0]["move"] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                    self.clusters[0]["fixed_rotate"] = False
                    self.clusters[0]["iteration_setting"] = STRICT_ITERTION_SETTING
                
                self.clusters[0]['center'] = cluster_center
                self.clusters[0]['radius'] = cluster_radius
            else:
                self.clusters = [{
                    "center": cluster_center, 
                    "radius": cluster_radius, 
                    "indices": np.array(range(len(indices))), 
                    "previous_rmsd": None, 
                    "move": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
                    "fixed_rotate": False, 
                    "iteration_setting": STRICT_ITERTION_SETTING
                }]

            self.logger.info(f"Using the center {cluster_center} and radius {cluster_radius}")

        solve_atoms = []
        # Snap each cluster center and radius onto the grid
        for idx, cluster in enumerate(self.clusters):
            cluster_center = cluster['center']
            cluster_radius = cluster['radius']
            cluster_indices = cluster['indices']

            # Snap to grid
            cluster_center = (np.floor(cluster_center / GRID_SIZES) + 0.5) * GRID_SIZES    
            cluster_radius = np.ceil(cluster_radius / np.mean(GRID_SIZES)) * np.mean(GRID_SIZES)
            self.clusters[idx]['center'] = cluster_center
            self.clusters[idx]['radius'] = cluster_radius

            cluster_positions = self._adjust_positions_to_center(positions, cluster_center, cell / 2, cell)        

            solve_atom = []
            for i in cluster_indices:
                solve_atom.append([indices[i], types[i], cluster_positions[i, 0], cluster_positions[i, 1], cluster_positions[i, 2]])
            solve_atoms.append(solve_atom)

            self.logger.info(f"Cluster label {idx}: center = {cluster_center}, radius = {cluster_radius:.4f}, size = {len(cluster_indices)}, setting is {cluster['iteration_setting']}")

        return solve_atoms

    def _adjust_positions_to_center(self, coords, cluster_center, refer_center, cell):
        """
        Adjust atom positions so that the largest cluster is close to the reference center
        """
        shift = refer_center - cluster_center
        adjusted_coords = coords + shift
        adjusted_coords = adjusted_coords % cell
        return adjusted_coords

    # ----------------------
    # main run: compute per-cluster RMSD via C_PIRMSD, build results
    # returns: results (list of (rmsd, move_atom, delta_rmsd) per cluster),
    #          total_rmsd (global CV), total_atoms (weighted)
    # ----------------------
    def run(self, frame, x, types, cell):
        atoms = []  # [id, type, x, y, z]
        self.logger.info(f"Loop {frame} starting")
        for idx, t in enumerate(types):
            pos = np.array(x[3*idx:3*(idx+1)])
            pos -= cell * np.round(pos/cell)
            atoms.append([float(idx+1), float(t), pos[0], pos[1], pos[2]])   
        
        # Find centers and radii of all clusters
        with timer("Compressing atoms", self.logger):
            solve_atoms = self._compress_atom(atoms, cell, self.search_clusters)  # id, type, x, y, z
            solve_atoms = [self._coincide(solve_atom) for solve_atom in solve_atoms]  # move_atom: id, type, vx, vy, vz

        results = []
        S_total = 0.0   # sum over clusters of (prefactor * S_cluster)
        W_total = 0.0   # sum over clusters of (prefactor * W_cluster)
        for i, solve_atom in enumerate(solve_atoms):
            cluster = self.clusters[i]
            iteration_setting = cluster.get("iteration_setting", LOOSE_ITERTION_SETTING)
            with timer(f"Processing cluster {i}", self.logger):
                if self.centerRMSD:
                    rmsd, move_atom, move = C_PIRMSD._BFGS_CENTER(
                        self.refer_structure,
                        solve_atom,
                        iteration_setting[0], iteration_setting[1], iteration_setting[2],
                        cluster.get("move", [0.0]*6),
                        cluster.get("fixed_rotate", False),
                        cluster["radius"] * self.scale_factor,
                        self.alpha,
                        self.comm.py2f()
                    )
                else:
                    rmsd, move_atom, move = C_PIRMSD._BFGS_NORMAL(
                        self.refer_structure,
                        solve_atom,
                        iteration_setting[0], iteration_setting[1], iteration_setting[2],
                        cluster.get("move", [0.0]*6),
                        cluster.get("fixed_rotate", False),
                        self.comm.py2f()
                    )
                cluster["move"] = move
            
            # iteration setting relaxation
            if cluster.get("iteration_setting") == STRICT_ITERTION_SETTING:
                cluster["iteration_setting"] = LOOSE_ITERTION_SETTING

            prev = cluster.get("previous_rmsd")
            delta_rmsd = 1.0 if prev is None else rmsd - prev
            cluster["previous_rmsd"] = rmsd
            
            # fixed rotate threshold
            cluster["fixed_rotate"] = True if rmsd < self.critical_rmsd else False
            
            # compute per-cluster prefactor (sigmoid of number of matched atoms)
            n_atoms_cluster = len(move_atom) if move_atom else 0
            prefactor = self._sigmoid(n_atoms_cluster, len(self.refer_structure), self.beta)

            # compute W_cluster = sum of per-atom weights; assume MOVE_ATOM entries include weight at index 8
            if move_atom:
                # robust handling: some MOVE_ATOM formats may not include weight; try-except
                try:
                    W_cluster = sum(float(entry[8]) for entry in move_atom)
                except Exception:
                    # fallback: use number of matched atoms if weight not present (legacy)
                    W_cluster = float(n_atoms_cluster)
            else:
                W_cluster = 0.0            
            
            # compute S_cluster from returned rmsd (rmsd^2 = S_cluster / W_cluster => S_cluster = rmsd^2 * W_cluster)
            S_cluster = (rmsd * rmsd) * W_cluster

            # aggregate (apply cluster-level prefactor)
            S_total += prefactor * S_cluster
            W_total += prefactor * W_cluster

            cluster['prefactor'] = prefactor
            cluster['W_cluster'] = W_cluster

            # keep results but do NOT modify move_atom in-place            
            results.append((rmsd, move_atom, delta_rmsd))
            
            self.clusters[i] = cluster
            self.logger.info(f"Loop {frame} cluster {i}: rmsd={rmsd:.6e}, ΔRMSD={delta_rmsd:.6e}, n_atoms={n_atoms_cluster}, prefactor={prefactor:.3f}, W_cluster={W_cluster:.6e}")

        # finalize global weighted rmsd (xi)
        W_total_safe = max(W_total, 1e-30)
        total_rmsd = math.sqrt(S_total / W_total_safe)

        if self.rank == 0:
            with open(self.logfile, 'a') as logout:
                print(f'The {frame} is finished, the RMSD is {total_rmsd}, the total weighted atoms is {W_total_safe}, the xi_min is {self.global_xi_min}', flush=True, file=logout)

        return results, total_rmsd, W_total_safe

if __name__ == "__main__":
    setting_file = sys.argv[1]
    lmpfile = "lmp.in"
    lmp_relaxfile = "lmp.relax.in"
    params_file = "params.json"
    log_file = "PIRMSD.log"

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    with open(params_file, 'r', encoding='utf-8') as f:
        params = json.load(f)
    simulation_parameters = params.get('Simulation_Parameters', {})
    INIT_ITER = simulation_parameters.get('init_iter', 0)
    INIT_TIME = simulation_parameters.get('init_time', 0.0)
    TIME_STEP = simulation_parameters.get('time_step', 0.5)
    thermodynamic_parameters = params.get('Thermodynamic_Parameters', {})
    T = thermodynamic_parameters.get('temperature', 453.0)

    with open(setting_file, 'r',encoding='utf-8') as rf:
        setting = json.load(rf)
    global_k = setting.get('k', 25.0)
    md_steps = setting.get("md_steps", 10000)
    niterations = setting.get('niterations', 500)
    relax_steps = setting.get('relax_steps', 2000)
    rmsd_scale = setting.get('rmsd_scale', 1.0)
    output_iteration = setting.get('output_iteration', 10)

    lmp_es_out_dir = Path(f"./PIRMSD_{INIT_ITER}")
    lmp_es_out_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        filename=lmp_es_out_dir / log_file,           # log file name
        level=logging.INFO,          # record level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # log format
        datefmt='%Y-%m-%d %H:%M:%S'  # time format
    )
    logger = logging.getLogger(f"PIRMSD_{INIT_ITER}")
    logger.setLevel(logging.INFO if rank == 0 else logging.WARNING)

    pirmsd = PIRMSD(comm, setting, logger)
    pirmsd.temperature = setting.get('temperature', T)
    pirmsd.bias_k = setting.get('bias_k', global_k)
    pirmsd.touch_logfile(lmp_es_out_dir)
    
    # try to resume
    previous_dir = Path(f"./PIRMSD_{int(INIT_ITER-1)}")
    start_iter = pirmsd.get_begin_iteration(previous_dir)

    # initialize LAMMPS and dumps
    lmp = lammps("mpi")
    lines = open(lmpfile,'r').readlines()
    for line in lines: 
        line = line.replace('${T}', str(T))                                 
        line = line.replace('${TIME_STEP}', str(TIME_STEP))    
        lmp.command(line)

    lmp_pirmsd_out_dump = lmp_es_out_dir / "lmp_pirmsd.lammpstrj"
    lmp.command(f"dump      1 all custom {md_steps*output_iteration} {lmp_pirmsd_out_dump} id mol type q xu yu zu vx vy vz") 
    lmp.command("dump_modify 1 sort id delay 1 pbc yes")  

    # prepare relax input file (root)
    current_time = INIT_TIME + md_steps*niterations*TIME_STEP*1.0E-6 # ns 
    if rank == 0:
        lines = open(lmp_relaxfile,'r').readlines()
        with open(f'lmp_{INIT_ITER}.relax.in', 'w') as wf:
            for line in lines: 
                line = line.replace('${T}', str(T))                                   
                line = line.replace('${TIME_STEP}', str(TIME_STEP))   
                line = line.replace('${relax_steps}', str(relax_steps))
                line = line.replace('${current_time}', f"{current_time:.4f}")   
                wf.write(line)

    # main sampling loop
    for iteration in range(start_iter, niterations):
        logger.info(f"Starting PIRMSD iteration {iteration}/{niterations}")
        nlocal = lmp.extract_global("nlocal")
        #nghost = lmp.extract_global("nghost")  
        x = lmp.gather_atoms("x", 1, 3)    
        global_types = lmp.gather_atoms("type", 0, 1)

        box = lmp.extract_box()   
        lx = box[1][0] - box[0][0] 
        ly = box[1][1] - box[0][1] 
        lz = box[1][2] - box[0][2] 
        cell = np.array([lx, ly, lz])

        # compute per-cluster RMSD and global total_rmsd
        results, total_rmsd, W_total = pirmsd.run(iteration, x, global_types, cell)

        # save clusters for restart occasionally
        if output_iteration > 0 and (iteration % output_iteration == 0 or iteration == niterations - 1): 
            pirmsd.write_clusters_infos(lmp_es_out_dir / f"cluster_{iteration}.json")

        # Ratcheting decision uses ONLY the global CV = total_rmsd
        # Broadcast a random number for Metropolis test
        if rank == 0:
            probability = np.random.random()
        else:
            probability = 0.0
        probability = comm.bcast(probability, root=0)        

        # compute Metropolis criterion based on Vb(ξ) = 1/2 k (ξ - ξ_min)^2
        xi_current = total_rmsd
        xi_old = pirmsd.previous_total_rmsd if hasattr(pirmsd, 'previous_total_rmsd') else xi_current
        if pirmsd.global_xi_min is None:
            pirmsd.global_xi_min = xi_current
            apply_bias = True
            reason = "init_global_xi_min"
            deltaVb = 0.0
        else:
            if total_rmsd < pirmsd.global_xi_min:
                pirmsd.global_xi_min = xi_current
                apply_bias = True
                reason = "downhill_new_min"
                deltaVb = 0.0
            else:
                bias_k = pirmsd.bias_k
                Vb_new = 0.5 * bias_k * (xi_current - pirmsd.global_xi_min)**2
                Vb_old = 0.5 * bias_k * (xi_old - pirmsd.global_xi_min)**2
                deltaVb = Vb_new - Vb_old

                if pirmsd.temperature and pirmsd.temperature > 0:
                    beta = 1.0 / (kB * pirmsd.temperature)
                else:
                    beta = 1.0

                # Metropolis decision: only need exp when deltaVb > 0; otherwise always accept
                if deltaVb <= 0.0:
                    apply_bias = True
                    reason = "downhill_apply_bias"
                else:
                    # Only compute probability for uphill moves (numerical stability)
                    # In parallel environments, u should be generated on rank0 and broadcast; here is a single-node example:
                    w = math.exp(-beta * deltaVb)  # 0 < w < 1
                    accept = (probability < w)

                    # draw/broadcasted probability 'probability' in [0,1)
                    # Metropolis decision:
                    #   if u < w -> accept uphill move (do NOT apply bias)
                    #   else     -> reject uphill move (apply bias to resist the uphill)
                    w = math.exp(-beta * deltaVb) 
                    if accept:
                        apply_bias = False           
                        reason = "uphill_accepted_no_bias"
                    else:
                        apply_bias = True
                        reason = "uphill_rejected_apply_bias"
        logger.info(f"apply bias : {apply_bias}, reason is {reason}, deltaVb is {deltaVb}, w is {math.exp(-beta * deltaVb)}, probability is {probability}\n")

        # update previous total rmsd for next iteration's old_xi       
        pirmsd.previous_total_rmsd = xi_current

        # If ratcheting decides to apply bias globally, we compute and apply cluster-specific bias contributions.
        # IMPORTANT: decision is global but forces are cluster-specific.
        if apply_bias and results:
            xi_ref = pirmsd.global_xi_min if pirmsd.global_xi_min is not None else xi_current
            k_bias = pirmsd.bias_k

            # Scalar term: - k*(xi - xi_ref)
            # This is the force coefficient on the reaction coordinate (multiplied by ∂ξ/∂x_i to get force on each atom)
            scalar_global  = - k_bias * (xi_current - xi_ref)
            dt_total = TIME_STEP * md_steps
            
            # Precompute denom using global xi and aggregated W_total (from above)
            Wtot = max(W_total, 1e-30)
            denom_global = 2.0 * xi_current * Wtot
            logger.info(f"Added bias force with strength {scalar_global/denom_global} ")
            
            tags = lmp.extract_atom('id')
            # Except for the “mass” property, the underlying storage will always be dimensioned for the range [0:nmax]. -> mass[tags[idx]]
            # the actual usable data may be only in the range [0:nlocal] or [0:nlocal][0:dim]. 
            mass = lmp.numpy.extract_atom('mass')
            local_types = lmp.extract_atom('type')
            v = lmp.extract_atom("v")

            # For each cluster, compute a cluster-specific scalar from the bias potential.
            for icluster, (cluster_rmsd, move_atom, delta_rmsd) in enumerate(results):
                cluster = pirmsd.clusters[icluster]
                pref_c = cluster.get('prefactor', 1.0)
                W_cluster = cluster.get('W_cluster', None)
        
                for idx in range(nlocal):
                    atom_i = tags[idx]
                    type_i = local_types[idx]
 
                    if type_i != Stype:
                        continue
                    # find atom in move_atom list for this cluster
                    entry = next((item for item in move_atom if int(item[0]) == atom_i), None)
                    if entry is None:
                        continue

                    # entry layout: [id, type, dSx, dSy, dSz, dWx, dWy, dWz, weight]
                    dS = np.array(entry[2:5], dtype=float)    # dS/dx, dS/dy, dS/dz
                    dW = np.array(entry[5:8], dtype=float)    # dW/dx, dW/dy, dW/dz

                    # gradient of global xi: prefactor multiplies numerator
                    grad_xi = pref_c * (dS - (xi_current*xi_current) * dW) / denom_global

                    # Force on atom = scalar_global * grad_xi
                    force_vec = scalar_global * grad_xi

                    # Convert to velocity impulse: Δv = (F/m) * dt_total
                    mass_i = mass[type_i]

                    delta_v = (force_vec / mass_i) * dt_total
                    # logging.error(f"{atom_i}: {idx_atom}")
                    v[idx][0] += delta_v[0]   
                    v[idx][1] += delta_v[1] 
                    v[idx][2] += delta_v[2]                      
                            
        lmp.command(f"run {md_steps}")

    lmp_es_out_dump = lmp_es_out_dir/"lmp_es.lammpstrj"
    lmp.command(f"dump        	es all custom 1 {lmp_es_out_dump} id mol type q xu yu zu vx vy vz") 
    lmp.command("dump_modify 	es sort id delay 1 pbc yes")
    lmp.command("run 0")
    lmp.command("undump 	    es ")        
    lmp.command("write_data lmp.data nocoeff")
