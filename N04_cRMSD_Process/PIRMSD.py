import numpy as np
from pathlib import Path
from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from pirmsd_numerics import centered_gradient
from collections import defaultdict
import math
import sys, json, re
from lammps import lammps
from mpi4py import MPI
import logging
import uuid
from ratchet_state import RatchetState, fingerprint, commit_bundle, load_bundle, restart_input
from matching_domains import DomainTracker, make_tracker
from ratchet_force import RatchetForce, orthogonal_box
from mpi_ratchet_force import DistributedRatchetForce

try:
    import C_PIRMSD
    if getattr(C_PIRMSD, "API_VERSION", 0) < 3:
        raise ImportError("Stale C_PIRMSD binary; rebuild with this Python interpreter")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    print("Please run: python setup.py build_ext --inplace first")
    exit(1)

"""
PIRMSD_<segment>/:
  latest.json + checkpoint_*.restart[.json]  # synchronized restart bundles
  lmp_pirmsd_<start_iter>_<uuid>.lammpstrj    # analysis: id mol type xu yu zu
  history.log                              # committed metric/bias observations
  PIRMSD.log, rienv_references/             # diagnostics and active RIEnv templates
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
        boundary = settings.get('boundary_score', {})
        self.boundary_score = boundary.get('status', True)
        self.mismatch_cutoff = float(boundary.get('mismatch_cutoff', 2.5))
        self.critical_score = float(boundary.get('critical_score', 0.2))
        if not np.isfinite(self.mismatch_cutoff) or self.mismatch_cutoff <= 0:
            raise ValueError('boundary_score.mismatch_cutoff must be positive and finite')
        if not 0 <= self.critical_score <= 1:
            raise ValueError('boundary_score.critical_score must be in [0, 1]')
        self.full_reference = [row for row in self.refer_structure if int(row[1]) == Stype]
        self.boundary_atom_count = None
        domain_settings = dict(settings.get('matching_domains', {}))
        self.use_matching_domains = self.boundary_score and domain_settings.pop('status', True)
        voronoi = domain_settings.get('core_voronoi', {})
        if self.use_matching_domains and voronoi.get('status') and voronoi.get('mode', 'persistent') == 'persistent':
            minimum_size = settings.get('cluster_min_size_exclusive', 50)
            if isinstance(minimum_size, bool) or not isinstance(minimum_size, int) or minimum_size < 0:
                raise ValueError('cluster_min_size_exclusive must be a nonnegative integer')
            domain_settings['cluster_pirmsd'] = dict(min_size_exclusive=minimum_size,
                beta=float(settings.get('beta', 5.)), reference_size=len(self.refer_structure))
        self.domain_tracker = make_tracker(domain_settings)
        self.reference_transition_active = False

        center_settings = settings.get('center', {})
        self.centerRMSD = center_settings.get('status', False)
        self.alpha = center_settings.get('alpha', 0.25)    
        self.scale_factor = center_settings.get('scale_factor', settings.get('scale_factor', 0.9))

        self.search_clusters = settings.get('search_clusters', False)
        self.beta = settings.get('beta', 5.0)

        # ratchet / bias defaults
        self.bias_k = settings.get('bias_k', settings.get('k', 25.0)) 
        self.mc_bias_probability_floor = settings.get('mc_bias_probability_floor', 0.)
        self.bias_potential = settings.get('bias_potential', 'quadratic')
        algorithm = settings.get('bias_algorithm')
        if algorithm is not None and algorithm not in ('continuous', 'ratchet'):
            raise ValueError('bias_algorithm must be continuous or ratchet')
        self.ratchet_enabled = settings.get('ratchet_enabled', algorithm != 'continuous')
        if not isinstance(self.ratchet_enabled, bool):
            raise ValueError('ratchet_enabled must be boolean')
        if algorithm is not None and self.ratchet_enabled != (algorithm == 'ratchet'):
            raise ValueError('bias_algorithm conflicts with ratchet_enabled')
        self.bias_algorithm = 'ratchet' if self.ratchet_enabled else 'continuous'
        if self.bias_potential not in ('quadratic', 'linear'):
            raise ValueError('bias_potential must be quadratic or linear')
        self.temperature = settings.get('temperature', None)

        # per-cluster data structure (list of dicts)
        # each cluster dict contains: center, radius, indices, previous_rmsd, move, fixed_rotate, iteration_setting, xi_min, bias_k
        self.clusters = [] 
        self.frame = 0
        self.start_frame = 0

        # historical global total RMSD for ratchet decision (not per-cluster)
        self.previous_total_rmsd = None
        self.global_xi_min = None  # mirror of the active ratchet epoch
        self.ratchet = RatchetState()
        self.metric_context = None
        self.reference_signature = fingerprint(self.refer_structure)

    # logging file handle path
    def touch_logfile(self, base_dir):
        self.logfile = base_dir / 'history.log'
        self.logger.info(f"History log will be written to {self.logfile}")

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
        if len(positions) == 0:
            return []
        # Store only radius neighbors, instead of allocating an N x N x 3 tensor.
        wrapped = np.asarray(positions) % cell
        tree = cKDTree(wrapped, boxsize=cell)
        neighbors = tree.query_ball_point(wrapped, eps)
        rows = np.repeat(np.arange(len(wrapped)), [len(n) for n in neighbors])
        cols = np.concatenate(neighbors).astype(int)
        delta = wrapped[rows] - wrapped[cols]
        delta -= cell * np.round(delta / cell)
        distances = np.maximum(np.linalg.norm(delta, axis=1), np.finfo(float).eps)
        distance_matrix = csr_matrix((distances, (rows, cols)), shape=(len(wrapped), len(wrapped)))
        # sklearn expects each sparse row sorted by distance, not column number.
        for i in range(len(wrapped)):
            a, b = distance_matrix.indptr[i:i+2]
            order = np.argsort(distance_matrix.data[a:b], kind='stable')
            distance_matrix.data[a:b] = distance_matrix.data[a:b][order]
            distance_matrix.indices[a:b] = distance_matrix.indices[a:b][order]
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
            return []  # No valid cluster found

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

        n_grids = (2 ** np.ceil(np.log2(np.maximum(1, cell / GRID_SIZES)))).astype(int)
        spacing = cell / n_grids
        
        # Initialize 3D grid counts
        grid_counts = np.zeros(n_grids)
        
        # Count atoms in each grid
        for coord in positions:
            normalized_coord = coord % cell
            grid_indices = (normalized_coord / spacing).astype(int)
            grid_indices = np.minimum(grid_indices, n_grids - 1)
            grid_counts[grid_indices[0], grid_indices[1], grid_indices[2]] += 1
        
        # Create Gaussian kernel
        cluster_radius = min(cell) * 0.05
        sigma_grids = cluster_radius / spacing
        
        # Create Gaussian kernel in frequency domain
        kernel = self._create_3d_gaussian_kernel_fft(n_grids, sigma_grids)
        
        # Convolution via FFT (automatically handles periodic boundaries)
        grid_fft = np.fft.fftn(grid_counts)
        kernel_fft = kernel
        convolved_fft = grid_fft * kernel_fft
        density = np.real(np.fft.ifftn(convolved_fft))

        # Find position of maximum density - use centroid of high density instead of simple max
        max_density = np.max(density)
        high_density_mask = density >= density_threshold * max_density

        # Use the maximum density grid as cluster center
        max_density_idx = np.unravel_index(np.argmax(density), density.shape)
        center_grid = np.array(max_density_idx, dtype=float)

        # Convert to real-space center
        center = (center_grid + 0.5) * spacing
        center = center % cell

        cluster_volume_grids = np.sum(high_density_mask)
        grid_volume = np.prod(spacing)
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
                types = []
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
            self.clusters = []
            return []

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
            if dbscan_method_center is None or not self.is_inside_cluster(dbscan_method_center, gird_density_center, cell, 0.8*cluster_radius):
                cluster_center = gird_density_center
            else:
                cluster_center = dbscan_method_center
            
            # cluster_center, cluster_radius = self._dbscan_method_center(positions, cell)
            if len(self.clusters) > 0:
                # If centroid changes significantly, reset optimization status
                delta_center = np.asarray(self.clusters[0]["center"]) - cluster_center
                delta_center -= cell * np.round(delta_center / cell)
                if np.linalg.norm(delta_center) > np.mean(GRID_SIZES):
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

            # Keep the measured center/radius; grid snapping creates artificial jumps.
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
    def run(self, frame, x, types, cell, rienv_scores=None):
        boundary = getattr(self, 'boundary_score', False)
        n_total = sum(int(t) == Stype for t in types)
        if boundary and n_total:
            if n_total > len(self.full_reference):
                raise ValueError('Boundary PIRMSD needs at least as many reference type-1 sites as simulated type-1 atoms; supply a larger reference')
            if self.boundary_atom_count != n_total:
                # Fixed across clusters/frames; never shrink to a fragment's size.
                self.refer_structure = self.full_reference[:n_total]
                self.refer_center = self._center(self.refer_structure)
                self.boundary_atom_count = n_total
        if boundary and getattr(self, 'use_matching_domains', False):
            return self._run_domains(frame, x, types, cell, rienv_scores)
        atoms = []  # [id, type, x, y, z]
        self.logger.info(f"Loop {frame} starting")
        for idx, t in enumerate(types):
            pos = np.array(x[3*idx:3*(idx+1)], dtype=float)
            pos -= cell * np.round(pos/cell)
            atoms.append([float(idx+1), float(t), pos[0], pos[1], pos[2]])   
        
        # Find centers and radii of all clusters
        with timer("Compressing atoms", self.logger):
            solve_atoms = self._compress_atom(atoms, cell, self.search_clusters)  # id, type, x, y, z
            solve_atoms = [self._coincide(solve_atom) for solve_atom in solve_atoms]  # move_atom: id, type, vx, vy, vz

        results = []
        S_total = 0.0   # sum over clusters of (prefactor * S_cluster)
        W_total = 0.0   # sum over clusters of (prefactor * W_cluster)
        contexts = []
        if boundary:
            if not n_total:
                self.metric_context = None
                return [], None, 0.0
            # All excluded/noise atoms have unit cost in the global population.
            S_total = float(n_total - sum(len(atoms) for atoms in solve_atoms))
            W_total = float(n_total)
        for i, solve_atom in enumerate(solve_atoms):
            cluster = self.clusters[i]
            iteration_setting = cluster.get("iteration_setting", LOOSE_ITERTION_SETTING)
            with timer(f"Processing cluster {i}", self.logger):
                if boundary:
                    rmsd, move_atom, move = C_PIRMSD._BFGS_BOUNDARY(
                        self.refer_structure, solve_atom,
                        iteration_setting[0], iteration_setting[1], iteration_setting[2],
                        cluster.get("move", [0.0]*6), cluster.get("fixed_rotate", False),
                        self.mismatch_cutoff, self.comm.py2f())
                elif self.centerRMSD:
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
            threshold = self.critical_score if boundary else self.critical_rmsd
            cluster["fixed_rotate"] = bool(self.fixed_rotate and rmsd < threshold) if boundary else rmsd < threshold
            
            n_atoms_cluster = len(move_atom)
            if boundary:
                # L_c = (missing slots + matched bounded losses) / N.
                # xi^2 = noise/N + sum_c (n_c/N)*L_c.
                prefactor = len(solve_atom) / n_total
                W_cluster = float(n_total)
                S_total += prefactor * rmsd*rmsd * W_cluster
            else:
                prefactor = self._sigmoid(n_atoms_cluster, len(self.refer_structure), self.beta)
                W_cluster = sum(float(entry[8]) for entry in move_atom)
                S_total += prefactor * rmsd*rmsd * W_cluster
                W_total += prefactor * W_cluster
            cluster['prefactor'] = prefactor
            cluster['W_cluster'] = W_cluster

            # Ordered matched IDs encode the greedy reference-site assignment.
            # Weight values vary with coordinates, but their defining radius/alpha
            # must stay fixed for historical values to be comparable.
            relative = np.asarray(solve_atom)[:, 2:5] - np.asarray(solve_atom)[0, 2:5]
            image_chart = np.rint(relative / cell).astype(int).tolist()
            contexts.append(dict(
                image_chart=image_chart,
                members=sorted(int(atom[0]) for atom in solve_atom),
                mapping=([(int(entry[9]), int(entry[0])) for entry in move_atom] if boundary else [int(entry[0]) for entry in move_atom]),
                move=list(move), radius=float(cluster['radius']*self.scale_factor) if self.centerRMSD and not boundary else None,
                alpha=float(self.alpha) if self.centerRMSD and not boundary else None,
                prefactor=float(prefactor)))
            # Chain rule for _coincide: every atom contributes to the moving centroid.
            move_atom = centered_gradient(move_atom, solve_atom)
            results.append((rmsd, move_atom, delta_rmsd))
            
            self.clusters[i] = cluster
            self.logger.info(f"Loop {frame} cluster {i}: rmsd={rmsd:.6e}, ΔRMSD={delta_rmsd:.6e}, n_atoms={n_atoms_cluster}, prefactor={prefactor:.3f}, W_cluster={W_cluster:.6e}")

        # No observations is an undefined metric, never a perfect match.
        if (not boundary and not results) or W_total <= 1e-30 or not np.isfinite([S_total, W_total]).all():
            self.metric_context = None
            return [], None, 0.0
        total_rmsd = math.sqrt(S_total / W_total)
        contexts.sort(key=lambda context: context['members'])
        self.metric_context = fingerprint(dict(reference=self.reference_signature, cell=np.asarray(cell).tolist(), clusters=contexts,
                                               metric='boundary_v1' if boundary else 'legacy_rmsd',
                                               population=n_total if boundary else None,
                                               cutoff=self.mismatch_cutoff if boundary else None))
        return results, total_rmsd, W_total

    def _run_domains(self, frame, x, types, cell, rienv_scores=None):
        xyz = np.asarray(x, dtype=float).reshape(-1, 3)
        population_ids = np.flatnonzero(np.asarray(types) == Stype)
        n = len(population_ids)
        if not n:
            self.metric_context = None
            self.reference_transition_active = False
            return [], None, 0.0
        if not np.isfinite(xyz).all():
            raise ValueError('Nonfinite coordinates in matching domains')
        if self.domain_tracker.settings.get('cluster_pirmsd'):
            components = error = None
            if self.rank == 0:
                try:
                    detected = self._dbscan_all_clusters(xyz[population_ids], cell)
                    components = [population_ids[c[2]].tolist() for c in detected]
                except Exception as exc:
                    error = str(exc)
            error, components = self.comm.bcast((error, components), root=0)
            if error is not None:
                raise RuntimeError('Physical cluster detection failed: ' + error)
        elif self.domain_tracker.settings.get('core_voronoi',{}).get('status',False):
            components = [population_ids.tolist()]  # No DBSCAN exclusion of free particles.
        elif self.search_clusters:
            detected = self._dbscan_all_clusters(xyz[population_ids], cell)
            components = [population_ids[np.asarray(c[2], dtype=int)].tolist() for c in detected]
        else:
            components = [population_ids.tolist()]
        core_settings = self.domain_tracker.core_selector.settings
        if core_settings['status']:
            interval = core_settings.get('sample_interval', 10)
            if int(interval) != interval or interval < 1:
                raise ValueError('RIEnv sample_interval must be a positive integer')
            payload = None
            if self.rank == 0:
                try:
                    scores = rienv_scores
                    if scores is None and (not self.domain_tracker.core_selector.frames or frame % interval == 0):
                        from rienv_provider import RIEnvProvider
                        if not hasattr(self, '_rienv_provider'):
                            directory = self.logfile.parent / 'rienv_references' if hasattr(self, 'logfile') else None
                            self._rienv_provider = RIEnvProvider(self.full_reference,
                                core_settings.get('provider'), directory)
                        scores = self._rienv_provider.compute(xyz[population_ids], np.asarray(cell))
                    payload = (None, scores)
                except Exception as exc:
                    payload = (str(exc), None)
            error, rienv_scores = self.comm.bcast(payload, root=0)
            if error:
                raise RuntimeError('RIEnv core evaluation failed: ' + error)
        self.domain_tracker.update(frame, xyz, cell, population_ids.tolist(), components,
                                   self.refer_structure, self.full_reference, self.mismatch_cutoff,
                                   C_PIRMSD, self.comm, LOOSE_ITERTION_SETTING, rienv_scores=rienv_scores)
        results, score, weight, self.clusters = self.domain_tracker.evaluate(xyz, cell, n, self.mismatch_cutoff)
        self.metric_context = self.domain_tracker.context(self.reference_signature, cell, self.mismatch_cutoff)
        self.reference_transition_active = self.domain_tracker.transition is not None
        self.logger.info('Spatial clusters=%s matching domains=%s transition=%s mix=%.6f event=%s protocol_score_work=%.9g',
                         len(components), len(self.domain_tracker.domains), self.reference_transition_active,
                         self.domain_tracker.last_mix, self.domain_tracker.last_event,
                         self.domain_tracker.protocol_score_work)
        return results, score, weight

    def record_bias_iteration(self, iteration, xyz, box):
        latest = self.force_driver.cache
        tracker = self.domain_tracker
        score = tracker.evaluate_global(xyz, box, len(tracker.population_ids), self.mismatch_cutoff)
        incoming = (tracker.transition or {}).get('physical_clusters', tracker.physical_clusters)
        with self.logfile.open('a') as stream:
            print(f"The {iteration} is finished, the RMSD is {latest['xi']}, "
                  f"the total weighted atoms is {latest['weight']}, the xi_min is {self.global_xi_min}, "
                  f"valid={latest['decision']['valid']}, epoch={self.ratchet.epoch}, "
                  f"reason={latest['decision']['reason']}, md_step={latest['key'][0]}, "
                  f"bias_energy={latest['energy']}, metric={tracker.metric_name}, "
                  f"cluster_eligible_Si={sum(map(len, tracker.physical_clusters))}, "
                  f"cluster_target_Si={sum(map(len, incoming))}, cluster_pirmsd_A={latest['xi']}, "
                  f"multidomain_pirmsd={score}", file=stream)

    def checkpoint(self, lmp, directory, next_iteration, rng, signature):
        """Collective LAMMPS write followed by a root-only atomic commit."""
        token = self.comm.bcast(uuid.uuid4().hex if self.rank == 0 else None, root=0)
        restart = Path(directory) / ('checkpoint_{}_{}.restart'.format(next_iteration, token))
        lmp.command('write_restart "{}"'.format(restart.resolve()))
        # write_restart can rebuild distributed mappings. Refresh before continuing.
        lmp.command('run 0')
        self.comm.Barrier()
        if hasattr(self, 'force_driver'):self.force_driver.check()
        self.commit_restart(directory, restart, next_iteration, rng, signature)

    def commit_restart(self, directory, restart, next_iteration, rng, signature):
        error = None
        if self.rank == 0:
            try:
                commit_bundle(directory, restart, dict(
                    signature=signature, next_iteration=next_iteration,
                    clusters=self.array_to_list(self.clusters), ratchet=self.ratchet.to_dict(),
                    matching_domains=self.domain_tracker.state() if self.use_matching_domains else None,
                    force_driver=self.force_driver.state() if hasattr(self, 'force_driver') else None,
                    rng=rng.bit_generator.state,
                    history_bytes=self.logfile.stat().st_size if self.logfile.exists() else 0))
            except Exception as exc:
                error = str(exc)
        error = self.comm.bcast(error, root=0)
        if error:
            raise RuntimeError('Checkpoint commit failed: '+error)


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
    branch_interval = setting.get('branch_update_steps')
    if 'branch_update_steps' in setting:
        for name, value in [('branch_update_steps', branch_interval),
                            ('md_steps', md_steps), ('niterations', niterations)]:
            if type(value) is not int or value < 1:
                raise ValueError(f'{name} must be a positive integer for single-run integration')
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

    # Both production bias algorithms use the persistent physical-cluster CV.
    # Older analysis backends remain available through the Python scoring API.
    domain_options = setting.setdefault('matching_domains', {})
    domain_options.setdefault('core_voronoi', {}).setdefault('status', True)
    pirmsd = PIRMSD(comm, setting, logger)
    if not pirmsd.domain_tracker.settings.get('cluster_pirmsd'):
        raise ValueError('Cluster PIRMSD bias requires boundary_score and persistent core_voronoi matching domains')
    pirmsd.temperature = setting.get('temperature', T)
    pirmsd.bias_k = setting.get('bias_k', global_k)
    pirmsd.touch_logfile(lmp_es_out_dir)
    
    # Only a committed bundle can resume the same sampling segment. A new major
    # iteration starts a fresh ratchet instead of inheriting unmatched old logs.
    lines = Path(lmpfile).read_text().splitlines(keepends=True)
    signature = fingerprint(dict(metric_version=17 if pirmsd.domain_tracker.settings.get('cluster_pirmsd') else 16, bias_integrator='single_run_external_v1' if branch_interval is not None else 'per_step_external_v1', settings=setting, timestep=TIME_STEP, temperature=T,
                                 niterations=niterations, mpi_size=size,
                                 reference=pirmsd.reference_signature, lmp_input=lines))
    checkpoint = load_bundle(lmp_es_out_dir, signature)
    rng = np.random.default_rng(setting.get('seed'))
    start_iter = 0
    if checkpoint is not None:
        start_iter = checkpoint['next_iteration']
        pirmsd.clusters = checkpoint['clusters']
        for cluster in pirmsd.clusters:
            if 'center' in cluster:
                cluster['center'] = np.asarray(cluster['center'])
            if 'indices' in cluster:
                cluster['indices'] = np.asarray(cluster['indices'], dtype=int)
        if pirmsd.use_matching_domains:
            if checkpoint.get('matching_domains') is None:
                raise ValueError('Missing persistent matching-domain checkpoint state')
            pirmsd.domain_tracker = make_tracker(pirmsd.domain_tracker.settings, checkpoint['matching_domains'])
        pirmsd.ratchet = RatchetState(**checkpoint['ratchet'])
        pirmsd.global_xi_min = pirmsd.ratchet.minimum
        pirmsd.previous_total_rmsd = pirmsd.ratchet.previous
        rng.bit_generator.state = checkpoint['rng']
        lines = restart_input(lines, lmp_es_out_dir/checkpoint['restart'])
        error = None
        if rank == 0:
            try:
                offset = checkpoint['history_bytes']
                if offset and (not pirmsd.logfile.exists() or pirmsd.logfile.stat().st_size < offset):
                    raise ValueError('Committed history is missing or truncated')
                with pirmsd.logfile.open('a+b') as stream:
                    stream.truncate(offset)
            except Exception as exc:
                error = str(exc)
        error = comm.bcast(error, root=0)
        if error:
            raise RuntimeError(error)
    elif pirmsd.logfile.exists() and pirmsd.logfile.stat().st_size:
        raise RuntimeError('Legacy history has no synchronized checkpoint; use a new PIRMSD directory')
    comm.Barrier()
    pirmsd.force_driver = DistributedRatchetForce(pirmsd, rng, checkpoint.get('force_driver') if checkpoint else None)
    pirmsd.force_driver.enabled = False
    lmp = lammps("mpi", comm=comm)
    bias_attached = False
    for line in lines:
        line = line.replace('${T}', str(T)).replace('${TIME_STEP}', str(TIME_STEP))
        if line.strip().startswith('fix ') or line.strip().startswith('fix\t'):
            if not bias_attached:
                pirmsd.force_driver.attach(lmp)
                bias_attached = True
        lmp.command(line)
    if not bias_attached:pirmsd.force_driver.attach(lmp)
    lmp.command('run 0')
    if checkpoint is None:
        pirmsd.checkpoint(lmp, lmp_es_out_dir, 0, rng, signature)

    dump_token = comm.bcast(uuid.uuid4().hex if rank == 0 else None, root=0)
    lmp_pirmsd_out_dump = lmp_es_out_dir / f"lmp_pirmsd_{start_iter}_{dump_token}.lammpstrj"
    lmp.command(f"dump      1 all custom {md_steps*max(1, output_iteration)} {lmp_pirmsd_out_dump} id mol type xu yu zu")
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

    pirmsd.force_driver.enabled = True

    if branch_interval is not None:
        from single_run_bias import run_bias_segment
        run_bias_segment(pirmsd, lmp, lmp_es_out_dir, start_iter, niterations,
                         md_steps, output_iteration, branch_interval, rng, signature)
    else:
        # main sampling loop
        for iteration in range(start_iter, niterations):
            logger.info(f"Starting PIRMSD iteration {iteration}/{niterations}")
            x = lmp.gather_atoms("x", 1, 3)
            global_types = lmp.gather_atoms("type", 0, 1)
            pirmsd.run(iteration, x, global_types, orthogonal_box(lmp))

            lmp.command(f"run {md_steps}")
            pirmsd.force_driver.check()
            final_xyz = np.asarray(lmp.gather_atoms('x', 1, 3)).reshape(-1, 3)
            boxlo, boxhi = lmp.extract_box()[:2]
            if rank == 0:
                pirmsd.record_bias_iteration(iteration, final_xyz, np.subtract(boxhi, boxlo))
            if (iteration+1) % max(1, output_iteration) == 0 or iteration == niterations-1:
                pirmsd.checkpoint(lmp, lmp_es_out_dir, iteration+1, rng, signature)

    lmp.command('unfix '+pirmsd.force_driver.fix_id)
    # The synchronized checkpoint retains the full MD state; lmp.data is
    # required by subsequent relaxation/MD. No duplicate terminal trajectory.
    lmp.command("write_data lmp.data nocoeff")
