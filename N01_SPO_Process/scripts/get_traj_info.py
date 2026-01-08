# %%
import sys, os
import MDAnalysis as mda
from MDAnalysis.analysis import distances, rdf
from MDAnalysis.lib.distances import minimize_vectors
import numpy as np
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
from scipy.integrate import simpson
from scipy.ndimage.filters import median_filter
from scipy.sparse import csr_matrix, lil_matrix, triu
from scipy import interpolate
import tqdm
import matplotlib.pyplot as plt
from multiprocessing import Pool
from functools import partial 
import itertools
import csv
from pathlib import Path

# %% [markdown]
# ### scipy.interpolate - find roots of discrete data & intersections of two discrete curves
# https://zhuanlan.zhihu.com/p/358435456

# %%
def numpy_scipy_find_roots_by_XY(X, Y):
    # X must be strictly monotonically increasing
    if np.all(np.diff(X) > 0) == True:
        pass
    else:
        raise Exception('X must be strictly monotonically increasing!')

    # Build spline curve from discrete XY data
    # BSpline(builtins.object)
    # tck: A spline, as returned by `splrep` or a BSpline object.
    tck = interpolate.make_interp_spline(x=X, y=Y, k=1) 

    # tuple(object)
    # tck: A spline, as returned by `splrep` or a BSpline object.
    # tck = interpolate.splrep(x=X, y=Y, k=1, s=0) 
    # k is the order of the spline, k=1 linear, k=2 quadratic polynomial, ...

    # Convert
    # class PPoly(_PPolyBase); Construct a piecewise polynomial from a spline
    piecewise_polynomial = interpolate.PPoly.from_spline(tck, extrapolate=None)

    # Find roots
    roots_X_ = piecewise_polynomial.roots()  # class ndarray(builtins.object)
    # Filter roots within the X range
    roots_X = roots_X_[np.where(np.logical_and(roots_X_ >= X[0], roots_X_ <= X[-1]))]

    return roots_X

# %%
# Find common definitional domain of X1 and X2
def numpy_find_commen_definitional_domain_X_by_X1X2(X1, X2):

    X1.sort()
    X2.sort()

    a = np.max([X1[0], X2[0]])
    b = np.min([X1[-1], X2[-1]])

    X_full = np.append(X1, X2)
    X_full = np.array(list(set(X_full)))  # remove duplicates
    X_full.sort()
    X_commen = X_full[np.where(np.logical_and(X_full >= a, X_full <= b))]
    return X_commen  # [2.  3.  3.2 4.  4.5 5. ]

# %%
def numpy_scipy_find_inersections_by_X1Y1X2Y2(X1, Y1, X2, Y2):
    # X1 and X2 must be strictly monotonically increasing
    if np.all(np.diff(X1) > 0) == True:
        pass
    else:
        raise Exception('X1 must be strictly monotonically increasing!')

    if np.all(np.diff(X2) > 0) == True:
        pass
    else:
        raise Exception('X2 must be strictly monotonically increasing!')

    # Put X1 and X2 on a common grid
    # Find common definitional domain (strictly monotonically increasing data)
    X_new = numpy_find_commen_definitional_domain_X_by_X1X2(X1=X1, X2=X2)
    # print(X_new)
    Y1_new = np.interp(X_new, X1, Y1)
    Y2_new = np.interp(X_new, X2, Y2)
    Y_new = Y1_new - Y2_new
    X_new = X1
    Y_new = Y1 - Y2 

    inersections_X = numpy_scipy_find_roots_by_XY(X=X_new, Y=Y_new)
    inersections_Y = np.interp(inersections_X, X1, Y1)
    return inersections_X, inersections_Y

# %%
def numpy_find_commen_definitional_domain_X_by_X1X2(X1, X2):
    X1.sort()
    X2.sort()

    a = np.max([X1[0], X2[0]])
    b = np.min([X1[-1], X2[-1]])

    X_full = np.append(X1, X2)
    X_full = np.array(list(set(X_full)))  # remove duplicates
    X_full.sort()
    X_commen = X_full[np.where(np.logical_and(X_full >= a, X_full <= b))]
    return X_commen

# %%
# https://stackoverflow.com/questions/11108869/optimizing-python-distance-calculation-while-accounting-for-periodic-boundary-co
def dist_PBC(x0, x1, dimensions):
    delta = np.abs(x0 - x1)
    delta = np.where(delta > 0.5 * dimensions, delta - dimensions, delta)
    return np.sqrt((delta ** 2).sum(axis=-1))

# %%
# Compute angle formed by P1-B-P2
def calc_angle(point1, bridge, point2):
    v1 = bridge - point1
    v1_norm = v1.dot(v1)**0.5
    v2 = bridge - point2
    v2_norm = v2.dot(v2)**0.5
    return np.arccos(np.dot(v1, v2)/(v1_norm*v2_norm))

# %%
# Compute angle formed by P1-B1-B2-P2
def calc_angle(point1, bridge1, bridge2, point2):
    v1 = bridge1 - point1
    v1_norm = v1.dot(v1)**0.5
    v2 = bridge2 - point2
    v2_norm = v2.dot(v2)**0.5
    return np.arccos(np.dot(v1, v2)/(v1_norm*v2_norm))

# %%
def crossing_points_list(X1, Y1, X2, Y2):
    # Smooth first
    Y1_smooth = savgol_filter(Y1, 50, 1)
    Y2_smooth = savgol_filter(Y2, 50, 1)

    X, Y = numpy_scipy_find_inersections_by_X1Y1X2Y2(X1, Y1_smooth, X2, Y2_smooth)
    return np.array([[x, y] for x, y in zip(X, Y)])      

# %%
r0        = 3.07
theta0    = 151.62
cutoff    = 1.6  # distance between dummy sites

class InfoFile():
    def __init__(self, topfile, trjfile, logfile, pool_threads=4, is_refer=False, is_gmx=True):
        self.dirname = os.path.dirname(topfile)
        self.filename = os.path.basename(topfile).rsplit('.', 2)[0]
        self.log  = logfile
        self.pool_threads = pool_threads
        self.is_gmx = is_gmx
        self.n_dummies = 4

        self.SI_O_CUTOFF = 2.0          # cutoff to decide which Si an O belongs to
        self.SI_SP_DIST  = 1.03171      # Si-SP distance

        if self.is_gmx:
            self.system = mda.Universe(topfile, trjfile)

            # TODO: get u_OB, u_OH, u_ON based on system information
            self.u_Si	= self.system.select_atoms('name Si*')
            
            u_SiH_OH = self.system.select_atoms('resname SiH and name O2 O4 O6 O8')

            u_SiN_OH = self.system.select_atoms('resname SiN and name O2 O4 O6')
            u_SiN_ON = self.system.select_atoms('resname SiN and name O8')

            u_2Si_OH = self.system.select_atoms('resname 2Si and name O1 O3 O5 O11 O 12')
            u_2Si_OB = self.system.select_atoms('resname 2Si and name O8')
            u_2Si_ON = self.system.select_atoms('resname 2Si and name O10')

            u_3Si_OH = self.system.select_atoms('resname 3Si and name O1 O3 O9 O10 O15 O16 O17')
            u_3Si_OB = self.system.select_atoms('resname 3Si and name O6 O13')
            u_3Si_ON = self.system.select_atoms('resname 3Si and name O8')

            u_4Si_OH = self.system.select_atoms('resname 4Si and name O1 O3 O8 O12 O14 O19 O22')
            u_4Si_OB = self.system.select_atoms('resname 4Si and name O6 O10 O13 O17')
            u_4Si_ON = self.system.select_atoms('resname 4Si and name O21')

            u_5Si_OH = self.system.select_atoms('resname 5Si and name O1 O3 O8 O9 O15 O19 O20 O26 O28')
            u_5Si_OB = self.system.select_atoms('resname 5Si and name O6 O12 O14 O17 O23')
            u_5Si_ON = self.system.select_atoms('resname 5Si and name O25')
        
            u_6Si_OH = self.system.select_atoms('resname 6Si and name O1 O3 O8 O9 O14 O15 O21 O25 O26 O32 O34')            
            u_6Si_OB = self.system.select_atoms('resname 6Si and name O6 O12 O18 O20 O23 O29')            
            u_6Si_ON = self.system.select_atoms('resname 6Si and name O31')        
            
            self.u_OH   = self.system.atoms[np.concatenate((u_SiH_OH.ix, u_SiN_OH.ix, u_2Si_OH.ix, u_3Si_OH.ix, u_4Si_OH.ix, u_5Si_OH.ix, u_6Si_OH.ix)).tolist()]
            self.u_OB   = self.system.atoms[np.concatenate((u_2Si_OB.ix, u_3Si_OB.ix, u_4Si_OB.ix, u_5Si_OB.ix, u_6Si_OB.ix)).tolist()]
            self.u_ON   = self.system.atoms[np.concatenate((u_SiN_ON.ix, u_2Si_ON.ix, u_3Si_ON.ix, u_4Si_ON.ix, u_5Si_ON.ix, u_6Si_ON.ix)).tolist()]
            self.u_O    = self.system.select_atoms('resname *Si SiH SiN and name O*')

            self.u_N    = self.system.select_atoms('resname TPA and name N*')
            u_C = self.system.select_atoms('resname TPA and name C3 C6 C13 C16 C23 C26 C33 C36')
            sorted_indices = np.argsort(u_C.ix)
            self.u_C    = self.system.atoms[u_C.ix[sorted_indices]]

            self.u_Na   = self.system.select_atoms("name Na")
            self.u_Br   = self.system.select_atoms("name Br")
            self.u_W    = self.system.select_atoms("resname WAT")

        else:
            self.system = mda.Universe(topfile, trjfile, topology_format='DATA', format='lammpsdump', atom_style='id resid type charge x y z')

            self.u_Si   = self.system.select_atoms('type 1')
            
            # split for kMC/MD
            self.u_OH   = self.system.select_atoms('type 2')
            self.u_OB   = self.system.select_atoms('type 2')
            self.u_ON   = self.system.select_atoms('type 2')
            self.u_O    = self.system.select_atoms('type 2')

            self.u_C    = self.system.select_atoms('type 4')
            self.u_N    = self.system.select_atoms('type 5')
            self.u_Na   = self.system.select_atoms("type 6")
            self.u_Br   = self.system.select_atoms("type 7")
            self.u_W    = self.system.select_atoms("type 8")

        self.mapping = {
            'Si': self.u_Si,
            'OH': self.u_OH,
            'OB': self.u_OB,
            'ON': self.u_ON,
            'O' : self.u_O,
            'N' : self.u_N,
            'C' : self.u_C,
            'Na': self.u_Na,
            'Br': self.u_Br,
        }

    # Write RDF to a specific file and return arrays
    def _write_rdf_xvg(self, of, bins, rdf_values):
        writer = csv.writer(of)
        writer.writerows(np.array([bins, rdf_values]).T)
 
    # Need two groups to compute RDF; optionally write to file
    def _get_rdf(self, lhg, rhg, range=(1.5,12.0), outfilename=None):
        """
        Compute RDF for two groups.

        Parameters:
        lhg, rhg: two AtomGroups
        outfilename: output filename (if not None)

        Returns:
        rdf: 2D array storing bins and values
        """ 
        # IDEA set lower bound 1.50 to exclude OB-OB interactions; OB-OB ~1.20 in CG model
        result = rdf.InterRDF(lhg, rhg, nbins=500, range=range, exclude_same="residue")    
        # result.run(start=-500)    # only use last 500 frames
        result.run()    # only use last 500 frames
        if outfilename:
            with open(outfilename, 'w') as of:
                self._write_rdf_xvg(of, result.bins, result.rdf)
        return [result.bins, result.rdf]   

    def _find_si_for_o(self, u_si, u_o, cutoff, dims):
            """
            Find the nearest Si atom for each O atom (return Si atom.id)
            """
            o_to_si = [-1 for _ in u_o.ix]

            # Compute distance matrix (with PBC)
            dist_matrix = distances.distance_array(u_si.positions, u_o.positions, box=dims)
            
            # For each O, find the closest Si (with cutoff)
            si_indices_closest = np.argmin(dist_matrix, axis=0)  # shape: (n_o,)
            for o_idx, si_idx in enumerate(si_indices_closest):
                if dist_matrix[si_idx, o_idx] < cutoff:
                    o_to_si[o_idx] = u_si[si_idx].id

            return np.array(o_to_si, dtype=np.int32)

    def _get_cg_positions(self, group_type, dims):
        """
        Return "CG-mapped" coordinates used for RDF calculations:
        - Non-GROMACS or CG trajectory: directly return original coordinates
        - GROMACS + O/OH/ON/OB: use Si->O direction with a fixed Si-SP distance 1.03171 Å
        - Other particles (Si/Na/C/N/...): still use original coordinates
        """
        # For LAMMPS CG part, directly use existing coordinates
        if not self.is_gmx:
            return self.mapping[group_type].positions

        # For GROMACS, map O-type atoms from AA to CG
        if group_type in ("O", "OH", "ON", "OB"):
            u_o  = self.mapping[group_type]
            u_si = self.u_Si

            # Find the nearest Si for each O (return Si atom.id)
            o_to_si_ids = self._find_si_for_o(u_si, u_o, self.SI_O_CUTOFF, dims)

            # Build Si atom.id -> index mapping table
            id2idx = {atom.id: idx for idx, atom in enumerate(u_si)}

            coords_si = u_si.positions
            coords_o  = u_o.positions

            # Si indices corresponding to each O; if not found, mark as -1
            si_indices = np.array([id2idx.get(si_id, -1) for si_id in o_to_si_ids], dtype=int)

            # Only map O atoms that have a valid Si
            valid_mask = si_indices >= 0
            if not np.any(valid_mask):
                # Extreme case: none matched, fall back to original O coordinates
                return coords_o

            si_pos_valid = coords_si[si_indices[valid_mask]]
            o_pos_valid  = coords_o[valid_mask]

            # Compute minimum-image vectors Si->O
            delta = o_pos_valid - si_pos_valid      # shape (n_valid, 3)
            delta = minimize_vectors(delta, box=dims)

            # Unit vectors
            norms = np.linalg.norm(delta, axis=1)
            # Avoid division by zero
            norms[norms < 1e-8] = 1.0
            directions = delta / norms[:, None]

            # CG positions: Si + fixed distance * direction
            cg_pos_valid = si_pos_valid + self.SI_SP_DIST * directions

            # Put results back into full array; keep original coords for unmatched O
            cg_pos = coords_o.copy()
            cg_pos[valid_mask] = cg_pos_valid

            return cg_pos

        # Other particles: Si/Na/C/N/Br etc -> keep original AA coordinates;
        # C will be turned into CG centers later in _get_centers
        return self.mapping[group_type].positions

    def _get_dist_matrix_mask(self, lhg_type, rhg_type, lhg_pos, rhg_pos, dims):
        # Compute distance matrix (with periodic boundary conditions)
        dist_matrix = distances.distance_array(lhg_pos, rhg_pos, box=dims)
        SI_O_CUTOFF = 2.0  # Si-O

        if (lhg_type.startswith('O') and lhg_type != 'OB') and (rhg_type.startswith('O') and rhg_type != 'OB'):
            # Find which Si each O belongs to
            if self.is_gmx:
                lhg_resids = self._find_si_for_o(self.u_Si, self.mapping[lhg_type], self.SI_O_CUTOFF, dims)
                rhg_resids = self._find_si_for_o(self.u_Si, self.mapping[rhg_type], self.SI_O_CUTOFF, dims)
            else:
                lhg_resids = self.mapping[lhg_type].resids
                rhg_resids = self.mapping[rhg_type].resids
            
            # lhg_resids = self._find_si_for_o(self.u_Si, self.mapping[lhg_type], SI_O_CUTOFF, dims)
            # rhg_resids = self._find_si_for_o(self.u_Si, self.mapping[rhg_type], SI_O_CUTOFF, dims)

            # Generate residue-ID comparison matrix (M x N)
            same_resid = lhg_resids[:, None] == rhg_resids[None, :]            

            # Generate valid distance mask (exclude atom pairs within the same residue)
            valid_mask = ~same_resid
            if lhg_type == rhg_type:
                np.fill_diagonal(valid_mask, False)  # exclude self-interactions

            # Extract valid distances
            return dist_matrix[valid_mask]
        else:
            if lhg_type == rhg_type:
                upper = np.triu_indices_from(dist_matrix, k=1)
                valid_dists = dist_matrix[upper]
            else:
                valid_dists = dist_matrix.flatten()
            return valid_dists
        
    # Keep this helper as a method (name comment unchanged)
    def _get_centers(self, group_type, positions, dimensions):
        """Compute geometric centers of atom pairs"""
        if group_type == 'C':
            delta = minimize_vectors(
                positions[1::2] - positions[0::2],
                box=dimensions
            )
            return positions[0::2] + delta / 2
        return positions


    def _process_frame(self, lhg_type, rhg_type, bin_edges, frame_idx):
        """Single-frame processing function"""

        self.system.trajectory[frame_idx]
        dims = self.system.dimensions

        # Key: first get CG-mapped coordinates based on type (AA -> CG)
        lhg_pos = self._get_cg_positions(lhg_type, dims)
        rhg_pos = self._get_cg_positions(rhg_type, dims)

        # For C use original "two C atoms -> one CG-C" center rule
        lhg_centers = self._get_centers(lhg_type, lhg_pos, dims)
        rhg_centers = self._get_centers(rhg_type, rhg_pos, dims)

        dist_matrix = self._get_dist_matrix_mask(lhg_type, rhg_type, lhg_centers, rhg_centers, dims)

        hist, _ = np.histogram(dist_matrix, bins=bin_edges)
        volume = np.prod(self.system.dimensions[:3])                

        return hist, len(dist_matrix), volume


    def _get_rdf_special(self, lhg_type, rhg_type, range=(1.5,12.0), outfilename=None):
        nbins = 500
        bin_edges = np.linspace(*range, nbins + 1)
        histogram = np.zeros(nbins)
        total_ref_pairs = 0.0
        total_volume = 0.0

        # Create process pool
        with Pool(processes=self.pool_threads) as pool:
            # results = pool.map(partial(self._process_frame, lhg_type, rhg_type, bin_edges, self.system.dimensions), self.system.trajectory[-500:])
            results = pool.map(partial(self._process_frame, lhg_type, rhg_type, bin_edges), [ts.frame for ts in self.system.trajectory])
            for hist, ref_pairs, volume in results:
                histogram += hist
                total_ref_pairs += ref_pairs
                total_volume += volume

        bin_centers = (bin_edges[:-1] + bin_edges[1:])/2
        bin_volumes = 4/3 * np.pi * (bin_edges[1:]**3 - bin_edges[:-1]**3)        
        rdf = histogram / (total_ref_pairs / total_volume * len(self.system.trajectory) * bin_volumes)

        if outfilename:
            with open(outfilename, 'w') as of:
                self._write_rdf_xvg(of, bin_centers, rdf)

        return [bin_centers, rdf] 

    def get_rdf(self, lhg_type, rhg_type, range=(1.5,12.0), outfilename=None):
        # IDEA For GROMACS C atoms we need AA->CG mapping, and exclude O atoms on the same Si
        if (self.is_gmx and (lhg_type == 'C' or rhg_type == 'C')) or \
            (lhg_type.startswith('O') and lhg_type != 'OB') and (rhg_type.startswith('O') and rhg_type != 'OB'):
            return self._get_rdf_special(lhg_type, rhg_type, range=range, outfilename=outfilename)

        """Return RDF for the specified pair"""
        return self._get_rdf(self.mapping[lhg_type], self.mapping[rhg_type], range=range, outfilename=outfilename)

    def _find_peaks_and_valleys(self, y):
        """Core helper to find peaks and valleys"""
        peaks, props = find_peaks(
            y,
            prominence=max(y)*0.02,
            width=3
        )
        valleys, _ = find_peaks(
            -y,
            prominence=max(y)*0.02,
            width=3
        )
        return peaks, valleys

    def _calculate_peak_area(self, x, y, peak_idx, valleys):
        """Compute integrated area for a single peak"""
        left_valleys = valleys[valleys < peak_idx]
        right_valleys = valleys[valleys > peak_idx]

        lv = left_valleys[-1] if left_valleys.size else 0
        rv = right_valleys[0] if right_valleys.size else len(x)-1

        return simpson(y[lv:rv], x=x[lv:rv])

    def get_peaks(self, bins, rdf, n_peaks=1, sorting='position'):
        """
        Get the first N characteristic peaks of an RDF.
        
        Parameters
        ----------
        bins : array-like
            RDF bin centers.
        rdf : array-like
            RDF values corresponding to `bins`.
        n_peaks : int
            Number of peaks requested.
        sorting : {'position', 'area'}
            Peak sorting mode.
        
        Returns
        -------
        peak_indices : list[int]
            Indices of the selected peaks (in the `bins` array).
        peak_properties : list[dict]
            A list of dictionaries describing each peak.
        """
        # Smooth RDF data
        smoothed = savgol_filter(rdf, window_length=51, polyorder=3)
        
        # Find peaks and valleys
        peaks, valleys = self._find_peaks_and_valleys(smoothed)
        
        # Compute properties for each peak
        peak_data = []
        for idx in peaks:
            area = self._calculate_peak_area(bins, smoothed, idx, valleys)
            peak_data.append({
                'index': idx,
                'position': bins[idx],
                'height': smoothed[idx],
                'area': area,
                'left_valley': max(valleys[valleys < idx], default=0),
                'right_valley': min(valleys[valleys > idx], default=len(bins)-1)
            })

        # Sorting logic
        if sorting == 'position':
            peak_data.sort(key=lambda x: x['position'])
        elif sorting == 'area':
            peak_data.sort(key=lambda x: -x['area'])
        else:
            raise ValueError("Unsupported sorting mode")

        # Return first N peaks
        valid_peaks = [p for p in peak_data if p['area'] > 0]
        return (
            [p['index'] for p in valid_peaks[:n_peaks]],
            valid_peaks[:n_peaks]
        )

    def _calc_angles(self, v1s, v2s):
        """Return angles between vectors v1 and v2"""
        angles = []
        for v1, v2 in zip(v1s, v2s):
            angles.append(np.arccos(np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))))
        return np.array(angles)

    def _get_sos_angles_cg(self, coords_si, coords_d, dims, r_dd):
        """Handle angle calculations for CG model"""
        # Generate D-D contact matrix
        mat_dd = distances.contact_matrix(coords_d, cutoff=r_dd, box=dims)
        
        # Exclude D-D contacts belonging to the same Si
        n_total = len(coords_d)
        for i in range(0, n_total, self.n_dummies):
            mat_dd[i:i+self.n_dummies, i:i+self.n_dummies] = False
        
        # Extract valid D-D pairs
        rows, cols = triu(mat_dd).nonzero()
        valid_pairs = [(r, c) for r, c in zip(rows, cols) 
                      if (r // self.n_dummies) != (c // self.n_dummies)]
        
        # Batch compute angles
        si_indices = np.array([(r//self.n_dummies, c//self.n_dummies) 
                             for r, c in valid_pairs])
        o_indices = np.array(valid_pairs)

        vec_sio1 = coords_d[o_indices[:,0]] - coords_si[si_indices[:,0]]  # D1 -> Si1
        vec_sio2 = coords_d[o_indices[:,1]] - coords_si[si_indices[:,1]]  # D2 -> Si2

        return self._calc_angles(vec_sio1, vec_sio2)

    def _manual_contact_matrix(self, coords1, coords2, cutoff, box):
        """
        Manually build a contact matrix with PBC considered.
        
        Parameters
        ----------
        coords1 : (M,3) array
            Coordinates of the first atom group.
        coords2 : (N,3) array
            Coordinates of the second atom group.
        cutoff : float
            Cutoff radius (Å).
        box : array-like
            Box dimensions [lx, ly, lz].
        
        Returns
        -------
        csr_matrix
            (M, N) sparse contact matrix.
        """
        # Input validation
        box = np.asarray(box[:3], dtype=np.float32)
        cutoff_sq = (cutoff ** 2)
        
        # Use optimized data types
        coords1 = np.asarray(coords1, dtype=np.float32)
        coords2 = np.asarray(coords2, dtype=np.float32)
        
        # Initialize sparse matrix
        n1, n2 = len(coords1), len(coords2)
        mat = lil_matrix((n1, n2), dtype=np.bool_)
        
        # Precompute inverse box lengths to speed up calculations
        inv_box = np.divide(1.0, box, where=(box != 0))
        
        # Check whether coords1 and coords2 belong to the same group
        same_group = np.allclose(coords1, coords2, atol=1e-10)    

        # Process in batches to optimize memory usage
        batch_size = 500  # tune according to memory
        for start in range(0, n1, batch_size):
            end = min(start + batch_size, n1)
            batch_coords = coords1[start:end]
            
            # Vectorized delta calculation (end-start, n1, 3)
            delta = batch_coords[:,None,:] - coords2[None,:,:]
            
            # Apply PBC minimum image
            delta -= np.round(delta * inv_box) * box
            
            # Compute squared distances
            dist_sq = np.sum(delta**2, axis=2)
            
            # Set contact flags in batch
            for batch_idx in range(batch_coords.shape[0]):
                global_idx = start + batch_idx
                contact_mask = dist_sq[batch_idx] < cutoff_sq
            
                # Exclude self-contacts
                if same_group:
                    contact_mask &= (np.arange(n2) != global_idx)
                
                contacts = np.where(contact_mask)[0]
                mat.rows[global_idx] = contacts.tolist()
                mat.data[global_idx] = [True]*len(contacts)
        
        return mat.tocsr()

    def _get_sos_angles_aa(self, coords_si, coords_o, dims, r_so):
        """Handle angle calculations for all-atom model"""
        # Generate Si-O contact matrix
        mat_so = self._manual_contact_matrix(coords_si, coords_o, cutoff=r_so, box=dims)
        csr_mat = csr_matrix(mat_so)

        # Identify bridging O atoms
        o_connectivity = csr_mat.getnnz(axis=0)
        bridge_o = np.where(o_connectivity == 2)[0]
        
        # Extract corresponding Si pairs
        si_pairs = np.array([csr_mat[:,o].nonzero()[0] for o in bridge_o])

        # Batch compute angles
        vec_sio1 = coords_o[bridge_o] - coords_si[si_pairs[:,0]]       # O -> Si1
        vec_sio2 = coords_o[bridge_o] - coords_si[si_pairs[:,1]]       # O -> Si2
        
        return self._calc_angles(vec_sio1, vec_sio2)

    # Special handling for angles: CG and AA handle angle calculations differently
    def _get_angles(self, u_si, u_o, dims, r_dd=None, r_so=None, ts=None):
        """Get all Si-O-Si (or Si-D-D-Si) angles.
    
        Parameters
        ----------
        u_si : AtomGroup
            MDAnalysis AtomGroup containing Si atoms.
        u_o : AtomGroup
            MDAnalysis AtomGroup containing O atoms.
        r_dd : float, optional
            D-D cutoff for CG model.
        r_so : float, optional
            Si-O cutoff for AA model.
        
        Returns
        -------
        angles_deg : ndarray
            Angles in degrees.
        """      
        coords_si = u_si.positions
        coords_o = u_o.positions
        if r_dd:
            angles = self._get_sos_angles_cg(coords_si, coords_o, dims, r_dd)
        elif r_so:
            angles = self._get_sos_angles_aa(coords_si, coords_o, dims, r_so)
        else:
            raise ValueError("Neither r_dd nor r_so is provided")

        return np.rad2deg(angles)

    def get_angles_rdf(self, r_dd=None, r_so=None, outfilename=None):
        # Use tqdm progress bar
        with Pool(processes = self.pool_threads) as pool:
            result = list(tqdm.tqdm(pool.imap(partial(self._get_angles, self.u_Si, self.u_O, self.system.dimensions, r_dd, r_so), self.system.trajectory), total=len(self.system.trajectory)))

        total_angles = np.array(list(itertools.chain(*result)))
        hist, bin_edges = np.histogram(total_angles, bins=720, range=(0,180.0))
        bins = np.mean(np.vstack([bin_edges[:-1], bin_edges[1:]]), axis=0) 
        if outfilename:
            with open(outfilename, 'w') as of:
                self._write_rdf_xvg(of, bins, hist)
        return [bins, hist]  

    def _get_qn(self, u_si, r_ss, dims, ts=None):
        """Qn statistics function.
    
        Parameters
        ----------
        u_si : AtomGroup
            MDAnalysis AtomGroup containing Si atoms.
        r_ss : float
            Si-Si cutoff radius.
        dims : array-like
            Box dimension information.
        
        Returns
        -------
        tuple
            Normalized statistics (Q0, Q1, Q2, Q3, Q4, Q5, C).
        """
        mat_ss = distances.contact_matrix(u_si.positions, cutoff=r_ss, box=dims)
        csr_mat = csr_matrix(mat_ss)  # convert to CSR format

        # Vectorized coordination number (exclude self entries)
        nnz_per_row = csr_mat.getnnz(axis=1)  # non-zero per row
        Q_values = nnz_per_row - 1  # subtract self term

        # Fast classification via histogram
        bins = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, np.inf]
        hist, _ = np.histogram(Q_values, bins=bins)
        Q_counts = hist.astype(float)
        n_atoms = Q_values.size

        # Compute C value (vectorized)
        weighted_sum = np.dot(Q_counts[1:5], [1, 2, 3, 4])
        C = 0.25 * weighted_sum / n_atoms

        # Normalize
        Q_normalized = Q_counts / n_atoms
        return (*Q_normalized, C)
    
    def get_qn_max_and_cross(self, r_ss):
        FILTER_SIZE = 10          # median filter window size
        SAVGOL_WINDOW = 10        # Savitzky-Golay filter window
        SAVGOL_ORDER = 1          # Savitzky-Golay polynomial order
        
		# Use tqdm and multiprocessing to compute Qn in parallel
        with Pool(processes=self.pool_threads) as pool:
            results = list(tqdm.tqdm(
                pool.imap(partial(
                    self._get_qn, 
                    self.u_Si, 
                    r_ss, 
                    self.system.dimensions
                ), self.system.trajectory),
                total=len(self.system.trajectory)
            ))

        # Convert to structured array for better performance
        qn_fields = ['q0', 'q1', 'q2', 'q3', 'q4', 'q5', 'c']
        qn_dtype = [(f, 'f8') for f in qn_fields]
        qn_data = np.array(results, dtype=qn_dtype)

        # Sort by c and remove duplicates
        sort_idx = np.argsort(qn_data['c'])
        qn_sorted = qn_data[sort_idx]
        unique_mask = np.r_[True, qn_sorted['c'][1:] != qn_sorted['c'][:-1]]
        qn_unique = qn_sorted[unique_mask]

        # Smooth data
        smooth_data = {}
        for q in qn_fields[:-1]:  # exclude 'c' column
            # Median filter
            med_filtered = median_filter(qn_unique[q], FILTER_SIZE)
            # Savitzky-Golay smoothing
            smooth_data[q] = savgol_filter(med_filtered, SAVGOL_WINDOW, SAVGOL_ORDER)

        # Find maxima and corresponding c for each Qn
        max_values = {}
        for q in qn_fields[:-1]:
            max_idx = np.argmax(qn_unique[q])
            max_values[q] = np.array([
                qn_unique[q][max_idx], 
                qn_unique['c'][max_idx]
            ])

        # Define Qn pairs to analyze crossings
        cross_pairs = [
            ('q0', 'q1'), ('q1', 'q2'), ('q2', 'q3'), ('q3', 'q4'),
            ('q0', 'q2'), ('q1', 'q3'), ('q2', 'q4'), ('q0', 'q3'),
            ('q1', 'q4'), ('q0', 'q4')
        ]

        # Compute all crossings
        cross_points = {}
        for q1, q2 in cross_pairs:
            key = f"{q1}_{q2}"
            crossings = crossing_points_list(
                qn_unique['c'], smooth_data[q1],
                qn_unique['c'], smooth_data[q2]
            )
            cross_points[key] = crossings[0] if crossings.size > 0 else np.array([[0.0, 1.0]])

        # Build extrema report
        max_report = "\n".join(
            [f"max_{q} = {val}" for q, val in max_values.items()] +
            [f"max_c = {np.max(qn_unique['c'])}"]
        )
        
        # Build crossing report
        cross_report = "\n".join(
            [f"{pair} = {pts}" for pair, pts in cross_points.items()]
        )
        
        print(f"*** Qn extrema information ***\n{max_report}")
        print(f"\n*** Crossing information ***\n{cross_report}")

        return  [max_values, cross_points]       


# %%
if __name__ == "__main__":
    current_dir = Path(sys.argv[1])
    nthreads = int(sys.argv[2])

    info = InfoFile(current_dir/"lmp.data", current_dir/"lmp.lammpstrj", current_dir/"info.log", nthreads, is_refer=False, is_gmx=False)
    # info = InfoFile(current_dir/"NPT.gro", current_dir/"NPT.xtc", current_dir/"info.log", nthreads, is_refer=True, is_gmx=True)

    print(f"Start processing Si-related RDF")
    Si_Si_rdf = info.get_rdf('Si', 'Si', outfilename=current_dir/"rdf_Si-Si.csv")
    Si_Na_rdf = info.get_rdf('Si', 'Na', outfilename=current_dir/"rdf_Si-Na.csv")
    Si_C_rdf  = info.get_rdf('Si',  'C', outfilename=current_dir/"rdf_Si-C.csv" , range=(1.5, 20.0))
    Si_N_rdf  = info.get_rdf('Si',  'N', outfilename=current_dir/"rdf_Si-N.csv" , range=(1.5, 20.0))
    plt.plot(Si_Si_rdf[0], Si_Si_rdf[1], label="Si-Si")
    plt.plot(Si_Na_rdf[0], Si_Na_rdf[1], label="Si-Na")
    plt.plot(Si_C_rdf[0], Si_C_rdf[1], label="Si-C")
    plt.plot(Si_N_rdf[0], Si_N_rdf[1], label="Si-N")
    plt.legend()
    plt.savefig(current_dir / "rdf_Si.png", dpi=1200)
    plt.cla()

    print(f"Start processing Na-related RDF")
    Na_O_rdf = info.get_rdf('Na', 'O', outfilename=current_dir/"rdf_Na-O.csv")
    plt.plot(Na_O_rdf[0], Na_O_rdf[1], label="Na-O")
    plt.plot(Si_Na_rdf[0], Si_Na_rdf[1], label="Si-Na")
    plt.legend()
    plt.savefig(current_dir / "rdf_Na.png", dpi=1200)
    plt.cla()

    print(f"Start processing C-related RDF")
    C_O_rdf  = info.get_rdf( 'C', 'O', outfilename=current_dir/"rdf_C-O.csv" , range=(1.5, 20.0))
    plt.plot(C_O_rdf[0], C_O_rdf[1], label="C-O")
    plt.plot(Si_C_rdf[0], Si_C_rdf[1], label="Si-C")
    plt.legend()
    plt.savefig(current_dir / "rdf_C.png", dpi=1200)
    plt.cla()

    print(f"Start processing N-related RDF")
    N_O_rdf  = info.get_rdf( 'N', 'O', outfilename=current_dir/"rdf_N-O.csv" , range=(1.5, 20.0))
    plt.plot(N_O_rdf[0], N_O_rdf[1], label="N-O")
    plt.plot(Si_N_rdf[0], Si_N_rdf[1], label="Si-N")
    plt.legend()
    plt.savefig(current_dir / "rdf_N.png", dpi=1200)
    plt.cla()
