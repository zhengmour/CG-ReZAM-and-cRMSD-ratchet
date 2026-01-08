import numpy as np
from scipy import signal, integrate, interpolate
from dtaidistance import dtw
import csv, json
import sys, re
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
from scipy.integrate import simpson
import math

# ======================
# Data preprocessing
# ======================
def preprocess_rdf(r1, g1, r2, g2, r_range=None):
    """Align two RDFs onto a common r grid."""
    # Determine common range
    r_min = max(r1.min(), r2.min()) if r_range is None else r_range[0]
    r_max = min(r1.max(), r2.max()) if r_range is None else r_range[1]
    
    # Create new grid
    r_new = np.linspace(r_min, r_max, max(len(r1), len(r2)))
    
    # Interpolation functions
    f1 = interpolate.interp1d(r1, g1, kind='linear', fill_value="extrapolate")
    f2 = interpolate.interp1d(r2, g2, kind='linear', fill_value="extrapolate")

    return r_new, f1(r_new), f2(r_new)

# ======================
# Similarity metrics
# ======================
def compute_mse(g1, g2):
    """Mean squared error."""
    return np.mean((g1 - g2)**2)

def compute_rmse(g1, g2):
    """Root mean squared error."""
    return np.sqrt(compute_mse(g1, g2))

def compute_pearson(g1, g2):
    """Pearson correlation coefficient in [-1, 1]."""
    return np.corrcoef(g1, g2)[0, 1]

def overlap_integral(r, g1, g2):
    """Compute overlap integral."""
    numerator = integrate.trapz(np.minimum(g1, g2), r)
    denominator = integrate.trapz(np.maximum(g1, g2), r)
    return numerator / denominator if denominator != 0 else 0.0

def compute_dtw_distance(g1, g2):
    """Dynamic Time Warping distance."""
    return dtw.distance_fast(g1, g2)

def kl_divergence(p, q, epsilon=1e-10):
    """KL divergence (requires normalized distributions)."""
    p = np.clip(p, epsilon, None)
    q = np.clip(q, epsilon, None)
    return np.sum(p * np.log2(p / q))

def js_divergence(p, q, epsilon=1e-10):
    """Jensen–Shannon divergence (the closer to 0, the more similar)."""
    p = np.clip(p, epsilon, None)
    q = np.clip(q, epsilon, None)

    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log2(p / m))
    kl_qm = np.sum(q * np.log2(q / m))
    return 0.5 * (kl_pm + kl_qm)

# ======================
# Peak analysis
# ======================
def _find_peaks_and_valleys(y):
    """Core helper to find peaks and valleys."""
    baseline = np.percentile(y, 25)
    prominence_th = max((np.max(y) - baseline) * 0.1, 0.02)  # at least 0.02 prominence

    peaks, _ = find_peaks(
        y,
        prominence=prominence_th,
        width=5,
        wlen=200
    )
    valleys, _ = find_peaks(
        -y,
        prominence=prominence_th,
        width=5,
        wlen=200
    )
    return peaks, valleys

def _calculate_peak_area(x, y, peak_idx, valleys):
    """Compute the integrated area of a single peak."""
    left_valleys = valleys[valleys < peak_idx]
    right_valleys = valleys[valleys > peak_idx]

    lv = left_valleys[-1] if left_valleys.size else 0
    rv = right_valleys[0] if right_valleys.size else len(x)-1

    return simpson(y[lv:rv], x=x[lv:rv])

def get_peaks(bins, rdf, n_peaks=1, sorting='position', min_position=1.8, min_area=0.1):
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
    min_position : float
        Minimum peak position (Å).
    min_area : float
        Minimum peak area threshold.

    Returns
    -------
    peak_indices : list[int]
        Indices of the selected peaks (in the `bins` array).
    peak_properties : list[dict]
        A list of dictionaries describing each peak.
    """
    # Adaptive smoothing (window = 1/10 of data length, enforced odd)
    window_length = len(bins) // 10
    window_length = window_length + 1 if window_length % 2 == 0 else window_length
    window_length = max(window_length, 11)  # minimum window of 11
    smoothed = savgol_filter(rdf, window_length=window_length, polyorder=3)
    
    # Find peaks and valleys
    peaks, valleys = _find_peaks_and_valleys(smoothed)

    # Compute peak properties and filter
    valid_peaks = []
    for idx in peaks:
        pos = bins[idx]
        if pos < min_position:
            continue  # skip peaks at too small r
            
        area = _calculate_peak_area(bins, smoothed, idx, valleys)
        if area < min_area:
            continue  # skip peaks with too small area
            
        valid_peaks.append({
            'index': idx,
            'position': pos,
            'height': smoothed[idx],
            'area': area,
            'left_valley': max(valleys[valleys < idx], default=0),
            'right_valley': min(valleys[valleys > idx], default=len(bins)-1)
        })

    # Sorting logic
    if sorting == 'position':
        valid_peaks.sort(key=lambda x: x['position'])
    elif sorting == 'area':
        valid_peaks.sort(key=lambda x: -x['area'])
    else:
        raise ValueError("Unsupported sorting mode")

    return (
        [p['index'] for p in valid_peaks[:n_peaks]],
        valid_peaks[:n_peaks]
    )

# ======================
# File sorting helpers
# ======================
def natural_sort_key(s):
    """Convert a string into a natural sort key tuple."""
    return [
        int(c) if c.isdigit() else c.lower()
        for c in re.split(r'(\d+)', s.name)
    ]

def get_sorted_rdf_files(folder):
    files = list(folder.glob("rdf*.csv"))
    return sorted(files, key=natural_sort_key)

# ======================
# Main program
# ======================
# Use Si-Si RDF as a primary reference and describe other RDFs relative to it
WEIGHTS_DICT = {
    "N-O"  : np.array([30.0, 0.0, 0.0, 10.0, 0.0, 0.0]),
    "Na-O" : np.array([0.0, 40.0, 0.0, 0.0, 0.0, 0.0]),
    "Si-C" : np.array([0.0, 0.0, 40.0, 0.0, 0.0, 0.0]),
    "Si-N" : np.array([10.0, 0.0, 0.0, 30.0, 0.0, 0.0]),
    "Si-Na": np.array([0.0, 10.0, 0.0, 0.0, 30.0, 0.0]),
    "Si-Si": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 40.0]),
}
WEIGHTS = np.array([5.0, 5.0, 10.0, 10.0, 20.0, 0.0])

if __name__ == "__main__":
    current_dir = Path(sys.argv[1])
    base_dir = Path(sys.argv[2])
    if len(sys.argv) > 3:
        base_key = sys.argv[3]
        element = base_key.split('-')[0]
    else:
        base_key = None
        element = None

    refer_dir = base_dir / "jobs/refer_files"
    score_file = current_dir / "score.txt"

    # === Read according to the latest elements.json structure ===
    # {
    #   "pairs": { "Si-Na": [...], ... },
    #   "lambda": 0.24
    # }
    with open(base_dir / "config/elements.json") as f:
        config = json.load(f)
    pairs = list(config["pairs"].keys())

    rdf_first_sigma_file = current_dir / "rdf_first_sigma_file.json"
    rdf_first_sigma = {element_pair + "_1": np.nan for element_pair in config["pairs"].keys()}

    current_rdf_files = get_sorted_rdf_files(current_dir)
    refer_rdf_files = get_sorted_rdf_files(refer_dir)
    
    if len(current_rdf_files) == len(refer_rdf_files):
        score = 0.0
        with open(score_file, 'w') as wf:
            values = []
            for current_rdf_file, refer_rdf_file in zip(current_rdf_files, refer_rdf_files):
                # Filenames are in the form: rdf_[element]_[other element].csv
                assert current_rdf_file.name == refer_rdf_file.name, "Current RDF file and reference RDF file are not the same file"
                basename = current_rdf_file.stem
                _, element_pair = basename.split('_', 1)
                element_pair_flatten = element_pair + "_1"

                with open(current_rdf_file, 'r') as cf, open(refer_rdf_file, 'r') as rf:
                    current_reader = csv.reader(cf)
                    refer_reader = csv.reader(rf)
                    current_rdf = np.array([row for row in current_reader], dtype=np.float64)
                    refer_rdf = np.array([row for row in refer_reader], dtype=np.float64)

                    if np.array_equal(current_rdf[:, 0], refer_rdf[:, 0], equal_nan=True):
                        bins_center = current_rdf[:, 0]
                        current_values = current_rdf[:, 1]
                        refer_values = refer_rdf[:, 1]
                    else:
                        bins_center, current_values, refer_values = preprocess_rdf(
                            current_rdf[:, 0], current_rdf[:, 1],
                            refer_rdf[:, 0], refer_rdf[:, 1]
                        )

                    # ===== Normalized L2 difference: 0 means identical, closer to 1 means more dissimilar =====
                    r_cut = 12.0

                    if r_cut is not None:
                        mask = bins_center <= r_cut
                    else:
                        mask = np.ones_like(bins_center, dtype=bool)

                    dr = np.diff(bins_center[mask])
                    dr = np.append(dr, dr[-1])

                    diff_sq = (current_values[mask] - refer_values[mask])**2
                    g_sq = refer_values[mask]**2

                    num = np.sum(diff_sq * dr)
                    den = np.sum(g_sq * dr) + 1e-12

                    value = num / den  # 0–1: 0 = most similar, closer to 1 = more different

                    # Do not extract peak positions for OB-related pairs
                    if (not np.all(np.isnan(current_rdf[:, 1]))) and \
                            not ((element_pair.startswith("OB") and element_pair.endswith("OH")) or 
                                (element_pair.startswith("OB") and element_pair.endswith("ON")) or 
                                (element_pair.startswith("O")  and element_pair.endswith("OB"))):
                        try:
                            current_peak_indice = get_peaks(current_rdf[:, 0], current_rdf[:, 1], n_peaks=1)[0][0]
                            current_peak = current_rdf[current_peak_indice][0]
                        except Exception:
                            current_peak = None
                            print(f"current rdf-{basename} has no peak")

                        try:
                            refer_peak_indice = get_peaks(refer_rdf[:, 0], refer_rdf[:, 1], n_peaks=1)[0][0]
                            refer_peak = refer_rdf[refer_peak_indice][0]
                        except Exception:
                            refer_peak = None
                            print(f"refer rdf-{basename} has no peak")

                        if refer_peak:
                            if not current_peak:
                                print(f"rdf-{basename} peak is {refer_peak}")
                                rdf_first_sigma[element_pair_flatten] = refer_peak / math.pow(2, 1/6)
                            else:
                                if abs(current_peak - refer_peak) < 1.0:
                                    peak = current_peak
                                else:
                                    peak = refer_peak
                                print(f"rdf-{basename} peak is {peak}")
                                rdf_first_sigma[element_pair_flatten] = peak / math.pow(2, 1/6)
    
                    plt.plot(current_rdf[:, 0], current_rdf[:, 1], label="current")
                    plt.plot(refer_rdf[:, 0], refer_rdf[:, 1], label="refer")
                    plt.legend()
                    png_file = current_dir / f"{basename}.png"
                    plt.savefig(png_file, dpi=1200)
                    plt.cla()

                print(f" element {element} with pair {element_pair}, value is {value} ")
                # Previously there was a weighted combination here; now we directly use `value` itself
                print(f" Final value (no weighting): element {element} with pair {element_pair}, value is {value} ")

                values.append(value)
                wf.write(f"{basename} {value}\n")
            
            weights = WEIGHTS_DICT.get(base_key, WEIGHTS)
            score = np.sum(np.array(values) * weights)
            wf.write(f"total {score}\n")
    else:
        # Penalty term when the number of files does not match
        with open(score_file, 'w') as wf:
            score = np.sum(WEIGHTS)
            wf.write(f"total {score}")

    # Replace NaN with None before writing, so JSON is valid
    rdf_first_sigma_clean = {
        k: (None if (isinstance(v, float) and np.isnan(v)) else v)
        for k, v in rdf_first_sigma.items()
    }

    # Remove first-sigma entries for Si-C/N and O-C/N type pairs
    for key in rdf_first_sigma_clean.keys():
        if key.startswith("N-O_") or key.startswith("Si-N_") or key.startswith("Na-O_"):
            rdf_first_sigma_clean[key] = None

    with open(rdf_first_sigma_file, 'w', encoding='utf-8') as wf:
        json.dump(rdf_first_sigma_clean, wf, indent=2)
