import numpy as np
import networkx as nx
from collections import deque
from scipy.spatial import cKDTree
from topology_constants import SiSi_distance, OBOB_distance, SiOB_distance, SiSi_neighbor_distance, HYSTERESIS_MARGIN
from tqdm import tqdm
import math

def ensure_cell_vector(cell):
    cell = np.asarray(cell)
    if cell.ndim == 1 and cell.size == 3:
        return cell.astype(float)
    elif cell.shape == (3,3):
        return np.array([cell[0,0], cell[1,1], cell[2,2]], dtype=float)
    else:
        raise ValueError("cell must be shape (3,) or (3,3)")

def pairwise_mic_distances(positions, cell):
    """Return N×N pairwise minimum-image distances (orthogonal box)."""
    positions = np.asarray(positions)
    cell_vec = ensure_cell_vector(cell)
    delta = positions[:, None, :] - positions[None, :, :]   # (N,N,3)
    frac = delta / cell_vec
    frac = frac - np.rint(frac)
    delta = frac * cell_vec
    dist = np.linalg.norm(delta, axis=-1)
    return dist

def get_adj_matrix(positions, cell, cutoff):
    dist = pairwise_mic_distances(positions, cell)
    sw = (dist < cutoff).astype(int)
    np.fill_diagonal(sw, 0)
    return sw


def tile_ob_positions(ob_positions, cell):
    """
    Tile OB positions to (-1..1) images (orthogonal box only),
    and return tiled_positions and tiled_orig_idx.
    tiled_positions.shape = (9*len(ob_positions), 3)
    tiled_orig_idx maps back to the original OB indices.
    """
    ob_positions = np.asarray(ob_positions)
    cell_vec = ensure_cell_vector(cell)
    shifts = [np.array([dx,dy,dz]) * cell_vec for dx in (-1,0,1) for dy in (-1,0,1) for dz in (-1,0,1)]
    tiled = []
    orig_idx = []
    for s_idx, shift in enumerate(shifts):
        for k, p in enumerate(ob_positions):
            tiled.append(p + shift)
            orig_idx.append(k)
    if len(tiled) == 0:
        return np.empty((0,3)), np.empty((0,), dtype=int)
    return np.vstack(tiled), np.array(orig_idx, dtype=int)


def build_graphs_direct(
    Si_pos,
    OB_pos,
    cell_vec,
    si_si_cutoff=SiSi_distance,
    ob_cutoff=OBOB_distance,
    siob_cutoff=SiOB_distance,
):
    """
    Build graph by pure distance + OB filtering (no hysteresis):
    - dist(Si-Si) < si_si_cutoff as candidate edges
    - With OB filtering: each of Si_i & Si_j must have nearby OB,
      and there must exist an OB-OB pair with distance < ob_cutoff.
    """
    Si_pos = np.asarray(Si_pos)
    N = len(Si_pos)

    # Si-Si MIC distance matrix
    dist = pairwise_mic_distances(Si_pos, cell_vec)

    # Candidate Si-Si bonds (upper triangle)
    candidate_mask = (dist < si_si_cutoff)
    idxs = np.transpose(np.nonzero(candidate_mask))

    bonds = set()

    # If OB filtering is disabled, directly add bonds
    if (ob_cutoff is None) or (OB_pos is None) or (len(OB_pos) == 0):
        for i, j in idxs:
            if i >= j:
                continue
            bonds.add(frozenset((i, j)))
        return bonds

    # Otherwise: pre-tile OB and build KDTree
    tiled_ob, tiled_idx = tile_ob_positions(OB_pos, cell_vec)
    if len(tiled_ob) == 0:
        # No OB available, return empty set
        return bonds

    ob_tree = cKDTree(tiled_ob)

    for i, j in idxs:
        if i >= j:
            continue

        si_i_pos = Si_pos[i]
        si_j_pos = Si_pos[j]

        ob_neighbors_i = ob_tree.query_ball_point(si_i_pos, siob_cutoff)
        ob_neighbors_j = ob_tree.query_ball_point(si_j_pos, siob_cutoff)

        if len(ob_neighbors_i) == 0 or len(ob_neighbors_j) == 0:
            continue

        connected = False
        for ti in ob_neighbors_i:
            for tj in ob_neighbors_j:
                if np.linalg.norm(tiled_ob[ti] - tiled_ob[tj]) < ob_cutoff:
                    connected = True
                    break
            if connected:
                break

        if not connected:
            continue

        bonds.add(frozenset((i, j)))

    return bonds


def build_graphs_hysteresis(Si_pos, OB_pos, cell_vec, prev_bonds, sisi_cutoff=SiSi_distance, ob_cutoff=OBOB_distance, siob_cutoff=SiOB_distance, create_margin=HYSTERESIS_MARGIN):
    """
    Sequential single-thread build: apply hysteresis to generate NetworkX Graph bonds for all frames.
    topologies: list of ( (Si_positions, OB_positions), cell, time )
    sisi_cutoff: nominal Si-Si neighbor distance (for create/break)
    create_margin: hysteresis margin (Å)
    ob_cutoff: OBOB_distance (for Si-OB-OB chemical filtering); if None, skip OB filtering
    siob_cutoff: Si-OB distance for neighborhood (if None, caller uses SiOB_distance from constants)
    Returns a set of bonds for the current frame.
    """
    create_cutoff = sisi_cutoff - create_margin   # Tighter criterion for creating bonds
    break_cutoff  = sisi_cutoff + create_margin   # Looser criterion for breaking bonds

    Si_pos = np.asarray(Si_pos)
    N = len(Si_pos)

    # Si-Si MIC distance matrix
    dist = pairwise_mic_distances(Si_pos, cell_vec)

    # Create / Break masks
    create_mask = (dist < create_cutoff)
    break_mask = (dist > break_cutoff)

    # OB precompute: tiled positions + KDTree (if needed)
    tiled_ob = None; tiled_idx = None; ob_tree = None
    if (ob_cutoff is not None) and (OB_pos is not None) and len(OB_pos) > 0:
        tiled_ob, tiled_idx = tile_ob_positions(OB_pos, cell_vec)
        ob_tree = cKDTree(tiled_ob)

    # Start from prev_bonds and apply break rule
    bonds = set()
    for b in prev_bonds:
        i, j = tuple(b)
        if break_mask[i, j]:
            # Break bond
            continue
        bonds.add(b)

    # Try to create new bonds from create_mask
    idxs = np.transpose(np.nonzero(create_mask))
    for i, j in idxs:
        if i >= j:
            continue
        b = frozenset((i, j))
        if b in bonds:
            continue
        # If OB chemical filtering is needed, require that Si_i and Si_j each have nearby OB images
        # and that there exists a pair with OB-OB distance < ob_cutoff
        if ob_tree is not None and ob_tree.n > 0:
            # Use tiled KDTree to query OB near each Si in the tiled space, returning tiled indices
            # Note: tiled_idx maps tiled index -> original OB index (not needed here unless deduplication is needed)
            si_i_pos = Si_pos[i]
            si_j_pos = Si_pos[j]
            ob_neighbors_i = ob_tree.query_ball_point(si_i_pos, siob_cutoff)
            ob_neighbors_j = ob_tree.query_ball_point(si_j_pos, siob_cutoff)
            if len(ob_neighbors_i) == 0 or len(ob_neighbors_j) == 0:
                # No OB support, reject bond creation
                continue
            connected = False
            # Compare tiled OB Cartesian distances (tiling already applied)
            for ti in ob_neighbors_i:
                for tj in ob_neighbors_j:
                    if np.linalg.norm(tiled_ob[ti] - tiled_ob[tj]) < ob_cutoff:
                        connected = True
                        break
                if connected:
                    break
            if not connected:
                # OB-OB filtering failed, skip creation
                continue
        # Passed all filters -> create bond
        bonds.add(b)

    return set(bonds)

# ---- chunked parallel orchestration helpers ----

def split_indices_into_chunks(n_frames, n_workers, chunk_size=None, overlap=2):
    """
    Split 0..n_frames-1 into chunks for parallel processing.
    If chunk_size is None: auto determine chunk_size = ceil(n_frames/n_workers)
    overlap: number of frames overlapped on each side to maintain hysteresis continuity
    Returns list of (start_idx, end_idx) inclusive for each chunk (with overlap)
    and also the core (non-overlapped) interval for trimming when merging.
    """
    if chunk_size is None:
        chunk_size = int(math.ceil(n_frames / n_workers))
    chunks = []
    for i in range(0, n_frames, chunk_size):
        s = i
        e = min(n_frames - 1, i + chunk_size - 1)
        s_ext = max(0, s - overlap)
        e_ext = min(n_frames - 1, e + overlap)
        chunks.append((s_ext, e_ext, s, e))
    return chunks

def process_chunk_build_graphs(chunk_frames,
                               sisi_cutoff=SiSi_distance,
                               siob_cutoff=SiOB_distance,
                               ob_ob_cutoff=OBOB_distance,
                               create_margin=HYSTERESIS_MARGIN,
                               debug=False):
    """
    Process one chunk sequentially, and return a list of bond-sets
    (each element is a set of frozenset pairs for one frame).
    For each frame, call build_graphs_* to perform graph construction with hysteresis.
    """
    results = []
    prev_bonds = set()  # Keep hysteresis within this chunk

    for local_idx, frame in enumerate(chunk_frames):
        # frame should be (positions, cell, time)
        try:
            frame_data, static = frame
        except Exception as e:
            raise RuntimeError(f"Bad frame format at local_idx {local_idx}: {e}")

        try:
            pos_all = frame_data["pos"]
            idx_Si = static["idx_Si"]
            idx_OB = static["idx_OB"]
            Si_pos = pos_all[idx_Si]
            OB_pos = pos_all[idx_OB]
            cell    = frame_data["cell"]
        except Exception as e:
            raise RuntimeError(f"Bad positions format at local_idx {local_idx}: {e}")

        cell_vec = ensure_cell_vector(cell)

        # ---------- First frame in the chunk: build by pure distance (direct) ----------
        if prev_bonds is None:
            # First frame of the chunk: pure distance + OB filtering
            new_bonds = build_graphs_direct(
                Si_pos, OB_pos, cell_vec,
                si_si_cutoff=sisi_cutoff,
                ob_cutoff=ob_ob_cutoff,
                siob_cutoff=siob_cutoff
            )
        else:
            # Subsequent frames: normal hysteresis
            new_bonds = build_graphs_hysteresis(
                Si_pos, OB_pos, cell_vec, prev_bonds,
                sisi_cutoff=sisi_cutoff,
                ob_cutoff=ob_ob_cutoff,
                siob_cutoff=siob_cutoff,
                create_margin=create_margin
            )
        # new_bonds must be set of frozenset
        if not isinstance(new_bonds, set):
            # If build_graphs_hysteresis returns other types, try converting or raise error
            try:
                new_bonds = set(new_bonds)
            except Exception:
                raise RuntimeError(f"build_graphs_hysteresis returned non-set at local_idx {local_idx}")

        results.append(new_bonds)
        prev_bonds = new_bonds

        if debug:
            print(f"[chunk] frame {local_idx}: Si_count={len(Si_pos)}, bonds_count={len(new_bonds)}")

    return results

def get_Qn_info(G):
    degrees = [d for _, d in G.degree()]
    return {q: degrees.count(q) for q in set(degrees)}

def get_chain_info(G):
    result = { 'linear': {}, 'branch': {}, 'cyclic': {} }
    for comp in nx.connected_components(G):
        sub = G.subgraph(comp)
        length = len(sub)
        degrees = [d for _, d in sub.degree()]

        if any(len(c) >= 3 for c in nx.cycle_basis(sub)):
            key = 'cyclic'
        elif max(degrees) > 2:
            key = 'branch'
        else:
            key = 'linear'
        
        if length not in result[key]:
            result[key][length] = 0
        result[key][length] += 1
        
    return result

def get_rings(G):
    """
    Find all independent chordless cycles in an undirected graph.
    Steps:
    1. Generate all continuous vertex triplets (A-B-C).
    2. For each triplet, search for the smallest cycle containing it (start from 3-membered).
    3. Ensure the cycle is chordless (no shortcuts).
    4. Collect all cycles and remove duplicates.
    """

    def _is_chordless(cycle, graph):
        """
        Check whether a cycle is chordless.
        For any pair of non-adjacent nodes in the cycle,
        there should be no edge between them.
        """
        n = len(cycle)
        for i in range(n):
            for j in range(i + 2, n):
                # Skip the pair (first, last), which are adjacent in a cycle
                if (i == 0) and (j == n - 1):
                    continue
                # If a non-adjacent pair has an edge, there is a chord
                if graph.has_edge(cycle[i], cycle[j]):
                    return False
        return True

    def _normalize_cycle(cycle):
        """
        Normalize cycle representation:
        1. Rotate so that the smallest node is at the first position.
        2. Choose the lexicographically smallest direction.
        """
        min_index = cycle.index(min(cycle))
        rotated = cycle[min_index:] + cycle[:min_index]
        reversed_rotated = [rotated[0]] + list(reversed(rotated[1:]))
        return min(tuple(rotated), tuple(reversed_rotated))

    # Step 1: generate all triplets (u, v, w)
    triplets = []
    for v in G.nodes():
        neighbors = list(G.neighbors(v))
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                u, w = neighbors[i], neighbors[j]
                triplets.append((u, v, w))

    # Store all found cycles (before deduplication)
    all_cycles = []
    
    # Step 2: search cycles for each triplet
    for triplet in triplets:
        u, v, w = triplet
        
        # Check for 3-membered ring
        if G.has_edge(u, w):
            cycle = [u, v, w]
            if _is_chordless(cycle, G):
                all_cycles.append(tuple(sorted(cycle)))
            # Already found smallest ring, skip larger search
            continue
        
        # BFS for larger rings (4 and above)
        queue = deque([(w, [w])])  # (current node, path)
        visited = set([u, v, w])
        found = False
        
        while queue and not found:
            current, path = queue.popleft()
            
            for neighbor in G.neighbors(current):
                # Found ring path (back to starting node u)
                if neighbor == u and len(path) >= 2:
                    cycle = [u, v] + path
                    if _is_chordless(cycle, G):
                        all_cycles.append(_normalize_cycle(cycle))
                        found = True  # Mark smallest ring found
                
                # Continue extending path
                elif neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
    
    # Step 3: deduplicate and return unique cycles
    unique_cycles = set()
    for cycle in all_cycles:
        # Normalize cycle representation
        unique_cycles.add(_normalize_cycle(cycle))
    return unique_cycles

def get_ring_info(G):
    unique_cycles = get_rings(G)
    if unique_cycles:
        cycles_sizes = [len(cycle) for cycle in unique_cycles]
        cycles_counts = {size: cycles_sizes.count(size) for size in set(cycles_sizes)}
        return cycles_counts
    return {}

def get_cluster_info(G):
    clusters = list(nx.connected_components(G))
    clusters = [c for c in clusters if len(c) >= 50]
    max_size = max(len(c) for c in clusters) if clusters else 0.0
    mean_size = np.mean([len(c) for c in clusters]) if clusters else 0.0
    num_clusters = len(clusters)
    return max_size, mean_size, num_clusters

def build_Graph_from_bonds(bonds, N):
    # Construct adjacency matrix and graph from bond sets
    adj = np.zeros((N, N), dtype=int)
    for b in bonds:
        i, j = tuple(b)
        adj[i, j] = adj[j, i] = 1

    return nx.from_numpy_array(adj)

def analyze_graph_frame(args):
    """
    Worker function for single-frame analysis (parallel).
    args: (frame_index, bonds, topology)
    returns: (time, (cluster_info, qn, chain, ring))
    """
    idx, bonds, topology = args
    frame_data, static = topology
    G = build_Graph_from_bonds(bonds, len(static["idx_Si"]))
    # Use helper functions: get_Qn_info, get_chain_info, get_ring_info, get_cluster_info
    qn = get_Qn_info(G)
    chain = get_chain_info(G)
    ring = get_ring_info(G)
    max_cluster, mean_cluster, number_clusters = get_cluster_info(G)
    # Keep return format consistent with the original process_trajectory
    cluster_info = [max_cluster, mean_cluster, number_clusters]  # The last components (e.g. amorphous) can be filled later if needed
    time = frame_data["time"]
    return (time, (cluster_info, qn, chain, ring))

def _classify_clusters_from_graph(G):
    """
    Classify clusters based on the bond graph G.
    Returns:
      - node2comp: node index -> connected-component ID
      - comp_info: component ID -> (size_si, ctype, comp_nodes_set)
    """
    comps = list(nx.connected_components(G))
    node2comp, comp_info = {}, {}
    for cid, comp in enumerate(comps):
        comp = set(comp)
        size_si = len(comp)
        if size_si == 1:
            ctype = 'monomer'
        elif size_si <= 8:            
            ctype = 'oligomer'
        elif size_si <= 50:
            ctype = 'polymer'
        else:
            ctype = None
        for n in comp:
            node2comp[n] = cid
        comp_info[cid] = (size_si, ctype, comp)
    return node2comp, comp_info


def build_clusters(si_positions, cell, cutoff):
    """
    Build clusters based on Si–Si distance.
    cutoff: SiSi_neighbor_distance
    Returns:
        node2cluster: dict, {atom_index: cluster_id}
        clusters: list of sets, each being a cluster of atom indices
    """
    cell_vec = ensure_cell_vector(cell)
    adj = get_adj_matrix(si_positions, cell_vec, cutoff)
    G = nx.from_numpy_array(adj)

    clusters = list(nx.connected_components(G))
    node2cluster = {}
    for cid, comp in enumerate(clusters):
        if len(comp) >= 50:
            for n in comp:
                node2cluster[n] = cid
        else:
            for n in comp:
                node2cluster[n] = None

    return node2cluster, clusters

def analyze_graph_pair_reaction(args):
    """
    worker function for reaction analysis between two consecutive frames (parallel)
    args: (idx, now_graph, next_graph, now_topology, next_topology)
    returns: (time, reactions_count_dict)
    """
    idx, now_bonds, next_bonds, now_top, next_top = args
    now_frame_data, now_static = now_top
    next_frame_data, next_static = next_top

    # ---- Time information: current frame vs next frame ----
    t_now  = float(now_frame_data["time"])
    t_next = float(next_frame_data["time"])

    now_G = build_Graph_from_bonds(now_bonds, len(now_static["idx_Si"]))
    next_G = build_Graph_from_bonds(next_bonds, len(next_static["idx_Si"]))

    now_Si_positions = now_frame_data["pos"][now_static["idx_Si"]]
    now_cell = now_frame_data["cell"]

    # degrees and edges from prebuilt graphs
    now_degrees = dict(now_G.degree())
    next_degrees = dict(next_G.degree())
    now_edges = set(frozenset(e) for e in now_G.edges())
    next_edges = set(frozenset(e) for e in next_G.edges())
    new_edges = next_edges - now_edges

    # cluster classification using helper _classify_clusters_from_graph
    now_node2comp, now_comp_info = _classify_clusters_from_graph(now_G)
    # Current clusters for intra-/inter-cluster classification using build_clusters
    # (based on SiSi_neighbor_distance)
    now_node2cluster, now_clusters = build_clusters(now_Si_positions, now_cell, SiSi_neighbor_distance)

    # Initialize counters
    reactions_count = {
        'monomer_monomer': 0,
        'monomer_oligomer': 0,
        'oligomer_oligomer': 0,
        'oligomer_polymer': 0,
        'polymer_polymer': 0,

        'linear_condensation': 0,
        'branched_condensation': 0,
        'inter_chain_crosslink': 0,

        'intra_cluster_condensation': 0,
        'cluster_cluster_interaction': 0,

        'ring_formation': 0,
    }

    for edge in new_edges:
        i, j = tuple(edge)
        qi_now, qj_now = now_degrees.get(i, 0), now_degrees.get(j, 0)

        # Bond-type classification
        if qi_now <= 1 and qj_now <= 1:
            reactions_count['linear_condensation'] += 1
        elif (qi_now <= 1 and qj_now == 2) or (qi_now == 2 and qj_now <= 1):
            reactions_count['branched_condensation'] += 1

        if qi_now >= 2 and qj_now >= 2:
            reactions_count['inter_chain_crosslink'] += 1

        # Loose cluster classification: intra vs inter
        ci = now_node2cluster.get(i, None)
        cj = now_node2cluster.get(j, None)
        if (ci is not None) and (cj is not None):
            if ci == cj:
                reactions_count['intra_cluster_condensation'] += 1
            else:
                reactions_count['cluster_cluster_interaction'] += 1

        # monomer/oligomer/polymer classification
        ti_id = now_node2comp.get(i, None)
        tj_id = now_node2comp.get(j, None)
        # defensive check
        if ti_id is not None and tj_id is not None:
            _, ti, _ = now_comp_info[ti_id]
            _, tj, _ = now_comp_info[tj_id]
    
            allowed = ['monomer', 'oligomer', 'polymer']
            # 如果有一个是 None（或不在 allowed 里），直接跳过
            if ti not in allowed or tj not in allowed:
                continue

            pair = tuple(sorted([ti, tj], key=lambda x: ['monomer','oligomer','polymer'].index(x)))
            if pair == ('monomer','monomer'):
                reactions_count['monomer_monomer'] += 1
            elif pair == ('monomer','oligomer'):
                reactions_count['monomer_oligomer'] += 1
            elif pair == ('oligomer','oligomer'):
                reactions_count['oligomer_oligomer'] += 1
            elif pair == ('oligomer','polymer'):
                reactions_count['oligomer_polymer'] += 1
            elif pair == ('polymer','polymer'):
                reactions_count['polymer_polymer'] += 1

    # Count ring formation: use get_rings and compare set differences
    now_cycles = get_rings(now_G)
    next_cycles = get_rings(next_G)
    reactions_count['ring_formation'] = len(next_cycles - now_cycles)

    # time stamp: use next frame time as reference
    return (t_next, reactions_count)
