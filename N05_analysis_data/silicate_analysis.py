# %%
import sys
import numpy as np
from multiprocessing import Pool
from pathlib import Path
from tqdm import tqdm

from topology_constants import MAX_LENGTH, REACTIONS
from topology_io import read_gro_files, read_xtc_files, read_data_files, read_dump_files, read_env_files, read_rmsd_files, read_rmsd_dump_files
from topology_utils import split_indices_into_chunks, process_chunk_build_graphs, analyze_graph_frame, analyze_graph_pair_reaction
from topology_plot import plot_qns, plot_c_qn, plot_chains, plot_rings, plot_Nrings, plot_reactions, plot_clusters, plot_rmsds, plot_envs
from topology_analysis import aggregate_topology_results, aggregate_topology_reactions_results
from topology_constants import SiSi_distance, OBOB_distance, SiOB_distance

def main(file_type: str, n_loop: int, traj_path: str = "./"):
    traj_path = Path(traj_path)
    print(f"Reading {file_type} files with {n_loop} processes...")

    # --- Step 1: Read topology files ---
    print("Step 1: Reading topology files...")
    if file_type == 'gro':
        topologies = read_gro_files(traj_path, n_loop)
    elif file_type == 'xtc':
        topologies = read_xtc_files(traj_path, n_loop)
    elif file_type in ['lmp', 'data']:
        topologies = read_data_files(traj_path, n_loop)
    elif file_type in ['dump', 'lammpstrj']:
        topologies = read_dump_files(traj_path, n_loop)
    elif file_type in ['rmsd_lammpstrj']:
        topologies = read_rmsd_dump_files(traj_path, n_loop)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")
    print(f"Loaded {len(topologies)} frames.")
    
    # --- Step 2: Read environment info (COLVAR) or RMSD files if available ---
    print("Step 2: Reading environment (COLVAR) and RMSD data...")
    envs_result = read_env_files(traj_path, n_loop)
    if envs_result:
        np.savetxt("Analysis__rienv.csv", np.column_stack(envs_result), delimiter=",", header="Time, Mean, Morethan", comments='')
        plot_envs(envs_result[0], envs_result[1], envs_result[2])

    rmsds_result = read_rmsd_files(traj_path, n_loop)
    if rmsds_result:
        np.savetxt("Analysis__pirmsd.csv", np.column_stack(rmsds_result), delimiter=",", header="Frame, RMSD", comments='')
        plot_rmsds(rmsds_result[0], rmsds_result[1])
        
    print("Environment and RMSD data processing finished.")

    # --- Step 3.1: Topology analysis ---
    # Number of overlapping frames at both ends of each chunk,
    # used to maintain hysteresis continuity (>= hysteresis persistence)
    overlap = 500
    print(f"Step 3.1: Splitting {len(topologies)} frames into chunks for {n_loop} workers (overlap={overlap})...")
    n_frames = len(topologies)
    chunks = split_indices_into_chunks(n_frames, n_loop, overlap=overlap)

    # Build chunk input lists: each item passed to process_chunk_build_graphs is a tuple:
    # (chunk_frames_list, si_si_cutoff, siob_cutoff, ob_ob_cutoff)
    chunk_inputs = []
    for (s_ext, e_ext, s_core, e_core) in chunks:
        # Slice topologies for this chunk (inclusive)
        chunk_frames = topologies[s_ext:e_ext+1]
        chunk_inputs.append((chunk_frames, SiSi_distance, SiOB_distance, OBOB_distance))

    print("Building bonds per frame within chunks (parallel)...")
    with Pool(n_loop) as pool:
        # Each returned item is a list of bonds-set for frames in that chunk (in the same order)
        chunk_results = list(tqdm(pool.starmap(process_chunk_build_graphs, chunk_inputs), total=len(chunk_inputs)))

    # Merge chunks: remove overlap at both ends of each chunk, keep only the core s..e segment
    print("Merging chunk results and trimming overlaps...")
    graphs_as_sets = []   # Final list of sets-of-frozenset pairs, one per frame in time order
    for idx, ((s_ext, e_ext, s_core, e_core), bonds_list) in enumerate(zip(chunks, chunk_results)):
        # bonds_list length == e_ext - s_ext + 1
        trim_front = s_core - s_ext
        trim_back  = e_ext - e_core
        # slice [trim_front : len - trim_back]
        if trim_back == 0:
            kept = bonds_list[trim_front:]
        else:
            kept = bonds_list[trim_front: len(bonds_list) - trim_back]
        graphs_as_sets.extend(kept)

    # Sanity check
    if len(graphs_as_sets) != n_frames:
        raise RuntimeError(f"Chunk merge error: expected {n_frames} frames after merge, got {len(graphs_as_sets)}")

    print(f"Built bonds for all {n_frames} frames.")

    # --- Step 3.2: Aggregate analysis data ---
    print("Step 3.2: Analyzing frames in parallel (using prebuilt graphs)...")
    frame_args = [(i, graphs_as_sets[i], topologies[i]) for i in range(len(topologies))]
    with Pool(n_loop) as pool:
        frame_results = list(tqdm(pool.map(analyze_graph_frame, frame_args), total=len(frame_args)))
    # Sort by time
    frame_results.sort(key=lambda x: x[0])
    # Extract results (keep consistent with original aggregate interface)
    results = [r for (_, r) in frame_results]
    print(f"Frame analysis completed on {len(results)} frames.")

    # --- Step 3.3: Aggregate analysis data (same as before) ---
    print("Step 3.3: Aggregating topology metrics...")
    times, Qns, chains, rings, cluster_infos = aggregate_topology_results(results, topologies)
    times = np.array(times) / 1000  # convert to ns
    print("Topology aggregation finished.")

    # --- Step 3.4: Save results ---
    print("Step 3.4: Writing CSV output files...")
    np.savetxt("Analysis__Qns.csv", np.column_stack([times, Qns]), delimiter=",", header="Time, Q0, Q1, Q2, Q3, Q4, Q5, C", comments='')
    np.savetxt("Analysis__Linear.csv", np.column_stack([times, chains[0]]), delimiter=",", header="Time, " + ", ".join([f"L{i+1}" for i in range(MAX_LENGTH)]), comments='')
    np.savetxt("Analysis__Branch.csv", np.column_stack([times, chains[1]]), delimiter=",", header="Time, " + ", ".join([f"B{i+1}" for i in range(MAX_LENGTH)]), comments='')
    np.savetxt("Analysis__Cyclic.csv", np.column_stack([times, chains[2]]), delimiter=",", header="Time, " + ", ".join([f"C{i+1}" for i in range(MAX_LENGTH)]), comments='')
    np.savetxt("Analysis__Rings.csv", np.column_stack([times, rings]), delimiter=",", header="Time, " + ", ".join([f"R{i+1}" for i in range(MAX_LENGTH)]), comments='')
    np.savetxt("Analysis__Cluster.csv", np.column_stack([times, cluster_infos]), delimiter=",", header="Time, Maximum Cluster Size, Mean Cluster Size, Number of Clusters, Maximum Amorphous Size, Mean Amorphous Size, Number of Amorphouses", comments='')
    print("CSV files for topology statistics saved.")

    # --- Step 3.5: Plot results ---
    print("Step 3.5: Generating topology plots...")
    plot_qns(times, Qns)
    plot_c_qn(Qns)

    plot_chains(times, chains)

    plot_rings(times, rings)
    plot_Nrings(times, rings)

    plot_clusters(times, cluster_infos)
    print("Topology plotting finished.")

    # --- Step 4.1: Reaction analysis in parallel (using prebuilt graphs, uniform time interval) ---
    print("Step 4.1: Analyzing reactions in parallel (using prebuilt graphs)...")

    # Frame times from topologies
    frame_times = np.array([topologies[i][0]["time"] for i in range(len(topologies))], dtype=float)
    if len(frame_times) < 2:
        raise RuntimeError("Not enough frames for reaction analysis.")

    # Use the maximum consecutive dt as the target interval
    dt_list = np.diff(frame_times)
    target_dt = np.max(dt_list)

    # Build index pairs (i, j) so that time[j] - time[i] ~ target_dt
    pair_indices = []
    i = 0
    while i < len(frame_times) - 1:
        target_time = frame_times[i] + target_dt
        j_candidates = np.where(frame_times >= target_time)[0]
        j_candidates = j_candidates[j_candidates > i]
        if len(j_candidates) == 0:
            break
        j = int(j_candidates[0])
        pair_indices.append((i, j))
        i = j

    if not pair_indices:
        raise RuntimeError("No valid frame pairs found for reaction analysis.")

    pair_args = []
    for k, (i, j) in enumerate(pair_indices):
        pair_args.append((k, graphs_as_sets[i], graphs_as_sets[j], topologies[i], topologies[j]))

    with Pool(n_loop) as pool:
        # pair_results: list of (time, reactions_dict)
        pair_results = list(tqdm(pool.map(analyze_graph_pair_reaction, pair_args), total=len(pair_args)))

    # Sort by time (the first element of each tuple)
    pair_results.sort(key=lambda x: x[0])
    print(f"Reaction analysis completed on {len(pair_results)} frame pairs.")

    # --- Step 4.2: Aggregate reaction metrics ---
    print("Step 4.2: Aggregating reaction metrics...")
    times, reactions_count = aggregate_topology_reactions_results(pair_results)
    print("Reaction aggregation finished.")

    # --- Step 4.3: Save reaction results ---
    print("Step 4.3: Writing reaction CSV output files...")
    row = [times] + [reactions_count[k] for k in REACTIONS.keys()]
    header = "Time, " + ", ".join(REACTIONS.keys())
    np.savetxt("Analysis__reactions.csv", np.column_stack(row), delimiter=",", header=header, comments='')
    print("CSV files for reactions saved.")

    # --- Step 4.4: Plot reaction results ---
    print("Step 4.4: Generating reaction plots...")
    plot_reactions(times, reactions_count)
    print("All analysis and plotting finished.")

if __name__ == "__main__":
    file_type = sys.argv[1]
    n_loop = int(sys.argv[2])
    traj_path = sys.argv[3] if len(sys.argv) > 3 else "./"
    main(file_type, n_loop, traj_path)
