import numpy as np
from topology_constants import QN_LENGTH, MAX_LENGTH, CHAIN_TYPE, CHAIN_LIST, REACTIONS

def aggregate_topology_results(results, topologies):
    frame_number = len(results)
    times = [topo[-1] for topo in topologies]

    cluster_infos = np.zeros((frame_number, 6))
    Qns = np.zeros((frame_number, QN_LENGTH + 1))
    chains = [np.zeros((frame_number, MAX_LENGTH)) for _ in range(CHAIN_TYPE)]
    rings = np.zeros((frame_number, MAX_LENGTH))

    for i, (cluster_info, Qn_info, chain_info, ring_info) in enumerate(results):
        cluster_infos[i] = cluster_info

        for q, count in Qn_info.items():
            if q < QN_LENGTH:
                Qns[i, q] = count / len(topologies[i][0][0])

        Qns[i, QN_LENGTH] = sum(j * Qns[i, j] for j in range(1, QN_LENGTH)) / 4.0

        for idx, key in enumerate(CHAIN_LIST):
            if key not in chain_info:
                continue
            for length, count in chain_info[key].items():
                if length - 1 < MAX_LENGTH:
                    chains[idx][i, length - 1] = count

        for size, count in ring_info.items():
            if size - 1 < MAX_LENGTH:
                rings[i, size - 1] = count

    return times, Qns, chains, rings, cluster_infos

def aggregate_topology_reactions_results(results, ):
    times = np.array([t for (t, _) in results], dtype=float) / 1000.0
    reactions_dicts = [d for (_, d) in results]
    reactions_count = {
        key: np.array([d[key] for d in reactions_dicts])
        for key in REACTIONS.keys()
    }
    return times, reactions_count

