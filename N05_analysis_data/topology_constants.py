# topology_constants.py

# --- Reaction settings ---
distance_threshold_reaction1 = 3.05  # Si-Si attack distance
distance_threshold_reaction2 = 1.70  # Si-O bond breaks
distance_threshold_reaction3 = 1.05  # OH and H form the distance of H2O

SiSi_distance = 3.30
OBOB_distance = 1.35
SiOB_distance = 1.20
SiSi_neighbor_distance = 4.80

REACTIONS = {
    'monomer_monomer': "Monomer+monomer",
    'monomer_oligomer': "Monomer+oligomer",
    'oligomer_oligomer': "Oligomer+oligomer",
    'oligomer_polymer': "Oligomer+polymer",
    'polymer_polymer': "Polymer+polymer",

    'linear_condensation': "Q0+Q1/Q1+Q1",
    'branched_condensation': "Q0+Q2/Q1+Q2",
    'inter_chain_crosslink': "Q2/Q3+Q2/Q3",

    'intra_cluster_condensation': "Intra-cluster condensation",
    'cluster_cluster_interaction': "Inter-cluster condensation",
    
    'ring_formation': "Ring formation",      
}

# --- Qn / Chain / Loop settings ---
QN_LENGTH = 6
MAX_LENGTH = 12
MAX_RING = 6
CHAIN_TYPE = 3
CHAIN_LIST = ['linear', 'branch', 'cyclic']

# --- Atom types ---
Si_type = 1
O_type  = [2, 3, 4]

# --- Plot style  ---
PLOT_COLORS_6 = [
    '#4d4d4d', '#e41a1c', '#377eb8',
    '#4daf4a', '#984ea3', '#ff7f00',  
]
PLOT_COLORS_12 = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    "#bcbd22", "#17becf", "#f781bf", "#a65628"
]

