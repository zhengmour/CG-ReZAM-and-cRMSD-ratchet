import numpy as np
import matplotlib.pyplot as plt
from topology_constants import QN_LENGTH, MAX_LENGTH, CHAIN_LIST, REACTIONS
from topology_constants import PLOT_COLORS_6, PLOT_COLORS_12

def plot_qns(times, Qns):
    for i in range(QN_LENGTH):
        plt.plot(times, Qns[:, i], label=f'Q{i}', color=PLOT_COLORS_6[i])
    plt.plot(times, Qns[:, -1], label='c', color=PLOT_COLORS_6[-1])
    plt.xlabel('Time/ns')
    plt.ylabel('Molar Distribution')
    plt.ylim((0.0, 1.0))
    plt.title('Silicate Speciation Qn')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Analysis_Qns.png')
    plt.clf()

def plot_c_qn(Qns):
    c = Qns[:, -1]
    for i in range(QN_LENGTH):
        plt.plot(c, Qns[:, i], label=f'Q{i}', color=PLOT_COLORS_6[i])
    plt.xlabel('Degree of Condensation')
    plt.ylabel('Molar Distribution')
    plt.xlim((0.0, 1.0))
    plt.ylim((0.0, 1.0))
    plt.title('Silicate Speciation Qn')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Analysis_c_Qns.png')
    plt.clf()    

def plot_chains(times, chains):
    for data, label in zip(chains, CHAIN_LIST):
        for i in range(1, MAX_LENGTH):
            plt.plot(times, data[:, i], label=f'{label[0]} {i + 1}', color=PLOT_COLORS_12[i])
        plt.xlabel('Time/ns')
        plt.ylabel(label)
        plt.title(f'{label} Length Distribution')
        plt.legend(ncol=2, fontsize=8)
        plt.tight_layout()
        plt.savefig(f'Analysis_Chains_{label.split()[0].lower()}.png')
        plt.clf()

def plot_rings(times, rings):
    for i in range(2, MAX_LENGTH):
        plt.plot(times, rings[:, i], label=f'Ring{i + 1}', color=PLOT_COLORS_12[i])
    plt.xlabel('Time/ns')
    plt.ylabel('Ring Count')
    plt.title('Ring Size Distribution')
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig('Analysis_Rings.png')
    plt.clf()

def plot_Nrings(times, rings):
    Nrings = np.sum(rings, axis=1)
    plt.plot(times, Nrings, label=f'Number of Rings')
    plt.xlabel('Time/ns')
    plt.ylabel('Number Of Rings')
    plt.title('Number Of Rings')
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig('Analysis_NRings.png')
    plt.clf()    

def plot_clusters(times, cluster_infos):
    plt.plot(times, cluster_infos[:, 0], label='Max Cluster Size', color='tab:blue')
    plt.xlabel('Time / ns')
    plt.ylabel('Cluster Size')
    plt.title('Maximum Cluster Size')
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig('Analysis_MaxCluster.png')
    plt.clf()   

    fig, ax1 = plt.subplots()
    # Left Y-axis (first dataset: mean)
    color1 = 'tab:blue'
    ax1.set_xlabel('Time / ns')
    ax1.set_ylabel('Cluster Size')
    ax1.plot(times, cluster_infos[:, 1], label='Mean Cluster Size', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor=color1)

    # Right Y-axis (second dataset: morethans)
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('Cluster Count')
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.plot(times, cluster_infos[:, 1], label='Number of Clusters', color='tab:green')

    # Plot title and layout
    plt.title('Mean Cluster Size and Cluster Count over Time')
    fig.tight_layout()
    plt.savefig('Analysis_MeanAndCountCluster.png')
    plt.clf()

def plot_rmsds(times, rmsds):
    plt.plot(times, rmsds, label=f'RMSD')
    plt.xlabel('Frame')
    plt.ylabel('Root Mean Square Deviation')
    plt.title('Root Mean Square Deviation')
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig('Analysis_pirmsd.png')
    plt.clf()

def plot_envs(times, means, morethans):
    fig, ax1 = plt.subplots()

    # Left Y-axis (first dataset: means)
    color1 = 'tab:blue'
    ax1.set_xlabel('Frame')
    ax1.set_ylabel('Mean of Env_CV', color=color1)
    ax1.plot(times, means, color=color1, label='Mean')
    ax1.tick_params(axis='y', labelcolor=color1)

    # Right Y-axis (second dataset: morethans)
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('The Number of Env_CV more than 0.70', color=color2)
    ax2.plot(times, morethans, color=color2, linestyle='--', label='Morethan')
    ax2.tick_params(axis='y', labelcolor=color2)

    # Plot title and layout
    plt.title('Env_CV and Exceeding Threshold over Time')
    fig.tight_layout()
    plt.savefig('Analysis_rienv.png')
    plt.clf()

def plot_reactions(times, reactions_count):
    for i ,(key, value) in enumerate(REACTIONS.items()):
        if key.endswith("_rate"):
            plt.plot(times, reactions_count[key], label=f'{value}', color=PLOT_COLORS_12[i])
    plt.xlabel('Time/ns')
    plt.ylabel('Number of reaction')
    plt.title('Silicte reactions')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Analysis_reactions.png')
    plt.clf() 
