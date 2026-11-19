import numpy as np
import matplotlib.pyplot as plt
from topology_constants import QN_LENGTH, MAX_LENGTH, CHAIN_LIST, REACTIONS
from topology_constants import PLOT_COLORS_6, PLOT_COLORS_12

TIME_LABEL = "Time / ns"

def plot_qns(times, Qns):
    for i in range(QN_LENGTH):
        plt.plot(times, Qns[:, i], label=f'Q{i}', color=PLOT_COLORS_6[i])
    plt.plot(times, Qns[:, -1], label='c', color=PLOT_COLORS_6[-1])
    plt.xlabel(TIME_LABEL)
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
        plt.xlabel(TIME_LABEL)
        plt.ylabel(label)
        plt.title(f'{label} Length Distribution')
        plt.legend(ncol=2, fontsize=8)
        plt.tight_layout()
        plt.savefig(f'Analysis_Chains_{label.split()[0].lower()}.png')
        plt.clf()

def plot_rings(times, rings):
    for i in range(2, MAX_LENGTH):
        plt.plot(times, rings[:, i], label=f'Ring{i + 1}', color=PLOT_COLORS_12[i])
    plt.xlabel(TIME_LABEL)
    plt.ylabel('Ring Count')
    plt.title('Ring Size Distribution')
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig('Analysis_Rings.png')
    plt.clf()

def plot_Nrings(times, rings):
    Nrings = np.sum(rings, axis=1)
    plt.plot(times, Nrings, label=f'Number of Rings')
    plt.xlabel(TIME_LABEL)
    plt.ylabel('Number Of Rings')
    plt.title('Number Of Rings')
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig('Analysis_NRings.png')
    plt.clf()    

def plot_clusters(times, cluster_infos):
    plt.plot(times, cluster_infos[:, 0], label='Max Cluster Size', color='tab:blue')
    plt.xlabel(TIME_LABEL)
    plt.ylabel('Cluster Size')
    plt.title('Maximum Cluster Size')
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig('Analysis_MaxCluster.png')
    plt.clf()   

    fig, ax1 = plt.subplots()
    # Left Y-axis (first dataset: mean)
    color1 = 'tab:blue'
    ax1.set_xlabel(TIME_LABEL)
    ax1.set_ylabel('Cluster Size')
    ax1.plot(times, cluster_infos[:, 1], label='Mean Cluster Size', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor=color1)

    # Right Y-axis (second dataset: morethans)
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('Cluster Count')
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.plot(times, cluster_infos[:, 2], label='Number of Clusters', color='tab:green')

    # Plot title and layout
    plt.title('Mean Cluster Size and Cluster Count over Time')
    fig.tight_layout()
    plt.savefig('Analysis_MeanAndCountCluster.png')
    plt.clf()

def plot_reactions(times, reactions_count):
    for i ,(key, value) in enumerate(REACTIONS.items()):
        plt.plot(times, reactions_count[key], label=value, color=PLOT_COLORS_12[i])
    plt.xlabel(TIME_LABEL)
    plt.ylabel('Number of reaction')
    plt.title('Silicte reactions')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Analysis_reactions.png')
    plt.clf() 


def plot_pirmsd_records(records):
    """Never join different metrics, segments, epochs, or protocol transitions."""
    if not records:
        return
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 8))
    x = [r['record'] for r in records]
    # A new epoch can start at every observation. Keep isolated points visible.
    for ax, field, unit in zip(axes, ('cluster_pirmsd_A', 'multidomain_pirmsd'),
                               ('Cluster PIRMSD (Å)', 'Multidomain PIRMSD (dimensionless)')):
        xs, ys, minima, transitions = [], [], [], []
        previous = None
        for row in records:
            index = row['record']
            primary = (field == 'cluster_pirmsd_A') == (row['metric'] == 'cluster_pirmsd_v1')
            value = row[field] if row['valid'] or not primary else None
            key = (row['segment'], row['metric'], row['epoch'])
            if key != previous or not row['comparable']:
                xs.append(np.nan)
                ys.append(np.nan)
            xs.append(index)
            ys.append(np.nan if value is None else value)
            previous = key if row['comparable'] and value is not None else None
            if primary and row['comparable'] and row['reason'] != 'continuous_bias' and row['xi_min'] is not None:
                minima.append((index, row['xi_min']))
            if value is not None and row['reason'] == 'reference_transition':
                transitions.append((index, value))
        ax.plot(xs, ys, '.-', markersize=2, linewidth=.7)
        if minima:
            ax.plot(*zip(*minima), '.', markersize=2, label='Comparable minimum')
        if transitions:
            ax.plot(*zip(*transitions), 'x', markersize=3, label='Reference transition')
        if minima or transitions:
            ax.legend()
        ax.set_ylabel(unit)
    axes[2].plot(x, [r['bias_energy'] for r in records], '.', markersize=2, color='tab:red')
    axes[2].set_ylabel('Bias energy (kcal/mol)')
    axes[2].set_xlabel('Observation index')
    fig.tight_layout()
    fig.savefig('Analysis_pirmsd.png')
    plt.close(fig)
