"""Offline core-first ownership experiment; not wired into the bias potential.

Cores are supplied, never inferred by partitioning the domain. All surrounding
particles must be supplied, including particles outside a DBSCAN cluster.
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path


def distances(xyz, box, cores, cutoff=4.8):
    xyz=np.asarray(xyz,float)%box
    n=len(xyz)
    if not cores:return np.empty((0,n)),np.empty((0,n))
    flat=[int(i) for core in cores for i in core]
    if any(not len(c) for c in cores) or len(set(flat))!=len(flat):
        raise ValueError('Cores must be nonempty and disjoint')
    if min(flat)<0 or max(flat)>=n:raise ValueError('Invalid core index')
    tree=cKDTree(xyz,boxsize=box)
    pairs=tree.query_pairs(cutoff,output_type='ndarray')
    r=np.concatenate((pairs[:,0],pairs[:,1]));c=np.concatenate((pairs[:,1],pairs[:,0]))
    graph=csr_matrix((np.ones(len(r)),(r,c)),shape=(n,n))
    hops=[];spatial=[]
    for core in cores:
        hops.append(shortest_path(graph,directed=False,unweighted=True,indices=core).min(axis=0))
        spatial.append(cKDTree(xyz[core],boxsize=box).query(xyz)[0])
    return np.asarray(hops),np.asarray(spatial)


def assign(hops, spatial, cores, layers=5, tt_distance=3.1, method='graph'):
    """Unique ownership; -1 means unassigned, never silently discarded.

    graph: nearest core by graph hops, then Euclidean distance, then core order.
    spatial: distance to nearest CORE ATOM <= layers * tt_distance (not center).
    halo: graph ownership plus spatial capture ONLY if no core is reachable.
    Core particles retain their supplied ownership. No c/R refitting occurs.
    """
    if layers<0 or tt_distance<=0:raise ValueError('Invalid growth extent')
    if method not in ('graph','spatial','halo'):raise ValueError('Unknown method')
    n=spatial.shape[1];owner=np.full(n,-1,int)
    if not len(cores):return owner,np.zeros(n,int)
    metric=spatial/tt_distance if method=='spatial' else hops.copy()
    if method=='halo':
        disconnected=~np.isfinite(hops).any(axis=0)
        metric[:,disconnected]=spatial[:,disconnected]/tt_distance
    eligible=metric<=layers
    for i in range(n):
        choices=np.flatnonzero(eligible[:,i])
        if len(choices):owner[i]=min(choices,key=lambda k:(metric[k,i],spatial[k,i],k))
    for k,core in enumerate(cores):owner[core]=k
    return owner,eligible.sum(axis=0)
