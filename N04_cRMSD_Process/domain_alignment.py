"""Birth-only greedy assignment / fixed-mapping BFGS, using unit quaternions.

Never differentiate across a greedy assignment update. Optimize squared bounded
loss in rotation-vector tangent coordinates plus optional lattice translation;
compose proper unit quaternions to enforce SO(3). Established domains use material transport instead.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree


def greedy_pairs(reference_positions, atom_positions, cutoff, reference_ids, atom_ids, box=None):
    """Global nearest-pair greedy assignment, with stable ID tie breaking.

    Store only pairs inside the bounded-loss support; the remaining slots are
    vacancies. Optional periodic trees reproduce minimum-image donor matching.
    """
    ref=np.asarray(reference_positions,dtype=float)
    atoms=np.asarray(atom_positions,dtype=float)
    if not len(ref) or not len(atoms):return []
    if box is not None:
        ref=ref%box;atoms=atoms%box
    neighbors=cKDTree(ref,boxsize=box).sparse_distance_matrix(
        cKDTree(atoms,boxsize=box),cutoff,output_type='coo_matrix')
    order=np.lexsort((np.asarray(atom_ids)[neighbors.col],
                      np.asarray(reference_ids)[neighbors.row],neighbors.data))
    taken_ref=np.zeros(len(ref),dtype=bool);taken_atom=np.zeros(len(atoms),dtype=bool)
    pairs=[]
    for entry in order:
        i,j=int(neighbors.row[entry]),int(neighbors.col[entry])
        if taken_ref[i] or taken_atom[j] or neighbors.data[entry]>=cutoff:continue
        taken_ref[i]=taken_atom[j]=True
        pairs.append((i,j,float(neighbors.data[entry])))
    return pairs


def rematch_occupied(domain, locked, xyz, box, cutoff, material):
    """Minimum total squared distance for the same occupied atoms at a branch update.

    Core pairs are fixed. Each peripheral atom may retain its old slot or
    take any free/non-core slot inside the minimum-image capture distance.
    """
    old = {atom: slot for slot, atom in domain['mapping']}
    atoms = np.array(sorted(old.keys() - locked), dtype=int)
    if not len(atoms):
        return
    fixed = [[slot, atom] for atom, slot in old.items() if atom in locked]
    slots = np.array(sorted(set(range(len(domain['targets']))) - {s for s, _ in fixed}))
    targets = np.asarray(domain['targets'])[slots] @ material['rotation'].T
    points = xyz[atoms] - material['center']
    edges = cKDTree(points % box, boxsize=box).sparse_distance_matrix(
        cKDTree(targets % box, boxsize=box), cutoff, output_type='coo_matrix')
    prior = np.searchsorted(slots, [old[i] for i in atoms])
    nearby = (edges.col != prior[edges.row]) & (edges.data < cutoff)
    delta = points - targets[prior]
    delta -= box * np.round(delta / box)
    fallback = np.einsum('ij,ij->i', delta, delta)
    rows = np.r_[edges.row[nearby], np.arange(len(atoms))]
    cols = np.r_[edges.col[nearby], prior]
    # Sparse matching ignores zero weights. Adding a constant per atom
    # preserves the optimum and makes exact coincidences valid edges.
    costs = 1.0 + np.r_[edges.data[nearby]**2, fallback]
    graph = csr_matrix((costs, (rows, cols)), shape=(len(atoms), len(slots)))
    idx, chosen = min_weight_full_bipartite_matching(graph)
    if np.array_equal(chosen, prior[idx]):
        return
    # Preserve existing identities for equal-cost symmetric permutations.
    new_cost = np.asarray(graph[idx, chosen]).ravel().sum()
    if new_cost < len(atoms) + fallback.sum() - 1e-10:
        domain['mapping'] = sorted(fixed + [[int(slots[j]), int(atoms[i])]
                                            for i, j in zip(idx, chosen)])


def stored_euler(rotation):
    """Legacy xyz angles, with an atan2 pitch and exact gimbal representative.

    Optimization uses quaternions. Older SciPy Euler conversion loses precision
    at pitch = +/- pi/2; atan2 avoids the square-root/acos cancellation there.
    """
    r=rotation.as_matrix()
    cosine=np.hypot(r[0,0],r[1,0])
    pitch=np.arctan2(-r[2,0],cosine)
    if cosine<1e-12:
        roll=np.arctan2(-r[1,2],r[1,1]);yaw=0.
    else:
        roll=np.arctan2(r[2,1],r[2,2]);yaw=np.arctan2(r[1,0],r[0,0])
    return [float(roll),float(pitch),float(yaw)]


def registration_seeds(base, solve, center, cutoff):
    """Core-only rigid triplet seeds: resolve lattice phase as well as rotation.

    A core centroid is generally not a lattice origin. Match a noncollinear
    local triplet to central reference motifs, then score using the whole core.
    No peripheral atoms participate and existing domains never call this search.
    """
    points=solve[:,2:5]-center
    if len(points)<3:return []
    anchor=int(np.argmin(np.linalg.norm(points,axis=1)))
    distances=np.linalg.norm(points-points[anchor],axis=1)
    near=[i for i in np.argsort(distances,kind='stable') if distances[i]>1e-8][:4]
    if len(near)<2:return []
    pair=max(((i,j) for i in near for j in near if i<j),
             key=lambda ij:np.linalg.norm(np.cross(points[ij[0]]-points[anchor],points[ij[1]]-points[anchor])))
    i,j=pair
    target=points[[anchor,i,j]]
    if np.linalg.matrix_rank(target-target.mean(axis=0))<2:return []
    lengths=[distances[i],distances[j],np.linalg.norm(points[i]-points[j])]
    tolerance=.2*cutoff;tree=cKDTree(base);seeds=[]
    for a in np.argsort(np.linalg.norm(base,axis=1),kind='stable')[:32]:
        neighbors=sorted(tree.query_ball_point(base[a],max(lengths[:2])+tolerance))
        first=[b for b in neighbors if b!=a and abs(np.linalg.norm(base[b]-base[a])-lengths[0])<=tolerance]
        second=[c for c in neighbors if c!=a and abs(np.linalg.norm(base[c]-base[a])-lengths[1])<=tolerance]
        for b in first:
            for c in second:
                if b==c or abs(np.linalg.norm(base[b]-base[c])-lengths[2])>tolerance:continue
                ref=base[[a,b,c]];rc=ref.mean(axis=0);tc=target.mean(axis=0)
                u,_,vt=np.linalg.svd((ref-rc).T@(target-tc))
                parity=np.eye(3);parity[2,2]=np.linalg.det(vt.T@u.T)
                matrix=vt.T@parity@u.T
                seeds.append((Rotation.from_matrix(matrix),tc-matrix@rc))
    return seeds


def align(reference, solve, cutoff, core, comm, iterations, fit_translation=False):
    reference=np.asarray(reference,dtype=float)
    solve=np.asarray(solve,dtype=float)
    center=reference[:,2:5].mean(axis=0)
    base=reference[:,2:5]-center
    outer,inner,tolerance=iterations
    rank,size=comm.Get_rank(),comm.Get_size()
    slots={int(row[0]):i for i,row in enumerate(reference)}
    atoms={int(row[0]):i for i,row in enumerate(solve)}
    def assign(q,translation):
        positions=base@q.as_matrix().T+center+translation
        pairs=greedy_pairs(positions,solve[:,2:5],cutoff,reference[:,0],solve[:,0])
        rows=[];loss=float(len(reference)-len(pairs))
        for i,j,distance in pairs:
            error=solve[j,2:5]-positions[i]
            u=distance**2/cutoff**2;z=1.-u
            loss+=u*(3.-3.*u+u*u)
            rows.append([solve[j,0],solve[j,1],*(6*z*z*error/cutoff**2),0.,0.,0.,1.,reference[i,0]])
        return np.sqrt(loss/len(reference)),rows
    # Global-index seeds produce the same starts regardless of MPI partition.
    starts=[]
    n_starts=max(1,int(outer))
    for index in range(rank,n_starts,size):
        q=Rotation.identity() if index==0 else Rotation.from_quat(np.random.default_rng(1729+index).normal(size=4))
        translation=np.zeros(3)
        value,rows=assign(q,translation)
        starts.append((value,index,q,translation,rows))
    if fit_translation:
        for index,(q,translation) in enumerate(registration_seeds(base,solve,center,cutoff)):
            if index%size!=rank:continue
            value,rows=assign(q,translation)
            starts.append((value,n_starts+index,q,translation,rows))
    # Refine the same two candidates as a serial search. Only scores and
    # global seed IDs cross ranks; the owner retains the assignment rows.
    local_starts = sorted(starts, key=lambda item: (item[0], item[1]))[:2]
    shortlists = comm.allgather([(s[0], s[1]) for s in local_starts])
    finalists = sorted(seed for shortlist in shortlists for seed in shortlist)[:2]
    priority = {index: order for order, (_, index) in enumerate(finalists)}
    best=(float('inf'),len(finalists),Rotation.identity(),np.zeros(3))
    for value,index,q,translation,rows in local_starts:
        if index not in priority:continue
        order=priority[index]
        if (value,order)<best[:2]:best=(value,order,q,translation.copy())
        for _ in range(3):
            if not rows or value<1e-12:break
            ref_ids=np.array([slots[int(row[9])] for row in rows])
            atom_ids=np.array([atoms[int(row[0])] for row in rows])
            anchor=q
            shift=translation.copy()
            def objective(increment):
                trial=Rotation.from_rotvec(increment[:3])*anchor
                error=solve[atom_ids,2:5]-center-base[ref_ids]@trial.as_matrix().T-shift
                if fit_translation:error-=increment[3:]
                u=np.sum(error*error,axis=1)/cutoff**2
                losses=np.where(u<1,u*(3-3*u+u*u),1.)
                return (len(reference)-len(rows)+losses.sum())/len(reference)
            fit=minimize(objective,np.zeros(6 if fit_translation else 3),method='BFGS',jac='3-point',
                         options=dict(maxiter=int(inner),gtol=min(float(tolerance),1e-8)))
            trial=Rotation.from_rotvec(fit.x[:3])*anchor
            new_translation=shift+fit.x[3:] if fit_translation else shift
            new_value,new_rows=assign(trial,new_translation)  # only between smooth optimizations
            if not np.isfinite(new_value) or new_value>=value-1e-12:break
            value,q,translation,rows=new_value,trial,new_translation,new_rows
            if (value,order)<best[:2]:best=(value,order,q,translation.copy())
    candidates=comm.allgather((best[0],best[1],rank))
    owner=min(candidates)[2]
    quaternion=comm.bcast(best[2].as_quat().tolist() if rank==owner else None,root=owner)
    q=Rotation.from_quat(quaternion)
    translation=np.array(comm.bcast(best[3].tolist() if rank==owner else None,root=owner))
    value,rows=assign(q,translation)
    # Keep the old serialized move layout for readers; no Euler-angle optimization.
    move=translation.tolist()+stored_euler(q)
    return value,rows,move
