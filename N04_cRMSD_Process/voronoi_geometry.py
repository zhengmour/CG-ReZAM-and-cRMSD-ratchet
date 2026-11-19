"""Periodic nearest-core ownership and branch-bound radial weights."""
import numpy as np
from scipy.spatial import cKDTree
from scipy.special import expit
from material_frame import minimum_image, frame, rotation_pullback


def radial_weights(distance, r_cut, alpha):
    return expit(-alpha*(np.asarray(distance)-r_cut))


def ownership(xyz, box, population_ids, cores, previous=None):
    """No radius/graph cutoff. Exact ties retain old DOMAIN, then persistent ID."""
    ids=np.asarray(population_ids,dtype=int)
    live=sorted((c for c in cores if c['status']!='retired'),key=lambda c:c['uid'])
    if not live:return {},{}
    d=np.asarray([cKDTree(np.asarray(xyz)[c['members']]%box,boxsize=box).query(
        np.asarray(xyz)[ids]%box)[0] for c in live])
    owners={};nearest={};previous=previous or {}
    for j,i in enumerate(ids):
        candidates=np.flatnonzero(np.abs(d[:,j]-d[:,j].min())<=1e-10)
        old=previous.get(int(i),previous.get(str(i)))
        k=min(candidates,key=lambda k:(live[k]['domain']!=old,live[k]['uid']))
        owners[int(i)]=int(live[k]['domain']);nearest[int(i)]=int(live[k]['uid'])
    return owners,nearest


def set_radius(domain,xyz,box,settings):
    """Initialize a branch once: robust domain radius; never per-force recalc."""
    material=frame(domain,xyz,box)
    distances=np.linalg.norm(minimum_image(xyz[domain['ids']]-material['center'],box),axis=1)
    radius=float(np.quantile(distances,settings['radius_quantile']))
    # Include all anchors in the scale estimate; do not let an isolated donor
    # set the whole domain radius. The estimator is part of the saved definition.
    core_radius=float(np.linalg.norm(minimum_image(xyz[domain['anchors']]-material['center'],box),axis=1).max())
    radius=max(radius,core_radius,settings['minimum_radius'])
    domain['r_cut']=settings['radius_scale']*radius
    domain['weight_alpha']=settings['weight_alpha']
    domain['radius_definition']=dict(quantile=settings['radius_quantile'],scale=settings['radius_scale'],
                                    initialized_radius=radius,minimum=settings['minimum_radius'])
    domain['radial_weighting']=True
    domain['slot_weights']=radial_weights(np.linalg.norm(domain['targets'],axis=1),domain['r_cut'],domain['weight_alpha']).tolist()
    domain['weight_radius']=domain['r_cut'];domain['weight_grid_offset']=[0.,0.,0.]


def evaluate_radial(domain,xyz,box,population,cutoff):
    """L=(N-n)+n*A/W, A=vacancy_weight+sum(w_i*ell_i).

    W=vacancy_weight+sum(w_i). Reference vacancy weights are fixed in the
    material frame; occupied weights use actual distances to c. The quotient,
    radial, center and proper-rotation derivatives are all included.
    """
    material=frame(domain,xyz,box);n=len(domain['ids'])
    ids=np.asarray(sorted(set(domain['ids'])|set(material['translation_ids'])|set(material['rotation_ids'])),int)
    index={int(a):j for j,a in enumerate(ids)};g=np.zeros((len(ids),3))
    mapping=np.asarray(domain['mapping'],int).reshape(-1,2)
    weights=np.asarray(domain['slot_weights']);occupied=set(mapping[:,0]) if len(mapping) else set()
    vacancy=float(sum(w for j,w in enumerate(weights) if j not in occupied))
    if not len(mapping):
        entries=np.zeros((len(ids),9));entries[:,0]=ids+1;entries[:,1]=1
        return float(population),entries.tolist()
    slots,atoms=mapping.T;targets=np.asarray(domain['targets'])[slots]@material['rotation'].T
    relative=minimum_image(xyz[atoms]-material['center'],box)
    r=np.linalg.norm(relative,axis=1)
    w=radial_weights(r,domain['r_cut'],domain['weight_alpha'])
    error=minimum_image(xyz[atoms]-material['center']-targets,box)
    u=np.sum(error*error,axis=1)/cutoff**2;z=np.maximum(0.,1.-u)
    ell=np.where(u<1,u*(3-3*u+u*u),1.)
    W=vacancy+float(w.sum());A=vacancy+float(w@ell)
    if W<=np.finfo(float).tiny:raise ValueError('Vanishing radial normalization')
    ratio=A/W;factor=n/W
    mismatch=factor*w[:,None]*6*z[:,None]**2*error/cutoff**2
    dw=-domain['weight_alpha']*(w*(1-w))[:,None]*relative/np.maximum(r[:,None],1e-30)
    weighted=factor*(ell-ratio)[:,None]*dw
    for atom,deriv in zip(atoms,mismatch+weighted):g[index[int(atom)]]+=deriv
    reaction=g.sum(axis=0)/len(material['translation_ids'])
    rotational=rotation_pullback(material,targets,mismatch)+rotation_pullback(material,np.zeros_like(targets),weighted)
    for a in material['translation_ids']:g[index[int(a)]]-=reaction
    for a,deriv in zip(material['rotation_ids'],rotational):g[index[int(a)]]+=deriv
    entries=np.zeros((len(ids),9));entries[:,0]=ids+1;entries[:,1]=1;entries[:,2:5]=g
    for a in atoms:entries[index[int(a)],8]=1
    return float(population-n+n*ratio),entries.tolist()
