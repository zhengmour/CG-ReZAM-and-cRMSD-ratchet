"""Explicit paper equations (2)--(7), with finite-data checks."""
import numpy as np


def relative_energies(energies,reference,families):
    e=np.asarray(energies,dtype=float).copy();r=np.asarray(reference,dtype=float).copy()
    for family in sorted(set(families)):
        ids=np.flatnonzero(np.asarray(families)==family)
        anchor=ids[np.argmin(r[ids])]
        e[ids]-=e[anchor];r[ids]-=r[anchor]
    return e,r


def qm_losses(predicted,reference):
    if set(predicted)!=set(reference) or not reference:raise ValueError('QM sample IDs must match exactly')
    keys=sorted(reference)
    energies=np.array([predicted[k]['energy'] for k in keys],dtype=float)
    truth=np.array([reference[k]['energy'] for k in keys],dtype=float)
    force_sq=np.array([predicted[k]['force_squared_error'] for k in keys],dtype=float)
    if not np.isfinite(np.r_[energies,truth,force_sq]).all() or np.any(force_sq<0):raise ValueError('Nonfinite/invalid QM data')
    relative,ref_relative=relative_energies(energies,truth,[reference[k]['family'] for k in keys])
    le=float(np.sqrt(np.mean((relative-ref_relative)**2)))
    # F_k in Eq. (2) is the complete configuration force vector.
    lf=float(np.sqrt(np.mean(force_sq)))
    if len(keys)<2 or np.ptp(energies)==0 or np.ptp(truth)==0:
        # Pearson is undefined. Never silently reward a constant energy surface.
        raise ValueError('Zero-energy Pearson loss requires two nonconstant energy arrays')
    correlation=float(np.corrcoef(energies,truth)[0,1])
    lzero=1.-min(1.,abs(correlation))
    return np.array([le,lf,lzero])


def mle_weights(losses,floor=1e-8):
    losses=np.asarray(losses,dtype=float)
    if not np.isfinite(losses).all() or np.any(losses<0):raise ValueError('Invalid MLE losses')
    # d/d(log sigma) [L/(2 sigma^2)+log sigma]=0 => sigma^2=L.
    return 1/(2*np.maximum(losses,floor))


def weighted_loss(losses,weights):
    losses=np.asarray(losses,dtype=float);weights=np.asarray(weights,dtype=float)
    if losses.shape!=weights.shape or not np.isfinite(losses).all():raise ValueError('Invalid weighted loss')
    return float(np.dot(losses,weights))


def rdf_distance(current,reference,lower=1.5,cutoff=12.,metric='squared_l2'):
    arrays=[]
    for data in (current,reference):
        data=np.asarray(data,dtype=float)
        if data.ndim!=2 or data.shape[1]!=2 or len(data)<2 or not np.isfinite(data).all():raise ValueError('Invalid RDF data')
        if np.any(np.diff(data[:,0])<=0) or np.any(data[:,1]<0):raise ValueError('RDF radii must increase and g(r) be nonnegative')
        # CSV radii are bin centers; tolerate at most half a boundary bin.
        if data[0,0]-(data[1,0]-data[0,0])/2>lower+1e-7 or data[-1,0]+(data[-1,0]-data[-2,0])/2<cutoff-1e-7:
            raise ValueError('RDF does not cover the configured radial interval')
        arrays.append(data)
    grid=np.unique(np.concatenate([np.array([lower,cutoff])]+[d[(d[:,0]>lower)&(d[:,0]<cutoff),0] for d in arrays]))
    g,ref=[np.interp(grid,d[:,0],d[:,1]) for d in arrays]
    trap=getattr(np,'trapezoid',None) or np.trapz
    denominator=float(trap(ref*ref,grid))
    if denominator<=0:raise ValueError('Zero reference RDF normalization')
    if metric=='squared_l2':numerator=trap((g-ref)**2,grid)
    elif metric=='paper_signed':numerator=trap(g*g-ref*ref,grid)
    else:raise ValueError('Unknown RDF loss')
    return float(numerator/denominator)
