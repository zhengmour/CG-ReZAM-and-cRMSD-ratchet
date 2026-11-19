"""Grid-gauged material translation and proper rotation with its exact pullback.

The grid point is a coordinate origin, not an imposed physical center. A fixed
anchor correspondence defines physical rotation; no permutation search occurs.
"""
import numpy as np
from scipy.spatial.transform import Rotation


def minimum_image(delta, box):
    return delta-box*np.round(delta/box)


def orientation_defined(points):
    points=np.asarray(points,dtype=float)
    if len(points)<3:return False
    # Center relative displacements, avoiding spurious rank from large origins.
    p=points-points[0];p-=p.mean(axis=0)
    singular=np.linalg.svd(p,compute_uv=False)
    return bool(singular[0]>1e-10 and singular[1]>1e-8*max(singular[0],1.))


def carrier_geometry(state, xyz, box):
    """Continuous carrier images and an immutable registration-time shape."""
    ids=np.asarray(state['carrier_ids'],dtype=int)
    previous=np.asarray(state['positions'])
    if 'carrier_reference' in state:
        reference=np.asarray(state['carrier_reference'])
        offset=np.asarray(state['center_offset'])
        base_rotation=np.asarray(state['reference_rotation'])
    else:
        # Upgrade translation-only state at its saved coordinates, not at xyz.
        center=np.asarray(state['center'])
        previous=center+minimum_image(previous-center,box)
        reference=previous-previous.mean(axis=0)
        offset=center-previous.mean(axis=0)
        base_rotation=np.asarray(state['rotation'])
    current=previous+minimum_image(xyz[ids]-previous,box)
    return ids,reference,current,offset,base_rotation


def fit_rotation(reference, current, transport):
    """Proper least-squares rotation and its differentiable stiffness."""
    if not transport:
        return np.eye(3),reference,None,np.zeros(3)
    u,_,vt=np.linalg.svd(reference.T@current)
    parity=np.eye(3);parity[2,2]=np.linalg.det(vt.T@u.T)
    rotation=vt.T@parity@u.T
    rotated=reference@rotation.T
    covariance=current.T@rotated
    covariance=.5*(covariance+covariance.T)
    stiffness=np.trace(covariance)*np.eye(3)-covariance
    eigenvalues=np.linalg.eigvalsh(stiffness)
    if eigenvalues[0]<=1e-10*max(eigenvalues[-1],1.):
        raise ValueError('Material orientation is underdetermined; tracking cohort collapsed or became degenerate')
    return rotation,rotated,stiffness,eigenvalues


def core_frame(domain, xyz, box):
    anchors=np.asarray(domain['anchors'],dtype=int)
    offsets=np.asarray(domain['anchor_offsets'],dtype=float)
    origin=xyz[anchors[0]]
    unwrapped=origin+offsets+minimum_image(xyz[anchors]-origin-offsets,box)
    center=unwrapped.mean(axis=0)
    spacing=np.asarray(domain.get('grid_spacing',[3.,3.,3.]),dtype=float)
    grid=spacing*np.floor(center/spacing+.5)
    residual=center-grid
    current=unwrapped-center
    reference=offsets-offsets.mean(axis=0)
    # Collinear/single-site anchors do not define a full material orientation.
    # Keep translation only for those births; never select an arbitrary SVD spin.
    transport=domain.get('orientation_transport',orientation_defined(reference))
    if transport:
        u,sv,vt=np.linalg.svd(reference.T@current)
        parity=np.eye(3);parity[2,2]=np.linalg.det(vt.T@u.T)
        rotation=vt.T@parity@u.T
        rotated=reference@rotation.T
        covariance=current.T@rotated
        covariance=.5*(covariance+covariance.T)
        stiffness=np.trace(covariance)*np.eye(3)-covariance
        eigenvalues=np.linalg.eigvalsh(stiffness)
        if eigenvalues[0]<=1e-10*max(eigenvalues[-1],1.):
            raise ValueError('Material orientation is underdetermined; anchor core collapsed or became degenerate')
    else:
        rotation=np.eye(3);rotated=reference;stiffness=None;eigenvalues=np.zeros(3)
    return dict(center=center,grid=grid,offset=residual,rotation=rotation,
                rotated_core=rotated,stiffness=stiffness,eigenvalues=eigenvalues)


def frame(domain, xyz, box):
    """Evaluate an accepted chart without making acceptance decisions."""
    state=domain.get('accepted_frame')
    if state is None:
        material=core_frame(domain,xyz,box)
        material['translation_ids']=domain['anchors']
        material['rotation_ids']=domain['anchors']
        return material
    ids,reference,current,offset,base_rotation=carrier_geometry(state,xyz,box)
    centroid=current.mean(axis=0)
    transport=state.get('orientation_transport',orientation_defined(reference))
    q,rotated,stiffness,eigenvalues=fit_rotation(reference,current-centroid,transport)
    rotated_offset=q@offset
    center=centroid+rotated_offset
    spacing=np.asarray(domain.get('grid_spacing',[3.,3.,3.]))
    grid=spacing*np.floor(center/spacing+.5)
    return dict(center=center,grid=grid,offset=center-grid,
                rotation=q@base_rotation,translation_ids=ids,rotation_ids=ids,
                rotation_center_offset=rotated_offset,carrier_positions=current,
                rotated_core=rotated,stiffness=stiffness,eigenvalues=eigenvalues)


def occupied_loss(domain, xyz, box, material, cutoff):
    """Same fixed slots/weights for both candidates; omit constant vacancies."""
    mapping=np.asarray(domain['mapping'],dtype=int).reshape(-1,2)
    if not len(mapping):return None
    slots,ids=mapping.T
    targets=np.asarray(domain['targets'])[slots]@material['rotation'].T
    errors=minimum_image(xyz[ids]-material['center']-targets,box)
    u=np.minimum(1.,np.sum(errors*errors,axis=1)/cutoff**2)
    weights=np.asarray(domain.get('slot_weights',[]),dtype=float)
    weights=weights[slots] if len(weights) else np.ones(len(slots))
    if weights.sum()<=0:return None
    return float(np.dot(weights,u*(3.-3.*u+u*u))/weights.sum())


def registration_score(domain, xyz, box, material, cutoff, context=None):
    """Full system PIRMSD with only this domain's c/R replaced.

    Same coordinates, mapping, cutoff, r_cut and weight *definition*; radial
    occupied weights and their denominator are recomputed at the trial center.
    Vacancies and all other domains are included, not an occupied-only proxy.
    """
    if context is not None and 'cluster_factors' in context:
        from cluster_pirmsd import cluster_terms
        s,w,_ = cluster_terms(domain,xyz,box,material,context['cluster_factors'])
        weight = context['other_weight'] + w
        return float(np.sqrt((context['other_numerator']+s)/weight)) if weight>1e-30 else None
    n=len(domain['ids']);context=context or dict(population=n,other_numerator=0.)
    population=context['population'];mapping=np.asarray(domain['mapping'],int).reshape(-1,2)
    if not len(mapping):loss=float(population)
    else:
        slots,atoms=mapping.T
        targets=np.asarray(domain['targets'])[slots]@material['rotation'].T
        errors=minimum_image(xyz[atoms]-material['center']-targets,box)
        u=np.minimum(1.,np.sum(errors*errors,axis=1)/cutoff**2)
        ell=u*(3.-3.*u+u*u)
        weights=np.asarray(domain.get('slot_weights',[]))
        if len(weights):
            occupied=set(slots);vacancy=float(sum(w for j,w in enumerate(weights) if j not in occupied))
            if domain.get('radial_weighting'):
                from scipy.special import expit
                radius=np.linalg.norm(minimum_image(xyz[atoms]-material['center'],box),axis=1)
                wi=expit(-domain['weight_alpha']*(radius-domain['r_cut']))
                denominator=vacancy+wi.sum()
            else:wi=weights[slots];denominator=weights.sum()
            if denominator<=np.finfo(float).tiny:raise ValueError('Vanishing registration normalization')
            loss=population-n+n*(vacancy+wi@ell)/denominator
        else:loss=population-len(atoms)+ell.sum()
    numerator=context['other_numerator']+n/population*loss
    return float(np.sqrt(max(0.,numerator/population)))


def update_accepted_frame(domain, xyz, box, cutoff, settings, propose=True, sample_id=None, score_context=None):
    """Called only at outer iterations. Rigid transport is not acceptance.

    Keep carrier identities fixed for the lifetime of this chart, including
    membership changes. Rebase their periodic displacements every outer step.
    Accepted registration changes increment the ratchet context revision.
    """
    old=frame(domain,xyz,box)
    state=domain.get('accepted_frame')
    if state is None:
        ids=list(domain['ids'])
        domain['frame_revision']=domain.get('frame_revision',0)+1
        decision=dict(reason='initialized',accepted=False)
    elif not propose:
        ids=state['carrier_ids']
        domain.pop('frame_switch',None)
        decision=dict(reason='reference_transition',accepted=False)
    else:
        ids=state['carrier_ids']
        old_loss=registration_score(domain,xyz,box,old,cutoff,score_context)
        decision=dict(reason='no_occupied_weight',accepted=False,old_loss=old_loss)
        try:
            candidate=core_frame(domain,xyz,box)
        except ValueError:
            candidate=None
            decision['reason']='degenerate_core'
        if settings.get('policy','three_conditions')=='three_conditions':
            from frame_switch import consider_switch, DEFAULTS
            policy=dict(DEFAULTS);policy.update(settings)
            decision,candidate,accepted_state=consider_switch(domain,xyz,box,cutoff,policy,old,candidate,sample_id,score_context)
            if decision['accepted']:
                old=candidate;state=accepted_state
                if domain.get('radial_weighting'):
                    from scipy.special import expit
                    domain['r_cut']=state['branch_r_cut']
                    domain['weight_radius']=domain['r_cut']
                    domain['slot_weights']=expit(-domain['weight_alpha']*(
                        np.linalg.norm(domain['targets'],axis=1)-domain['r_cut'])).tolist()
                domain['frame_revision']=domain.get('frame_revision',0)+1
                domain.pop('frame_switch',None)
        elif candidate is not None and old_loss is not None:
            new_loss=registration_score(domain,xyz,box,candidate,cutoff,score_context)
            gain=old_loss-new_loss
            threshold=max(settings['min_absolute_gain'],settings['min_relative_gain']*old_loss)
            accepted=bool(gain>threshold)
            decision.update(reason='accepted' if accepted else 'insufficient_gain',
                            accepted=accepted,new_loss=new_loss,gain=gain,threshold=threshold,
                            center_correction_A=float(np.linalg.norm(minimum_image(candidate['center']-old['center'],box))),
                            rotation_correction_deg=float(np.degrees(np.arccos(np.clip(
                                (np.trace(candidate['rotation']@old['rotation'].T)-1)/2,-1.,1.)))))
            if accepted:
                old=candidate
                domain['frame_revision']=domain.get('frame_revision',0)+1
    if state is None:
        current=old['center']+minimum_image(xyz[ids]-old['center'],box)
    else:
        _,reference,current,offset,base_rotation=carrier_geometry(state,xyz,box)
    if state is None or (decision['accepted'] and settings.get('policy','three_conditions')=='instant_gain'):
        reference=current-current.mean(axis=0)
        offset=old['center']-current.mean(axis=0)
        base_rotation=old['rotation']
    # Do not refit the reference shape on rejection: that would silently
    # change the metric gradient and accumulate deformation-induced rotations.
    transport=bool(state.get('orientation_transport',orientation_defined(reference))) if state is not None else bool(orientation_defined(reference))
    domain['accepted_frame']=dict(carrier_ids=ids,positions=current.tolist(),
        center=old['center'].tolist(),rotation=old['rotation'].tolist(),
        carrier_reference=reference.tolist(),center_offset=offset.tolist(),
        reference_rotation=base_rotation.tolist(),
        orientation_transport=transport)
    decision['score_kind']=('cluster_pirmsd_A' if score_context and 'cluster_factors' in score_context
                            else 'full_pirmsd_common_definition')
    domain['frame_acceptance']=decision


def rotation_pullback(material, targets, site_gradient):
    """dL/d(anchor positions) through the fixed-correspondence polar rotation.

    A small lab-frame rotation obeys K dtheta = sum_i (R b_i) x dx_i.
    This includes torque reaction even when donors are not anchor-core atoms.
    """
    if material['stiffness'] is None:
        return np.zeros_like(material['rotated_core'])
    # The accepted origin need not coincide with the carrier centroid. Its
    # rotated lever arm contributes to the reference-position derivative too.
    levers=targets+material.get('rotation_center_offset',np.zeros(3))
    torque=-np.cross(levers,site_gradient).sum(axis=0)
    multiplier=np.linalg.solve(material['stiffness'].T,torque)
    return np.cross(multiplier,material['rotated_core'])


def diagnostics(domain, material, lattice_rotation):
    quaternion=Rotation.from_matrix(material['rotation']@lattice_rotation).as_quat()
    previous=domain.get('orientation_quaternion')
    if previous is not None and np.dot(quaternion,previous)<0:
        quaternion=-quaternion
    domain['grid_point']=material['grid'].tolist()
    domain['center_offset']=material['offset'].tolist()
    domain['orientation_quaternion']=quaternion.tolist()  # x,y,z,w; sign-continuous
    domain['frame_eigenvalues']=material['eigenvalues'].tolist()


DYNAMIC_FRAME_KEYS={'grid_point','center_offset','orientation_quaternion','frame_eigenvalues',
                    'accepted_frame','frame_acceptance','frame_switch'}

def structural_state(value):
    """Grid/physical rotation transport must not create a new ratchet epoch."""
    if isinstance(value,dict):
        result={k:structural_state(v) for k,v in value.items() if k not in DYNAMIC_FRAME_KEYS}
        if 'accepted_frame' in value:
            state=value['accepted_frame']
            # Bind the ratchet to the actual material registration, not just
            # an incremented revision. Display c/R and periodic image rebasing
            # are dynamic; the carrier shape and lattice registration are not.
            keys=('carrier_ids','carrier_reference','center_offset',
                  'reference_rotation','orientation_transport')
            if 'carrier_reference' not in state:
                keys=('carrier_ids','positions','center','rotation')
            result['accepted_frame_definition']={k:state[k] for k in keys if k in state}
        return result
    if isinstance(value,list):return [structural_state(v) for v in value]
    return value
