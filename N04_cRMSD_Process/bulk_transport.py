"""Bind a domain reference to a fixed whole-cluster material cohort at birth.

Core identity and frame transport identity are deliberately different. A change
of cohort is a new reference definition, never an implicit per-step operation.
"""
import numpy as np
from rienv_core import groups, unwrap
from material_frame import frame, minimum_image, orientation_defined


def bind_cluster(domain, xyz, box, population_ids, link_cutoff):
    old=frame(domain,xyz,box)
    parts=groups(population_ids,xyz,box,link_cutoff)
    anchors=set(domain['anchors'])
    members=next((c for c in parts if anchors<=set(c)),None)
    if members is None:raise ValueError('Core does not belong to one physical cluster')
    bind_members(domain, members, xyz, box, link_cutoff, old)


def bind_members(domain, members, xyz, box, link_cutoff, old):
    """Transport a newly accepted branch with its fixed physical cohort."""
    kind='cluster_rigid'
    try:
        current=unwrap(members,xyz,box,link_cutoff)
    except ValueError as exc:
        if 'winds around' not in str(exc):raise
        # A winding periodic network has no unique global rigid rotation.
        # Mean continuous displacement still defines translation. Never silently
        # fall back to the local core rotation for this case.
        current=np.asarray(xyz)[members].copy();kind='winding_translation_only'
    center=current.mean(axis=0);reference=current-center
    transport=kind=='cluster_rigid' and orientation_defined(reference)
    if kind=='cluster_rigid' and not transport:kind='degenerate_translation_only'
    offset=minimum_image(old['center']-center,box)
    domain['accepted_frame']=dict(carrier_ids=members,positions=current.tolist(),
        carrier_reference=reference.tolist(),center_offset=offset.tolist(),
        reference_rotation=old['rotation'].tolist(),center=(center+offset).tolist(),
        rotation=old['rotation'].tolist(),orientation_transport=bool(transport))
    domain['bulk_transport']=dict(schema=1,kind=kind,cohort_ids=members,
        link_cutoff=float(link_cutoff),cohort_policy='fixed_birth_cluster')


def bind_core(domain, xyz, box, pose):
    """Carry the accepted lattice with its registration anchors only."""
    ids = list(domain['anchors'])
    offsets = np.asarray(domain['anchor_offsets'])
    origin = xyz[ids[0]]
    current = origin + offsets + minimum_image(xyz[ids] - origin - offsets, box)
    centroid = current.mean(axis=0)
    reference = current - centroid
    if not orientation_defined(reference):
        raise ValueError('Degenerate core cannot carry a lattice rotation')
    offset = minimum_image(pose['center'] - centroid, box)
    domain['accepted_frame'] = dict(carrier_ids=ids, positions=current.tolist(),
        carrier_reference=reference.tolist(), center_offset=offset.tolist(),
        reference_rotation=pose['rotation'].tolist(), center=(centroid+offset).tolist(),
        rotation=pose['rotation'].tolist(), orientation_transport=True)
    domain.pop('bulk_transport', None)
    domain.pop('frame_switch', None)
    domain['core_transport'] = True
    domain['frame_revision'] = domain.get('frame_revision', 0) + 1
