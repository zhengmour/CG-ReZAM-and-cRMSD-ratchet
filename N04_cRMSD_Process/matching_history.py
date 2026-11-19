"""Re-evaluate retained ratchet configurations after boundary assignment updates."""
import copy
import numpy as np
from material_frame import structural_state, frame
from ratchet_state import fingerprint


def assignment_reference_definition(domains):
    # Ownership and slot assignment may change at a boundary. Keep every
    # reference/core/carrier/weight-scale field in the acceptance identity.
    definitions=structural_state(domains)
    for d in definitions:
        d.pop('mapping',None)
        d.pop('ids',None)
    return definitions


def registration_context(tracker, reference, box, cutoff):
    domains=assignment_reference_definition(tracker.domains)
    if hasattr(tracker, 'registration_identity'):
        domains = [tracker.registration_identity(), domains]
    return fingerprint(dict(reference=reference,box=np.asarray(box).tolist(),cutoff=cutoff,
                            population=tracker.population_ids,domains=domains))


def capture_geometry(snapshot, tracker, xyz, box, registration):
    ids=np.asarray(tracker.population_ids,dtype=int)
    snapshot.update(registration_context=registration,geometry=dict(ids=ids.tolist(),
        positions=xyz[ids].tolist(),natoms=len(xyz),box=np.asarray(box).tolist(),
        carrier_images={str(d['uid']):frame(d,xyz,box)['carrier_positions'].tolist() for d in tracker.domains}))


def rebase_history(ratchet, tracker, xyz, box, cutoff, context, registration):
    """Keep the previous/minimum configurations; never equate remapping to motion.

    The new baseline is the better of the two retained historical configurations
    under the new ownership and mapping. It is not claimed to be the minimum over discarded
    trajectory frames, which cannot be reconstructed from a two-state ratchet.
    """
    if ratchet.context==context or ratchet.context is None:return None
    states=[ratchet.minimum_state,ratchet.previous_state]
    if any(s is None or s.get('registration_context')!=registration or 'geometry' not in s for s in states):return None
    from ratchet_force import fixed_metric,metric_snapshot
    converted=[]
    for old in states:
        geometry=old['geometry'];np.testing.assert_array_equal(geometry['box'],box)
        coords=np.zeros((geometry['natoms'],3));coords[geometry['ids']]=geometry['positions']
        historical=copy.deepcopy(tracker)
        for d in historical.domains:
            if 'carrier_images' in geometry:
                d['accepted_frame']['positions']=copy.deepcopy(geometry['carrier_images'][str(d['uid'])])
        value,weight,_,_=fixed_metric(historical,coords,box,cutoff)
        if value is None or not np.isfinite(value):raise ValueError('Invalid retained configuration after boundary reassignment')
        new=metric_snapshot(historical,coords,box,old['timestep'],context,value,weight)
        capture_geometry(new,historical,coords,box,registration);converted.append(new)
    before=[s['value'] for s in states]
    ratchet.minimum_state=copy.deepcopy(min(converted,key=lambda s:s['value']))
    ratchet.previous_state=converted[1]
    ratchet.minimum=ratchet.minimum_state['value'];ratchet.previous=converted[1]['value']
    ratchet.context=context;ratchet.epoch+=1
    return dict(old_values=before,new_values=[s['value'] for s in converted],
                minimum_timestep=ratchet.minimum_state['timestep'],previous_timestep=ratchet.previous_state['timestep'])
