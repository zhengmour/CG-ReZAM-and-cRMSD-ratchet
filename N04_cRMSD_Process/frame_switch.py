"""Outer-iteration reference switching after stagnation and stable superiority."""
import copy
import numpy as np
from material_frame import (frame, carrier_geometry, registration_score, minimum_image,
                            fit_rotation, structural_state, orientation_defined)
from ratchet_state import fingerprint
from rienv_core import groups

DEFAULTS=dict(policy='three_conditions',stagnation_window=20,stagnation_windows=2,
              stagnation_absolute_gain=1e-4,stagnation_relative_gain=.001,
              candidate_confirmations=3,core_link_cutoff=4.8,
              core_min_rank_ratio=.01,core_max_rmsd=.25)


def core_positions(domain,xyz,box):
    ids=np.asarray(domain['anchors'],dtype=int)
    offsets=np.asarray(domain['anchor_offsets'])
    origin=xyz[ids[0]]
    return origin+offsets+minimum_image(xyz[ids]-origin-offsets,box)


def stable_core(domain,xyz,box,reference,settings):
    anchors=domain['anchors']
    if len(anchors)<3 or not set(anchors)<=set(domain['ids']):
        return False,dict(core_reason='missing_or_small_core')
    if len(groups(anchors,xyz,box,settings['core_link_cutoff']))!=1:
        return False,dict(core_reason='disconnected_core')
    current=core_positions(domain,xyz,box);current-=current.mean(axis=0)
    spectrum=np.linalg.eigvalsh(current.T@current)
    ratio=float(spectrum[-2]/max(spectrum[-1],1e-30))
    if ratio<settings['core_min_rank_ratio']:
        return False,dict(core_reason='ill_conditioned_core',core_rank_ratio=ratio)
    if reference is None:reference=current
    try:
        _,rotated,_,_=fit_rotation(np.asarray(reference),current,True)
    except ValueError:
        return False,dict(core_reason='degenerate_core_fit',core_rank_ratio=ratio)
    rmsd=float(np.sqrt(np.mean(np.sum((current-rotated)**2,axis=1))))
    return rmsd<=settings['core_max_rmsd'],dict(
        core_reason='stable' if rmsd<=settings['core_max_rmsd'] else 'deformed_core',
        core_rank_ratio=ratio,core_rmsd=rmsd)


def consider_switch(domain,xyz,box,cutoff,settings,old,proposal,sample_id=None,score_context=None):
    """Only the frozen candidate is scored after birth, never a refitted one."""
    score=registration_score(domain,xyz,box,old,cutoff,score_context)
    chart = structural_state(domain)
    if score_context and 'cluster_definition' in score_context:
        chart = [chart, score_context['cluster_definition']]
    definition=fingerprint(chart)
    watch=domain.get('frame_switch')
    if watch is None or watch['definition']!=definition:
        watch=dict(definition=definition,best=score,window_start=score,steps=0,
                   stagnant_windows=0,candidate=None,last_sample=None)
        domain['frame_switch']=watch
    if sample_id is not None and watch['last_sample']==sample_id:
        return dict(reason='duplicate_sample',accepted=False),None,None
    watch['last_sample']=sample_id
    if score is None:
        watch['candidate']=None
        return dict(reason='no_occupied_weight',accepted=False),None,None
    if watch['best'] is None:
        watch.update(best=score,window_start=score,steps=0)
    watch['best']=min(watch['best'],score)
    tolerance=max(settings['stagnation_absolute_gain'],
                  settings['stagnation_relative_gain']*watch['window_start'])
    if watch['window_start']-watch['best']>tolerance:
        watch['stagnant_windows']=0
    watch['steps']+=1
    if watch['steps']>=settings['stagnation_window']:
        improvement=watch['window_start']-watch['best']
        tolerance=max(settings['stagnation_absolute_gain'],
                      settings['stagnation_relative_gain']*watch['window_start'])
        watch['stagnant_windows']=watch['stagnant_windows']+1 if improvement<=tolerance else 0
        watch.update(last_window_gain=improvement,window_start=watch['best'],steps=0)
    stagnant=watch['stagnant_windows']>=settings['stagnation_windows']
    pending=watch['candidate']
    stable,diagnostic=stable_core(domain,xyz,box,None if pending is None else pending['core_reference'],settings)
    decision=dict(reason='waiting_candidate',accepted=False,old_loss=score,
                  old_stagnant=stagnant,stagnant_windows=watch['stagnant_windows'],
                  candidate_better=False,core_stable=stable,**diagnostic)
    if not stable or proposal is None:
        watch['candidate']=None
        decision['reason']='unstable_core' if not stable else 'degenerate_core'
        return decision,None,None
    if pending is None:
        gain=score-registration_score(domain,xyz,box,proposal,cutoff,score_context)
        threshold=max(settings['min_absolute_gain'],settings['min_relative_gain']*score)
        if gain<=threshold:
            decision.update(reason='insufficient_gain',gain=gain,threshold=threshold)
            return decision,None,None
        ids,_,current,_,_=carrier_geometry(domain['accepted_frame'],xyz,box)
        reference=current-current.mean(axis=0)
        state=dict(carrier_ids=ids.tolist(),positions=current.tolist(),
                   carrier_reference=reference.tolist(),center_offset=(proposal['center']-current.mean(axis=0)).tolist(),
                   reference_rotation=proposal['rotation'].tolist(),center=proposal['center'].tolist(),
                   rotation=proposal['rotation'].tolist(),orientation_transport=domain['accepted_frame'].get('orientation_transport',bool(orientation_defined(reference))))
        if domain.get('radial_weighting'):
            # Freeze the WHOLE proposal, including its radius, at candidate birth.
            # Acceptance uses common old slot weights above/below; changing the
            # weighting scale alone can never manufacture a registration gain.
            spec=domain['radius_definition']
            distances=np.linalg.norm(minimum_image(xyz[domain['ids']]-proposal['center'],box),axis=1)
            extent=np.linalg.norm(minimum_image(xyz[domain['anchors']]-proposal['center'],box),axis=1).max()
            state['branch_r_cut']=float(spec['scale']*max(np.quantile(distances,spec['quantile']),extent,spec['minimum']))
        core=core_positions(domain,xyz,box);core-=core.mean(axis=0)
        pending=dict(state=state,core_reference=core.tolist(),confirmations=0)
        watch['candidate']=pending
    shadow=dict(domain,accepted_frame=pending['state'])
    try:
        candidate=frame(shadow,xyz,box)
    except ValueError:
        watch['candidate']=None
        decision.update(reason='degenerate_candidate',core_stable=False)
        return decision,None,None
    new_loss=registration_score(domain,xyz,box,candidate,cutoff,score_context)
    gain=score-new_loss
    threshold=max(settings['min_absolute_gain'],settings['min_relative_gain']*score)
    if gain<=threshold:
        watch['candidate']=None
        decision.update(reason='candidate_lost_advantage',new_loss=new_loss,gain=gain,threshold=threshold)
        return decision,None,None
    # A radius change must also pass the full score after committing the new
    # radius/slot weights. Require improvement under BOTH the common old
    # definition and the proposed final definition; never exploit a rescaling.
    committed_loss=new_loss
    if domain.get('radial_weighting'):
        from scipy.special import expit
        final_domain=dict(domain,r_cut=pending['state']['branch_r_cut'])
        final_domain['slot_weights']=expit(-domain['weight_alpha']*(
            np.linalg.norm(domain['targets'],axis=1)-final_domain['r_cut'])).tolist()
        committed_loss=registration_score(final_domain,xyz,box,candidate,cutoff,score_context)
        if score-committed_loss<=threshold:
            watch['candidate']=None
            decision.update(reason='committed_definition_not_better',new_loss=new_loss,
                            committed_loss=committed_loss,gain=gain,threshold=threshold)
            return decision,None,None
    pending['confirmations']+=1
    # Rebase periodic images without changing the candidate's registration.
    pending['state']['positions']=candidate['carrier_positions'].tolist()
    confirmed=pending['confirmations']>=settings['candidate_confirmations']
    accepted=stagnant and confirmed and stable
    decision.update(reason='accepted' if accepted else 'waiting_stagnation' if not stagnant else 'waiting_confirmation',
                    accepted=accepted,candidate_better=confirmed,confirmations=pending['confirmations'],
                    new_loss=new_loss,committed_loss=committed_loss,gain=gain,threshold=threshold,
                    center_correction_A=float(np.linalg.norm(minimum_image(candidate['center']-old['center'],box))),
                    rotation_correction_deg=float(np.degrees(np.arccos(np.clip(
                        (np.trace(candidate['rotation']@old['rotation'].T)-1)/2,-1.,1.)))))
    if domain.get('radial_weighting'):
        decision.update(old_r_cut=domain['r_cut'],candidate_r_cut=pending['state']['branch_r_cut'])
    return decision,candidate,copy.deepcopy(pending['state']) if accepted else None
