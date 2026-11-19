"""Opt-in compact-cluster experiment: one persistent central geometric core.

This is not an automatic morphology classifier. The caller explicitly treats
the target population as one domain, including isolated peripheral particles.
"""
import copy
import numpy as np
from core_voronoi import CoreVoronoiTracker
from rienv_core import groups, unwrap
from material_frame import structural_state
from ratchet_state import fingerprint


class CentralCoreTracker(CoreVoronoiTracker):
    def __init__(self, settings=None, state=None):
        super().__init__(settings, state)
        self.central_state=copy.deepcopy((state or {}).get('central_core'))
        if state and self.initialized and self.central_state is None:
            raise ValueError('Cannot import multi-core state into central-core experiment')

    def state(self):
        s=super().state();s['central_core']=copy.deepcopy(self.central_state)
        return s

    def central_members(self, xyz, box, ids):
        parts=groups(ids,xyz,box,self.core_selector.settings['link_cutoff'])
        largest=max(parts,key=lambda c:(len(c),-min(c)))
        points=unwrap(largest,xyz,box,self.core_selector.settings['link_cutoff'])
        center=points.mean(axis=0)
        radii=np.linalg.norm(points-center,axis=1)
        order=sorted(range(len(largest)),key=lambda j:(radii[j],largest[j]))
        requested=self.policy.get('central_core_size',12)
        if int(requested)!=requested or requested<self.core_selector.settings['min_core']:
            raise ValueError('central_core_size must be an integer >= min_core')
        maximum=min(len(largest),int(self.policy.get('central_core_max_size',64)))
        for n in range(int(requested),maximum+1):
            members=sorted(largest[j] for j in order[:n])
            if self._geometry(members,xyz,box):
                return members,dict(schema=1,selection='nearest_to_unwrapped_largest_component_centroid',
                    center_at_birth=(center%box).tolist(),component_size=len(largest),
                    component_sizes=sorted(map(len,parts),reverse=True),requested_size=int(requested),
                    selected_size=n,core_extent_from_cluster_center=float(max(radii[j] for j in order[:n])),
                    gyration_eigenvalues=np.linalg.eigvalsh((points-center).T@(points-center)/len(points)).tolist())
        raise ValueError('No connected nondegenerate central core within size limit')

    def update(self,frame,xyz,box,population_ids,components,reference,full_reference,cutoff,core,comm,iterations,rienv_scores=None):
        if self.last_frame is not None and frame<=self.last_frame:raise ValueError('Nonincreasing core lifecycle frame')
        if self.population_ids is not None and list(population_ids)!=self.population_ids:raise ValueError('Changed target identities')
        self.population_ids=list(population_ids);self.last_frame=frame;self.last_event='retained'
        if not len(population_ids):return
        self.core_selector.observe(frame,population_ids,rienv_scores)
        if rienv_scores is not None:self.observations+=1
        self.refresh_frames(xyz,box,cutoff)
        if self.transition is not None:self._advance(xyz,box,cutoff);return
        if not self.initialized:
            if self.observations<self.policy['initialize_samples']:
                self.last_event='initial_observation';return
            members,metadata=self.central_members(xyz,box,population_ids)
            d=self._new_domain(members,xyz,box,reference,full_reference,cutoff,core,comm,iterations)
            c=self._make_core(members,d['uid'],False)
            c['status']='geometric';d['provisional']=False
            self.domains,self.owners=self._partition([d],[c],xyz,box,cutoff,True)
            self.cores=[c];self.central_state=metadata;self.initialized=True
            self._event('central_initialized',cores=1,particles=len(members));return
        # Membership is fixed for this controlled experiment. Existing material
        # transport, three-condition frame acceptance and assignment transitions
        # remain unchanged; no RIEnv-based birth/growth/retirement is attempted.
        if len(self.cores)!=1 or len(self.domains)!=1:raise ValueError('Central-core invariant broken')
        if self.suspended:return
        ds,owners=self._partition(self.domains,self.cores,xyz,box,cutoff)
        definition=fingerprint(structural_state(ds));old=fingerprint(structural_state(self.domains))
        if definition!=old:
            if self._confirmed('assignment',definition,self.policy['assignment_confirmations']):
                self._start(ds,self.cores,owners,'assignment')
        else:self.monitors.pop('assignment',None)
