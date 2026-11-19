"""Persistent PIRMSD matching domains, independent of spatial components.

A takeover retains a parent's material anchor, orientation and occupied slots.
Frozen squared-score numerators and gradients are mixed, never atom coordinates
or Euler angles. This module does not define an environment descriptor.
"""
import copy
import numpy as np
from ratchet_state import fingerprint
from material_frame import frame as material_frame, rotation_pullback, diagnostics, structural_state, update_accepted_frame, carrier_geometry
from domain_alignment import align, greedy_pairs, stored_euler
from rienv_core import CoreSelector, unwrap
from frame_switch import DEFAULTS as FRAME_SWITCH_DEFAULTS
from material_frame import orientation_defined
from scipy.spatial.transform import Rotation


def minimum_image(delta, box):
    return delta - box*np.round(delta/box)


def rotation(move):
    a,b,c=move[3:]
    rx=np.array([[1,0,0],[0,np.cos(a),-np.sin(a)],[0,np.sin(a),np.cos(a)]])
    ry=np.array([[np.cos(b),0,np.sin(b)],[0,1,0],[-np.sin(b),0,np.cos(b)]])
    rz=np.array([[np.cos(c),-np.sin(c),0],[np.sin(c),np.cos(c),0],[0,0,1]])
    return rz@ry@rx



def evaluate(domain, xyz, box, population, cutoff):
    """Return unnormalized slot loss and its derivative at a frozen mapping."""
    if domain.get('retired'):
        entries=np.zeros((len(domain['ids']),9));entries[:,0]=np.asarray(domain['ids'])+1;entries[:,1]=1
        return float(population),entries.tolist()
    if domain.get('radial_weighting'):
        from voronoi_geometry import evaluate_radial
        return evaluate_radial(domain,xyz,box,population,cutoff)
    material=material_frame(domain,xyz,box)
    n=len(domain['ids'])
    ids=np.asarray(sorted(set(domain['ids'])|set(material['translation_ids'])|set(material['rotation_ids'])),dtype=int)
    index={int(atom):row for row,atom in enumerate(ids)}
    gradient=np.zeros((len(ids),3))
    targets=np.asarray(domain['targets'])@material['rotation'].T
    mapping=domain['mapping']
    weights=np.asarray(domain.get('slot_weights',[]))
    weighted=len(weights)>0
    # Fixed reference weights plus explicit population coverage prevent a small
    # perfect core from erasing missing boundary sites or fragment penalties.
    normalization=float(weights.sum()) if weighted else 1.
    if weighted:
        occupied={slot for slot,_ in mapping}
        missing=sum(w for slot,w in enumerate(weights) if slot not in occupied)
        loss=float(population-n)+n*missing/normalization
    else:loss=float(population-len(mapping))
    matched_targets=[];site_gradients=[]
    for slot,atom in mapping:
        # Grid gauge: retain the continuous center offset when the grid switches.
        relative=(xyz[atom]-material['grid'])-material['offset']
        error=minimum_image(relative-targets[slot],box)
        u=float(error@error)/cutoff**2
        z=max(0.,1.-u)
        site_loss=u*(3.-3.*u+u*u) if u<1 else 1.
        coefficient=n*weights[slot]/normalization if weighted else 1.
        if weighted:loss+=coefficient*site_loss
        else:loss+=site_loss
        deriv=coefficient*6*z*z*error/cutoff**2
        gradient[index[atom]]+=deriv
        # On the selected periodic branch the target vector controls rotation.
        matched_targets.append(targets[slot]);site_gradients.append(deriv)
    translation_ids=material['translation_ids']
    reaction=gradient.sum(axis=0)/len(translation_ids)
    rotational=rotation_pullback(material,np.asarray(matched_targets).reshape(-1,3),
                                 np.asarray(site_gradients).reshape(-1,3))
    for atom in translation_ids:gradient[index[atom]]-=reaction
    for j,atom in enumerate(material['rotation_ids']):
        gradient[index[atom]]+=rotational[j]
    entries=np.zeros((len(ids),9))
    entries[:,0]=ids+1;entries[:,1]=1;entries[:,2:5]=gradient
    for _,atom in mapping:entries[index[atom],8]=1.
    return loss,entries.tolist()


class DomainTracker:
    def __init__(self, settings=None, state=None):
        self.settings=dict(confirmations=3,transition_iterations=20,min_gain=0.005,
                           retain_fraction=0.9,max_alignment_loss_increase=0.02,grid_spacing=[3.,3.,3.])
        self.settings.update(settings or {})
        self.frame_acceptance=dict(status=False,min_absolute_gain=1e-4,min_relative_gain=0.01)
        self.frame_acceptance.update(FRAME_SWITCH_DEFAULTS)
        self.frame_acceptance.update(self.settings.get('frame_acceptance',{}))
        if self.frame_acceptance['policy'] not in ('three_conditions','instant_gain'):
            raise ValueError('Unknown frame acceptance policy')
        for key in ('stagnation_window','stagnation_windows','candidate_confirmations'):
            value=self.frame_acceptance[key]
            if int(value)!=value or value<1:raise ValueError(key+' must be a positive integer')
        for key in ('min_absolute_gain','min_relative_gain','stagnation_absolute_gain',
                    'stagnation_relative_gain','core_max_rmsd','core_min_rank_ratio','core_link_cutoff'):
            if not np.isfinite(self.frame_acceptance[key]) or self.frame_acceptance[key]<0:
                raise ValueError('frame_acceptance '+key+' must be finite and nonnegative')
        if self.frame_acceptance['core_link_cutoff']<=0 or not 0<self.frame_acceptance['core_min_rank_ratio']<=1:
            raise ValueError('Invalid core stability geometry thresholds')
        for key in ('confirmations','transition_iterations'):
            if int(self.settings[key])!=self.settings[key] or self.settings[key]<1:
                raise ValueError(key+' must be a positive integer')
        if not 0<=self.settings['retain_fraction']<=1 or self.settings['min_gain']<0 or self.settings['max_alignment_loss_increase']<0:
            raise ValueError('Invalid takeover thresholds')
        spacing=np.asarray(self.settings['grid_spacing'],dtype=float)
        if spacing.shape!=(3,) or not np.isfinite(spacing).all() or np.any(spacing<=0):
            raise ValueError('grid_spacing must contain three finite positive lengths')
        state=copy.deepcopy(state or {})
        self.domains=state.get('domains',[])
        self.transition=state.get('transition')
        self.pending=state.get('pending')
        self.next_id=state.get('next_id',0)
        self.population_ids=state.get('population_ids')
        self.last_frame=state.get('last_frame')
        self.last_mix=state.get('last_mix',0.)
        self.protocol_score_work=state.get('protocol_score_work',0.)
        self.last_event='initial'
        self.core_selector=CoreSelector(self.settings.get('rienv_core'),state.get('core_selector'))

    def state(self):
        return copy.deepcopy(dict(domains=self.domains,transition=self.transition,
            pending=self.pending,next_id=self.next_id,population_ids=self.population_ids,
            last_frame=self.last_frame,last_mix=self.last_mix,
            protocol_score_work=self.protocol_score_work,core_selector=self.core_selector.state()))

    def birth(self, ids, xyz, box, reference, full_reference, cutoff, core, comm, iterations, anchor_ids=None, provisional=False):
        ids=sorted(ids)
        anchors=sorted(anchor_ids if anchor_ids is not None else ids)
        coords=unwrap(anchors,xyz,box,self.core_selector.settings['link_cutoff']) if anchor_ids is not None else xyz[anchors]
        offsets=(coords-coords[0]) if anchor_ids is not None else minimum_image(coords-coords[0],box)
        coords=coords[0]+offsets
        reference=np.asarray(reference)
        center=reference[:,2:5].mean(axis=0)
        solve=np.column_stack((np.asarray(anchors)+1,np.ones(len(anchors)),coords-coords.mean(axis=0)+center))
        _,rows,move=align(reference,solve,cutoff,core,comm,iterations,fit_translation=anchor_ids is not None)
        pool=np.asarray(full_reference)
        targets=(pool[:,2:5]-center)@rotation(move).T+np.asarray(move[:3])
        slots={int(row[0]):i for i,row in enumerate(pool)}
        domain=dict(uid=self.next_id,ids=ids,anchors=anchors,anchor_offsets=offsets.tolist(),
                    move=list(move),targets=targets.tolist(),grid_spacing=list(self.settings['grid_spacing']),
                    orientation_transport=bool(orientation_defined(offsets)),
                    mapping=sorted([slots[int(row[9])],int(row[0])-1] for row in rows))
        if anchor_ids is not None:
            matched={atom for _,atom in domain['mapping']}
            coherent=self.core_selector.coherent_groups(sorted(set(anchors)&matched),xyz,box)
            if not coherent and not provisional:return None
            chosen=anchors if provisional else max(coherent,key=lambda c:(len(c),tuple(-i for i in c)))
            if chosen!=anchors:domain=self.reanchor(domain,chosen,xyz,box)
            else:self.core_selector.set_weights(domain,xyz,box)
            outer=sorted(set(ids)-{atom for _,atom in domain['mapping']})
            if outer:domain=self.candidate(domain,dict(ids=outer,uid=domain['uid']),xyz,box,cutoff,core)
            domain.pop('parents',None)
        if self.core_selector.settings['status']:domain['provisional']=bool(provisional)
        if self.frame_acceptance['status']:
            update_accepted_frame(domain,xyz,box,cutoff,self.frame_acceptance)
        self.next_id+=1
        diagnostics(domain,material_frame(domain,xyz,box),rotation(domain['move']))
        self.domains.append(domain)
        return domain

    def reanchor(self,domain,anchors,xyz,box):
        material=material_frame(domain,xyz,box)
        mapped={atom:slot for slot,atom in domain['mapping']}
        targets=np.asarray(domain['targets'])@material['rotation'].T
        expected=targets[[mapped[i] for i in anchors]]
        coords=material['center']+expected+minimum_image(xyz[anchors]-material['center']-expected,box)
        center=coords.mean(axis=0)
        new=copy.deepcopy(domain)
        new['anchors']=list(anchors);new['anchor_offsets']=(coords-coords[0]).tolist()
        new['targets']=(targets+material['center']-center).tolist()
        new['move']=[0.,0.,0.]+stored_euler(Rotation.from_matrix(material['rotation']@rotation(domain['move'])))
        new['orientation_transport']=True
        if 'accepted_frame' in domain:
            # Changing anchor identities must preserve the accepted physical
            # lattice. A later outer-step proposal may change its registration.
            state=new['accepted_frame']
            _,reference,current,offset,base_rotation=carrier_geometry(state,xyz,box)
            state['center']=material['center'].tolist()
            state['rotation']=np.eye(3).tolist()
            state['positions']=current.tolist()
            state['carrier_reference']=reference.tolist()
            state['center_offset']=offset.tolist()
            state['reference_rotation']=(base_rotation@material['rotation'].T).tolist()
            new['targets']=targets.tolist()
        self.core_selector.set_weights(new,xyz,box)
        return new

    def candidate(self, dominant, other, xyz, box, cutoff, core):
        """Keep the dominant lattice; fit only donor atoms into unused slots."""
        target=copy.deepcopy(dominant)
        target.pop('core_update',None)
        target['ids']=sorted(set(dominant['ids'])|set(other['ids']))
        target['parents']=[dominant['uid'],other['uid']]
        occupied={slot for slot,_ in dominant['mapping']}
        material=material_frame(dominant,xyz,box)
        center=material['grid']+material['offset']
        pool=np.asarray(dominant['targets'])@material['rotation'].T
        available=[i for i in range(len(pool)) if i not in occupied]
        donor=sorted(other['ids'])
        pairs=greedy_pairs(pool[available],xyz[donor]-center,cutoff,available,donor,box=box)
        for slot_row,atom_row,_ in pairs:
            target['mapping'].append([available[slot_row],donor[atom_row]])
        target['mapping'].sort()
        return target

    def _same_component(self, a, b, components):
        needed=set(a['ids'])|set(b['ids'])
        return any(needed<=set(component) for component in components)

    def update(self, frame, xyz, box, population_ids, components, reference,
               full_reference, cutoff, core, comm, iterations, rienv_scores=None):
        """Advance once per sampling frame; evaluation itself is side-effect free."""
        if self.last_frame is not None and frame<=self.last_frame:
            raise ValueError('Matching domains must advance once per increasing frame')
        self.last_frame=frame
        self.last_event='retained'
        if self.population_ids is not None and self.population_ids!=list(population_ids):
            raise ValueError('Persistent matching domains require fixed target atom identities')
        self.population_ids=list(population_ids)
        n=len(population_ids)
        if not n:return
        if self.core_selector.settings['status']:
            self.core_selector.observe(frame,population_ids,rienv_scores)
        self.refresh_frames(xyz,box,cutoff)
        if self.transition is not None:
            t=self.transition
            u=min(1.,max(0.,(frame-t['start'])/self.settings['transition_iterations']))
            mix=u*u*u*(10.-15.*u+6.*u*u)
            old=sum(len(d['ids'])/n*evaluate(d,xyz,box,n,cutoff)[0] for d in t['old'])
            new=len(t['new']['ids'])/n*evaluate(t['new'],xyz,box,n,cutoff)[0]
            self.protocol_score_work+=(mix-self.last_mix)*(new-old)/n
            self.last_mix=mix
            if u>=1:
                retired={d['uid'] for d in t['old']}
                self.domains=[d for d in self.domains if d['uid'] not in retired]+([] if t['new'].get('retired') else [t['new']])
                self.transition=None;self.pending=None;self.last_mix=0.
                self.last_event='takeover_complete'
            # A started transition remains frozen even if contact flickers.
            # No new transition or births until it finishes.
            return
        owned={i for d in self.domains for i in d['ids']}
        for component in components:
            newborn=sorted(set(component)-owned)
            if newborn:
                if self.core_selector.settings['status']:
                    selections=[(region,anchors,False) for region,anchors in
                        self.core_selector.initial_domains(newborn,xyz,box)]
                    claimed={i for region,_,_ in selections for i in region}
                    selections += [(region,anchors,True) for region,anchors in
                        self.core_selector.provisional_domains(sorted(set(newborn)-claimed),xyz,box)]
                else:selections=[(newborn,None,False)]
                for region,anchors,provisional in selections:
                    born=self.birth(region,xyz,box,reference,full_reference,cutoff,core,comm,iterations,anchors,provisional)
                    if born is None:
                        born=self.birth(region,xyz,box,reference,full_reference,cutoff,core,comm,iterations,anchors,True)
                    if born is not None:owned.update(region);self.last_event='domain_birth'
        costs={d['uid']:len(d['ids'])/n*evaluate(d,xyz,box,n,cutoff)[0] for d in self.domains}
        proposals=[]
        if self.core_selector.settings['status']:
            for domain in self.domains:
                chosen,reason=self.core_selector.select(domain,xyz,box)
                if reason=='insufficient_history':continue
                if chosen is None:
                    # Loss of RIEnv qualification must not discard the material chart.
                    if domain.get('provisional'):continue
                    target=copy.deepcopy(domain);target['provisional']=True
                else:
                    if chosen==domain['anchors'] and not domain.get('provisional'):continue
                    target=self.reanchor(domain,chosen,xyz,box)
                    target['provisional']=False
                target['core_update']=True;target['parents']=[domain['uid']]
                key=fingerprint(dict(parents=target['parents'],anchors=chosen,provisional=target.get('provisional',False)))
                proposals.append((0.,key,target,[domain]))
        # Let previously vacant atoms join their own lattice through the same
        # confirmed transition; freezing a domain must not freeze vacancies forever.
        for domain in self.domains:
            matched={atom for _,atom in domain['mapping']}
            vacant=sorted(set(domain['ids'])-matched)
            if not vacant:continue
            target=self.candidate(domain,dict(ids=vacant,uid=domain['uid']),xyz,box,cutoff,core)
            target['parents']=[domain['uid']]
            new=len(target['ids'])/n*evaluate(target,xyz,box,n,cutoff)[0]
            gain=(costs[domain['uid']]-new)/n
            if gain>self.settings['min_gain']:
                key=fingerprint(dict(parents=target['parents'],mapping=target['mapping']))
                proposals.append((gain,key,target,[domain]))
        for i,a in enumerate(self.domains):
            for b in self.domains[i+1:]:
                if not self._same_component(a,b,components):continue
                for dominant,other in ((a,b),(b,a)):
                    target=self.candidate(dominant,other,xyz,box,cutoff,core)
                    retained=len(a['mapping'])+len(b['mapping'])
                    if len(target['mapping'])<self.settings['retain_fraction']*retained:continue
                    old=costs[a['uid']]+costs[b['uid']]
                    target_loss=evaluate(target,xyz,box,n,cutoff)[0]
                    # Coverage gain must not hide a newly damaged grain interior.
                    def alignment_loss(d):
                        unweighted=dict(d);unweighted.pop('slot_weights',None)
                        return evaluate(unweighted,xyz,box,n,cutoff)[0]-(n-len(d['mapping']))
                    parent_loss=alignment_loss(a)+alignment_loss(b)
                    old_alignment=parent_loss/max(1,retained)
                    new_alignment=alignment_loss(target)/max(1,len(target['mapping']))
                    if new_alignment>old_alignment+self.settings['max_alignment_loss_increase']:continue
                    new=len(target['ids'])/n*target_loss
                    gain=(old-new)/n
                    if gain>self.settings['min_gain']:
                        key=fingerprint(dict(parents=target['parents'],mapping=target['mapping']))
                        proposals.append((gain,key,target,[a,b]))
        if not proposals:
            self.pending=None
            return
        gain,key,target,parents=max(proposals,key=lambda p:(p[2].get('core_update',False),p[0],p[1]))
        count=self.pending['count']+1 if self.pending and self.pending['key']==key else 1
        self.pending=dict(key=key,count=count,gain=gain)
        if count>=self.settings['confirmations']:
            target['uid']=self.next_id;self.next_id+=1
            self.transition=dict(old=copy.deepcopy(parents),new=target,start=frame)
            self.last_mix=0.;self.last_event='takeover_started'

    def frame_score_context(self, domain, xyz, box, cutoff):
        n=len(self.population_ids or domain['ids'])
        owned={i for d in self.domains for i in d['ids']}
        other=float(n-len(owned))
        for d in self.domains:
            if d['uid']!=domain['uid']:
                other+=len(d['ids'])/n*evaluate(d,xyz,box,n,cutoff)[0]
        return dict(population=n,other_numerator=other)

    def refresh_frames(self, xyz, box, cutoff=None):
        domains=list(self.domains)
        if self.transition is not None:
            domains+=self.transition['old']+[self.transition['new']]
        for domain in domains:
            if domain.get('retired'):continue
            # Freeze reference acceptance during existing topology transitions.
            if self.frame_acceptance['status'] and self.transition is None:
                if cutoff is None:raise ValueError('Frame acceptance requires cutoff')
                update_accepted_frame(domain,xyz,box,cutoff,self.frame_acceptance,sample_id=self.last_frame,
                                      score_context=self.frame_score_context(domain,xyz,box,cutoff))
            elif self.frame_acceptance['status'] and 'accepted_frame' in domain:
                update_accepted_frame(domain,xyz,box,cutoff,self.frame_acceptance,propose=False)
            diagnostics(domain,material_frame(domain,xyz,box),rotation(domain['move']))

    def branches(self):
        if self.transition is None:return [(1.,d) for d in self.domains]
        t=self.transition;retired={d['uid'] for d in t['old']}
        return ([(1.,d) for d in self.domains if d['uid'] not in retired]+
                [(1.-self.last_mix,d) for d in t['old']]+[(self.last_mix,t['new'])])

    def evaluate(self, xyz, box, population, cutoff):
        owned={i for d in self.domains for i in d['ids']}
        numerator=float(population-len(owned))
        results=[];clusters=[]
        for blend,domain in self.branches():
            loss,entries=evaluate(domain,xyz,box,population,cutoff)
            prefactor=blend*len(domain['ids'])/population
            numerator+=prefactor*loss
            results.append((float(np.sqrt(loss/population)),entries,0.))
            clusters.append(dict(domain_id=domain['uid'],prefactor=prefactor,W_cluster=float(population)))
        return results,float(np.sqrt(max(0.,numerator/population))),float(population),clusters

    def context(self, reference_signature, box, cutoff):
        # Spatial component labels deliberately do not enter this fingerprint.
        transport=dict(frame_transport='gated_rigid_carriers_v2') if self.frame_acceptance['status'] else {}
        return fingerprint(dict(metric='rienv_core_domains_v4' if self.core_selector.settings['status'] else 'grid_material_domains_v2',reference=reference_signature,
            population=self.population_ids,box=np.asarray(box).tolist(),cutoff=cutoff,
            domains=structural_state(self.domains),transition=structural_state(self.transition),**transport))


def make_tracker(settings=None,state=None):
    if (settings or {}).get('core_voronoi',{}).get('status',False):
        mode=settings['core_voronoi'].get('mode','persistent')
        if mode=='single_center':
            from central_core import CentralCoreTracker
            return CentralCoreTracker(settings,state)
        if mode!='persistent':raise ValueError('Unknown core_voronoi mode: '+str(mode))
        if settings.get("cluster_pirmsd"):
            from cluster_pirmsd import ClusterTracker
            return ClusterTracker(settings,state)
        from core_voronoi import CoreVoronoiTracker
        return CoreVoronoiTracker(settings,state)
    return DomainTracker(settings,state)
