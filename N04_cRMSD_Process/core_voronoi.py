"""Persistent core lifecycle + unbounded periodic Voronoi matching domains.

Topology events are outer-step transactions. A quintic frozen-branch transition
suspends the ratchet; checkpoint state includes candidates, identities and lineage.
"""
import copy
from collections import deque
import numpy as np
from scipy.spatial import cKDTree
from bulk_transport import bind_core, bind_cluster
from domain_alignment import rematch_occupied
from matching_domains import DomainTracker, evaluate, minimum_image, rotation
from material_frame import frame as material_frame, structural_state, orientation_defined, update_accepted_frame, diagnostics
from rienv_core import groups, neighbors, unwrap
from voronoi_geometry import ownership, set_radius
from ratchet_state import fingerprint

DEFAULTS=dict(initialize_samples=3,allow_provisional=True,max_initial_provisional=8,
              seed_radius=10.,seed_separation=20.,birth_confirmations=3,
              growth_confirmations=3,retire_confirmations=6,merge_confirmations=5,
              merge_distance=9.6,merge_tolerance=1.,merge_fraction=.9,
              minimum_radius=4.8,radius_quantile=.9,radius_scale=.8,weight_alpha=.25,
              assignment_confirmations=2,instant_local_assignment=True,core_anchored_rematching=False)


class CoreVoronoiTracker(DomainTracker):
    def __init__(self,settings=None,state=None):
        settings=copy.deepcopy(settings or {})
        settings.setdefault('rienv_core',{})['status']=True
        settings.setdefault('frame_acceptance',{})['status']=True
        settings['frame_acceptance']['policy']='three_conditions'
        super().__init__(settings,state)
        self.policy=dict(DEFAULTS);self.policy.update(settings.get('core_voronoi',{}))
        for key in ('initialize_samples','max_initial_provisional','birth_confirmations','growth_confirmations',
                    'retire_confirmations','merge_confirmations','assignment_confirmations'):
            if int(self.policy[key])!=self.policy[key] or self.policy[key]<1:raise ValueError('Invalid '+key)
        for key in ('seed_radius','seed_separation','merge_distance','merge_tolerance','minimum_radius','weight_alpha'):
            if not np.isfinite(self.policy[key]) or self.policy[key]<=0:raise ValueError('Invalid '+key)
        for key in ('radius_quantile','radius_scale','merge_fraction'):
            if not 0<self.policy[key]<=1:raise ValueError('Invalid '+key)
        state=copy.deepcopy(state or {})
        if state and state.get('core_voronoi_schema')!=2:raise ValueError('Legacy core-transport state requires a new validated bulk-reference initialization')
        self.cores=state.get('cores',[]);self.next_core_id=state.get('next_core_id',0)
        self.initialized=state.get('initialized',False);self.observations=state.get('observations',0)
        self.monitors=state.get('monitors',{});self.events=state.get('events',[])
        self.owners={int(k):v for k,v in state.get('owners',{}).items()}
        self.suspended=state.get('suspended',[])
        if self.transition is not None:
            self.transition['new_owners']={int(k):v for k,v in self.transition['new_owners'].items()}

    def state(self):
        s=super().state();s.update(core_voronoi_schema=2,cores=self.cores,next_core_id=self.next_core_id,
            initialized=self.initialized,observations=self.observations,monitors=self.monitors,
            events=self.events,owners=self.owners,suspended=self.suspended)
        return copy.deepcopy(s)

    def _event(self,kind,**details):
        self.events.append(dict(frame=self.last_frame,kind=kind,**details));self.last_event=kind

    def _confirmed(self,key,signature,count):
        previous=self.monitors.get(key)
        n=previous['count']+1 if previous and previous['signature']==signature else 1
        self.monitors[key]=dict(signature=signature,count=n,last_frame=self.last_frame)
        return n>=count

    def _geometry(self,members,xyz,box):
        if len(members)<self.core_selector.settings['min_core']:return False
        try:p=unwrap(members,xyz,box,self.core_selector.settings['link_cutoff'])
        except ValueError:return False
        e=np.linalg.eigvalsh((p-p.mean(axis=0)).T@(p-p.mean(axis=0)))
        return bool(orientation_defined(p) and e[-2]>=self.core_selector.settings['rank_ratio']*e[-1])

    def _seeds(self,ids,xyz,box,scores,maximum=None):
        """Local connected geometric cores, never periodic domain chart fragments."""
        ids=sorted(ids);graph=neighbors(ids,xyz,box,self.core_selector.settings['link_cutoff'])
        claimed=set();result=[];centers=[];minimum=self.core_selector.settings['min_core']
        order=sorted(ids,key=lambda i:(-scores.get(i,0.),i))
        for seed in order:
            if seed in claimed:continue
            if any(np.linalg.norm(minimum_image(xyz[seed]-c,box))<self.policy['seed_separation'] for c in centers):continue
            todo=deque([seed]);seen={seed};members=[]
            while todo:
                i=todo.popleft()
                if i in claimed or np.linalg.norm(minimum_image(xyz[i]-xyz[seed],box))>self.policy['seed_radius']:continue
                members.append(i)
                if len(members)>=minimum and self._geometry(sorted(members),xyz,box):break
                for j in graph[i]:
                    if j not in seen:seen.add(j);todo.append(j)
            members=sorted(members)
            if not self._geometry(members,xyz,box):continue
            result.append(members);claimed.update(members);centers.append(xyz[seed].copy())
            if maximum is not None and len(result)>=maximum:break
        return result

    def _make_core(self,members,domain,provisional):
        uid=self.next_core_id;self.next_core_id+=1
        return dict(uid=uid,members=sorted(members),domain=domain,status='provisional' if provisional else 'mature',
                    born=self.last_frame,parents=[],history=[dict(frame=self.last_frame,event='born')])

    def _new_domain(self,members,xyz,box,reference,full_reference,cutoff,core,comm,iterations):
        # Core selects the initial lattice. Its subsequent natural motion is
        # carried by the whole physical cluster, not the local core or a cell.
        d=super().birth(members,xyz,box,reference,full_reference,cutoff,core,comm,iterations,members,True)
        if self.policy['core_anchored_rematching']:
            bind_core(d,xyz,box,material_frame(d,xyz,box))
        else:
            bind_cluster(d,xyz,box,self.population_ids,self.core_selector.settings['link_cutoff'])
        self.domains.remove(d)
        return d

    def _partition(self,domains,cores,xyz,box,cutoff,initialize_radius=False):
        owners,_=ownership(xyz,box,self.population_ids,cores,self.owners)
        new=[]
        for old in domains:
            d=copy.deepcopy(old);ids=sorted(i for i,k in owners.items() if k==d['uid'])
            if not ids:continue
            d['ids']=ids;d['core_ids']=sorted(c['uid'] for c in cores if c['domain']==d['uid'] and c['status']!='retired')
            owned=set(ids)
            d['mapping']=[pair for pair in d['mapping'] if pair[1] in owned]
            # Existing carriers must still belong to the domain. A dominant
            # core retirement is handled by explicit reference replacement.
            carriers=set(d['anchors'] if d.get('bulk_transport') else d.get('accepted_frame',{}).get('carrier_ids',d['anchors']))
            if not carriers<=set(ids):raise ValueError('Core transaction would strand a material carrier')
            if self.policy['core_anchored_rematching']:
                locked=set(d['anchors'])
                for c in cores:
                    if c['uid'] in d['core_ids']:
                        locked.update(c['members'])
                rematch_occupied(d,locked,xyz,box,cutoff,material_frame(d,xyz,box))
            matched={a for _,a in d['mapping']};donors=sorted(set(ids)-matched)
            if donors:d=super().candidate(d,dict(ids=donors,uid=d['uid']),xyz,box,cutoff,None)
            d.pop('parents',None)
            if initialize_radius or 'r_cut' not in d:set_radius(d,xyz,box,self.policy)
            new.append(d)
        return new,owners

    def _start(self,new,cores,owners,event):
        # Boundary ownership/slot changes do not suspend the whole system.
        # Core identities, c/R/r_cut and carrier definitions must still agree.
        if event=='assignment' and self.policy['instant_local_assignment']:
            from matching_history import assignment_reference_definition
            if assignment_reference_definition(self.domains)==assignment_reference_definition(new) and cores==self.cores:
                assigned={i:d['uid'] for d in new for i in d['ids']}
                if (len(assigned)!=sum(len(d['ids']) for d in new) or
                    set(assigned)!=set(self.population_ids) or assigned!=owners):
                    raise ValueError('Boundary assignment must uniquely cover the target population')
                changed=sum(self.owners.get(i)!=owner for i,owner in owners.items())
                self.domains=copy.deepcopy(new);self.owners=copy.deepcopy(owners)
                self.monitors.pop('assignment',None)
                self.last_mix=0.;self._event('local_assignment_committed',boundary_particles=changed);return
        self.transition=dict(old=copy.deepcopy(self.domains),new=copy.deepcopy(new),
                             new_cores=copy.deepcopy(cores),new_owners=owners,start=self.last_frame,event=event)
        self.last_mix=0.;self.monitors={};self._event(event+'_started')

    def _advance(self,xyz,box,cutoff):
        t=self.transition;u=min(1.,(self.last_frame-t['start'])/self.settings['transition_iterations'])
        mix=u**3*(10-15*u+6*u*u);n=len(self.population_ids)
        def score(ds):
            owned={i for d in ds for i in d['ids']}
            return (n-len(owned)+sum(len(d['ids'])/n*evaluate(d,xyz,box,n,cutoff)[0] for d in ds))/n
        if self.force_valid(xyz,box):self.protocol_score_work+=(mix-self.last_mix)*(score(t['new'])-score(t['old']))
        self.last_mix=mix
        if u>=1:
            self.domains=t['new'];self.cores=t['new_cores'];self.owners=t['new_owners']
            self.transition=None;self.last_mix=0.;self._event(t['event']+'_committed')

    def branches(self):
        if self.transition is None:return [(1.,d) for d in self.domains]
        return [(1-self.last_mix,d) for d in self.transition['old']]+[(self.last_mix,d) for d in self.transition['new']]

    def force_valid(self,xyz,box):
        for blend,d in self.branches():
            if blend==0:continue
            try:material_frame(d,xyz,box)
            except (ValueError,np.linalg.LinAlgError):return False
        return bool(self.domains or (self.transition and self.transition['new']))

    def refresh_frames(self,xyz,box,cutoff=None):
        self.suspended=[]
        domains=self.domains if self.transition is None else self.transition['old']+self.transition['new']
        for d in domains:
            try:
                update_accepted_frame(d,xyz,box,cutoff,self.frame_acceptance,
                                      propose=self.transition is None,sample_id=self.last_frame,
                                      score_context=self.frame_score_context(d,xyz,box,cutoff) if cutoff is not None and self.transition is None else None)
                diagnostics(d,material_frame(d,xyz,box),rotation(d['move']))
            except (ValueError,np.linalg.LinAlgError) as exc:
                self.suspended.append(dict(domain=d['uid'],reason=str(exc)))

    def evaluate(self,xyz,box,population,cutoff):
        if not self.force_valid(xyz,box):return [],None,0.,[]
        # Both endpoint partitions cover all particles whenever any core exists.
        numerator=0.;results=[];clusters=[]
        for blend,d in self.branches():
            if blend==0:
                entries=np.zeros((len(d['ids']),9));entries[:,0]=np.asarray(d['ids'])+1
                loss=float(population)
            else:loss,entries=evaluate(d,xyz,box,population,cutoff)
            prefactor=blend*len(d['ids'])/population;numerator+=prefactor*loss
            results.append((float(np.sqrt(loss/population)),np.asarray(entries).tolist(),0.))
            clusters.append(dict(domain_id=d['uid'],prefactor=prefactor,W_cluster=float(population)))
        if self.transition:
            if not self.transition['old']:numerator+=(1-self.last_mix)*population
            if not self.transition['new']:numerator+=self.last_mix*population
        return results,float(np.sqrt(max(0.,numerator/population))),float(population),clusters

    def context(self,reference_signature,box,cutoff):
        return fingerprint(dict(metric='core_voronoi_bulk_radial_v2',reference=reference_signature,
            population=self.population_ids,box=np.asarray(box).tolist(),cutoff=cutoff,
            domains=structural_state(self.domains),
            transition=None if self.transition is None else dict(old=structural_state(self.transition['old']),new=structural_state(self.transition['new']))))

    def _compatible(self,a,b,xyz,box,cutoff):
        """Common lattice phase/orientation test, not distance-only merging."""
        ma=material_frame(a,xyz,box);mb=material_frame(b,xyz,box)
        atoms=sorted(set(a['anchors']+b['anchors']))
        ta=(np.asarray(a['targets'])@ma['rotation'].T+ma['center'])%box
        tb=(np.asarray(b['targets'])@mb['rotation'].T+mb['center'])%box
        pa=ta[cKDTree(ta,boxsize=box).query(xyz[atoms]%box)[1]]
        pb=tb[cKDTree(tb,boxsize=box).query(xyz[atoms]%box)[1]]
        agree=np.linalg.norm(minimum_image(pa-pb,box),axis=1)<=self.policy['merge_tolerance']
        fits=(np.linalg.norm(minimum_image(xyz[atoms]-pa,box),axis=1)<cutoff)&(np.linalg.norm(minimum_image(xyz[atoms]-pb,box),axis=1)<cutoff)
        return float(np.mean(agree&fits))>=self.policy['merge_fraction']

    def update(self,frame,xyz,box,population_ids,components,reference,full_reference,cutoff,core,comm,iterations,rienv_scores=None):
        if self.last_frame is not None and frame<=self.last_frame:raise ValueError('Nonincreasing core lifecycle frame')
        if self.population_ids is not None and list(population_ids)!=self.population_ids:raise ValueError('Changed target identities')
        self.population_ids=list(population_ids);self.last_frame=frame;self.last_event='retained'
        if not len(population_ids):return
        fresh=rienv_scores is not None
        self.core_selector.observe(frame,population_ids,rienv_scores)
        if fresh:self.observations+=1
        if self.policy['core_anchored_rematching']:
            # Upgrade inherited bulk-carried branches once, at an update boundary.
            domains = self.domains if not self.transition else (
                self.domains + self.transition['old'] + self.transition['new'])
            for d in domains:
                if not d.get('core_transport'):
                    bind_core(d,xyz,box,material_frame(d,xyz,box))
        self.refresh_frames(xyz,box,cutoff)
        if self.transition is not None:self._advance(xyz,box,cutoff);return
        scores=dict(zip(population_ids,rienv_scores)) if fresh else {}
        qualified=set(self.core_selector.qualified(population_ids))
        args=(xyz,box,reference,full_reference,cutoff,core,comm,iterations)
        if not self.initialized:
            if self.observations<self.policy['initialize_samples']:self.last_event='initial_observation';return
            mature=self._seeds(qualified,xyz,box,scores)
            seeds=mature or (self._seeds(population_ids,xyz,box,scores,self.policy['max_initial_provisional']) if self.policy['allow_provisional'] else [])
            ds=[];cs=[]
            for members in seeds:
                d=self._new_domain(members,*args);c=self._make_core(members,d['uid'],not bool(mature))
                d['provisional']=c['status']=='provisional';ds.append(d);cs.append(c)
            self.domains,self.owners=self._partition(ds,cs,xyz,box,cutoff,True)
            self.cores=cs;self.initialized=True;self._event('initialized',cores=len(cs),provisional=not bool(mature));return
        # Quality confirmation counters consume fresh observations only.
        live=[c for c in self.cores if c['status']!='retired']
        if fresh:
            for c in live:
                valid=self._geometry(c['members'],xyz,box)
                mature_quality=all(self.core_selector.samples[-1].get(str(i),0.)>=self.core_selector.settings['leave'] for i in c['members'])
                bad=not valid or (c['status']=='mature' and not mature_quality)
                key='retire:%d'%c['uid']
                if not bad:self.monitors.pop(key,None)
                elif self._confirmed(key,str(c['uid']),self.policy['retire_confirmations']):
                    cs=copy.deepcopy(self.cores);retired=next(q for q in cs if q['uid']==c['uid']);retired['status']='retired'
                    retired['history'].append(dict(frame=frame,event='retired'))
                    survivors=[q for q in cs if q['domain']==c['domain'] and q['status']!='retired']
                    ds=[copy.deepcopy(d) for d in self.domains if d['uid']!=c['domain']]
                    if survivors:
                        # Reference replacement only when the retired core carried it.
                        old=next(d for d in self.domains if d['uid']==c['domain'])
                        survivor_atoms={i for q in survivors for i in q['members']}
                        if set(old['anchors'])<=survivor_atoms:ds.append(copy.deepcopy(old))
                        else:
                            d=self._new_domain(survivors[0]['members'],*args)
                            for q in survivors:q['domain']=d['uid']
                            ds.append(d)
                    ds,owners=self._partition(ds,cs,xyz,box,cutoff)
                    self._start(ds,cs,owners,'retirement');return
                if c['status']=='provisional' and set(c['members'])<=qualified and valid:
                    c['status']='mature';c['history'].append(dict(frame=frame,event='promoted'))
                    for d in self.domains:
                        if d['uid']==c['domain']:d['provisional']=False
                    self._event('core_promoted',core=c['uid'])
            # Grow a core only by persistent qualified neighbors. Carriers and
            # r_cut remain unchanged: growth is not a registration refit.
            claimed={i for c in live for i in c['members']}
            for c in live:
                candidates=sorted(qualified-claimed)
                if candidates:
                    dd=cKDTree(xyz[c['members']]%box,boxsize=box).query(xyz[candidates]%box)[0]
                    addition=[i for i,d in zip(candidates,dd) if d<=self.core_selector.settings['link_cutoff']]
                else:addition=[]
                key='growth:%d'%c['uid']
                if addition and self._geometry(sorted(c['members']+addition),xyz,box):
                    if self._confirmed(key,fingerprint(addition),self.policy['growth_confirmations']):
                        cs=copy.deepcopy(self.cores);q=next(q for q in cs if q['uid']==c['uid']);q['members']=sorted(q['members']+addition)
                        q['history'].append(dict(frame=frame,event='growth',added=addition))
                        ds,owners=self._partition(self.domains,cs,xyz,box,cutoff)
                        self._start(ds,cs,owners,'growth');return
                else:self.monitors.pop(key,None)
            # Births remain detectable inside an already fully owned Voronoi cell.
            excluded=set(claimed)
            for i in qualified-claimed:
                if any(np.min(np.linalg.norm(minimum_image(xyz[c['members']]-xyz[i],box),axis=1))<self.policy['seed_separation'] for c in live):excluded.add(i)
            candidates=self._seeds(qualified-excluded,xyz,box,scores)
            keys=set()
            for members in candidates:
                key='birth:'+fingerprint(members);keys.add(key)
                if self._confirmed(key,key,self.policy['birth_confirmations']):
                    d=self._new_domain(members,*args);c=self._make_core(members,d['uid'],False);d['provisional']=False
                    cs=copy.deepcopy(self.cores)+[c];ds,owners=self._partition(self.domains+[d],cs,xyz,box,cutoff)
                    self._start(ds,cs,owners,'birth');return
            for key in list(self.monitors):
                if key.startswith('birth:') and key not in keys:self.monitors.pop(key)
        # Distance/contact alone is insufficient: require common lattice phase.
        for ia,a in enumerate(self.domains):
            for b in self.domains[ia+1:]:
                ca=[c for c in live if c['domain']==a['uid']];cb=[c for c in live if c['domain']==b['uid']]
                ai=[i for c in ca for i in c['members']];bi=[i for c in cb for i in c['members']]
                distance=cKDTree(xyz[ai]%box,boxsize=box).query(xyz[bi]%box)[0].min()
                key='merge:%d:%d'%(a['uid'],b['uid'])
                if distance>self.policy['merge_distance'] or not self.force_valid(xyz,box):self.monitors.pop(key,None);continue
                if not self._compatible(a,b,xyz,box,cutoff):self.monitors.pop(key,None);continue
                if not self._confirmed(key,key,self.policy['merge_confirmations']):continue
                target=copy.deepcopy(a);target['uid']=self.next_id;self.next_id+=1
                target['lineage']=a.get('lineage',[a['uid']])+b.get('lineage',[b['uid']])
                cs=copy.deepcopy(self.cores)
                for c in cs:
                    if c['domain'] in (a['uid'],b['uid']) and c['status']!='retired':
                        c['domain']=target['uid'];c['history'].append(dict(frame=frame,event='domain_merge',parents=[a['uid'],b['uid']]))
                ds=[d for d in self.domains if d['uid'] not in (a['uid'],b['uid'])]+[target]
                ds,owners=self._partition(ds,cs,xyz,box,cutoff)
                merged=next(d for d in ds if d['uid']==target['uid'])
                set_radius(merged,xyz,box,self.policy)  # Explicit new branch; never mutate old radius.
                self._start(ds,cs,owners,'merge');return
        if self.suspended:return
        # Stable core list, evolving outer-boundary ownership and vacancy fill.
        ds,owners=self._partition(self.domains,self.cores,xyz,box,cutoff)
        definition=fingerprint(structural_state(ds));old=fingerprint(structural_state(self.domains))
        if definition!=old:
            if self.policy['core_anchored_rematching']:
                # A moving peripheral slot must not restart ownership confirmation.
                definition=fingerprint(owners)
            local_swap=self.policy['core_anchored_rematching'] and owners==self.owners
            if local_swap or self._confirmed('assignment',definition,self.policy['assignment_confirmations']):
                self._start(ds,self.cores,owners,'assignment')
        else:self.monitors.pop('assignment',None)
