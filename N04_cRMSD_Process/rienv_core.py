"""Persistent high-RIEnv connected cores.
Scores must come from EnvironmentSimilarityKernel, never a PIRMSD proxy.
"""
from collections import deque
import numpy as np
from scipy.spatial import cKDTree
from material_frame import minimum_image, frame


def neighbors(ids,xyz,box,cutoff):
    ids=list(ids)
    if not ids:return {}
    near=cKDTree(np.asarray(xyz)[ids]%box,boxsize=box).query_ball_point(np.asarray(xyz)[ids]%box,cutoff)
    return {atom:sorted(ids[j] for j in near[i] if ids[j]!=atom) for i,atom in enumerate(ids)}


def groups(ids,xyz,box,cutoff):
    graph=neighbors(ids,xyz,box,cutoff);unseen=set(ids);result=[]
    while unseen:
        seed=min(unseen);unseen.remove(seed);todo=[seed];part=[]
        while todo:
            i=todo.pop();part.append(i)
            for j in graph[i]:
                if j in unseen:unseen.remove(j);todo.append(j)
        result.append(sorted(part))
    return result


def unwrap(ids,xyz,box,cutoff):
    graph=neighbors(ids,xyz,box,cutoff)
    positions={ids[0]:np.asarray(xyz[ids[0]]).copy()};queue=deque([ids[0]])
    while queue:
        i=queue.popleft()
        for j in graph[i]:
            proposed=positions[i]+minimum_image(xyz[j]-xyz[i],box)
            if j not in positions:positions[j]=proposed;queue.append(j)
            elif np.linalg.norm(proposed-positions[j])>1e-6:
                raise ValueError('RIEnv core winds around the periodic box; no single material chart')
    if len(positions)!=len(ids):raise ValueError('RIEnv anchor core is not connected')
    return np.array([positions[i] for i in ids])


def local_regions(ids,xyz,box,cutoff):
    """Cover periodic winding networks by connected, nonwinding local charts.

    Preserve ordinary components. Only winding components are partitioned;
    no atoms are dropped. Each patch spans < half a box per axis, so all of
    its bond images agree with the same local coordinate chart.
    """
    output=[]
    for component in groups(ids,xyz,box,cutoff):
        try:
            unwrap(component,xyz,box,cutoff)
            output.append(component);continue
        except ValueError as exc:
            if 'winds around' not in str(exc):raise
        graph=neighbors(component,xyz,box,cutoff);remaining=set(component)
        while remaining:
            seed=min(remaining);origin=xyz[seed];lo=np.zeros(3);hi=np.zeros(3)
            todo=deque([seed]);queued={seed};patch=[]
            while todo:
                i=todo.popleft()
                if i not in remaining:continue
                delta=minimum_image(xyz[i]-origin,box)
                lower=np.minimum(lo,delta);upper=np.maximum(hi,delta)
                if np.any(upper-lower>.45*np.asarray(box)):continue
                lo,hi=lower,upper;remaining.remove(i);patch.append(i)
                for j in graph[i]:
                    if j in remaining and j not in queued:queued.add(j);todo.append(j)
            output.append(sorted(patch))
    return output


class CoreSelector:
    def __init__(self,settings=None,state=None):
        self.settings=dict(status=False,enter=.8,leave=.7,history=3,min_core=12,
            link_cutoff=4.8,geometry_cutoff=2.5,rank_ratio=.01,
            weight_alpha=.25,weight_padding=4.8)
        self.settings.update(settings or {})
        s=self.settings
        if not 0<=s['leave']<=s['enter']<=1:raise ValueError('RIEnv thresholds must satisfy 0 <= leave <= enter <= 1')
        if s['min_core']<3 or s['history']<1:raise ValueError('RIEnv core requires >=3 sites and positive history')
        if s['link_cutoff']<=0 or s['geometry_cutoff']<=0 or s['weight_alpha']<=0 or s['weight_padding']<0:
            raise ValueError('Invalid RIEnv core distances/weights')
        state=state or {};self.samples=state.get('samples',[]);self.frames=state.get('frames',[])

    def state(self):return dict(samples=self.samples,frames=self.frames)

    def observe(self,frame_id,ids,scores):
        if scores is None:return
        if self.frames and frame_id<=self.frames[-1]:raise ValueError('Duplicate/stale RIEnv observation')
        scores=np.asarray(scores,dtype=float)
        if scores.shape!=(len(ids),) or not np.isfinite(scores).all() or np.any(scores< -1e-6):
            raise ValueError('RIEnv scores must be finite nonnegative per-target values')
        self.samples.append({str(i):float(max(q,0.)) for i,q in zip(ids,scores)})
        self.frames.append(int(frame_id))
        self.samples=self.samples[-self.settings['history']:];self.frames=self.frames[-self.settings['history']:]

    def qualified(self,ids,previous=()):
        if len(self.samples)<self.settings['history']:return []
        old=set(previous);out=[];s=self.settings
        for atom in ids:
            q=np.array([sample.get(str(atom),-1.) for sample in self.samples])
            if np.all(q>=s['enter']) or (atom in old and np.count_nonzero(q>=s['leave'])>=max(1,s['history']-1)):
                out.append(atom)
        return out

    def coherent_groups(self,ids,xyz,box):
        output=[]
        for group in local_regions(ids,xyz,box,self.settings['link_cutoff']):
            if len(group)<self.settings['min_core']:continue
            points=unwrap(group,xyz,box,self.settings['link_cutoff'])
            e=np.linalg.eigvalsh(np.cov(points.T))
            if e[-1]>0 and e[1]>=self.settings['rank_ratio']*e[-1]:output.append(group)
        return output

    def initial_domains(self,component,xyz,box):
        regions=local_regions(component,xyz,box,self.settings['link_cutoff'])
        if len(regions)>1:
            return [item for region in regions for item in self.initial_domains(region,xyz,box)]
        cores=self.coherent_groups(self.qualified(component),xyz,box)
        cores.sort(key=lambda c:(-len(c),c))
        graph=neighbors(component,xyz,box,self.settings['link_cutoff'])
        owner={};queue=deque()
        for k,core in enumerate(cores):
            for atom in core:owner[atom]=k;queue.append(atom)
        while queue:
            i=queue.popleft()
            for j in graph[i]:
                if j not in owner:owner[j]=owner[i];queue.append(j)
        return [(sorted(i for i,k in owner.items() if k==index),core) for index,core in enumerate(cores)]

    def provisional_domains(self, component, xyz, box):
        """Birth-only geometric anchors; never reselect them on score rankings."""
        result=[]
        for region in local_regions(component,xyz,box,self.settings['link_cutoff']):
            graph=neighbors(region,xyz,box,self.settings['link_cutoff'])
            points=unwrap(region,xyz,box,self.settings['link_cutoff'])
            center=points.mean(axis=0)
            seed=region[int(np.argmin(np.linalg.norm(points-center,axis=1)))]
            queue=deque([seed]);seen={seed};anchors=[]
            while queue:
                i=queue.popleft();anchors.append(i)
                for j in graph[i]:
                    if j not in seen:seen.add(j);queue.append(j)
                if len(anchors)>=self.settings['min_core']:
                    coords=unwrap(anchors,xyz,box,self.settings['link_cutoff'])
                    e=np.linalg.eigvalsh(np.cov(coords.T))
                    if e[-1]>0 and e[1]>=self.settings['rank_ratio']*e[-1]:break
            result.append((region,sorted(anchors)))
        return result

    def select(self,domain,xyz,box):
        if len(self.samples)<self.settings['history']:return domain['anchors'],'insufficient_history'
        material=frame(domain,xyz,box);targets=np.asarray(domain['targets'])@material['rotation'].T
        mapped={atom:slot for slot,atom in domain['mapping']}
        eligible=self.qualified(domain['ids'],() if domain.get('provisional') else domain['anchors'])
        eligible=[i for i in eligible if i in mapped and np.linalg.norm(minimum_image(
            xyz[i]-material['center']-targets[mapped[i]],box))<self.settings['geometry_cutoff']]
        candidates=self.coherent_groups(eligible,xyz,box)
        if not candidates:return None,'core_unreliable'
        previous=set(domain['anchors'])
        chosen=max(candidates,key=lambda c:(len(previous.intersection(c)),len(c),tuple(-i for i in c)))
        return chosen,'validated_connected_core'

    def set_weights(self,domain,xyz,box):
        material=frame(domain,xyz,box)
        # Pick a real grid point close to the core at (re)anchoring, then transport
        # that grid point with the material. Lab-grid bookkeeping is a gauge.
        offset=(material['grid']-material['center'])@material['rotation']
        # Rotation derivatives may now use the larger tracking cohort. Spatial
        # weights must still be centered on the selected core, not that cohort.
        anchor_offsets=np.asarray(domain['anchor_offsets'])
        anchor_reference=anchor_offsets-anchor_offsets.mean(axis=0)
        radius=max(np.linalg.norm(anchor_reference,axis=1))+self.settings['weight_padding']
        r=np.linalg.norm(np.asarray(domain['targets'])-offset,axis=1)
        weights=1/(1+np.exp(np.clip(self.settings['weight_alpha']*(r-radius),-60,60)))
        domain['slot_weights']=weights.tolist()
        domain['weight_grid_offset']=offset.tolist()
        domain['weight_radius']=float(radius)
