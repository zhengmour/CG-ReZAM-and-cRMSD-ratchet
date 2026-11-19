"""Per-MD-step PIRMSD forces on a low-frequency, frozen matching chart.

The stochastic gate is sampled once per physical timestep/chart. Repeated
LAMMPS setup callbacks and restart force reconstruction reuse that decision.
No velocity, mass, timestep multiplier, or RIEnv derivative enters this force.
"""
import copy
import numpy as np
from ratchet_state import RatchetState
from material_frame import frame as material_frame, minimum_image

KB_REAL = 0.0019872041


def metric_snapshot(tracker, xyz, box, timestep, context, xi, weight):
    """Concrete c/R values bound to an immutable metric-definition fingerprint.

    The checkpoint's domain tracker supplies the referenced mappings, weights
    and carrier templates; these are included in context, not duplicated at
    every MD step. Lab-frame motion does not change that context.
    """
    frames=[]
    for blend,domain in tracker.branches():
        if domain.get('retired') or blend==0:continue
        material=material_frame(domain,xyz,box)
        frames.append(dict(domain_id=domain['uid'],blend=blend,
                           frame_revision=domain.get('frame_revision',0),
                           center=material['center'].tolist(),rotation=material['rotation'].tolist(),
                           r_cut=domain.get('r_cut'),weight_alpha=domain.get('weight_alpha')))
    return dict(schema=1,context=context,value=xi,weight=weight,
                coordinate_dependent_weight=bool(tracker.settings.get('cluster_pirmsd')),
                timestep=int(timestep),frames=frames)


def orthogonal_box(lmp):
    box=lmp.extract_box()
    if any(abs(float(t))>1e-12 for t in box[2:5]) or not all(box[5]):
        raise ValueError('PIRMSD requires an orthogonal fully periodic box')
    return np.asarray(box[1])-np.asarray(box[0])


def fixed_metric(tracker, xyz, box, cutoff):
    """Pure evaluation: never select cores, remap, advance mixing or history."""
    n=len(tracker.population_ids or [])
    if not n:return None,0.,np.zeros_like(xyz),np.zeros(6)
    results,xi,weight,clusters=tracker.evaluate(xyz,box,n,cutoff)
    gradient=np.zeros_like(xyz);tensor=np.zeros((3,3))
    if xi is None or xi<=1e-15:return xi,weight,gradient,np.zeros(6)
    for (_,domain),(_,entries,_),cluster in zip(tracker.branches(),results,clusters):
        if cluster['prefactor']==0:
            continue
        rows=np.asarray(entries)
        ids=rows[:,0].astype(int)-1
        grad=cluster['prefactor']*(rows[:,2:5]-xi*xi*rows[:,5:8])/(2*xi*weight)
        np.add.at(gradient,ids,grad)
        if not domain.get('retired') and cluster['prefactor']!=0:
            material=material_frame(domain,xyz,box)
            center=material['center']
            # Consistent local images are essential for an internal-force virial.
            targets=np.asarray(domain['targets'])@material['rotation'].T
            offsets=minimum_image(xyz[ids]-center,box)
            index={atom:i for i,atom in enumerate(ids)}
            for slot,atom in domain['mapping']:
                offsets[index[atom]]=targets[slot]+minimum_image(xyz[atom]-center-targets[slot],box)
            if 'accepted_frame' in domain:
                for atom,pos in zip(material['translation_ids'],material['carrier_positions']):
                    offsets[index[atom]]=pos-center
            else:
                anchors=np.asarray(domain['anchors'],dtype=int)
                origin=xyz[anchors[0]];old=np.asarray(domain['anchor_offsets'])
                unwrapped=origin+old+minimum_image(xyz[anchors]-origin-old,box)
                for atom,pos in zip(anchors,unwrapped):offsets[index[atom]]=pos-center
            tensor+=offsets.T@grad
    virial=tensor[[0,1,2,0,0,1],[0,1,2,1,2,2]]
    return xi,weight,gradient,virial


class RatchetForce:
    def __init__(self,pirmsd,rng,state=None,trace=None):
        if not pirmsd.use_matching_domains:
            raise ValueError('Per-step bias requires boundary_score and persistent matching_domains')
        if not np.isfinite(pirmsd.temperature) or pirmsd.temperature<=0:
            raise ValueError('Ratchet temperature must be positive')
        if not np.isfinite(pirmsd.bias_k) or pirmsd.bias_k<0:
            raise ValueError('Ratchet force constant must be finite and nonnegative')
        self.mc_bias_probability_floor = getattr(pirmsd,'mc_bias_probability_floor',0.)
        if not np.isfinite(self.mc_bias_probability_floor) or not 0<=self.mc_bias_probability_floor<=1:
            raise ValueError('mc_bias_probability_floor must be in [0, 1]')
        self.potential = getattr(pirmsd, 'bias_potential', 'quadratic')
        self.ratchet_enabled = getattr(pirmsd, 'ratchet_enabled', True)
        if not isinstance(self.ratchet_enabled, bool):
            raise ValueError('ratchet_enabled must be boolean')
        if not self.ratchet_enabled and self.potential != 'linear':
            raise ValueError('Continuous PIRMSD bias requires bias_potential=linear')
        if self.potential not in ('quadratic', 'linear'):
            raise ValueError('Unknown ratchet potential: '+str(self.potential))
        self.pirmsd=pirmsd;self.rng=rng;self.trace=trace
        self.cache=copy.deepcopy((state or {}).get('cache'))
        self.evaluations=(state or {}).get('evaluations',0)
        self.error=None;self.lmp=None;self.fix_id='pirmsd_bias'
        self._chart_revision=None;self._context=None;self._registration=None
        self.enabled=True

    def state(self):return dict(cache=copy.deepcopy(self.cache),evaluations=self.evaluations)

    def evaluate(self,timestep,xyz,box):
        p=self.pirmsd
        tracker=p.domain_tracker
        revisions=tuple((d['uid'],d.get('frame_revision',0),id(d.get('accepted_frame')),d.get('r_cut'),d.get('weight_alpha'))
                        for _,d in tracker.branches())
        revision=(id(tracker),tracker.last_frame,id(tracker.transition),float(tracker.last_mix),tuple(box),revisions)
        if revision!=self._chart_revision:
            self._context=tracker.context(p.reference_signature,box,p.mismatch_cutoff)
            if getattr(tracker,'policy',{}).get('instant_local_assignment',False):
                from matching_history import registration_context
                self._registration=registration_context(tracker,p.reference_signature,box,p.mismatch_cutoff)
            self._chart_revision=revision
        context=self._context
        # Include mixing progress: forces change even when branch identities do not.
        key=[int(timestep),context,float(p.domain_tracker.last_mix),'joint_metric_state_v2',self.ratchet_enabled]
        if self.cache is not None and self.cache['key']==key:return self.cache
        xi,weight,gradient,virial=fixed_metric(p.domain_tracker,xyz,box,p.mismatch_cutoff)
        random_value=float(self.rng.random())
        previous=p.ratchet.previous
        previous_context=p.ratchet.context
        snapshot=None
        if tracker.transition is None and xi is not None:
            snapshot=metric_snapshot(tracker,xyz,box,timestep,context,xi,weight)
        mapping_rebase=None
        if snapshot is not None and getattr(tracker,'policy',{}).get('instant_local_assignment',False):
            from matching_history import capture_geometry,rebase_history
            capture_geometry(snapshot,tracker,xyz,box,self._registration)
            mapping_rebase=rebase_history(p.ratchet,tracker,xyz,box,p.mismatch_cutoff,context,self._registration)
            if mapping_rebase is not None:
                previous=p.ratchet.previous;previous_context=p.ratchet.context
        decision=p.ratchet.decide(xi,weight,context,p.bias_k,
            1/(KB_REAL*p.temperature),random_value,transition=p.domain_tracker.transition is not None,
            potential=self.potential,metric_state=snapshot,bias_probability_floor=self.mc_bias_probability_floor)
        reference=p.ratchet.minimum
        scalar=energy=0.
        if not self.ratchet_enabled:
            # V=k*xi, F=-k*grad(xi): history remains diagnostic only.
            # Frozen branch blends have valid derivatives during transitions too.
            valid=xi is not None and np.isfinite(xi) and weight>0
            apply=bool(valid and p.bias_k>0)
            decision=dict(valid=bool(valid),apply=apply,
                          reason='continuous_bias' if apply else 'continuous_inactive',
                          delta=0.,probability=0. if apply else 1.)
            if apply:scalar=-p.bias_k;energy=p.bias_k*xi
        elif decision['apply']:
            displacement=xi-reference
            if self.potential == 'linear':
                scalar=-p.bias_k
                energy=p.bias_k*displacement
            else:
                scalar=-p.bias_k*displacement
                energy=.5*p.bias_k*displacement**2
        forces=scalar*gradient
        if not np.isfinite(forces).all() or not np.isfinite(energy):
            raise ValueError('Nonfinite PIRMSD bias force/energy')
        p.global_xi_min=p.ratchet.minimum;p.previous_total_rmsd=p.ratchet.previous
        self.evaluations+=1
        self.cache=dict(key=key,xi=xi,weight=weight,reference=reference,previous=previous,
            reference_context=p.ratchet.context,previous_context=previous_context,
            decision=decision,mapping_rebase=mapping_rebase,energy=energy,forces=forces.tolist(),virial=(scalar*virial).tolist(),
            random_value=random_value,ratchet=p.ratchet.to_dict())
        if self.trace is not None:self.trace(copy.deepcopy(self.cache))
        return self.cache

    def attach(self,lmp):
        """Install BEFORE rigid/constraint fixes so setup half-kicks see bias."""
        self.lmp=lmp
        if lmp.extract_global('units')!='real':
            raise ValueError('PIRMSD bias parameters currently require LAMMPS units real')
        lmp.command('fix '+self.fix_id+' all external pf/callback 1 1')
        lmp.set_fix_external_callback(self.fix_id,self.callback)
        lmp.command('fix_modify '+self.fix_id+' energy yes virial yes')

    def callback(self,caller,timestep,nlocal,tags,x,forces):
        # Never let ctypes swallow an exception and silently continue with stale forces.
        forces[:]=0.
        if not self.enabled:
            self.lmp.fix_external_set_energy_global(self.fix_id,0.)
            self.lmp.fix_external_set_virial_global(self.fix_id,[0.]*6)
            return
        p=self.pirmsd;comm=p.comm
        xyz=np.asarray(self.lmp.gather_atoms('x',1,3)).reshape(-1,3)
        payload=None
        if p.rank==0:
            try:
                if self.error:raise RuntimeError(self.error)
                payload=(None,self.evaluate(timestep,xyz,orthogonal_box(self.lmp)),self.evaluations)
            except Exception as exc:payload=(str(exc),None,self.evaluations)
        error,cache,count=comm.bcast(payload,root=0)
        if error:
            self.error=error
            self.lmp.force_timeout()
            self.lmp.fix_external_set_energy_global(self.fix_id,0.)
            self.lmp.fix_external_set_virial_global(self.fix_id,[0.]*6)
            return
        self.cache=cache;self.evaluations=count
        p.ratchet=RatchetState(**cache['ratchet'])
        p.global_xi_min=p.ratchet.minimum;p.previous_total_rmsd=p.ratchet.previous
        forces[:]=np.asarray(cache['forces'])[np.asarray(tags).reshape(-1).astype(int)-1]
        self.lmp.fix_external_set_energy_global(self.fix_id,cache['energy'])
        self.lmp.fix_external_set_virial_global(self.fix_id,cache['virial'])

    def check(self):
        if self.error:raise RuntimeError('PIRMSD force callback failed; no checkpoint committed: '+self.error)
