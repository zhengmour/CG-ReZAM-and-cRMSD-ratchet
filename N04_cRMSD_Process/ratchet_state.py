"""PIRMSD ratchet epochs and committed restart bundles (no environment CV)."""
import hashlib
import copy
import json
import math
import os
from pathlib import Path
import uuid


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


class RatchetState:
    def __init__(self, context=None, minimum=None, previous=None, epoch=0,
                 minimum_state=None, previous_state=None):
        self.context, self.minimum, self.previous, self.epoch = context, minimum, previous, epoch
        self.minimum_state=copy.deepcopy(minimum_state)
        self.previous_state=copy.deepcopy(previous_state)

    def decide(self, value, weight, context, k, beta, random_value, transition=False, potential='quadratic', metric_state=None, bias_probability_floor=0.):
        if not math.isfinite(bias_probability_floor) or not 0<=bias_probability_floor<=1:
            raise ValueError('MC bias probability floor must be in [0, 1]')
        if potential not in ('quadratic', 'linear'):
            raise ValueError('Unknown ratchet potential: '+str(potential))
        if context is None or value is None or not math.isfinite(value) or value < 0 or not math.isfinite(weight) or weight <= 1e-30:
            return dict(valid=False, apply=False, reason='invalid_metric', delta=0., probability=1.)
        if transition:
            # Reference mixing is protocol work, not spontaneous structural progress.
            return dict(valid=True, apply=False, reason='reference_transition', delta=0., probability=1.)
        if metric_state is not None:
            if metric_state.get('context')!=context or metric_state.get('value')!=value or metric_state.get('weight')!=weight:
                raise ValueError('Joint PIRMSD/frame snapshot does not match evaluated metric')
            fingerprint(metric_state)  # Reject nonfinite/unserializable snapshots.
        elif self.minimum_state is not None:
            raise ValueError('A joint-state ratchet requires a frame snapshot on every evaluation')
        legacy_baseline=metric_state is not None and (self.minimum_state is None or self.previous_state is None)
        if self.context != context or self.minimum is None or legacy_baseline:
            self.context = context
            self.minimum = self.previous = value
            self.minimum_state=copy.deepcopy(metric_state)
            self.previous_state=copy.deepcopy(metric_state)
            self.epoch += 1
            return dict(valid=True, apply=False, reason='new_context_baseline', delta=0., probability=1.)
        if metric_state is not None:
            for stored,scalar in ((self.minimum_state,self.minimum),(self.previous_state,self.previous)):
                if stored.get('context')!=context or stored.get('value')!=scalar or (not metric_state.get('coordinate_dependent_weight') and stored.get('weight')!=weight):
                    raise ValueError('Ratchet history mixes different PIRMSD/frame definitions')
        if value <= self.previous:
            reason = 'new_minimum' if value < self.minimum else 'downhill_no_bias'
            if value < self.minimum:self.minimum_state=copy.deepcopy(metric_state)
            self.minimum = min(self.minimum,value)
            delta, probability, apply = 0., 1., False
        else:
            delta = (k*(value-self.previous) if potential == 'linear' else
                     .5*k*((value-self.minimum)**2-(self.previous-self.minimum)**2))
            probability = (1.-bias_probability_floor if k>0 else 1.)*math.exp(min(0., -beta*delta))
            apply = k>0 and random_value >= probability
            reason = 'apply_bias' if apply else 'uphill_accepted_no_bias'
        self.previous = value
        # This is the previous *observed* MD state. apply_bias is not an MD
        # rollback; keep the minimum's registration frozen on an uphill step.
        self.previous_state=copy.deepcopy(metric_state)
        return dict(valid=True, apply=apply, reason=reason, delta=delta, probability=probability)

    def to_dict(self):
        return dict(context=self.context, minimum=self.minimum, previous=self.previous, epoch=self.epoch,
                    minimum_state=copy.deepcopy(self.minimum_state),previous_state=copy.deepcopy(self.previous_state))


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def commit_bundle(directory, restart, state):
    """Publish only after restart and full JSON state exist. Old bundles survive."""
    directory, restart = Path(directory), Path(restart)
    if not restart.is_file() or restart.stat().st_size == 0:
        raise ValueError('Missing/empty LAMMPS restart')
    with restart.open('rb') as stream:
        os.fsync(stream.fileno())
    state = dict(state, restart=restart.name, restart_sha256=file_digest(restart), schema=1)
    state_path = directory / (restart.name+'.json')
    atomic_json(state_path, state)
    atomic_json(directory/'latest.json', dict(state=state_path.name, sha256=file_digest(state_path)))


def load_bundle(directory, signature):
    directory = Path(directory)
    manifest = directory/'latest.json'
    if not manifest.exists():
        return None
    pointer = json.loads(manifest.read_text())
    state_path = directory/pointer['state']
    if file_digest(state_path) != pointer['sha256']:
        raise ValueError('Checkpoint state checksum mismatch')
    state = json.loads(state_path.read_text())
    if state['schema'] != 1 or state['signature'] != signature:
        raise ValueError('Checkpoint settings/reference/input mismatch; use a new PIRMSD directory')
    restart = directory/state['restart']
    if file_digest(restart) != state['restart_sha256']:
        raise ValueError('LAMMPS restart checksum mismatch')
    return state


def restart_input(lines, restart):
    """Recreate fixes with original IDs so LAMMPS can restore their restart state."""
    result, count = [], 0
    for line in lines:
        words = line.split('#', 1)[0].split()
        if words and words[0] == 'read_data':
            result.append('read_restart "{}"\n'.format(Path(restart).resolve()))
            count += 1
        else:
            result.append(line)
    if count != 1:
        raise ValueError('Expected exactly one read_data command in lmp.in')
    return result
