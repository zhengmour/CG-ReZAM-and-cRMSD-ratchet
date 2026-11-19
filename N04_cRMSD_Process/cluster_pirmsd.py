"""Physical-cluster PIRMSD on persistent core/Voronoi lattice registrations."""

import copy
from collections import Counter
import numpy as np
from scipy.special import expit
from scipy.spatial.transform import Rotation
from bulk_transport import bind_core, bind_members
from domain_alignment import stored_euler
from matching_domains import rotation
from rienv_core import unwrap

from core_voronoi import CoreVoronoiTracker
from material_frame import frame, minimum_image, rotation_pullback
from ratchet_state import fingerprint
from voronoi_geometry import radial_weights, set_radius


def cluster_terms(domain, xyz, box, material, factors, derivatives=False):
    """Weighted occupied-site S,W and their material-frame derivatives."""
    pairs = [(s, i) for s, i in domain['mapping'] if i in factors]
    rows = None
    if derivatives:
        ids = np.asarray(
            sorted(
                set(domain['ids'])
                | set(material['translation_ids'])
                | set(material['rotation_ids'])
            ),
            dtype=int,
        )
        rows = np.zeros((len(ids), 9))
        rows[:, 0] = ids + 1
    if not pairs:
        return 0.0, 0.0, rows
    slots, atoms = np.asarray(pairs).T
    targets = np.asarray(domain['targets'])[slots] @ material['rotation'].T
    relative = minimum_image(xyz[atoms] - material['center'], box)
    errors = minimum_image(relative - targets, box)
    radius = np.linalg.norm(relative, axis=1)
    w = radial_weights(radius, domain['r_cut'], domain['weight_alpha'])
    a = np.asarray([factors[i] for i in atoms])
    aw = a * w
    squared = np.einsum('ij,ij->i', errors, errors)
    if derivatives:

        def pullback(columns, mismatch, radial):
            g = rows[:, columns]
            direct = mismatch + radial
            g[np.searchsorted(ids, atoms)] += direct
            carriers = material['translation_ids']
            g[np.searchsorted(ids, carriers)] -= direct.sum(axis=0) / len(carriers)
            torque = rotation_pullback(material, targets, mismatch)
            torque += rotation_pullback(material, np.zeros_like(targets), radial)
            g[np.searchsorted(ids, material['rotation_ids'])] += torque

        dw = (-domain['weight_alpha'] * aw * (1 - w) / np.maximum(radius, 1e-30))[
            :, None
        ] * relative
        pullback(slice(2, 5), 2 * aw[:, None] * errors, squared[:, None] * dw)
        pullback(slice(5, 8), np.zeros_like(errors), dw)
        rows[np.searchsorted(ids, atoms), 8] = aw
    return float(aw @ squared), float(aw.sum()), rows


def topology(old, new):
    """Link parents/children by dominant overlap in each direction.

    A single exchanged surface atom between otherwise intact clusters must
    not count as both a merger and a split.
    """
    before = [set(c) for c in old]
    after = [set(c) for c in new]
    overlap = [[len(a & b) for b in after] for a in before]
    links = set()
    for i, row in enumerate(overlap):
        if row and max(row):
            links.add((i, int(np.argmax(row))))
    for j in range(len(after)):
        column = [row[j] for row in overlap]
        if column and max(column):
            links.add((int(np.argmax(column)), j))
    parents = [[i for i in range(len(old)) if (i, j) in links] for j in range(len(new))]
    children = Counter(i for i, j in links)
    events = []
    if any(not p for p in parents):
        events.append('birth')
    if any(len(p) > 1 for p in parents):
        events.append('merge')
    if any(n > 1 for n in children.values()):
        events.append('split')
    if len(children) < len(old):
        events.append('disappearance')
    if not events:
        events.append('membership')
    return events, parents


class ClusterTracker(CoreVoronoiTracker):
    metric_name = 'cluster_pirmsd_v1'

    def __init__(self, settings, state=None):
        super().__init__(settings, state)
        self.cluster_policy = settings['cluster_pirmsd']
        state = state or {}
        self.physical_clusters = state.get('physical_clusters', [])
        self.cluster_candidate = state.get('cluster_candidate')
        self.cluster_epoch = state.get('cluster_epoch', 0)
        self._factor_cache = {}
        if state and 'physical_clusters' not in state:
            # Import the material reference, discard bounded-score stagnation history.
            for _, domain in self.branches():
                domain.pop('frame_switch', None)

    def state(self):
        return dict(
            super().state(),
            physical_clusters=copy.deepcopy(self.physical_clusters),
            cluster_candidate=copy.deepcopy(self.cluster_candidate),
            cluster_epoch=self.cluster_epoch,
        )

    def update(
        self,
        frame,
        xyz,
        box,
        population_ids,
        components,
        reference,
        full_reference,
        cutoff,
        core,
        comm,
        iterations,
        rienv_scores=None,
    ):
        self._factor_cache.clear()
        super().update(
            frame,
            xyz,
            box,
            population_ids,
            components,
            reference,
            full_reference,
            cutoff,
            core,
            comm,
            iterations,
            rienv_scores,
        )
        if self.transition is not None or not self.initialized:
            self.cluster_candidate = None
            return
        observed = sorted(
            sorted(c) for c in components if len(c) > self.cluster_policy['min_size_exclusive']
        )
        if observed == self.physical_clusters:
            self.cluster_candidate = None
            return
        events, parents = topology(self.physical_clusters, observed)
        # A 100 -> 60 + 40 split still changes the surviving child's carriers,
        # even though the second physical fragment falls below the score cutoff.
        _, all_parents = topology(self.physical_clusters, components)
        child_counts = Counter(i for lineage in all_parents for i in lineage)
        split_parents = {i for lineage in parents for i in lineage if child_counts[i] > 1}
        if split_parents and 'split' not in events:
            events = [event for event in events if event != 'membership'] + ['split']
        # Confirm the same lineage, allowing a moving outer surface.
        signature = sorted(tuple(p) for p in parents)
        previous = self.cluster_candidate
        count = 1
        if (
            previous
            and previous.get('events') == events
            and previous['parents'] == [list(p) for p in signature]
        ):
            consistent, _ = topology(previous['observed'], observed)
            if consistent == ['membership']:
                count += previous['count']
        self.cluster_candidate = dict(
            parents=[list(p) for p in signature], observed=observed, count=count, events=events
        )
        required = dict(
            birth=self.policy['birth_confirmations'],
            merge=self.policy['merge_confirmations'],
            split=self.policy['merge_confirmations'],
            disappearance=self.policy['retire_confirmations'],
            membership=self.policy['assignment_confirmations'],
        )
        if count < max(required[event] for event in events):
            return
        if events == ['membership']:
            # Membership can drift in a growing cluster on every outer frame.
            # Keep its accepted registration and re-score the two historical
            # configurations with the new membership before deciding uphill.
            self.physical_clusters = observed
            self._event('cluster_membership_committed', sizes=[len(c) for c in observed])
            self.cluster_candidate = None
            return
        domains, cores, owners = self.domains, self.cores, self.owners
        if 'split' in events:
            domains, cores, owners = self._split_branches(
                observed, parents, split_parents, xyz, box, cutoff
            )
        self._start(domains, cores, owners, 'cluster_' + '_'.join(events))
        self.transition['physical_clusters'] = observed
        self.transition['cluster_parents'] = parents
        self.cluster_candidate = None

    def _split_branches(self, observed, parents, split_parents, xyz, box, cutoff):
        """Give persistent split children their parent's lattice at the split geometry."""
        fragments = [
            part for part, lineage in zip(observed, parents) if split_parents.intersection(lineage)
        ]
        cores = copy.deepcopy(self.cores)
        domains = list(self.domains)
        replacements = set()
        for part in fragments:
            members = set(part)
            live = [c for c in cores if c['status'] != 'retired' and set(c['members']) <= members]
            if not live and self.policy['allow_provisional']:
                claimed = {i for c in cores if c['status'] != 'retired' for i in c['members']}
                seeds = self._seeds(members - claimed, xyz, box, {}, 1)
                votes = Counter(self.owners[i] for i in part if i in self.owners)
                if seeds and votes:
                    seed = self._make_core(seeds[0], votes.most_common(1)[0][0], True)
                    cores.append(seed)
                    live = [seed]
            for old_id in sorted({c['domain'] for c in live}):
                old = next(d for d in self.domains if d['uid'] == old_id)
                associated = [c for c in live if c['domain'] == old_id]
                anchor = associated[0]['members']
                if (
                    set(old['anchors']) <= members
                    and set(old['accepted_frame']['carrier_ids']) <= members
                ):
                    continue
                branch = copy.deepcopy(old)
                branch['uid'] = self.next_id
                self.next_id += 1
                inherited = frame(old, xyz, box)
                core_positions = unwrap(
                    anchor, xyz, box, self.core_selector.settings['link_cutoff']
                )
                center = core_positions.mean(axis=0)
                shift = minimum_image(inherited['center'] - center, box)
                branch['targets'] = (
                    np.asarray(old['targets']) @ inherited['rotation'].T + shift
                ).tolist()
                branch['move'] = [0.0, 0.0, 0.0] + stored_euler(
                    Rotation.from_matrix(inherited['rotation'] @ rotation(old['move']))
                )
                branch['anchors'] = list(anchor)
                branch['anchor_offsets'] = (core_positions - core_positions[0]).tolist()
                branch['mapping'] = [[s, i] for s, i in old['mapping'] if i in members]
                branch['lineage'] = list(old.get('lineage', [old_id]))
                branch['split_parent'] = old_id
                branch.pop('frame_switch', None)
                pose = dict(center=center, rotation=np.eye(3))
                if self.policy['core_anchored_rematching']:
                    bind_core(branch, xyz, box, pose)
                else:
                    bind_members(branch, part, xyz, box, 5.0, pose)
                for c in associated:
                    c['domain'] = branch['uid']
                    c['history'].append(
                        dict(frame=self.last_frame, event='domain_split', parent=old_id)
                    )
                domains.append(branch)
                replacements.add(branch['uid'])
        active = {c['domain'] for c in cores if c['status'] != 'retired'}
        domains, owners = self._partition(
            [d for d in domains if d['uid'] in active], cores, xyz, box, cutoff
        )
        for d in domains:
            if d['uid'] in replacements:
                set_radius(d, xyz, box, self.policy)
        return domains, cores, owners

    def _compatible(self, a, b, xyz, box, cutoff):
        # A crystal reference can merge only inside one accepted physical cluster.
        anchors = set(a['anchors']) | set(b['anchors'])
        return any(
            anchors.issubset(part) for part in self.physical_clusters
        ) and super()._compatible(a, b, xyz, box, cutoff)

    def _advance(self, xyz, box, cutoff):
        transition = self.transition
        super()._advance(xyz, box, cutoff)
        if self.transition is None and 'physical_clusters' in transition:
            self.physical_clusters = transition['physical_clusters']
            self.cluster_epoch += 1
            self.events[-1]['cluster_parents'] = transition['cluster_parents']

    def _factors(self, domains, components):
        key = (id(domains), id(components))
        cached = self._factor_cache.get(key)
        if cached is not None and cached[0] is domains and cached[1] is components:
            return cached[2]
        matched = {i for d in domains for _, i in d['mapping']}
        result = {}
        for group in components:
            atoms = matched.intersection(group)
            a = float(
                expit(
                    self.cluster_policy['beta']
                    * (len(atoms) / self.cluster_policy['reference_size'] - 0.5)
                )
            )
            result.update(dict.fromkeys(atoms, a))
        self._factor_cache[key] = (domains, components, result)
        return result

    def _endpoints(self):
        if self.transition is None:
            return [(1.0, self.domains, self.physical_clusters)]
        t = self.transition
        return [
            (1 - self.last_mix, t['old'], self.physical_clusters),
            (self.last_mix, t['new'], t.get('physical_clusters', self.physical_clusters)),
        ]

    def context(self, reference_signature, box, cutoff):
        return fingerprint(
            [
                super().context(reference_signature, box, cutoff),
                self.registration_identity(),
                self.physical_clusters,
                self.transition.get('physical_clusters') if self.transition else None,
            ]
        )

    def registration_identity(self):
        return [self.metric_name, self.cluster_policy, self.cluster_epoch]

    def _score_definition(self):
        # Membership and mappings are fixed during one outer-step assessment.
        # Keep moving coordinates out of the stagnation-window identity.
        return fingerprint(
            [
                self.cluster_epoch,
                self.physical_clusters,
                [
                    (d['uid'], d['mapping'], d.get('frame_revision'), d.get('r_cut'))
                    for d in self.domains
                ],
            ]
        )

    def frame_score_context(self, domain, xyz, box, cutoff):
        factors = self._factors(self.domains, self.physical_clusters)
        other_s = other_w = 0.0
        for d in self.domains:
            if d['uid'] != domain['uid']:
                s, w, _ = cluster_terms(d, xyz, box, frame(d, xyz, box), factors)
                other_s += s
                other_w += w
        return dict(
            cluster_factors=factors,
            other_numerator=other_s,
            other_weight=other_w,
            cluster_definition=self._score_definition(),
        )

    def evaluate_global(self, xyz, box, population, cutoff):
        return super().evaluate(xyz, box, population, cutoff)[1]

    def evaluate(self, xyz, box, population, cutoff):
        if not self.force_valid(xyz, box):
            return [], None, 0.0, []
        total = weight = 0.0
        results, clusters = [], []
        for blend, domains, components in self._endpoints():
            factors = self._factors(domains, components)
            for d in domains:
                s = w = 0.0
                rows = []
                if blend:
                    s, w, rows = cluster_terms(
                        d, xyz, box, frame(d, xyz, box), factors, derivatives=True
                    )
                total += blend * s
                weight += blend * w
                results.append((float(np.sqrt(s / w)) if w else None, rows, 0.0))
                clusters.append(dict(domain_id=d['uid'], prefactor=blend, W_cluster=w))
        score = float(np.sqrt(total / weight)) if weight > 1e-30 else None
        return results, score, weight, clusters
