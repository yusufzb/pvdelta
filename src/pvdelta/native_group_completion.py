"""Research-only completion from four observed local cells, without old scale anchors."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from itertools import combinations

import numpy as np

from pvdelta.convex_geometry import convex_iou
from pvdelta.native_affine_groups import _basis, _cycles
from pvdelta.native_completion import _observed_sides, _validate
from pvdelta.native_line_groups import _valid_affine
from pvdelta.regions import polygon_to_mask
from pvdelta.support_partition import partition_measurement_support


def _direct_evidence(proposal):
    if not _observed_sides(proposal):
        return False
    if proposal['source_method'] == 'closed_contour':
        return True
    if proposal['source_method'] != 'four_observed_lines' or proposal.get('support_mode') != 'segments':
        return False
    original = np.asarray(proposal.get('original_side_support', []), float)
    return bool(original.shape == (4, 3) and np.isfinite(original).all()
                and (original[:, 0] == 1).all() and (original[:, 1] <= 1+1e-9).all()
                and np.array_equal(np.minimum(original[:, 1], 1), proposal['side_coverage'])
                and np.array_equal(original[:, 2], proposal['side_max_gap_fraction']))


def complete_local_native_groups(baseline, alternative, *, native_shape,
                                 native_temperature_sha256, margin_px=2, max_overlap_iou=.05):
    """Append unreviewed geometry backed by actual four-cell affine cycles.

    This opt-in alternative is NOT called by production extraction. The caller
    must bind baseline and alternative to the same source bytes and native grid.
    The native digest argument checks the extractor's supplied identity, not
    physical provenance. No labels, temperature ranking, CAD or filled cells
    are used. Regular non-PV structures can pass; PV semantics stay unverified.

    Every group member must have direct closed-contour or four-segment evidence
    and be a clean addition or an existing duplicate with nonempty support.
    Invalid/conflicting members invalidate the whole group. Old envelopes,
    IDs and native measurement interiors are preserved, even at excluded edges.
    """
    if not isinstance(baseline, dict) or not isinstance(alternative, dict):
        raise TypeError('Baseline and alternative must be geometry objects')
    if len(native_shape) != 2 or any(type(n) is not int or n < 1 for n in native_shape):
        raise ValueError('Positive native dimensions required')
    if (not isinstance(native_temperature_sha256, str) or len(native_temperature_sha256) != 64
            or any(c not in '0123456789abcdef' for c in native_temperature_sha256)
            or alternative.get('native_temperature_sha256') != native_temperature_sha256
            or alternative.get('native_shape') != list(native_shape)):
        raise ValueError('Alternative and caller native identity must match')
    if baseline.get('native_temperature_sha256', native_temperature_sha256) != native_temperature_sha256:
        raise ValueError('Baseline native identity differs from the caller')
    if (alternative.get('schema') != 'pvdelta.native-line-affine-groups/v1'
            or alternative.get('status') != 'unreviewed_line_affine_geometry'
            or alternative.get('radiometric_resampling') is not False
            or alternative.get('measurement_authorised') is not False
            or alternative.get('semantic_panel_membership') != 'unverified'):
        raise ValueError('Unreviewed native line-group evidence required')
    if (isinstance(max_overlap_iou, bool) or not np.isfinite(max_overlap_iou)
            or not 0 <= max_overlap_iou <= .05):
        raise ValueError('Envelope overlap limit must be finite in [0,.05]')
    _validate(baseline, native_shape)
    _validate(alternative, native_shape)
    groups = alternative.get('groups')
    if (not isinstance(groups, list) or len(groups) > 4096 or len(alternative['proposals']) > 512):
        raise ValueError('Bounded groups (4096) and observations (512) required')
    candidates = {p['proposal_id']:p for p in alternative['proposals']}
    group_ids, memberships = set(), set()
    for group in groups:
        if not isinstance(group, dict):
            raise TypeError('Group entries must be objects')
        gid, members = group.get('group_id'), group.get('proposal_ids')
        if (not isinstance(gid, str) or not gid or gid in group_ids or not isinstance(members, list)
                or len(members) != 4 or any(not isinstance(m, str) or m not in candidates for m in members)
                or len(set(members)) != 4 or tuple(sorted(members)) in memberships):
            raise ValueError('Unique group IDs and four distinct observed members required')
        group_ids.add(gid)
        memberships.add(tuple(sorted(members)))
    for p in candidates.values():
        actual = sorted(g['group_id'] for g in groups if p['proposal_id'] in g['proposal_ids'])
        if (not isinstance(p.get('affine_group_ids'), list) or sorted(p['affine_group_ids']) != actual):
            raise ValueError('Group membership must agree in both directions')
    retained = baseline['proposals']
    old_masks = [polygon_to_mask(p['polygon_native_xy'], native_shape) for p in retained]
    old_support = partition_measurement_support(old_masks, margin_px=margin_px)
    protected = np.zeros(native_shape, bool)
    for support in old_support:
        protected |= support.interior
    masks, reasons, duplicates = {}, {}, {}
    for pid, p in candidates.items():
        q = np.asarray(p['polygon_native_xy'], float)
        overlaps = [convex_iou(q, old['polygon_native_xy']) for old in retained]
        if not _direct_evidence(p) or not _valid_affine(q, native_shape):
            reasons[pid] = 'insufficient_direct_observed_geometry'
        elif any(v > .65 for v in overlaps):
            matches = [i for i, v in enumerate(overlaps) if v > .65 and old_support[i].interior.any()]
            if matches:
                duplicates[pid] = retained[max(matches, key=lambda i:overlaps[i])]['proposal_id']
                reasons[pid] = 'duplicates_existing_geometry'
            else:
                reasons[pid] = 'existing_duplicate_has_no_measurement_interior'
        elif any(v > max_overlap_iou for v in overlaps):
            reasons[pid] = 'overlaps_existing_envelope'
        else:
            mask = polygon_to_mask(q, native_shape)
            if (mask & protected).any():
                reasons[pid] = 'touches_existing_measurement_pixels'
            else:
                masks[pid] = mask
    certified, group_reasons = [], {}
    for group in sorted(groups, key=lambda g:g['group_id']):
        gid, members = group['group_id'], sorted(group['proposal_ids'])
        if any(m not in masks and m not in duplicates for m in members):
            group_reasons[gid] = 'group_member_unavailable'
            continue
        cycles, status = _cycles([{'quad':np.asarray(candidates[m]['polygon_native_xy'], float)} for m in members])
        if status != 'complete' or cycles != [(0, 1, 2, 3)]:
            group_reasons[gid] = 'not_four_actual_affine_cycle_members'
        else:
            certified.append((gid, members))
    supported = {m for _, members in certified for m in members if m in masks}
    conflicted = set()
    for first, second in combinations(sorted(supported), 2):
        if convex_iou(candidates[first]['polygon_native_xy'], candidates[second]['polygon_native_xy']) > max_overlap_iou:
            conflicted.update((first, second))
    accepted_groups = []
    for gid, members in certified:
        if conflicted.intersection(members):
            group_reasons[gid] = 'conflicting_member_groups_all_withheld'
        else:
            accepted_groups.append((gid, members))
    # Empty interiors must not remain invisible votes for other new cells.
    while True:
        selected = sorted({m for _, members in accepted_groups for m in members if m in masks},
            key=lambda m:(*np.asarray(candidates[m]['polygon_native_xy']).mean(axis=0)[::-1], m))
        supports = partition_measurement_support(old_masks+[masks[m] for m in selected], margin_px=margin_px)
        empty = {m for m, s in zip(selected, supports[len(retained):]) if not s.interior.any()}
        if not empty:
            break
        for gid, members in accepted_groups:
            if empty.intersection(members):
                group_reasons[gid] = 'group_member_has_no_measurement_interior'
        accepted_groups = [(gid, members) for gid, members in accepted_groups if not empty.intersection(members)]
    if any(not np.array_equal(a.interior, b.interior) for a, b in zip(old_support, supports)):
        raise RuntimeError('Local group completion changed old measurement pixels')
    output = deepcopy(baseline)
    assigned = dict(duplicates)
    used = {p['proposal_id'] for p in retained}
    number = 1
    for pid in selected:
        while f'C{number:04d}' in used:
            number += 1
        assigned[pid] = f'C{number:04d}'
        used.add(assigned[pid])
    for pid in selected:
        p = deepcopy(candidates[pid]);quad = np.asarray(p['polygon_native_xy'], float)
        member_groups = [(gid, members) for gid, members in accepted_groups if pid in members]
        shared = {m for _, members in member_groups for m in members if m != pid}
        inverse = np.linalg.inv(_basis(quad));neighbours = []
        for other in sorted(shared):
            delta = inverse@(np.asarray(candidates[other]['polygon_native_xy']).mean(axis=0)-quad.mean(axis=0))
            if ((.9 <= abs(delta[0]) <= 1.3 and abs(delta[1]) <= .15)
                    or (.9 <= abs(delta[1]) <= 1.3 and abs(delta[0]) <= .15)):
                neighbours.append(assigned[other])
        p.update(proposal_id=assigned[pid], proposal_origin='observed_local_group_completion',
            neighbour_ids=neighbours, measurement_authorised=False, semantic_panel_membership='unverified',
            completion_support={'alternative_proposal_id':pid,
                'local_group_ids':[gid for gid, _ in member_groups],
                'reference_polygons_native_xy':[candidates[m]['polygon_native_xy'] for m in sorted(shared)],
                'scale_basis':'four_actual_local_affine_cells', 'semantic_panel_membership':'unverified'})
        output['proposals'].append(p)
    rejected = [dict(deepcopy(p), group_completion_rejection_reason=reasons.get(pid, 'no_surviving_observed_group'))
                for pid, p in sorted(candidates.items()) if pid not in selected]
    output.update(structure_pass='baseline_with_unreviewed_local_group_completion', measurement_authorised=False,
        local_group_completion={'status':'unreviewed_geometry', 'baseline_proposals':len(retained),
            'alternative_proposals':len(candidates), 'retained_additions':len(selected),
            'validated_groups':len(certified), 'retained_groups':len(accepted_groups),
            'group_rejection_reasons':dict(sorted(group_reasons.items())),
            'rejection_counts':dict(Counter(p['group_completion_rejection_reason'] for p in rejected)),
            'rejected_proposals':rejected, 'old_measurement_pixels_lost':0,
            'old_geometry_and_ids_unchanged':True, 'native_temperature_sha256':native_temperature_sha256,
            'radiometric_resampling':False, 'semantic_panel_membership':'unverified',
            'measurement_authorised':False, 'efficacy_accepted':False})
    return output
