"""Opt-in observed-geometry completion that cannot replace baseline support."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy

import numpy as np

from pvdelta.convex_geometry import convex_iou
from pvdelta.native_boundaries import _neighbours
from pvdelta.native_lines import _closed_scale_references
from pvdelta.regions import polygon_to_mask
from pvdelta.support_partition import partition_measurement_support


def _validate(result, shape):
    if not isinstance(result, dict) or not isinstance(result.get('proposals'), list):
        raise TypeError('A boundary result with a proposals list is required')
    ids = set()
    for proposal in result['proposals']:
        identifier = proposal.get('proposal_id')
        if not isinstance(identifier, str) or not identifier or identifier in ids:
            raise ValueError('Unique nonempty proposal IDs required within each result')
        ids.add(identifier)
        quad = np.asarray(proposal.get('polygon_native_xy'), float)
        if (quad.shape != (4, 2) or not np.isfinite(quad).all() or (quad < 0).any()
                or (quad[:, 0] > shape[1]).any() or (quad[:, 1] > shape[0]).any()):
            raise ValueError('Finite convex quadrilaterals inside the native frame required')
        convex_iou(quad, quad)
        polygon_to_mask(quad, shape)


def _observed_sides(proposal):
    if proposal.get('source_method') == 'native_level_regions':
        from pvdelta.native_level_regions import supported_level_region

        return supported_level_region(proposal)
    if proposal.get('source_method') == 'closed_contour':
        votes = proposal.get('threshold_votes')
        return type(votes) is int and 2 <= votes <= 3
    if proposal.get('source_method') == 'four_native_transitions':
        # Mixed transitions require two observed ridges; a four-step warm patch
        # is not sufficient to append a panel, even with nearby scale priors.
        models = proposal.get('side_model_families')
        evidence = proposal.get('native_transition_evidence', {})
        if (not isinstance(models, list) or len(models) != 4
                or any(m not in ('ridge', 'step') for m in models) or models.count('ridge') < 2
                or not isinstance(evidence, dict)
                or evidence.get('schema') != 'pvdelta.native-transition-evidence/v1'
                or evidence.get('purpose') != 'geometry_only' or evidence.get('measurement_authorised') is not False):
            return False
        edges = evidence.get('edges')
        if not isinstance(edges, list) or len(edges) != 4:
            return False
        try:
            for index, (edge, model) in enumerate(zip(edges, models)):
                if (not isinstance(edge, dict) or edge.get('status') != 'supported_native_transition'
                        or edge.get('family') != model or type(edge.get('strip_consensus_count')) is not int
                        or not 3 <= edge['strip_consensus_count'] <= 4
                        or not np.isfinite([edge['offset_px'], edge['coverage'], edge['largest_gap_fraction']]).all()
                        or abs(edge['offset_px']) > .75
                        or abs(edge['coverage']-proposal['side_coverage'][index]) > 1e-9
                        or abs(edge['largest_gap_fraction']-proposal['side_max_gap_fraction'][index]) > 1e-9):
                    return False
        except (KeyError, IndexError, TypeError, ValueError):
            return False
    elif proposal.get('source_method') not in ('four_observed_lines', 'observed_lines_and_native_ridges'):
        return False
    coverage = np.asarray(proposal.get('side_coverage', []), float)
    gaps = np.asarray(proposal.get('side_max_gap_fraction', []), float)
    return bool(coverage.shape == gaps.shape == (4,) and np.isfinite(coverage).all()
                and np.isfinite(gaps).all() and (coverage >= .8).all() and (coverage <= 1).all()
                and (gaps >= 0).all() and (gaps <= .18).all())


def complete_native_proposals(baseline, alternative, *, native_shape, margin_px=2,
                              max_overlap_iou=.05, manual_panel_mask=None):
    """Append only observed alternatives with existing, disjoint scale evidence.

    Both inputs must come from actual native geometry searches on the SAME
    source/grid; the caller must verify source and temperature hashes. This
    geometry-only helper cannot establish image provenance or panel semantics.
    It never sees reference labels, RGB, CAD or temperatures, and invents no
    missing sides or grid cells. Opt-in closed level regions have their own
    two-adjacent-level occupancy/persistence evidence, not segment-side evidence.
    The baseline must already be suitable for the
    caller's use: preserving its geometry is not proof that it is correct.

    Existing proposal dictionaries/IDs are retained exactly. Alternatives may
    share excluded envelope edges but cannot touch ANY old measurement pixel.
    Two baseline scale references (one adjacent) are required by default. An
    explicit manual_panel_mask can replace adjacency: every candidate envelope
    pixel must belong to that same-grid user panel support. Unknown/non-panel
    pixels must be False; source identity and annotation meaning are caller duties.
    Two observed scale references remain mandatory. Alternatives
    cannot vote for each other; conflicting eligible alternatives are all
    withheld rather than choosing a winner. All supports are partitioned again
    and checked bit-for-bit before returning unreviewed geometry.
    """
    if (len(native_shape) != 2 or any(type(n) is not int or n < 1 for n in native_shape)):
        raise ValueError('A positive native (height, width) grid is required')
    if (isinstance(max_overlap_iou, bool) or not np.isfinite(max_overlap_iou)
            or not 0 <= max_overlap_iou <= .05):
        raise ValueError('max_overlap_iou must be finite in [0,.05]')
    _validate(baseline, native_shape)
    _validate(alternative, native_shape)
    if manual_panel_mask is not None:
        manual_panel_mask = np.asarray(manual_panel_mask)
        if manual_panel_mask.dtype != np.bool_ or manual_panel_mask.shape != tuple(native_shape):
            raise ValueError('Manual panel support must be boolean on the native grid')
    retained = baseline['proposals']
    old_masks = [polygon_to_mask(p['polygon_native_xy'], native_shape) for p in retained]
    old_supports = partition_measurement_support(old_masks, margin_px=margin_px)
    protected = np.zeros(native_shape, bool)
    for support in old_supports:
        protected |= support.interior
    eligible = []
    rejected = []

    def reject(proposal, reason):
        rejected.append(dict(deepcopy(proposal), completion_rejection_reason=reason,
                             measurement_authorised=False))

    ordered = sorted(alternative['proposals'], key=lambda p: (
        *np.asarray(p['polygon_native_xy'], float).mean(axis=0)[::-1], p['proposal_id']))
    for proposal in ordered:
        quad = np.asarray(proposal['polygon_native_xy'], float)
        overlaps = [convex_iou(quad, p['polygon_native_xy']) for p in retained]
        if any(iou > .65 for iou in overlaps):
            reject(proposal, 'duplicates_existing_geometry')
            continue
        if not _observed_sides(proposal):
            reject(proposal, 'insufficient_observed_side_evidence')
            continue
        if any(iou > max_overlap_iou for iou in overlaps):
            reject(proposal, 'overlaps_existing_envelope')
            continue
        mask = polygon_to_mask(quad, native_shape)
        if manual_panel_mask is not None and not manual_panel_mask[mask].all():
            reject(proposal, 'outside_explicit_manual_panel_support')
            continue
        if (mask & protected).any():
            reject(proposal, 'touches_existing_measurement_pixels')
            continue
        pool = sorted((p for p, support in zip(retained, old_supports)
                       if _observed_sides(p) and support.interior.any()),
                      key=lambda p: not _neighbours(quad, np.asarray(p['polygon_native_xy'])))
        references = _closed_scale_references(quad, pool)
        if not references or (manual_panel_mask is None and not any(
                _neighbours(quad, np.asarray(p)) for p in references)):
            reject(proposal, 'requires_two_baseline_scale_references_one_adjacent')
            continue
        eligible.append((proposal, mask, references))
    conflicted = set()
    for i, (proposal, _, _) in enumerate(eligible):
        for j in range(i+1, len(eligible)):
            if convex_iou(proposal['polygon_native_xy'], eligible[j][0]['polygon_native_xy']) > max_overlap_iou:
                conflicted.update((i, j))
    additions = []
    for index, (proposal, mask, references) in enumerate(eligible):
        if index in conflicted:
            reject(proposal, 'conflicting_alternative_envelopes_all_withheld')
        else:
            additions.append((proposal, mask, references))
    supports = partition_measurement_support(old_masks + [r[1] for r in additions], margin_px=margin_px)
    nonempty = []
    for item, support in zip(additions, supports[len(retained):]):
        if support.interior.any():
            nonempty.append(item)
        else:
            reject(item[0], 'no_unambiguous_measurement_interior')
    supports = partition_measurement_support(old_masks + [r[1] for r in nonempty], margin_px=margin_px)
    if any(not np.array_equal(old.interior, new.interior) for old, new in zip(old_supports, supports)):
        raise RuntimeError('Completion changed baseline measurement support')
    output = deepcopy(baseline)
    used_ids = {p['proposal_id'] for p in retained}
    next_id = 1
    for proposal, _, references in nonempty:
        while f'E{next_id:04d}' in used_ids:
            next_id += 1
        identifier = f'E{next_id:04d}'
        used_ids.add(identifier)
        added = deepcopy(proposal)
        added.update(proposal_id=identifier, proposal_origin='supported_native_completion',
            neighbour_ids=[p['proposal_id'] for p in retained if _neighbours(
                np.asarray(proposal['polygon_native_xy']), np.asarray(p['polygon_native_xy']))],
            completion_support={'alternative_proposal_id': proposal['proposal_id'],
                'alternative_origin': proposal.get('proposal_origin'),
                'reference_polygons_native_xy': references,
                'semantic_panel_membership': 'unverified'}, measurement_authorised=False)
        output['proposals'].append(added)
    contour_count = sum(p.get('source_method') == 'closed_contour' for p in output['proposals'])
    output.update(structure_pass='baseline_with_supported_native_completion',
        contour_proposal_count=contour_count, line_proposal_count=len(output['proposals'])-contour_count,
        measurement_authorised=False,
        native_completion={'status': 'unreviewed_geometry', 'margin_px': margin_px,
            'max_overlap_iou': max_overlap_iou, 'baseline_proposals': len(retained),
            'alternative_proposals': len(alternative['proposals']), 'retained_additions': len(nonempty),
            'old_geometry_and_ids_unchanged': True, 'old_measurement_pixels_lost': 0,
            'alternative_config': deepcopy(alternative.get('config')),
            'rejection_counts': dict(Counter(p['completion_rejection_reason'] for p in rejected)),
            'rejected_proposals': rejected, 'radiometric_resampling': False,
            'semantic_panel_membership': 'unverified', 'measurement_authorised': False})
    if manual_panel_mask is not None:
        import hashlib

        output['native_completion']['reference_policy'] = 'two_observed_scales_and_explicit_manual_surface'
        output['native_completion']['manual_panel_mask_sha256'] = hashlib.sha256(
            manual_panel_mask.tobytes()).hexdigest()
    return output
