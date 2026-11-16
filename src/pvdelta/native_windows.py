"""Bounded local native-geometry search; no resized radiometry or inferred grid."""
from __future__ import annotations

from copy import deepcopy

import numpy as np

from pvdelta.convex_geometry import convex_iou
from pvdelta.native_boundaries import NativeBoundaryConfig, _neighbours
from pvdelta.native_lines import (
    _area_axis,
    _closed_scale_references,
    _compose_hybrid,
    _line_pass,
    native_line_proposals,
)


def _translate_proposal(proposal, window):
    x, y, right, bottom = window
    result = deepcopy(proposal)
    result['polygon_native_xy'] = (np.asarray(result['polygon_native_xy'])+[x, y]).tolist()
    if 'internal_structure_resolution' in result:
        resolution = result['internal_structure_resolution']
        resolution['reference_polygons_native_xy'] = [
            (np.asarray(p)+[x, y]).tolist() for p in resolution['reference_polygons_native_xy']]
    result.update(proposal_origin='local_native_window', native_window_xyxy=[x, y, right, bottom])
    return result


def _cross_panel_references(proposal, baseline):
    """Find two retained global panels crossed orthogonally by a local envelope."""
    quad = np.asarray(proposal['polygon_native_xy'], float)
    area, _, _, axis = _area_axis(quad)
    references = []
    for panel in baseline:
        other = np.asarray(panel['polygon_native_xy'], float)
        other_area, _, _, other_axis = _area_axis(other)
        if abs(float(axis@other_axis)) > .2:
            continue
        iou = convex_iou(quad, other)
        intersection = iou*(area+other_area)/(1+iou)
        if intersection/other_area < .25:
            continue
        for previous, previous_intersection in references:
            if (convex_iou(other, previous) <= .05 and _neighbours(other, previous)
                    and (intersection+previous_intersection)/area >= .8):
                return [previous.tolist(), other.tolist()]
        references.append((other, intersection))
    return []


def _withhold_cross_panel_locals(result, global_result, cfg):
    """Retain conflicting local evidence for review instead of consuming panel ROI.

    Only still-retained, disjoint compatible global neighbours can supply this
    conflict evidence. This does not establish that the global panels are true.
    """
    baseline = [p for p in global_result['proposals'] if any(
        p['polygon_native_xy'] == q['polygon_native_xy'] for q in result['proposals'])]
    retained, conflicts = [], []
    for proposal in result['proposals']:
        references = _cross_panel_references(proposal, baseline) if (
            proposal['proposal_origin'] == 'local_native_window') else []
        if references:
            conflicts.append(dict(deepcopy(proposal),
                reason='orthogonal_local_envelope_crosses_two_global_panels',
                reference_polygons_native_xy=references, semantic_panel_membership='unverified',
                measurement_authorised=False))
        else:
            retained.append(proposal)
    if conflicts:
        filtered = dict(result, proposals=retained)
        selected = _compose_hybrid(global_result, filtered, cfg)
        for proposal in selected['proposals']:
            previous = next((p for p in global_result['proposals'] if
                p['polygon_native_xy'] == proposal['polygon_native_xy']), None)
            proposal['proposal_origin'] = previous['proposal_origin'] if previous else 'local_native_window'
        selected.update(structure_pass='hybrid_with_local_windows',
            pre_conflict_selection_rejections={
                'superseded_compound_proposals':result['superseded_compound_proposals'],
                'composition_rejected_proposals':result['composition_rejected_proposals']})
    else:
        selected = dict(result)
    selected['local_orientation_conflicts'] = conflicts
    selected['local_window_search'] = dict(result['local_window_search'],
                                          orientation_conflicts_withheld=len(conflicts))
    return selected


def _supplement_raw_structure(baseline, raw, cfg):
    """Admit only disjoint raw envelopes supported by two retained scale references.

    Raw proposals have already passed all hybrid-side/divider/border/neighbour
    gates. References are geometry evidence, not semantic approval. New raw
    proposals cannot vote for each other or replace existing geometry.
    """
    retained = baseline['proposals']
    additions, rejected = [], []
    for proposal in raw['proposals']:
        if proposal['source_method'] not in ('four_observed_lines', 'observed_lines_and_native_ridges'):
            rejected.append(dict(deepcopy(proposal), reason='raw_requires_four_supported_sides',
                                 measurement_authorised=False))
            continue
        quad = np.asarray(proposal['polygon_native_xy'], float)
        overlaps = [convex_iou(quad, p['polygon_native_xy']) for p in retained]
        if any(iou > .65 for iou in overlaps):
            continue
        scale_pool = sorted(retained, key=lambda p: not _neighbours(
            quad, np.asarray(p['polygon_native_xy'])))
        references = _closed_scale_references(quad, scale_pool)
        adjacent = any(_neighbours(quad, np.asarray(p)) for p in references)
        if any(iou > .05 for iou in overlaps) or not references or not adjacent:
            rejected.append(dict(deepcopy(proposal),
                reason='raw_requires_disjoint_retained_scale_and_adjacent_references',
                measurement_authorised=False))
            continue
        additions.append(dict(deepcopy(proposal), proposal_origin='raw_structure_supplement',
            raw_scale_support={'reference_polygons_native_xy':references,
                               'semantic_panel_membership':'unverified'}))
    result = _compose_hybrid(baseline, dict(raw, proposals=additions,
        structure_pass='global_only_hybrid_supplement'), cfg) if additions else baseline
    for proposal in result['proposals']:
        previous = next((p for p in retained if p['polygon_native_xy'] == proposal['polygon_native_xy']), None)
        proposal['proposal_origin'] = previous['proposal_origin'] if previous else 'raw_structure_supplement'
    # Preserve earlier window/conflict evidence, even when this pass adds nothing.
    contour_count = sum(p['source_method']=='closed_contour' for p in result['proposals'])
    output = dict(baseline, proposals=result['proposals'],
        contour_proposal_count=contour_count,
        line_proposal_count=len(result['proposals'])-contour_count,
        structure_pass='local_windows_with_supported_raw_structure',
        raw_structure_supplement={'status':'unreviewed_geometry',
            'raw_input_proposals':len(raw['proposals']), 'eligible_additions':len(additions),
            'retained_additions':sum(p['proposal_origin']=='raw_structure_supplement' for p in result['proposals']),
            'rejected_proposals':rejected,
            'composition_rejections':result['composition_rejected_proposals'] if additions else [],
            'superseded_compounds':result['superseded_compound_proposals'] if additions else [],
            'radiometric_resampling':False, 'measurement_authorised':False})
    return output


def native_window_proposals(temperature_c, *, config=None, line_config=None,
                            raw_supplement=False):
    """Supplement unchanged global hybrid search with four overlapping windows.

    Windows cover two thirds of each native dimension (rounded up), anchored at
    source corners. All existing observed-side, border, shape and neighbour gates
    run in each window; area bounds are relative to that window. Native offsets
    translate geometry and reference provenance, never temperatures. Global
    geometry takes priority over near duplicates. Optional raw supplementation
    checks a separate global hybrid pass without local contrast, requiring two existing scale
    references, one adjacent reference, and no overlap with existing envelopes.
    Remaining conflicts stay
    unreviewed; this function does not authorise measurement or replace findings.
    """
    if not isinstance(raw_supplement, bool):
        raise TypeError('raw_supplement must be boolean')
    cfg = config or NativeBoundaryConfig()
    global_result = native_line_proposals(temperature_c, config=cfg,
                                          line_config=line_config, support_mode='hybrid')
    values = np.asarray(temperature_c)
    height, width = values.shape
    if min(height, width) < 24:
        return dict(global_result, local_window_search={'status':'native_grid_too_small', 'windows':[]})
    tile_h, tile_w = (2*height+2)//3, (2*width+2)//3
    additional, diagnostics = [], []
    for y in (0, height-tile_h):
        for x in (0, width-tile_w):
            window = [x, y, x+tile_w, y+tile_h]
            local = native_line_proposals(values[y:y+tile_h, x:x+tile_w], config=cfg,
                                          line_config=line_config, support_mode='hybrid')
            additional.extend(_translate_proposal(p, window) for p in local['proposals'])
            diagnostics.append({'native_window_xyxy':window, 'proposal_count':len(local['proposals']),
                'status':local['line_status'], 'composition':local['composition_diagnostics']})
    extra = dict(global_result, proposals=additional, structure_pass='four_native_windows')
    result = _compose_hybrid(global_result, extra, cfg)
    # Composition's baseline is the existing global hybrid, not segment-only.
    # Preserve its original provenance and explicitly label local supplements.
    for proposal in result['proposals']:
        existing = next((p for p in global_result['proposals'] if
            p['polygon_native_xy'] == proposal['polygon_native_xy']), None)
        if existing is not None:
            proposal['proposal_origin'] = existing['proposal_origin']
        else:
            proposal['proposal_origin'] = 'local_native_window'
    result.update(structure_pass='hybrid_with_local_windows',
        local_window_search={'status':'unreviewed_local_geometry', 'windows':diagnostics,
            'global_input_proposals':len(global_result['proposals']),
            'local_input_proposals':len(additional), 'source_shape':[height, width],
            'radiometric_resampling':False, 'measurement_authorised':False},
        global_search_rejections={
            'superseded_compound_proposals':global_result['superseded_compound_proposals'],
            'composition_rejected_proposals':global_result['composition_rejected_proposals']})
    selected = _withhold_cross_panel_locals(result, global_result, cfg)
    if not raw_supplement:
        return selected
    raw = _line_pass(values, config=cfg, line_config=line_config,
                     local_contrast=False, support_mode='hybrid')
    return _supplement_raw_structure(selected, raw, cfg)
