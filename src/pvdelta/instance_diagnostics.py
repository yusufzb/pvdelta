"""Post-prediction geometric error patterns against explicit full-panel drafts."""
from __future__ import annotations

from collections import Counter

import numpy as np

from pvdelta.instance_benchmark import evaluate_panel_instances
from pvdelta.regions import polygon_to_mask


def diagnose_panel_instances(reference, proposals, *, source_sha256, native_shape,
                             iou_threshold=.75, ignore_fraction=.9):
    """Describe absent coverage, merged/split-like and duplicate-like geometry.

    The scorer validates exhaustive native identity, geometry and disjoint
    references first. Diagnostics never alter predictions, references, matching,
    measurement masks or review status. Thresholds below describe overlap
    patterns, not validated causes, actual panel counts or fault diagnoses.
    """
    proposals = list(proposals)
    score = evaluate_panel_instances(reference, proposals, source_sha256=source_sha256,
        native_shape=native_shape, iou_threshold=iou_threshold, ignore_fraction=ignore_fraction)
    panels = sorted(reference['panels'], key=lambda r: r['instance_id'])
    proposed = sorted(proposals, key=lambda p: p['proposal_id'])
    truth = [polygon_to_mask(r['polygon_native_xy'], native_shape) for r in panels]
    predicted = [polygon_to_mask(p['polygon_native_xy'], native_shape) for p in proposed]
    truth_union = np.zeros(native_shape, bool)
    prediction_union = np.zeros(native_shape, bool)
    ignored = np.zeros(native_shape, bool)
    for mask in truth:
        truth_union |= mask
    for mask in predicted:
        prediction_union |= mask
    for region in reference.get('ignore', []):
        ignored |= polygon_to_mask(region['polygon_native_xy'], native_shape)
    intersections = np.array([[int((p & t).sum()) for t in truth] for p in predicted], dtype=int)
    intersections = intersections.reshape(len(predicted), len(truth))
    truth_sizes = np.array([int(m.sum()) for m in truth])
    prediction_sizes = np.array([int(m.sum()) for m in predicted])
    # Empty input sets are valid; polygons with no native centres fail in rasterisation.
    coverage = np.divide(intersections, truth_sizes, out=np.zeros(intersections.shape, float),
                         where=truth_sizes > 0)
    purity = np.divide(intersections, prediction_sizes[:, None], out=np.zeros(intersections.shape, float),
                       where=prediction_sizes[:, None] > 0)
    matched_predictions = {m['proposal_id'] for m in score['matches']}
    ignored_predictions = set(score['ignored_proposal_ids'])
    prediction_rows = []
    for i, proposal in enumerate(proposed):
        mask, size = predicted[i], int(prediction_sizes[i])
        in_truth, in_ignore = int((mask & truth_union).sum()), int((mask & ignored).sum())
        outside = int((mask & ~(truth_union | ignored)).sum())
        merged = [panels[j]['instance_id'] for j in range(len(truth)) if coverage[i, j] >= .8]
        status = ('no_native_pixel_support' if size == 0 else
                  'matched' if proposal['proposal_id'] in matched_predictions else
                  'ignored' if proposal['proposal_id'] in ignored_predictions else
                  'unmatched_mostly_outside_reference_and_ignore' if outside/size >= .9 else
                  'unmatched_overlapping_reference_or_ignore')
        prediction_rows.append({'proposal_id': proposal['proposal_id'], 'score_status': status,
            'native_pixels': size, 'pixels_in_full_references': in_truth, 'pixels_in_ignore': in_ignore,
            'pixels_outside_reference_and_ignore': outside,
            'fraction_outside_reference_and_ignore': outside/size if size else None,
            'merge_like': len(merged) >= 2, 'merge_like_reference_ids': merged if len(merged) >= 2 else [],
            'reference_intersections': [{'instance_id': panels[j]['instance_id'],
                'native_pixels': int(intersections[i,j]),
                'fraction_of_reference': float(coverage[i,j]) if truth_sizes[j] else None,
                'fraction_of_proposal': float(purity[i,j]) if size else None}
                for j in range(len(truth)) if intersections[i,j] > 0]})
    reference_rows = []
    matched_references = {m['instance_id'] for m in score['matches']}
    for j, panel in enumerate(panels):
        size = int(truth_sizes[j])
        covered = int((truth[j] & prediction_union).sum())
        duplicates = [proposed[i]['proposal_id'] for i in range(len(predicted))
                      if coverage[i,j] >= .8 and purity[i,j] >= .8]
        fragments = [i for i in range(len(predicted)) if .1 <= coverage[i,j] < .8 and purity[i,j] >= .8]
        union = np.zeros(native_shape, bool)
        for i in fragments:
            union |= predicted[i] & truth[j]
        union_count = int(union.sum())
        repeated_pixels = int(intersections[fragments,j].sum())-union_count
        split = bool(len(fragments) >= 2 and size > 0 and union_count/size >= .8
                     and repeated_pixels/size <= .1)
        reference_rows.append({'instance_id': panel['instance_id'], 'native_pixels': size,
            'matched': panel['instance_id'] in matched_references,
            'covered_by_any_proposal_pixels': covered,
            'fraction_covered_by_any_proposal': covered/size if size else None,
            'absent_native_coverage': size > 0 and covered == 0,
            'duplicate_like': len(duplicates) >= 2,
            'duplicate_like_proposal_ids': duplicates if len(duplicates) >= 2 else [],
            'split_like': split,
            'split_like_proposal_ids': [proposed[i]['proposal_id'] for i in fragments] if split else [],
            'fragment_union_pixels': union_count, 'fragment_repeated_pixels': repeated_pixels})
    return {'schema': 'pvdelta.panel-instance-diagnostics/v1', 'scores': score,
        'reference_review_status': reference['review_status'],
        'proposal_status_counts': dict(Counter(r['score_status'] for r in prediction_rows)),
        'merge_like_proposals': sum(r['merge_like'] for r in prediction_rows),
        'split_like_references': sum(r['split_like'] for r in reference_rows),
        'duplicate_like_references': sum(r['duplicate_like'] for r in reference_rows),
        'references_without_native_coverage': sum(r['absent_native_coverage'] for r in reference_rows),
        'full_reference_pixels': int(truth_union.sum()), 'ignore_pixels': int(ignored.sum()),
        'ignore_fraction_of_frame': float(ignored.mean()),
        'proposal_diagnostics': prediction_rows, 'reference_diagnostics': reference_rows,
        'pattern_rules': {'merge_reference_coverage_min': .8, 'duplicate_reference_coverage_and_purity_min': .8,
            'split_fragment_reference_coverage_range': [.1, .8], 'split_fragment_purity_min': .8,
            'split_union_reference_coverage_min': .8, 'split_repeated_reference_fraction_max': .1},
        'scope': 'Overlap patterns under declared reference/ignore labels; not independent semantic validation, physical inventory, thermal accuracy or fault/IEC approval',
        'measurement_authorised': False}
