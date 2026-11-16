"""Frozen development-region evaluation; unfinished frames never become negatives."""
from __future__ import annotations

import numpy as np

from pvdelta.hotspot_benchmark import evaluate_hotspot_regions
from pvdelta.instance_benchmark import _maximum_matching
from pvdelta.panel_annotation import _canonical, _sha
from pvdelta.region_annotation import validate_region_export
from pvdelta.region_polygon import simple_polygon_mask
from pvdelta.regions import polygon_to_mask


def freeze_region_protocol(pack, predictions, *, version=1):
    """Select existing predictions without references; bind rules and exact images."""
    if predictions.get('schema') != 'pvdelta.frozen-panel-predictions/v1':
        raise ValueError('Frozen predictions required')
    rows = predictions['frames']
    by_source = {row['source_sha256']: row for row in rows}
    if len(by_source) != len(rows):
        raise ValueError('Duplicate prediction sources')
    frames = []
    for frame in pack['frames']:
        row = by_source.get(frame['source_sha256'])
        shape = [frame['native_height'], frame['native_width']]
        if row is None or row.get('native_shape') != shape:
            raise ValueError('Missing predictions or wrong native grid')
        regions = [{'region_id': f['finding_id'], 'polygon_native_xy': f['polygon_native_xy']}
                   for f in row['findings']]
        evaluate_hotspot_regions([], regions, native_shape=shape)
        for region in regions:
            if not polygon_to_mask(region['polygon_native_xy'], shape).any():
                raise ValueError('Prediction extent contains no native pixels')
        frames.append({'source_sha256': frame['source_sha256'], 'case_id': frame['case_id'],
                       'native_shape': shape, 'predictions': regions})
    if version not in (1, 2):
        raise ValueError('Unsupported protocol version')
    protocol = {'schema': f'pvdelta.frozen-region-protocol/v{version}', 'pack_token': pack['pack_token'],
                'prediction_document_sha256': _sha(_canonical(predictions)),
                'rules': {'primary_iou': .5, 'auxiliary_iou': .1, 'ignore_fraction': .9,
                          'matching': 'one_to_one_maximum_cardinality',
                          'geometry': 'native_convex_review_extents_not_segmentation',
                          'ignore': 'unmatched_predictions_only_native_pixel_fraction',
                          'incomplete': 'unscored_not_negative'}, 'frames': frames}
    if version == 2:
        protocol['rules']['geometry'] = 'native_simple_polygon_pixel_centre_iou'
    return dict(protocol, protocol_sha256=_sha(_canonical(protocol)))


def _raster_scores(references, predictions, *, native_shape, iou_threshold):
    truth = [simple_polygon_mask(r['polygon_native_xy'], native_shape) for r in references]
    predicted = [simple_polygon_mask(r['polygon_native_xy'], native_shape) for r in predictions]
    ref_ids = [r['region_id'] for r in references]
    pred_ids = [r['region_id'] for r in predictions]
    overlaps = np.array([[int((p & r).sum())/int((p | r).sum()) for r in truth]
                         for p in predicted]).reshape(len(predicted), len(truth))
    matches = _maximum_matching(overlaps, iou_threshold, pred_ids, ref_ids)
    covered = set(matches.values())
    tp = len(matches)
    return {'iou_threshold': iou_threshold, 'reference_hotspot_boxes': len(references),
            'predicted_hotspot_regions': len(predictions), 'true_positives': tp,
            'false_positives': len(predicted)-tp, 'false_negatives': len(truth)-tp,
            'recall': tp/len(truth) if truth else None,
            'matches': [{'prediction_id': pred_ids[i], 'reference_id': ref_ids[j],
                         'iou': float(overlaps[i, j])} for i, j in sorted(matches.items())],
            'unmatched_prediction_ids': [r for i, r in enumerate(pred_ids) if i not in matches],
            'missed_reference_ids': [r for i, r in enumerate(ref_ids) if i not in covered]}


def evaluate_region_work(pack, content, protocol):
    """Score validated human declarations against a separately frozen baseline.

    No panel-mask eligibility filter is applied to reference regions. Counts are
    frame-local and cannot certify physical identity, electrical faults or IEC.
    """
    body = {k: v for k, v in protocol.items() if k != 'protocol_sha256'}
    if (protocol.get('schema') not in ('pvdelta.frozen-region-protocol/v1', 'pvdelta.frozen-region-protocol/v2')
            or protocol.get('pack_token') != pack['pack_token']
            or protocol.get('protocol_sha256') != _sha(_canonical(body))):
        raise ValueError('Frozen protocol or pack identity changed')
    checked = validate_region_export(pack, content)
    # Reconstruct with the published v1 constants to reject even a rehashed rule change.
    geometry = ('native_simple_polygon_pixel_centre_iou' if protocol['schema'].endswith('/v2')
                else 'native_convex_review_extents_not_segmentation')
    if protocol['rules'] != {'primary_iou': .5, 'auxiliary_iou': .1, 'ignore_fraction': .9,
                            'matching': 'one_to_one_maximum_cardinality',
                            'geometry': geometry,
                            'ignore': 'unmatched_predictions_only_native_pixel_fraction',
                            'incomplete': 'unscored_not_negative'}:
        raise ValueError('Unsupported evaluation rules')
    rows = protocol['frames']
    if len(rows) != len(pack['frames']) or any(
            r.get('source_sha256') != f['source_sha256']
            or r.get('native_shape') != [f['native_height'], f['native_width']]
            for r, f in zip(rows, pack['frames'])):
        raise ValueError('Protocol population differs from the pack')
    reports = []
    totals = {name: {'true_positives': 0, 'false_positives': 0, 'false_negatives': 0,
                     'ignored_predictions': 0} for name in ('primary', 'auxiliary')}
    for ref, row in zip(checked['frames'], rows):
        report = {'case_id': ref['case_id'], 'source_sha256': ref['source_sha256'],
                  'status': 'declared_reviewed' if ref['review_complete'] else 'pending',
                  'predictions': len(row['predictions'])}
        reports.append(report)
        if not ref['review_complete']:
            continue
        shape = row['native_shape']
        ignored = np.zeros(shape, bool)
        for region in ref['ignore']:
            ignored |= polygon_to_mask(region['polygon_native_xy'], shape)
        for name, threshold in (('primary', .5), ('auxiliary', .1)):
            scorer = _raster_scores if protocol['schema'].endswith('/v2') else evaluate_hotspot_regions
            score = scorer(ref['regions'], row['predictions'],
                                             native_shape=shape, iou_threshold=threshold)
            suppressed = []
            for region in row['predictions']:
                if region['region_id'] not in score['unmatched_prediction_ids']:
                    continue
                mask = polygon_to_mask(region['polygon_native_xy'], shape)
                if not mask.any():
                    raise ValueError('Prediction extent contains no native pixels')
                if (mask & ignored).sum()/mask.sum() >= .9:
                    suppressed.append(region['region_id'])
            score['unmatched_prediction_ids'] = [r for r in score['unmatched_prediction_ids']
                                                  if r not in suppressed]
            score['false_positives'] = len(score['unmatched_prediction_ids'])
            score['ignored_prediction_ids'] = suppressed
            score['ignored_predictions'] = len(suppressed)
            tp, fp = score['true_positives'], score['false_positives']
            score['precision'] = tp/(tp+fp) if tp+fp else None
            report[name] = score
            for key in totals[name]:
                totals[name][key] += score[key]
    complete = checked['completed_frames']
    for total in totals.values():
        tp, fp, fn = (total[k] for k in ('true_positives', 'false_positives', 'false_negatives'))
        total.update(precision=tp/(tp+fp) if tp+fp else None,
                     recall=tp/(tp+fn) if tp+fn else None)
    return {'schema': 'pvdelta.development-region-evaluation/v1',
            'protocol_sha256': protocol['protocol_sha256'],
            'geometry': geometry,
            'reference_export_sha256': _sha(content.encode() if isinstance(content, str) else content),
            'scored_frames': complete, 'pending_frames': len(reports)-complete,
            'status': 'partial_development_scores' if complete else 'awaiting_reference_review',
            'totals': totals if complete else None, 'frames': reports,
            'independent_review_verified': False, 'stage1_accepted': False,
            'scope': 'Frozen baseline versus declared frame-local review extents; no healthy-panel count'}
