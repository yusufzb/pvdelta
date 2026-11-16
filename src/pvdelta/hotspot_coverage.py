"""Locate labelled hotspot misses without treating labels as module geometry."""
from __future__ import annotations

from collections import Counter

import numpy as np

from pvdelta.convex_geometry import convex_iou
from pvdelta.hotspot_benchmark import evaluate_hotspot_regions
from pvdelta.regions import polygon_to_mask
from pvdelta.support_partition import partition_measurement_support


def diagnose_hotspot_coverage(references, predictions, envelopes, *, native_shape,
                              margin_px=2, iou_threshold=.1):
    """Separate envelope coverage, excluded support and candidate localisation.

    Inputs are frame-local region dictionaries with unique ``region_id`` and
    ``polygon_native_xy``. Envelopes must be actual model output, never supplied
    hotspot boxes. Their disjoint measurement supports are reconstructed using
    the same partition rule as analysis. Bounding-box overlap is diagnostic;
    it does not establish that a labelled hot pixel was measured correctly.

    Call only after predictions are frozen. No detector settings, temperatures,
    findings or labels are changed. Unmatched published boxes are not evidence
    that a physical fault exists, nor are empty labels proof of healthy panels.
    """
    references, predictions, envelopes = list(references), list(predictions), list(envelopes)
    score = evaluate_hotspot_regions(references, predictions, native_shape=native_shape,
                                     iou_threshold=iou_threshold)
    # Reuse strict finite, convex, in-frame geometry and unique-ID validation.
    evaluate_hotspot_regions(envelopes, [], native_shape=native_shape)
    masks = [polygon_to_mask(p['polygon_native_xy'], native_shape) for p in envelopes]
    supports = partition_measurement_support(masks, margin_px=margin_px)
    envelope_union = np.zeros(native_shape, bool)
    measurement_union = np.zeros(native_shape, bool)
    for mask, support in zip(masks, supports):
        envelope_union |= mask
        measurement_union |= support.interior
    matched = {row['reference_id'] for row in score['matches']}
    rows = []
    for ref in references:
        mask = polygon_to_mask(ref['polygon_native_xy'], native_shape)
        pixels = int(mask.sum())
        in_envelope = int((mask & envelope_union).sum())
        in_measurement = int((mask & measurement_union).sum())
        best = max((convex_iou(ref['polygon_native_xy'], p['polygon_native_xy'])
                    for p in predictions), default=0.)
        bucket = ('matched' if ref['region_id'] in matched else
                  'no_proposed_envelope_overlap' if not in_envelope else
                  'envelope_present_but_measurement_excluded' if not in_measurement else
                  'candidate_unmatched_one_to_one' if best >= iou_threshold else
                  'candidate_overlap_below_iou_gate' if best > 0 else
                  'measurement_present_no_candidate_overlap')
        polygon = np.asarray(ref['polygon_native_xy'], dtype=float)
        edge_distance = min(float(polygon[:, 0].min()), float(polygon[:, 1].min()),
                            float(native_shape[1]-polygon[:, 0].max()),
                            float(native_shape[0]-polygon[:, 1].max()))
        rows.append({'reference_id': ref['region_id'], 'diagnostic_bucket': bucket,
                     'reference_native_pixels': pixels, 'envelope_overlap_pixels': in_envelope,
                     'measurement_overlap_pixels': in_measurement,
                     'fraction_in_measurement_support': in_measurement/pixels if pixels else None,
                     'best_candidate_iou': best, 'frame_edge_distance_px': edge_distance})
    return {'schema': 'pvdelta.hotspot-coverage/v1', 'scores': score, 'references': rows,
            'counts': dict(Counter(r['diagnostic_bucket'] for r in rows)),
            'proposed_envelope_pixels': int(envelope_union.sum()),
            'measurement_pixels': int(measurement_union.sum()), 'margin_px': margin_px,
            'scope': 'Published-box coverage diagnostics only; no semantic panel or field fault validation',
            'measurement_authorised': False}
