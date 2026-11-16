"""One-to-one full-panel evaluation on explicitly exhaustive native references."""
from __future__ import annotations

import numpy as np

from pvdelta.convex_geometry import convex_iou
from pvdelta.regions import polygon_to_mask


def _polygons(items, shape, id_key):
    polygons, identifiers = [], []
    for item in items:
        identifier = item.get(id_key)
        if not isinstance(identifier, str) or not identifier or identifier in identifiers:
            raise ValueError('unique nonempty instance/proposal IDs required')
        polygon = np.asarray(item.get('polygon_native_xy'), float)
        if (polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3
                or not np.isfinite(polygon).all() or (polygon < 0).any()
                or (polygon[:, 0] > shape[1]).any() or (polygon[:, 1] > shape[0]).any()):
            raise ValueError('finite native polygons must stay inside the source grid')
        convex_iou(polygon, polygon)  # reject degenerate, unordered or non-convex annotations
        polygons.append(polygon)
        identifiers.append(identifier)
    return polygons, identifiers


def _maximum_matching(overlaps, threshold, proposal_ids, instance_ids):
    """Maximum-cardinality matching; IoU orders traversal, not a total-IoU optimum."""
    edges = [sorted(np.flatnonzero(row >= threshold),
                    key=lambda j: (-row[j], instance_ids[j])) for row in overlaps]
    by_instance, by_proposal = {}, {}
    order = sorted(range(len(proposal_ids)),
                   key=lambda i: (-max(overlaps[i], default=0), proposal_ids[i]))
    for initial in order:
        queue, seen, previous, free = [initial], {initial}, {}, None
        for proposal in queue:
            for instance in edges[proposal]:
                if instance in previous:
                    continue
                previous[instance] = proposal
                if instance not in by_instance:
                    free = instance
                    break
                occupant = by_instance[instance]
                if occupant not in seen:
                    seen.add(occupant)
                    queue.append(occupant)
            if free is not None:
                break
        while free is not None:
            proposal = previous[free]
            old_instance = by_proposal.get(proposal)
            by_instance[free] = proposal
            by_proposal[proposal] = free
            free = old_instance
    return by_proposal


def evaluate_panel_instances(reference, proposals, *, source_sha256, native_shape,
                             iou_threshold=.5, ignore_fraction=.9):
    """Count full-panel TP/FP/FN without treating sparse patches as full labels.

    Reference must explicitly declare exhaustive full-frame labelling. Full
    instances are matched once each by maximum cardinality above the IoU gate.
    Only unmatched predictions almost entirely inside explicit ignore regions
    are excluded. An ignored partial panel is never a positive full instance.
    These geometric scores cannot authorise radiometry or field acceptance.
    """
    if (not isinstance(source_sha256, str) or len(source_sha256) != 64
            or any(c not in '0123456789abcdef' for c in source_sha256)
            or reference.get('source_sha256') != source_sha256):
        raise ValueError('reference/source SHA256 identity mismatch')
    if (len(native_shape) != 2 or any(type(v) is not int or v < 1 for v in native_shape)
            or reference.get('coordinate_space') != 'thermal_native'
            or (reference.get('native_height'), reference.get('native_width')) != tuple(native_shape)):
        raise ValueError('reference must match the native source grid')
    if (reference.get('schema') != 'pvdelta.native-panel-instances/v1'
            or reference.get('coverage') != 'exhaustive_full_frame'
            or not isinstance(reference.get('panels'), list)
            or reference.get('review_status') not in ('developer_draft', 'independently_reviewed')):
        raise ValueError('explicit exhaustive full-panel reference and review status required')
    if any(isinstance(v, bool) or not np.isfinite(v) or not 0 < v <= 1
           for v in (iou_threshold, ignore_fraction)):
        raise ValueError('IoU and ignore fractions must lie in (0, 1]')
    truth, instance_ids = _polygons(reference['panels'], native_shape, 'instance_id')
    predicted, proposal_ids = _polygons(proposals, native_shape, 'proposal_id')
    ignored, _ = _polygons(reference.get('ignore', []), native_shape, 'region_id')
    truth_union = np.zeros(native_shape, bool)
    for polygon in truth:
        mask = polygon_to_mask(polygon, native_shape)
        if (truth_union & mask).any():
            raise ValueError('full reference instances must not overlap')
        truth_union |= mask
    ignore = np.zeros(native_shape, bool)
    for polygon in ignored:
        ignore |= polygon_to_mask(polygon, native_shape)
    if (ignore & truth_union).any():
        raise ValueError('ignore regions must not overlap full reference instances')
    overlaps = np.array([[convex_iou(p, q) for q in truth] for p in predicted])
    overlaps = overlaps.reshape(len(predicted), len(truth))
    matched = _maximum_matching(overlaps, iou_threshold, proposal_ids, instance_ids)
    ignored_predictions, false_positives = [], []
    for index, polygon in enumerate(predicted):
        if index in matched:
            continue
        mask = polygon_to_mask(polygon, native_shape)
        # A polygon smaller than the native raster must not be silently ignored.
        fraction = int((mask & ignore).sum())/int(mask.sum()) if mask.any() else 0
        (ignored_predictions if fraction >= ignore_fraction else false_positives).append(proposal_ids[index])
    matched_instances = set(matched.values())
    false_negatives = [identifier for index, identifier in enumerate(instance_ids)
                       if index not in matched_instances]
    tp, fp, fn = len(matched), len(false_positives), len(false_negatives)
    return {'status':'full_instance_reference_scores',
        'reference_review_status':reference['review_status'], 'source_sha256':source_sha256,
        'scope':'frame-local full-panel geometry; not physical site inventory or anomaly accuracy',
        'iou_threshold':float(iou_threshold), 'ignore_fraction':float(ignore_fraction),
        'matching':'maximum_cardinality; IoU-ordered traversal, not maximum total IoU',
        'reference_instances':len(truth), 'predictions':len(predicted),
        'true_positives':tp, 'false_positives':fp, 'false_negatives':fn,
        'precision':tp/(tp+fp) if tp+fp else None,
        'recall':tp/(tp+fn) if tp+fn else None,
        'matches':[{'proposal_id':proposal_ids[i], 'instance_id':instance_ids[j],
                    'iou':float(overlaps[i, j])} for i, j in sorted(matched.items())],
        'false_positive_ids':false_positives, 'false_negative_ids':false_negatives,
        'ignored_proposal_ids':ignored_predictions, 'ignored_pixels':int(ignore.sum()),
        'measurement_authorised':False}
