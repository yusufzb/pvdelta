"""Sparse native semantic reference checks; never infers full-mask accuracy."""
from __future__ import annotations

import numpy as np

from pvdelta.regions import polygon_to_mask


def evaluate_labeled_support(reference, masks, *, source_sha256, native_shape):
    """Evaluate coverage/intrusion only on explicitly labelled source pixels.

    Most source pixels may remain unlabelled. Therefore neither module recall
    nor whole-mask precision is reported. A consistent labelled subset cannot
    promote an unreviewed proposal to a verified panel mask.
    """
    if (not isinstance(source_sha256, str) or len(source_sha256) != 64
            or any(c not in '0123456789abcdef' for c in source_sha256)
            or reference.get('source_sha256') != source_sha256):
        raise ValueError('reference/source SHA256 identity mismatch')
    expected_shape = (reference.get('native_height'), reference.get('native_width'))
    if (reference.get('coordinate_space') != 'thermal_native' or tuple(native_shape) != expected_shape
            or len(native_shape) != 2 or any(type(n) is not int or n < 1 for n in native_shape)):
        raise ValueError('reference must match the native source grid')
    labels = {label: np.zeros(native_shape, bool) for label in ('panel', 'non_panel', 'ignore')}
    ids = set()
    region_masks = []
    for region in reference.get('regions', []):
        label, region_id = region.get('label'), region.get('region_id')
        if label not in labels or not isinstance(region_id, str) or not region_id or region_id in ids:
            raise ValueError('semantic regions require known labels and unique nonempty IDs')
        ids.add(region_id)
        polygon = np.asarray(region['polygon_native_xy'], float)
        if (polygon.ndim != 2 or polygon.shape[1] != 2 or not np.isfinite(polygon).all()
                or (polygon < 0).any() or (polygon[:, 0] > native_shape[1]).any()
                or (polygon[:, 1] > native_shape[0]).any()):
            raise ValueError('reference vertices must stay inside the native source')
        mask = polygon_to_mask(region['polygon_native_xy'], native_shape)
        labels[label] |= mask
        region_masks.append((region_id, label, mask))
    if np.any(labels['panel'] & labels['non_panel']):
        raise ValueError('panel and non-panel reference labels overlap')
    panel = labels['panel'] & ~labels['ignore']
    ground = labels['non_panel'] & ~labels['ignore']
    known = panel | ground
    union = np.zeros(native_shape, bool)
    proposals = []
    for mask in masks:
        if not isinstance(mask, np.ndarray) or mask.dtype != np.bool_ or mask.shape != tuple(native_shape):
            raise ValueError('proposal masks must be boolean native-grid arrays')
        union |= mask
        positive, negative = int((mask & panel).sum()), int((mask & ground).sum())
        proposals.append({'labeled_panel_pixels': positive, 'labeled_non_panel_pixels': negative,
            'unlabeled_pixels': int((mask & ~known).sum()),
            'status': 'contradicts_non_panel_label' if negative else (
                'consistent_on_labeled_subset' if positive else 'unlabeled')})
    panel_hit, ground_hit = int((union & panel).sum()), int((union & ground).sum())
    regions = []
    for region_id, label, mask in region_masks:
        selected = mask & ~labels['ignore']
        total, covered = int(selected.sum()), int((union & selected).sum())
        regions.append({'region_id': region_id, 'label': label, 'labeled_pixels': total,
                        'covered_pixels': covered, 'coverage': covered/total if total else None})
    return {'status': 'sparse_reference_only', 'reference_review_status': reference.get('review_status', 'unreviewed'),
        'labeled_panel_pixels': int(panel.sum()), 'covered_panel_pixels': panel_hit,
        'labeled_non_panel_pixels': int(ground.sum()), 'covered_non_panel_pixels': ground_hit,
        'panel_patch_coverage': panel_hit/int(panel.sum()) if panel.any() else None,
        'non_panel_patch_intrusion': ground_hit/int(ground.sum()) if ground.any() else None,
        'predicted_pixels': int(union.sum()), 'unlabeled_predicted_pixels': int((union & ~known).sum()),
        'proposals': proposals, 'regions': regions, 'measurement_authorised': False}
