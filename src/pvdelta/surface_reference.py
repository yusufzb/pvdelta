"""User-declared panel surfaces, distinct from individual modules and hotspots."""
from __future__ import annotations

import json

import numpy as np

from pvdelta.panel_annotation import _canonical, validate_annotation_export
from pvdelta.region_polygon import simple_polygon_mask
from pvdelta.regions import polygon_to_mask


def import_surface_work(pack, content, *, confirmed_meaning):
    if confirmed_meaning != 'panel_surface_and_non_panel':
        raise ValueError('Explicit user confirmation of surface-label meaning required')
    document = json.loads(content)
    if (document.get('schema') != 'pvdelta.thermal-region-work/v1'
            or not isinstance(document.get('frames'), list)):
        raise ValueError('Original native editor export required')
    adapted = dict(document, schema='pvdelta.panel-annotation-work/v1', frames=[])
    for work in document['frames']:
        if not isinstance(work.get('regions'), list) or not isinstance(work.get('ignore'), list):
            raise TypeError('Explicit panel and non-panel region lists required')
        reason = work.get('empty_reason')
        if isinstance(reason, str) and work['regions'] and not reason.strip():
            reason = 'Panel surfaces supplied'
        adapted['frames'].append(dict(work, panels=[], ignore=[], empty_reason=reason))
    checked = validate_annotation_export(pack, _canonical(adapted))
    supplied = {w['source_sha256']: w for w in document['frames']}
    frames, masks = [], {}
    for ref in checked['frames']:
        work = supplied[ref['source_sha256']]
        shape = (ref['native_height'], ref['native_width'])
        layers, polygons = {}, {}
        for name in ('regions', 'ignore'):
            union, seen = np.zeros(shape, bool), set()
            polygons[name] = []
            for item in work[name]:
                identifier = item.get('region_id')
                if not isinstance(identifier, str) or not identifier or identifier in seen:
                    raise ValueError('Unique region IDs required')
                seen.add(identifier)
                polygon = item.get('polygon_native_xy')
                union |= simple_polygon_mask(polygon, shape)
                polygons[name].append({'region_id': identifier, 'polygon_native_xy': polygon})
            layers[name] = union
        # Explicit four-valued support: unmapped pixels are never negative labels.
        labels = layers['regions'].astype(np.uint8) + 2*layers['ignore'].astype(np.uint8)
        masks[ref['source_sha256']] = labels
        frames.append({k: ref[k] for k in ('case_id', 'source_file', 'source_sha256',
                        'native_temperature_sha256', 'display_png_sha256',
                        'native_height', 'native_width', 'reviewer', 'note')} |
                      {'review_complete': work['exhaustive'], 'empty_reason': work['empty_reason'],
                       'panel_polygons': polygons['regions'], 'non_panel_polygons': polygons['ignore'],
                       'pixel_counts': {str(i): int((labels == i).sum()) for i in range(4)}})
    return {'schema': 'pvdelta.manual-surface-reference/v1', 'pack_token': pack['pack_token'],
            'meaning_confirmed_by_user': True, 'classes': {'0': 'unlabelled', '1': 'panel',
                '2': 'non_panel', '3': 'conflicting_labels'}, 'frames': frames,
            'independent_review_verified': False, 'measurement_authorised': False}, masks


def audit_surface_predictions(reference, labels, predictions):
    if labels.shape != (reference['native_height'], reference['native_width']):
        raise ValueError('Native mask shape mismatch')
    if predictions['source_sha256'] != reference['source_sha256']:
        raise ValueError('Source identity mismatch')
    if predictions.get('native_shape') != list(labels.shape):
        raise ValueError('Prediction grid mismatch')
    if not reference['review_complete']:
        return {'case_id': reference['case_id'], 'status': 'pending', 'findings': None, 'modules': None}

    def counts(mask):
        values = labels[mask]
        if not len(values):
            raise ValueError('Empty prediction support')
        c = {name: int((values == i).sum()) for i, name in enumerate(
            ('unlabelled_pixels', 'panel_pixels', 'non_panel_pixels', 'conflict_pixels'))}
        state = ('panel_only' if c['panel_pixels'] == len(values) else
                 'non_panel_only' if c['non_panel_pixels'] == len(values) else
                 'mixed' if c['panel_pixels'] and c['non_panel_pixels'] else 'unresolved')
        return dict(c, pixels=len(values), support=state)

    modules, findings = [], []
    covered = np.zeros(labels.shape, bool)
    for module in predictions['proposals']:
        mask = polygon_to_mask(module['polygon_native_xy'], labels.shape)
        covered |= mask
        modules.append({'proposal_id': module['proposal_id'], **counts(mask)})
    for finding in predictions['findings']:
        pixels = finding.get('region_pixels_yx')
        if (not isinstance(pixels, list) or not pixels
                or any(not isinstance(p, list) or len(p) != 2
                       or any(type(v) is not int for v in p)
                       or not 0 <= p[0] < labels.shape[0] or not 0 <= p[1] < labels.shape[1]
                       for p in pixels) or len({tuple(p) for p in pixels}) != len(pixels)):
            raise ValueError('Unique native candidate pixel coordinates required')
        mask = np.zeros(labels.shape, bool)
        yx = np.asarray(pixels)
        mask[yx[:, 0], yx[:, 1]] = True
        findings.append({'finding_id': finding['finding_id'], **counts(mask)})
    panel = labels == 1
    return {'case_id': reference['case_id'], 'status': 'user_declared_surface_audit',
            'modules': modules, 'findings': findings, 'panel_pixels': int(panel.sum()),
            'panel_pixels_in_proposed_envelopes': int((panel & covered).sum()),
            'panel_pixels_without_proposed_envelopes': int((panel & ~covered).sum()),
            'scope': 'Frame-local surface support; not hotspot accuracy or unique module coverage'}
