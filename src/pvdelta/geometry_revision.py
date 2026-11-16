"""Immutable native-geometry versions with fresh, source-bound ROI analysis."""
from __future__ import annotations

import hashlib
import json
import platform
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import numpy as np

from pvdelta.config import DetectionConfig
from pvdelta.convex_geometry import convex_iou
from pvdelta.models import ThermalFrame
from pvdelta.pipeline import analyze_annotations
from pvdelta.regions import ModuleAnnotation, polygon_to_mask
from pvdelta.support_partition import partition_measurement_support

_ENGINE = ('geometry_revision.py', 'pipeline.py', 'detect.py', 'reference.py', 'regions.py',
           'support_partition.py', 'masking.py', 'normalize.py', 'conditions.py', 'quality.py',
           'models.py', 'config.py', 'report.py', 'convex_geometry.py')


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _digest(value):
    return hashlib.sha256(value).hexdigest()


def _array_sha(value):
    return _digest(value.tobytes(order='C'))


def _valid_hash(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def revision_engine_sources():
    from pvdelta.surface_constraints import surface_engine_binding

    return {name: _digest(Path(__file__).with_name(name).read_bytes())
            for name in (*_ENGINE,'surface_constraints.py')} | surface_engine_binding()


def _geometry(geometry, shape):
    if not isinstance(geometry, dict) or not isinstance(geometry.get('proposals'), list):
        raise TypeError('Geometry requires a proposals list')
    if len(geometry['proposals']) > 512:
        raise ValueError('At most 512 frame observations supported')
    proposals = deepcopy(geometry['proposals'])
    ids = set()
    for p in proposals:
        identifier = p.get('proposal_id') if isinstance(p, dict) else None
        if not isinstance(identifier, str) or not identifier.strip() or identifier in ids:
            raise ValueError('Unique nonempty proposal IDs required')
        ids.add(identifier)
        polygon = p.get('polygon_native_xy')
        if (not isinstance(polygon, list) or len(polygon) != 4
                or any(not isinstance(point, list) or len(point) != 2
                       or any(type(v) not in (int, float) for v in point) for point in polygon)):
            raise ValueError('Four finite numeric native corners required')
        q = np.asarray(polygon, float)
        if (not np.isfinite(q).all() or (q < 0).any()
                or (q[:, 0] > shape[1]).any() or (q[:, 1] > shape[0]).any()):
            raise ValueError('Native corners must stay inside the source grid')
        convex_iou(q, q)
        if not polygon_to_mask(polygon, shape).any():
            raise ValueError('Geometry has no native envelope pixels')
        if p.get('visibility', 'full') not in ('full', 'partial'):
            raise ValueError('Visibility must be full or partial')
        peer = p.get('peer_group')
        if peer is not None and (not isinstance(peer, str) or not peer.strip()):
            raise ValueError('Explicit peer group must be text or None')
    _canonical(proposals)
    return {'proposals': proposals}


def verify_geometry_version(version):
    if (not isinstance(version, dict) or version.get('schema') != 'pvdelta.native-geometry-version/v1'
            or version.get('measurement_authorised') is not False
            or version.get('main_scan_updated') is not False or version.get('database_written') is not False
            or version.get('independent_review_verified') is not False):
        raise ValueError('An unauthorised, immutable geometry-version document is required')
    payload = {k: v for k, v in version.items() if k != 'version_id'}
    if version.get('version_id') != 'sha256:' + _digest(_canonical(payload)):
        raise ValueError('Geometry version identity mismatch')
    return version['version_id']


def apply_geometry_patch(parent, patch):
    """Replace/remove/add explicit IDs without carrying stale edge/GPS evidence."""
    verify_geometry_version(parent)
    if (not isinstance(patch, dict) or patch.get('schema') != 'pvdelta.native-geometry-patch/v1'
            or patch.get('parent_version_id') != parent['version_id']
            or patch.get('source_sha256') != parent['source']['source_sha256']
            or patch.get('native_temperature_sha256') != parent['source']['native_temperature_sha256']
            or not isinstance(patch.get('operations'), list) or not patch['operations']
            or len(patch['operations']) > 512):
        raise ValueError('Patch must bind this exact parent, source and native grid')
    proposals = deepcopy(parent['geometry']['proposals'])
    seen = set()
    for op in patch['operations']:
        if (not isinstance(op, dict) or op.get('op') not in ('add', 'replace', 'remove')
                or not isinstance(op.get('proposal_id'), str) or not op['proposal_id'].strip()
                or op['proposal_id'] in seen or not isinstance(op.get('reason'), str) or not op['reason'].strip()
                or set(op)-{'op', 'proposal_id', 'reason', 'polygon_native_xy', 'visibility', 'peer_group'}):
            raise ValueError('Each explicit patch ID needs one operation and a reason')
        identifier = op['proposal_id']
        seen.add(identifier)
        existing = next((i for i, p in enumerate(proposals) if p['proposal_id'] == identifier), None)
        if (op['op'] == 'add') == (existing is not None):
            raise ValueError('Add requires a new ID; replace/remove require an existing ID')
        if op['op'] == 'remove':
            if set(op)-{'op', 'proposal_id', 'reason'}:
                raise ValueError('Remove only accepts an ID and a reason')
            proposals.pop(existing)
            continue
        old = proposals[existing] if existing is not None else {}
        changed = {'proposal_id': identifier, 'polygon_native_xy': op.get('polygon_native_xy'),
            'visibility': op.get('visibility', old.get('visibility', 'full')),
            'peer_group': op.get('peer_group', old.get('peer_group')),
            'proposal_origin': 'explicit_geometry_revision', 'revision_reason': op['reason'].strip(),
            'parent_proposal_sha256': _digest(_canonical(old)) if old else None,
            'semantic_panel_membership': 'unverified', 'measurement_authorised': False}
        if existing is None:
            proposals.append(changed)
        else:
            proposals[existing] = changed
    return _geometry({'proposals': proposals}, tuple(parent['source']['native_shape']))


def _provenance(value):
    if (not isinstance(value, dict) or value.get('kind') not in ('automatic', 'AI_developer_draft', 'user_edit')
            or not isinstance(value.get('actor'), str) or not value['actor'].strip()
            or not isinstance(value.get('note', ''), str)
            or type(value.get('reference_assisted', False)) is not bool):
        raise ValueError('Declared actor, supported provenance kind and note required')
    return {'kind': value['kind'], 'actor': value['actor'].strip(), 'note': value.get('note', '').strip(),
            'reference_assisted': value.get('reference_assisted', False), 'human_review_verified': False}


def analyze_geometry_version(frame, geometry, *, measurement_parameters, provenance,
                             config=None, margin_px=2, direction='hot', parent=None, patch=None):
    """Recompute the ENTIRE frame: neighbouring masks and explicit peers may change.

    Source bytes, original native values, valid mask, capture context, settings
    and engine are bound. Parent results are never copied into new analysis.
    Existing versions remain readable; an engine/settings change needs a new
    baseline so a paired geometry comparison does not mix algorithm changes.
    Authorisation and independent-review claims are never inferred from edits.
    """
    from pvdelta.surface_constraints import apply_configured_surface

    frame=apply_configured_surface(frame)
    if not isinstance(frame, ThermalFrame) or not _valid_hash(frame.source_hash):
        raise ValueError('ThermalFrame with a source SHA256 required')
    if not isinstance(measurement_parameters, dict) or not measurement_parameters:
        raise ValueError('Explicit decoder/measurement parameters required')
    source_path = Path(frame.source_path).expanduser().resolve()
    source_hash = _digest(source_path.read_bytes())
    if source_hash != frame.source_hash:
        raise ValueError('Original source file SHA256 mismatch')
    cfg = config or DetectionConfig()
    if not isinstance(cfg, DetectionConfig) or type(margin_px) is not int or margin_px < 0:
        raise ValueError('DetectionConfig and nonnegative integer margin required')
    if direction not in ('hot', 'cold', 'both'):
        raise ValueError('Direction must be hot, cold or both')
    declared_geometry = geometry
    geometry = _geometry(geometry, frame.temperature_c.shape)
    source = {'source_sha256': source_hash, 'source_path': str(source_path), 'frame_id': frame.frame_id,
        'native_temperature_sha256': _array_sha(frame.temperature_c), 'native_shape': list(frame.temperature_c.shape),
        'native_dtype': frame.temperature_c.dtype.str,
        'valid_mask_sha256': _array_sha(frame.valid_mask) if frame.valid_mask is not None else None,
        'capture_time': frame.capture_time.isoformat() if frame.capture_time else None,
        'camera_profile': asdict(frame.camera_profile) if frame.camera_profile else None,
        'raw_metadata': deepcopy(frame.raw_metadata), 'measurement_parameters': deepcopy(measurement_parameters)}
    if frame.raw_metadata.get('pvdelta_surface_constraint'):
        source['surface_constraint']=frame.raw_metadata['pvdelta_surface_constraint']
    settings = {'config': asdict(cfg), 'margin_px': margin_px, 'direction': direction,
                'support_policy': 'shared_envelope_pixels_excluded_from_all_ROIs'}
    engine = {'sources': revision_engine_sources(), 'numpy': np.__version__, 'python': platform.python_version()}
    declared = _provenance(provenance)
    source, settings, engine, declared = json.loads(_canonical([source, settings, engine, declared]))
    for key, expected in (('native_shape', source['native_shape']),
                          ('native_temperature_sha256', source['native_temperature_sha256']),
                          ('source_sha256', source['source_sha256'])):
        if key in declared_geometry and declared_geometry[key] != expected:
            raise ValueError('Declared geometry source/native identity differs from this frame')
    if parent is not None:
        verify_geometry_version(parent)
        if parent['source'] != source or parent['settings'] != settings or parent['engine'] != engine:
            raise ValueError('Parent source, native data, context, settings or engine changed; rebuild a baseline')
        if patch is None or apply_geometry_patch(parent, patch) != geometry:
            raise ValueError('Child geometry must equal its exact bound patch')
    elif patch is not None:
        raise ValueError('A patch requires its parent version')
    masks = [polygon_to_mask(p['polygon_native_xy'], frame.temperature_c.shape) for p in geometry['proposals']]
    supports = partition_measurement_support(masks, margin_px=margin_px)
    annotations = [ModuleAnnotation(p['proposal_id'], tuple(map(tuple, p['polygon_native_xy'])),
        visibility=p.get('visibility', 'full'), peer_group=p.get('peer_group')) for p in geometry['proposals']]
    reports = analyze_annotations(frame, annotations, config=cfg, direction=direction,
        module_masks={p['proposal_id']:s.interior for p, s in zip(geometry['proposals'], supports)})
    summaries = []
    for p, support in zip(geometry['proposals'], supports):
        effective = support.interior & np.isfinite(frame.temperature_c)
        if frame.valid_mask is not None:
            effective &= frame.valid_mask
        summaries.append({'proposal_id': p['proposal_id'], 'envelope_sha256': _array_sha(support.envelope),
            'interior_sha256': _array_sha(support.interior), 'effective_sha256': _array_sha(effective),
            'envelope_pixels': int(support.envelope.sum()), 'interior_pixels': int(support.interior.sum()),
            'ambiguous_pixels': int(support.excluded_ambiguous.sum()), 'effective_pixels': int(effective.sum())})
    if (_digest(source_path.read_bytes()) != source_hash or _array_sha(frame.temperature_c) != source['native_temperature_sha256']
            or (_array_sha(frame.valid_mask) if frame.valid_mask is not None else None) != source['valid_mask_sha256']):
        raise RuntimeError('Source/native values or valid mask changed during reanalysis')
    version = {'schema': 'pvdelta.native-geometry-version/v1', 'parent_version_id': parent['version_id'] if parent else None,
        'source': source, 'settings': settings, 'engine': engine, 'geometry': geometry, 'patch': deepcopy(patch),
        'provenance': declared, 'mask_summaries': summaries, 'module_reports': [asdict(r) for r in reports],
        'scope': 'Versioned provisional frame observations; no physical inventory, healthy-panel, fault or IEC acceptance',
        'measurement_authorised': False, 'database_written': False, 'main_scan_updated': False,
        'independent_review_verified': False}
    # Normalise tuples exactly as they will be saved, so readback compares deterministically.
    version = json.loads(_canonical(version))
    version['version_id'] = 'sha256:' + _digest(_canonical(version))
    return version


def geometry_version_impact(parent, child):
    verify_geometry_version(parent)
    verify_geometry_version(child)
    if (child['parent_version_id'] != parent['version_id'] or child['source'] != parent['source']
            or child['settings'] != parent['settings'] or child['engine'] != parent['engine']):
        raise ValueError('Comparable parent/child versions required')
    old = {p['proposal_id']: p for p in parent['geometry']['proposals']}
    new = {p['proposal_id']: p for p in child['geometry']['proposals']}
    shared = old.keys() & new.keys()
    geometry_changed = sorted(i for i in shared if old[i]['polygon_native_xy'] != new[i]['polygon_native_xy'])
    annotations_changed = sorted(i for i in shared if any(old[i].get(k, 'full' if k == 'visibility' else None)
        != new[i].get(k, 'full' if k == 'visibility' else None) for k in ('visibility', 'peer_group')))
    masks = [{s['proposal_id']: s for s in version['mask_summaries']} for version in (parent, child)]
    mask_changed = sorted(i for i in shared if masks[0][i]['effective_sha256'] != masks[1][i]['effective_sha256'])
    reports = [{r['module']['module_id']: r for r in version['module_reports']} for version in (parent, child)]
    return {'added_ids': sorted(new.keys()-old.keys()), 'removed_ids': sorted(old.keys()-new.keys()),
        'geometry_changed_ids': geometry_changed, 'annotation_changed_ids': annotations_changed,
        'effective_masks_changed_ids': mask_changed,
        'unchanged_geometry_with_changed_mask_ids': sorted(set(mask_changed)-set(geometry_changed)),
        'reports_changed_ids': sorted(i for i in shared if reports[0][i] != reports[1][i]),
        'parent_version_id': parent['version_id'], 'child_version_id': child['version_id'],
        'all_child_ROIs_reanalyzed': True, 'native_values_unchanged': True, 'main_scan_updated': False}


def save_geometry_version(version, output):
    """Content-addressed local JSON; never overwrite a prior result or touch DuckDB."""
    identifier = verify_geometry_version(version)
    directory = Path(output).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / ('geometry-version-' + identifier[7:] + '.json')
    encoded = _canonical(version)
    try:
        with path.open('xb') as stream:
            stream.write(encoded)
    except FileExistsError:
        if path.read_bytes() != encoded:
            raise ValueError('Existing immutable version differs; refusing overwrite')
    return path


def load_geometry_version(path):
    version = json.loads(Path(path).read_bytes())
    verify_geometry_version(version)
    return version
