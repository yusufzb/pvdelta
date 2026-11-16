"""Read-only, source-bound native boundary comparisons for local review."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from pvdelta.decode import RJpeg
from pvdelta.masking import summarize_temperature
from pvdelta.native_windows import native_window_proposals
from pvdelta.regions import polygon_to_mask
from pvdelta.support_partition import partition_measurement_support

_ENGINE_FILES = (
    'native_review.py', 'native_windows.py', 'native_lines.py', 'native_boundaries.py',
    'edge_support.py', 'convex_geometry.py', 'junction_registration.py',
    'support_partition.py', 'regions.py', 'masking.py', 'reference.py', 'boundaries.py', 'decode.py',
)


@dataclass(frozen=True)
class NativeBoundaryPreview:
    report: dict
    baseline_image: object
    proposed_image: object
    support_image: object


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def boundary_engine_sources():
    from pvdelta.surface_constraints import surface_engine_binding

    return {name:_sha(Path(__file__).with_name(name))
            for name in (*_ENGINE_FILES,'surface_constraints.py')} | surface_engine_binding()


def boundary_preview_token(path, sdk_library, *, expected_sha256=None):
    """Invalidate a preview on source, selected SDK path or engine-file changes."""
    source = Path(path).expanduser().resolve()
    source_hash = _sha(source)
    if expected_sha256 is not None and source_hash != expected_sha256:
        raise ValueError('Thermal source SHA256 differs from the indexed record; re-index or select the original file')
    code = boundary_engine_sources()
    configured_sdk = sdk_library or os.environ.get('PVDELTA_DIRP_LIB')
    sdk = Path(configured_sdk).expanduser().resolve() if configured_sdk else None
    sdk_state = []
    if sdk is not None:
        files = [sdk] if sdk.is_file() else sorted(sdk.glob('*.dll')) + sorted(sdk.glob('*.so*'))
        sdk_state = [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in files if p.is_file()]
    identity = {'path':str(source), 'source_sha256':source_hash,
                'sdk_library':str(sdk), 'sdk_files':sdk_state, 'engine_files':code,
                'margin_px':2, 'raw_supplement':True}
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def indexed_native_frames(database):
    """List source identities without creating a store or running migrations."""
    from pvdelta.store import connect_store

    connection = connect_store(Path(database), read_only=True)
    try:
        cursor = connection.execute('''SELECT c.capture_id, c.source_path AS indexed_source_path,
            c.source_hash, c.run_id, r.source_root,
            CAST(c.capture_time_utc AS VARCHAR) AS captured_at
            FROM captures c LEFT JOIN inspection_runs r ON r.run_id=c.run_id
            WHERE c.image_kind='thermal'
            ORDER BY c.capture_time_utc NULLS LAST, c.source_path, c.capture_id''')
        columns = [c[0] for c in cursor.description]
        rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
        for row in rows:
            indexed = Path(row['indexed_source_path'])
            row['source_name'] = indexed.name
            row['source_path'] = str(indexed) if indexed.is_absolute() else None
            if not indexed.is_absolute() and row['source_root']:
                root = Path(row['source_root']).expanduser()
                resolved = (root/indexed).resolve()
                if root.is_absolute() and resolved.is_relative_to(root.resolve()):
                    row['source_path'] = str(resolved)
        return rows
    finally:
        connection.close()


def native_boundary_preview(path, sdk_library, *, expected_sha256=None):
    """Compare both modes on one genuine SDK grid; never writes findings or masks."""
    from PIL import Image, ImageDraw

    source = Path(path).expanduser().resolve()
    token = boundary_preview_token(source, sdk_library, expected_sha256=expected_sha256)
    source_hash = _sha(source)
    with RJpeg(source, sdk_library) as image:
        values = image.temperature()
        display = image.processed_rgb()
        parameters = asdict(image.measurement_params)
    if (values.ndim != 2 or not np.isfinite(values).all() or display.dtype != np.uint8
            or display.shape != (*values.shape, 3)):
        raise ValueError('SDK render and finite temperature grid must have identical native dimensions')
    digest = hashlib.sha256(values.tobytes()).hexdigest()
    baseline = native_window_proposals(values)
    proposed = native_window_proposals(values, raw_supplement=True)
    shape = values.shape

    def partition(result):
        return partition_measurement_support([
            polygon_to_mask(p['polygon_native_xy'], shape) for p in result['proposals']], margin_px=2)

    before, supports = partition(baseline), partition(proposed)
    old_union = np.zeros(shape, bool)
    union = np.zeros(shape, bool)
    overlap = np.zeros(shape, np.int32)
    for s in before:
        old_union |= s.interior
    table = []
    support_pixels = display.copy()
    for proposal, support in zip(proposed['proposals'], supports):
        if (union & support.interior).any():
            raise ValueError('Preview measurement supports overlap')
        union |= support.interior
        overlap += support.envelope
        masked = support.masked_temperature(values)
        np.testing.assert_array_equal(masked[support.interior], values[support.interior])
        if not np.isnan(masked[~support.interior]).all():
            raise ValueError('Excluded pixels must be NaN')
        # Visibility aid only; the untouched grid supplies every statistic.
        support_pixels[support.interior] = (
            .65*display[support.interior] + .35*np.array([0, 220, 220])).astype(np.uint8)
        table.append(dict(proposal_id=proposal['proposal_id'],
            origin=proposal['proposal_origin'], **asdict(summarize_temperature(values, support))))
    shared = overlap > 1
    support_pixels[shared] = [255, 0, 255]
    lost = old_union & ~union
    exact_lost = [p['proposal_id'] for p in baseline['proposals'] if not any(
        p['polygon_native_xy']==q['polygon_native_xy'] for q in proposed['proposals'])]

    def outlines(result):
        picture = Image.fromarray(display.copy())
        draw = ImageDraw.Draw(picture)
        for p in result['proposals']:
            q = [tuple(v) for v in p['polygon_native_xy']]
            colour = '#ffad33' if p.get('proposal_origin')=='raw_structure_supplement' else '#00ffff'
            draw.line(q+[q[0]], fill=colour, width=1)
        return picture

    if (digest != hashlib.sha256(values.tobytes()).hexdigest()
            or token != boundary_preview_token(source, sdk_library, expected_sha256=source_hash)):
        raise ValueError('Source or engine changed during preview; run it again')
    report = {'schema':'pvdelta.native-boundary-preview/v1', 'preview_token':token,
        'source_name':source.name, 'source_sha256':source_hash,
        'native_temperature_sha256':digest, 'native_shape':list(shape),
        'engine_source_sha256':boundary_engine_sources(),
        'radiometric_parameters':parameters, 'margin_px':2,
        'baseline_proposals':len(baseline['proposals']), 'proposals':len(proposed['proposals']),
        'raw_additions':sum(p.get('proposal_origin')=='raw_structure_supplement' for p in proposed['proposals']),
        'lost_baseline_proposal_ids':exact_lost,
        'shared_envelope_pixels':int(shared.sum()), 'unique_interior_pixels':int(union.sum()),
        'empty_interiors':sum(not s.interior.any() for s in supports),
        'old_interior_pixels_excluded':int(lost.sum()),
        'old_pixels_lost_outside_shared_envelopes':int((lost & ~shared).sum()),
        'native_values_unchanged':True, 'descriptive_statistics':table,
        'baseline':baseline, 'proposed':proposed,
        'semantic_panel_membership':'unverified', 'measurement_authorised':False,
        'database_written':False, 'anomaly_rerun':False,
        'scope':'Single-frame geometry review; not unique physical panels, anomaly accuracy or IEC acceptance'}
    return NativeBoundaryPreview(report, outlines(baseline), outlines(proposed), Image.fromarray(support_pixels))
