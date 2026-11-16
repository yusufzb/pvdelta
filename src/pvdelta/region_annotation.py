"""Manual thermal review regions, separate from detector findings and panel masks."""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path

from pvdelta.panel_annotation import (
    _canonical,
    _hash,
    _sha,
    load_annotation_pack,
    validate_annotation_export,
)
from pvdelta.region_polygon import simple_polygon_mask


def load_region_pack(path):
    """Reuse unresolved native images; bind optional unregistered RGB context too."""
    from PIL import Image

    path = Path(path).expanduser().resolve()
    pack = load_annotation_pack(path)
    document = json.loads(path.read_bytes())
    for frame, source in zip(pack['frames'], document['frames']):
        if 'rgb_context_image' not in source:
            continue
        rgb = (path.parent / source['rgb_context_image']).resolve()
        if not rgb.is_relative_to(path.parent):
            raise ValueError('RGB context must stay inside its reference pack')
        if (not _hash(source.get('rgb_source_sha256'))
                or not isinstance(source.get('rgb_source_file'), str)):
            raise ValueError('RGB source filename and hash are required')
        raw = rgb.read_bytes()
        with Image.open(io.BytesIO(raw)) as image:
            if image.format not in ('JPEG', 'PNG'):
                raise ValueError('RGB context must be JPEG or PNG')
            mime = 'image/jpeg' if image.format == 'JPEG' else 'image/png'
            image.load()
        frame.update(rgb_source_file=source['rgb_source_file'],
                     rgb_source_sha256=source['rgb_source_sha256'],
                     rgb_display_sha256=_sha(raw),
                     rgb_data_url=f'data:{mime};base64,' + base64.b64encode(raw).decode())
    identity = {k: v for k, v in pack.items() if k != 'frames'}
    identity['frames'] = [{k: v for k, v in f.items() if not k.endswith('data_url')}
                          for f in pack['frames']]
    pack.update(schema='pvdelta.thermal-region-annotation-input/v1',
                pack_token=_sha(_canonical(identity)))
    return pack


def region_editor_html(pack):
    template = Path(__file__).with_name('region_annotation_editor.html').read_text('utf-8')
    return template.replace('__PVDELTA_PACK__', json.dumps(pack, allow_nan=False)
                            .replace('<', '\\u003c'))


def validate_region_export(pack, content):
    """Validate simple native polygons, including concave regions; grant no truth status.

    Regions mark thermal patterns requiring review, not a fault diagnosis. Unknown
    areas remain explicit ignores. Exhaustive completion is only a declaration.
    """
    document = json.loads(content)
    if (not isinstance(document, dict)
            or document.get('schema') != 'pvdelta.thermal-region-work/v1'
            or not isinstance(document.get('frames'), list)):
        raise ValueError('A thermal-region editor export is required')
    adapted = dict(document, schema='pvdelta.panel-annotation-work/v1', frames=[])
    for work in document['frames']:
        if not isinstance(work, dict) or not isinstance(work.get('regions'), list):
            raise TypeError('Explicit thermal regions list required')
        # Reuse identity/completion validation without applying convex panel geometry.
        # Nonempty regions do not need the empty-frame reason required for zero panels.
        reason = work.get('empty_reason')
        if isinstance(reason, str) and work['regions'] and not reason.strip():
            reason = 'Nonempty thermal review regions'
        adapted['frames'].append(dict(work, panels=[], ignore=[], empty_reason=reason))
    checked = validate_annotation_export(pack, _canonical(adapted))
    by_source = {w['source_sha256']: w for w in document['frames']}
    for ref, source in zip(checked['frames'], pack['frames']):
        import numpy as np

        complete = ref['ready_to_score']
        work = by_source[ref['source_sha256']]
        if not isinstance(work.get('ignore'), list):
            raise TypeError('Explicit uncertain regions list required')
        shapes = {}
        union = np.zeros((ref['native_height'], ref['native_width']), bool)
        for name in ('regions', 'ignore'):
            shapes[name], identifiers = [], set()
            for item in work[name]:
                if not isinstance(item, dict):
                    raise TypeError('Thermal regions must be objects')
                identifier = item.get('region_id')
                if not isinstance(identifier, str) or not identifier or identifier in identifiers:
                    raise ValueError('Unique nonempty region IDs required')
                identifiers.add(identifier)
                polygon = item.get('polygon_native_xy')
                mask = simple_polygon_mask(polygon, union.shape)
                if (union & mask).any():
                    raise ValueError('Reference and uncertain regions must not overlap')
                union |= mask
                shapes[name].append({'region_id': identifier, 'polygon_native_xy': polygon})
        ref.pop('panels', None)
        ref.pop('draft_panels', None)
        regions = shapes['regions']
        ref.update(ignore=shapes['ignore'], empty_reason=work['empty_reason'].strip())
        ref.update({k: v for k, v in source.items() if k.startswith('rgb_') and k != 'rgb_data_url'})
        ref.update(schema='pvdelta.native-thermal-region-reference/v1',
                   regions=regions if complete else None,
                   annotation_method='manual_native_thermal_region_canvas',
                   scope='Visible native polygon review regions; not confirmed faults',
                   coverage='declared_full_frame' if complete else 'unresolved',
                   ready_to_score=False, review_complete=complete)
        if not complete:
            ref['draft_regions'] = regions
    checked.update(schema='pvdelta.thermal-region-drafts/v1',
                   scoring_protocol_frozen=False)
    return checked


def save_region_export(pack_path, content, output):
    """Rebind all images at save time and retain each content-addressed version."""
    checked = validate_region_export(load_region_pack(pack_path), content)
    encoded = _canonical(checked)
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    path = output / ('thermal-region-references-' + _sha(encoded) + '.json')
    try:
        with path.open('xb') as stream:
            stream.write(encoded)
    except FileExistsError:
        if path.read_bytes() != encoded:
            raise ValueError('Existing reference version differs; refusing to overwrite')
    return path, checked
