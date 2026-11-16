"""Prediction-free native panel annotation and source-bound draft validation."""
from __future__ import annotations

import base64
import hashlib
import io
import json
from pathlib import Path

from pvdelta.instance_benchmark import evaluate_panel_instances


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _hash(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def load_annotation_pack(path):
    """Bind an unmarked PNG to an unresolved source/native template, never a detector."""
    from PIL import Image

    path = Path(path).expanduser().resolve()
    content = path.read_bytes()
    document = json.loads(content)
    if (not isinstance(document, dict)
            or document.get('schema') != 'pvdelta.pending-panel-reference-pack/v1'
            or not isinstance(document.get('frames'), list) or not document['frames']):
        raise ValueError('An unresolved native panel reference pack is required')
    frames, seen = [], set()
    for frame in document['frames']:
        if not isinstance(frame, dict):
            raise TypeError('Pack frames must be objects')
        if (frame.get('schema') != 'pvdelta.native-panel-instances/v1'
                or frame.get('coordinate_space') != 'thermal_native'
                or any(not _hash(frame.get(k)) for k in ('source_sha256', 'native_temperature_sha256'))
                or frame.get('source_sha256') in seen
                or any(type(frame.get(k)) is not int or frame[k] < 1 for k in ('native_height', 'native_width'))
                or not isinstance(frame.get('case_id'), str) or not frame['case_id']
                or frame.get('panels') is not None or frame.get('review_status') != 'unreviewed'
                or frame.get('coverage') != 'unresolved'):
            raise ValueError('Unique unresolved sources with native hashes and dimensions are required')
        seen.add(frame['source_sha256'])
        if not isinstance(frame.get('image'), str):
            raise TypeError('A local unmarked native PNG is required')
        image_path = (path.parent / frame['image']).resolve()
        if not image_path.is_relative_to(path.parent):
            raise ValueError('Native PNG must stay inside its reference pack')
        raw = image_path.read_bytes()
        with Image.open(io.BytesIO(raw)) as image:
            if image.format != 'PNG' or image.size != (frame['native_width'], frame['native_height']):
                raise ValueError('Unmarked PNG must match the exact native grid')
            image.load()
        frames.append({'case_id': frame['case_id'], 'source_file': frame.get('source_file'),
            'source_sha256': frame['source_sha256'], 'native_temperature_sha256': frame['native_temperature_sha256'],
            'native_height': frame['native_height'], 'native_width': frame['native_width'],
            'display_png_sha256': _sha(raw), 'image_data_url': 'data:image/png;base64,' + base64.b64encode(raw).decode()})
    identity = {'manifest_sha256': _sha(content), 'frames': [{k:v for k,v in f.items() if k != 'image_data_url'}
                                                         for f in frames]}
    return {'schema': 'pvdelta.panel-annotation-input/v1', 'pack_token': _sha(_canonical(identity)),
                'manifest_sha256': _sha(content), 'frames': frames}


def annotation_editor_html(pack):
    """Single local HTML editor; display zoom never changes exported native coordinates."""
    template = Path(__file__).with_name('panel_annotation_editor.html').read_text(encoding='utf-8')
    return template.replace('__PVDELTA_PACK__', json.dumps(pack, allow_nan=False).replace('<', '\\u003c'))


def validate_annotation_export(pack, content):
    """Validate every shape before any save; completed frames remain developer drafts.

    Completion is a reviewer's declaration, not an independent ground-truth audit.
    Partial frames keep panels:null so sparse work cannot masquerade as negatives.
    """
    try:
        document = json.loads(content)
        _canonical(document)
    except (ValueError, TypeError, UnicodeError) as exc:
        raise ValueError('Annotation export must be finite JSON') from exc
    if (not isinstance(document, dict) or document.get('schema') != 'pvdelta.panel-annotation-work/v1'
            or document.get('pack_token') != pack['pack_token']
            or not isinstance(document.get('frames'), list)
            or len(document['frames']) != len(pack['frames'])):
        raise ValueError('Annotation export must match this exact native pack')
    supplied = {}
    for work in document['frames']:
        if (not isinstance(work, dict) or not _hash(work.get('source_sha256'))
                or work['source_sha256'] in supplied):
            raise ValueError('Annotation frames must have unique source identities')
        supplied[work.get('source_sha256')] = work
    references = []
    for frame in pack['frames']:
        work = supplied.get(frame['source_sha256'])
        if (work is None or any(work.get(k) != frame[k] for k in
                ('native_temperature_sha256', 'native_height', 'native_width', 'display_png_sha256'))
                or any(type(work.get(k)) is not int for k in ('native_height', 'native_width'))):
            raise ValueError('Source, native grid or displayed image identity changed')
        if (type(work.get('exhaustive')) is not bool or not isinstance(work.get('panels'), list)
                or not isinstance(work.get('ignore'), list)
                or any(not isinstance(work.get(k), str) for k in ('reviewer', 'note', 'empty_reason'))):
            raise ValueError('Explicit completion, panel/ignore lists and reviewer text are required')
        # Whitelist shape fields; imported review/measurement authorisation claims are discarded.
        shapes = {}
        for name, id_key in (('panels', 'instance_id'), ('ignore', 'region_id')):
            shapes[name] = []
            for shape in work[name]:
                if not isinstance(shape, dict):
                    raise TypeError('Annotation shapes must be objects')
                polygon = shape.get('polygon_native_xy')
                if (not isinstance(polygon, list) or len(polygon) != 4
                        or any(not isinstance(p, list) or len(p) != 2
                               or any(type(v) not in (int, float) for v in p) for p in polygon)):
                    raise ValueError('Four ordered finite native corners are required')
                shapes[name].append({id_key:shape.get(id_key), 'polygon_native_xy':polygon})
        ref = {'schema': 'pvdelta.native-panel-instances/v1', 'coordinate_space': 'thermal_native',
            'source_sha256': frame['source_sha256'], 'native_temperature_sha256': frame['native_temperature_sha256'],
            'source_file': frame['source_file'], 'case_id': frame['case_id'],
            'native_height': frame['native_height'], 'native_width': frame['native_width'],
            'display_png_sha256': frame['display_png_sha256'], 'panels': shapes['panels'], 'ignore': shapes['ignore'],
            'coverage': 'exhaustive_full_frame', 'review_status': 'developer_draft'}
        # Reuse the production scorer's bounds, convexity, IDs and raster overlap checks even for partial drafts.
        evaluate_panel_instances(ref, [], source_sha256=frame['source_sha256'],
                                 native_shape=(frame['native_height'], frame['native_width']))
        complete = work['exhaustive']
        if complete and not work['reviewer'].strip():
            raise ValueError('Completed frames require the actual reviewer name')
        if complete and not shapes['panels'] and not work['empty_reason'].strip():
            raise ValueError('A zero-panel complete frame requires an explicit empty-frame reason')
        ref.update(reviewer=work['reviewer'].strip() or None, note=work['note'].strip(),
            empty_reason=work['empty_reason'].strip(), annotation_method='manual_native_canvas',
            ready_to_score=complete, independent_review_verified=False, measurement_authorised=False)
        if not complete:
            ref.update(panels=None, draft_panels=shapes['panels'], coverage='unresolved', review_status='unreviewed')
        references.append(ref)
    return {'schema': 'pvdelta.panel-annotation-drafts/v1', 'pack_token': pack['pack_token'], 'frames': references,
        'completed_frames': sum(f['ready_to_score'] for f in references),
        'independent_review_verified': False, 'database_written': False, 'measurement_authorised': False}


def save_annotation_export(pack_path, content, output):
    """Re-read the pack at save time; save a version without overwriting old work or touching a DB."""
    pack = load_annotation_pack(pack_path)
    validated = validate_annotation_export(pack, content)
    encoded = _canonical(validated)
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    path = output / ('panel-references-' + _sha(encoded) + '.json')
    try:
        with path.open('xb') as stream:
            stream.write(encoded)
    except FileExistsError:
        if path.read_bytes() != encoded:
            raise ValueError('Existing reference version differs; refusing to overwrite')
    return path, validated
