"""Source-bound before/after scoring against explicit full-panel references."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from pvdelta.boundary_review_store import boundary_result_identity
from pvdelta.instance_benchmark import evaluate_panel_instances


@dataclass(frozen=True)
class BoundaryReferenceComparison:
    report: dict
    baseline_image: object
    proposed_image: object


def validation_engine_sources():
    return {name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ('boundary_validation.py','instance_benchmark.py','convex_geometry.py',
                         'regions.py','boundary_review_store.py')}


def reference_validation_token(report, reference_bytes):
    identity = {'result_id':boundary_result_identity(report),
                'reference_file_sha256':hashlib.sha256(reference_bytes).hexdigest(),
                'scorer_source_sha256':validation_engine_sources()}
    return hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()


def _load_reference(content, source_sha256):
    try:
        document = json.loads(content)
    except (ValueError, UnicodeError) as exc:
        raise ValueError('Reference must be a valid JSON document') from exc
    if not isinstance(document, dict):
        raise TypeError('Reference must be an object or an object containing frames')
    if 'frames' in document:
        frames = document['frames']
        if not isinstance(frames,list) or any(not isinstance(f,dict) for f in frames):
            raise ValueError('Reference frames must be a list of objects')
        matches = [f for f in frames if f.get('source_sha256')==source_sha256]
        if len(matches)!=1:
            raise ValueError('Reference bundle requires exactly one frame with this source SHA256')
        reference = matches[0]
    else:
        reference = document
    if any(type(reference.get(k)) is not int for k in ('native_height','native_width')):
        raise ValueError('Reference requires integer native grid dimensions')
    try:
        canonical = json.dumps(reference,sort_keys=True,separators=(',',':'),allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError('Reference must contain finite JSON values') from exc
    return reference, hashlib.sha256(canonical.encode()).hexdigest()


def _overlay(picture, reference, proposals, score):
    from PIL import ImageDraw

    marked = picture.copy().convert('RGB')
    draw = ImageDraw.Draw(marked)
    matched = {m['proposal_id'] for m in score['matches']}
    ignored = set(score['ignored_proposal_ids'])

    def outline(item, colour, label, dashed=False):
        points = [tuple(p) for p in item['polygon_native_xy']]
        if dashed:
            for start,end in zip(points, points[1:]+points[:1]):
                length = ((end[0]-start[0])**2+(end[1]-start[1])**2)**.5
                if length==0:
                    continue
                for distance in range(0,int(length)+1,10):
                    fractions = (distance/length,min(distance+5,length)/length)
                    segment = [tuple(a+(b-a)*t for a,b in zip(start,end)) for t in fractions]
                    draw.line(segment,fill=colour,width=2)
        else:
            draw.line(points+points[:1],fill=colour,width=2)
        # Instance identifiers may contain Unicode; keep labels ASCII for PIL's default font.
        text = label.encode('ascii','replace').decode('ascii')
        position = (max(0,min(marked.width-1,int(points[0][0]))),
                    max(0,min(marked.height-12,int(points[0][1]))))
        draw.text(position,text,fill=colour,stroke_width=1,stroke_fill='black')

    for p in proposals:
        pid = p['proposal_id']
        kind,colour = ('TP','#35e88b') if pid in matched else (
            ('IGN','#a6a6a6') if pid in ignored else ('FP','#ffad33'))
        outline(p,colour,f'{kind}:{pid}')
    missed = set(score['false_negative_ids'])
    for p in reference['panels']:
        if p['instance_id'] in missed:
            outline(p,'#ff5555',f"FN:{p['instance_id']}",dashed=True)
    return marked


def compare_boundary_reference(preview, reference_bytes):
    """Score existing native geometry; never decode, change labels, or save decisions.

    The declared reference review status is metadata, not proof of independence.
    Fixed IoU .5/.75 are geometry diagnostics, not IEC acceptance thresholds.
    """
    report = preview.report
    token = reference_validation_token(report,reference_bytes)
    reference, reference_hash = _load_reference(reference_bytes,report['source_sha256'])
    declared_native_hash = reference.get('native_temperature_sha256')
    if 'native_temperature_sha256' in reference:
        if (not isinstance(declared_native_hash, str) or len(declared_native_hash) != 64
                or any(c not in '0123456789abcdef' for c in declared_native_hash)):
            raise ValueError('Reference native temperature SHA256 must be a lowercase hex digest')
        if declared_native_hash != report['native_temperature_sha256']:
            raise ValueError('Reference/native temperature SHA256 mismatch; check the measurement profile')
    shape = tuple(report['native_shape'])
    for image in (preview.baseline_image,preview.proposed_image):
        if image.size!=(shape[1],shape[0]):
            raise ValueError('Reference overlays require the exact native image dimensions')
    scores = {}
    try:
        for mode in ('baseline','proposed'):
            scores[mode] = {str(threshold):evaluate_panel_instances(reference,report[mode]['proposals'],
                source_sha256=report['source_sha256'],native_shape=shape,iou_threshold=threshold)
                for threshold in (.5,.75)}
    except (TypeError, KeyError, AttributeError) as exc:
        raise ValueError('Invalid panel reference or proposal structure') from exc
    result = {
        'schema':'pvdelta.boundary-reference-comparison/v1', 'validation_token':token,
        'boundary_result_id':boundary_result_identity(report),
        'source_name':report['source_name'],'source_sha256':report['source_sha256'],
        'native_temperature_sha256':report['native_temperature_sha256'],'native_shape':list(shape),
        'reference_file_sha256':hashlib.sha256(reference_bytes).hexdigest(),
        'selected_reference_sha256':reference_hash,
        'declared_reference_review_status':reference['review_status'],
        'reference_provenance':reference.get('annotation_method'),
        'reference_correctness_verified':False,
        'reference_native_temperature_binding':
            'matched' if declared_native_hash is not None else 'not_supplied',
        'mismatch_interpretation':'Unmatched predictions/references relative to supplied labels. '
            'A mismatch does not establish a detector error; inspect original sources and label placement.',
        'scorer_source_sha256':validation_engine_sources(),
        'scores':scores,'overlay_iou_threshold':.75,
        'measurement_authorised':False,'database_written':False,'anomaly_rerun':False,
        'scope':'Frame-local full-panel geometry against supplied labels; not verified independent '
                'validation, physical panel inventory, anomaly accuracy or IEC acceptance',
    }
    pictures = [_overlay(picture,reference,report[mode]['proposals'],scores[mode]['0.75'])
                for mode,picture in (('baseline',preview.baseline_image),('proposed',preview.proposed_image))]
    return BoundaryReferenceComparison(result,*pictures)
