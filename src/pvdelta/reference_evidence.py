"""Reproduce saved candidates before auditing their reference geometry."""
from __future__ import annotations

import hashlib
import json
from dataclasses import fields
from pathlib import Path

import cv2
import numpy as np

from pvdelta.candidate_stability import candidate_stability
from pvdelta.candidate_triage import native_support_masks
from pvdelta.config import DetectionConfig
from pvdelta.decode import RJpeg
from pvdelta.detect import assess_intra_module_anomalies
from pvdelta.measurement_profile import apply_saved_measurement_profile, read_measurement_profile
from pvdelta.models import ModuleObservation, ThermalFrame
from pvdelta.reference import erode
from pvdelta.store import connect_store


def reference_geometry(values, reference, background, points, *, minimum_pixels):
    """Pixel-centre hull interpolation is spatial support, not fault probability."""
    values, reference, background = map(np.asarray, (values, reference, background))
    points = np.asarray(points)
    if (values.ndim != 2 or reference.shape != values.shape or reference.dtype != np.bool_
            or background.shape != values.shape or reference.sum() < 3
            or points.ndim != 2 or points.shape[1] != 2 or not len(points)
            or not np.issubdtype(points.dtype, np.integer) or (points < 0).any()
            or (points[:, 0] >= values.shape[0]).any() or (points[:, 1] >= values.shape[1]).any()
            or len(np.unique(points, axis=0)) != len(points)
            or type(minimum_pixels) is not int or minimum_pixels < 1):
        raise ValueError('Valid reference mask and distinct native candidate pixels required')
    yy, xx = points.T
    if reference[yy, xx].any():
        raise ValueError('Candidate contaminates its reference')
    if not (np.isfinite(values[reference]).all() and np.isfinite(background[reference]).all()
            and np.isfinite(values[yy, xx]).all() and np.isfinite(background[yy, xx]).all()):
        raise ValueError('Reference and candidate temperatures must be finite')
    hull = cv2.convexHull(np.argwhere(reference)[:, ::-1].astype(np.float32))
    if cv2.contourArea(hull) <= 0:
        raise ValueError('Reference cannot support a two-dimensional plane')
    outside = sum(cv2.pointPolygonTest(hull, (float(x), float(y)), False) < 0 for y, x in points)
    errors = values[reference] - background[reference]
    residuals = values[yy, xx] - background[yy, xx]
    return {'reference_pixels': int(reference.sum()), 'candidate_pixels': len(points),
        'reference_mad_k': float(np.median(np.abs(errors - np.median(errors)))),
        'reference_residual_p95_abs_k': float(np.percentile(np.abs(errors), 95)),
        'extrapolated_pixels': outside, 'extrapolated_fraction': outside / len(points),
        'background_span_on_candidate_k': float(np.ptp(background[yy, xx])),
        'median_residual_k': float(np.median(residuals)), 'peak_residual_k': float(np.max(residuals)),
        'minimum_pixel_support': len(points) == minimum_pixels,
        'reference_status': 'extrapolation_review' if outside else 'interpolated_reference',
        'fault_confirmed': False}


def check_temperature_evidence(database, run_id, observation_id, *, sdk_library=None):
    """Fresh saved-profile SDK read; refuse nonreproducing records. No writes."""
    database = Path(database).resolve()
    summary = json.loads(database.with_name('summary.json').read_bytes())
    if summary.get('schema') != 'pvdelta.native-flight/v1' or summary.get('analysis_run_id') != run_id:
        raise ValueError('Matching native-flight run settings required')
    settings = summary['settings']
    config = DetectionConfig(**{f.name: settings[f.name] for f in fields(DetectionConfig)})
    c = connect_store(database, read_only=True)
    try:
        c.execute('BEGIN TRANSACTION')
        row = c.execute('''SELECT m.capture_id,m.frame_module_id,m.pixel_polygon_json,
            c.source_path,r.source_root,c.source_hash FROM module_observations m
            JOIN captures c USING(capture_id) JOIN inspection_runs r ON r.run_id=c.run_id
            WHERE m.analysis_run_id=? AND m.observation_id=?''', [run_id, observation_id]).fetchone()
        if row is None:
            raise ValueError('Selected observation is outside the analysis run')
        cid, mid, polygon, relative, root, source_hash = row
        records = c.execute('''SELECT finding_id,region_pixels_yx_json,delta_t_k,
            absolute_temperature_c,n_reference_px FROM findings
            WHERE analysis_run_id=? AND observation_id=? ORDER BY finding_id''',
            [run_id, observation_id]).fetchall()
        if not records:
            raise ValueError('No saved candidates for this observation')
        mask = native_support_masks(c, run_id, cid, 0)[mid]
        profile = read_measurement_profile(c, run_id, cid)
        c.execute('COMMIT')
    finally:
        c.close()
    base = Path(root).resolve()
    path = (base / relative).resolve()
    if not path.is_relative_to(base):
        raise ValueError('Source is outside the indexed flight folder')
    with RJpeg(path, sdk_library) as reader:
        values = apply_saved_measurement_profile(reader, path, profile)
    original = hashlib.sha256(values.tobytes()).hexdigest()
    frame = ThermalFrame(cid, str(path), source_hash, values, np.isfinite(values), values.shape[1], values.shape[0])
    assessment = assess_intra_module_anomalies(frame, ModuleObservation(mid, cid, json.loads(polygon)),
        mask, config=config, include_background=True)
    if assessment.reasons or assessment.reference_mask is None:
        raise ValueError('Background replay unavailable: ' + '; '.join(assessment.reasons))

    def key(points):
        return tuple(sorted(map(tuple, points)))

    reproduced = {key(f.region_pixels_yx): f for f in assessment.findings if f.anomaly_type == 'hot_region'}
    saved = {key(json.loads(row[1])) for row in records}
    if len(saved) != len(records) or set(reproduced) != saved:
        raise ValueError('Current detector does not reproduce the complete saved hot-region population')
    results = []
    for fid, encoded, delta, temperature, count in records:
        points = json.loads(encoded)
        finding = reproduced[key(points)]
        if (abs(finding.delta_t_c - delta) > 1e-9 or finding.absolute_t_c != temperature
                or finding.n_reference_px != count):
            raise ValueError('Current detector does not reproduce saved candidate measurements')
        results.append({'finding_id': fid, **reference_geometry(values, assessment.reference_mask,
            assessment.background_c, points, minimum_pixels=config.min_region_px)})
    if hashlib.sha256(values.tobytes()).hexdigest() != original:
        raise RuntimeError('Diagnostic changed native temperatures')
    sensitivity = candidate_stability(values, erode(mask, config.edge_erosion_px),
        assessment.reference_mask, {row[0]: json.loads(row[1]) for row in records}, config=config)
    return {'schema': 'pvdelta.reference-evidence/v1', 'analysis_run_id': run_id,
        'observation_id': observation_id, 'source_file': path.name, 'source_sha256': source_hash,
        'profile_id': profile['profile_id'], 'native_temperature_sha256': original,
        'settings': settings, 'complete_saved_population_reproduced': True, 'regions': results,
        'sensitivity': sensitivity,
        'note': 'Reference geometry only. No panel identity, glare clearance or fault diagnosis.',
        'database_written': False}
