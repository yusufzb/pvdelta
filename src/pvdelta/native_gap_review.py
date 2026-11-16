"""Separate review evidence from unresolved native boundaries, never inventory."""
from __future__ import annotations

import hashlib
from copy import deepcopy
from dataclasses import asdict

import numpy as np

from pvdelta.convex_geometry import convex_iou
from pvdelta.models import ThermalFrame
from pvdelta.native_completion import _observed_sides, _validate
from pvdelta.native_lines import _area_axis, _neighbours
from pvdelta.pipeline import analyze_annotations
from pvdelta.reference import erode
from pvdelta.regions import ModuleAnnotation, polygon_to_mask


def export_native_frame_gaps(source_path, diagnostic, measurement_profile, directory, *, sdk_library=None):
    """Review rejected observed boundaries on the exact saved native profile, separately.

    Accepted panel gates are unchanged. Unresolved regions can expose a missed
    thermal candidate without assigning it a module, location or fault verdict.
    """
    from pathlib import Path

    from pvdelta.decode import RJpeg
    from pvdelta.gap_review_bundle import write_gap_bundle
    from pvdelta.measurement_profile import apply_saved_measurement_profile
    from pvdelta.native_review import boundary_engine_sources
    from pvdelta.native_search_trace import trace_native_search
    from pvdelta.native_windows import native_window_proposals

    if diagnostic.get('geometry_method') != 'native_observed_boundaries':
        raise ValueError('Source-bound native structural geometry required')
    source = Path(source_path)
    with RJpeg(source, sdk_library) as reader:
        values = apply_saved_measurement_profile(reader, source, measurement_profile)
    frame = ThermalFrame(source.stem, str(source.resolve()), measurement_profile['source_sha256'],
        values, None, values.shape[1], values.shape[0])
    baseline = diagnostic['native_geometry']
    collector = NativeGapReview(frame, baseline)
    with trace_native_search(values, collector):
        repeated = native_window_proposals(values, raw_supplement=True)
    if repeated['proposals'] != baseline['proposals']:
        raise ValueError('Native boundary engine or proposal geometry changed; reanalyse this frame first')
    report = collector.evaluate()
    report['measurement_profile_id'] = measurement_profile['profile_id']
    hashes = boundary_engine_sources() | {name: hashlib.sha256(
        Path(__file__).with_name(name).read_bytes()).hexdigest() for name in
        ('native_gap_review.py', 'native_search_trace.py', 'gap_review_bundle.py', 'measurement_profile.py')}
    folder = write_gap_bundle(directory, [(report, values, source,
        'DJI SDK native temperatures reconstructed from the saved analysis profile; unresolved boundary',
        measurement_profile)], engine_source_sha256=hashes)
    return folder, report


def _context(quad, baseline, centres=None):
    _, short, long, axis = _area_axis(quad)
    normal = np.array([-axis[1], axis[0]])
    along, across = [], []
    if centres is None:
        centres = np.asarray([np.asarray(p['polygon_native_xy']).mean(axis=0) for p in baseline]).reshape(-1, 2)
    nearby = np.flatnonzero(np.linalg.norm(centres-quad.mean(axis=0), axis=1) <= long*1.6)
    for index in nearby:
        p = baseline[index]
        if not _observed_sides(p):
            continue
        other = np.asarray(p['polygon_native_xy'], float)
        if not _neighbours(quad, other) or convex_iou(quad, other) > .05:
            continue
        delta = other.mean(axis=0)-quad.mean(axis=0)
        a, b = abs(float(delta@axis)), abs(float(delta@normal))
        if abs(a-long) < long*.3 and b < short*.25:
            along.append(p)
        if abs(b-short) < short*.3 and a < long*.25:
            across.append(p)
    for a in along:
        for b in across:
            if a['proposal_id'] != b['proposal_id'] and convex_iou(a['polygon_native_xy'], b['polygon_native_xy']) <= .05:
                return [a['proposal_id'], b['proposal_id']]
    return []


class NativeGapReview:
    """Bounded trace sink; no references, RGB, CAD or inferred panel grid.

    Retain actual hybrid side-support rejections with two passing sides and
    disjoint observed neighbours along BOTH module axes. Review temperatures
    on provisional native hypotheses only. Nothing changes accepted masks,
    module counts, fault counts or health assessment. Use default .8/.18 sides.
    """

    def __init__(self, frame: ThermalFrame, baseline, *, max_hypotheses=512, max_analyses=64):
        for name, value, limit in (('max_hypotheses', max_hypotheses, 4096), ('max_analyses', max_analyses, 256)):
            if type(value) is not int or not 1 <= value <= limit:
                raise ValueError(f'{name} must be integer 1..{limit}')
        if max_analyses > max_hypotheses:
            raise ValueError('analysis limit cannot exceed hypothesis limit')
        if not isinstance(frame, ThermalFrame) or not np.isfinite(frame.temperature_c).all():
            raise ValueError('A finite native ThermalFrame is required')
        _validate(baseline, frame.temperature_c.shape)
        self.frame, self.baseline = frame, deepcopy(baseline['proposals'])
        self.centres = np.asarray([np.asarray(p['polygon_native_xy']).mean(axis=0) for p in self.baseline]).reshape(-1, 2)
        self.max_hypotheses, self.max_analyses = max_hypotheses, max_analyses
        self.passes, self.candidates = {}, {}
        self.counters = {'side_rejections_seen': 0, 'orthogonal_context_hypotheses': 0,
                         'budget_evictions': 0, 'overlap_rejections': 0}
        self.digest = hashlib.sha256(frame.temperature_c.tobytes()).hexdigest()

    @staticmethod
    def _rank(candidate):
        s = candidate['support']
        return (-sum(bool(r[0]) for r in s), -sum(float(r[1]) for r in s),
                sum(float(r[2]) for r in s), *np.asarray(candidate['polygon_native_xy']).ravel())

    def __call__(self, event):
        if event.get('event') == 'pass_begin':
            self.passes[event['pass_id']] = deepcopy(event)
            return
        if event.get('event') != 'candidate' or event.get('stage') != 'side_support':
            return
        info = self.passes.get(event['pass_id'])
        if not info or info.get('support_mode') != 'hybrid':
            return
        self.counters['side_rejections_seen'] += 1
        support = np.asarray(event.get('support'), float)
        if (support.shape != (4, 3) or not np.isfinite(support).all()
                or ((support[:, 1:] < -1e-9) | (support[:, 1:] > 1+1e-9)).any()
                or not np.isin(support[:, 0], (0, 1)).all()):
            raise ValueError('Four actual side-support tuples required')
        # Analytic interval unions can exceed one by floating-point roundoff.
        # Reject material invalid evidence; normalize only machine-scale error.
        support[:, 1:] = np.clip(support[:, 1:], 0, 1)
        passing = (support[:, 1] >= .8) & (support[:, 2] <= .18)
        if not np.array_equal(passing, support[:, 0].astype(bool)):
            raise ValueError('Side status disagrees with coverage/gap')
        if passing.sum() < 2:
            return
        quad = np.asarray(event['polygon_native_xy'], float)
        if quad.shape != (4, 2) or not np.isfinite(quad).all():
            raise ValueError('Finite quadrilateral trace event required')
        if ((quad < 3).any() or (quad[:, 0] >= self.frame.width-3).any()
                or (quad[:, 1] >= self.frame.height-3).any()):
            return
        _validate({'proposals': [{'proposal_id': 'gap', 'polygon_native_xy': quad.tolist()}]}, self.frame.temperature_c.shape)
        refs = _context(quad, self.baseline, self.centres)
        if not refs:
            return
        if any(convex_iou(quad, p['polygon_native_xy']) > .05 for p in self.baseline):
            self.counters['overlap_rejections'] += 1
            return
        record = {'polygon_native_xy': quad.tolist(), 'support': support.tolist(),
            'baseline_context_ids': refs, 'source_pass': deepcopy(info), 'reason': 'four_side_support_unresolved'}
        key = tuple(np.round(quad, 3).ravel())
        previous = self.candidates.get(key)
        if previous is not None and self._rank(previous) <= self._rank(record):
            return
        self.counters['orthogonal_context_hypotheses'] += 1
        self.candidates[key] = record
        if len(self.candidates) > self.max_hypotheses:
            worst = max(self.candidates, key=lambda k: self._rank(self.candidates[k]))
            del self.candidates[worst]
            self.counters['budget_evictions'] += 1

    def evaluate(self):
        if hashlib.sha256(self.frame.temperature_c.tobytes()).hexdigest() != self.digest:
            raise RuntimeError('Native source changed during gap review')
        occupied = np.zeros(self.frame.temperature_c.shape, bool)
        for p in self.baseline:
            occupied |= polygon_to_mask(p['polygon_native_xy'], occupied.shape)
        selected, duplicate_count = [], 0
        for candidate in sorted(self.candidates.values(), key=self._rank):
            if any(convex_iou(candidate['polygon_native_xy'], p['polygon_native_xy']) > .8 for p in selected):
                duplicate_count += 1
                continue
            selected.append(candidate)
        withheld = max(0, len(selected)-self.max_analyses)
        evidence, analyses, seen = [], [], []
        for index, candidate in enumerate(selected[:self.max_analyses]):
            quad = candidate['polygon_native_xy']
            mask = erode(polygon_to_mask(quad, occupied.shape), 2) & ~occupied
            identifier = f'gap-{index+1:03d}'
            annotation = ModuleAnnotation(identifier, tuple(map(tuple, quad)))
            report = analyze_annotations(self.frame, [annotation], module_masks={identifier: mask}, direction='hot')[0]
            analyses.append({'hypothesis_id': identifier, 'candidate': candidate,
                'assessment_status': report.assessment_status, 'reasons': list(report.reasons),
                'native_measurement_pixels': int(mask.sum()), 'hot_evidence_count': len(report.findings)})
            for finding in report.findings:
                pixels = frozenset(map(tuple, finding.region_pixels_yx or []))
                if not pixels or any(len(pixels & prior)/len(pixels | prior) >= .5 for prior in seen):
                    continue
                seen.append(pixels)
                native = np.asarray(sorted(pixels), dtype='<i4')
                if occupied[native[:, 0], native[:, 1]].any():
                    raise RuntimeError('Gap evidence overlaps baseline envelopes')
                evidence_id = hashlib.sha256(self.frame.source_hash.encode()+native.tobytes()).hexdigest()[:24]
                evidence.append({'evidence_id': evidence_id, 'status': 'unassigned_thermal_evidence_review_required',
                    'source_frame_id': self.frame.frame_id, 'source_hash': self.frame.source_hash,
                    'source_name': self.frame.source_path, 'hypothesis_id': identifier,
                    'provisional_polygon_native_xy': quad, 'geometry_evidence': candidate,
                    'native_finding': asdict(finding), 'module_identity': None, 'panel_location': None,
                    'confirmed_fault': False, 'measurement_authorised': False,
                    'warning': 'Unresolved boundary: possible panel/ground/reflection; excluded from panel/fault/healthy totals.'})
        if hashlib.sha256(self.frame.temperature_c.tobytes()).hexdigest() != self.digest:
            raise RuntimeError('Native source changed during gap evaluation')
        return {'schema': 'pvdelta.native-gap-review/v1', 'source_hash': self.frame.source_hash,
            'native_temperature_sha256': self.digest, 'counters': deepcopy(self.counters),
            'stored_hypotheses': len(self.candidates), 'near_duplicate_hypotheses': duplicate_count,
            'hypotheses_withheld_by_analysis_budget': withheld, 'max_hypotheses': self.max_hypotheses,
            'max_analyses': self.max_analyses, 'analyses': analyses, 'evidence': evidence,
            'baseline_unchanged': True, 'production_enabled': False,
            'scope': 'Separate review queue; not accepted modules, confirmed findings, inventory or health assessment'}
