"""Source-bound, provisional RGB frame checks using complete observed module quads.

Candidate pixels/temperatures never fit the model. The selected module and a
fixed spatial subset of neighbours are held out. Repetitive arrays still need
independent correspondence review even when these image checks pass.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from pvdelta.convex_geometry import convex_iou
from pvdelta.rgb_preview import quadratic_points


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def pair_check_binding(thermal, rgb, profile, diagnostic, module_id, registration):
    return {'thermal_sha256': hashlib.sha256(Path(thermal).read_bytes()).hexdigest(),
        'rgb_sha256': hashlib.sha256(Path(rgb).read_bytes()).hexdigest(),
        'profile_id': profile['profile_id'], 'native_sha256': profile['native_temperature_sha256'],
        'geometry_sha256': digest(diagnostic), 'prior_sha256': digest(registration),
        'module_id': module_id, 'engine_sources': {name: hashlib.sha256(
            Path(__file__).with_name(name).read_bytes()).hexdigest() for name in
            ('rgb_pair_check.py', 'rgb_preview.py', 'table_boundaries.py', 'boundaries.py',
             'local_rgb_transfer.py', 'alignment.py', 'measurement_profile.py', 'decode.py',
             'table_surface.py', 'convex_geometry.py', 'tracking.py', 'junction_registration.py')}}


def _aligned_corners(predicted, observed):
    # Cyclic corner origin is arbitrary. Preserve winding; do not permit a reflection.
    from pvdelta.tracking import quadrilateral_center

    quadrilateral_center(observed)
    def cross(q):
        a, b = q[1]-q[0], q[2]-q[1]
        return float(a[0]*b[1]-a[1]*b[0])
    if cross(predicted)*cross(observed) < 0:
        observed = observed[::-1]
    return min((np.roll(observed, k, axis=0) for k in range(4)),
               key=lambda q: np.linalg.norm(q-predicted, axis=1).mean())


def explain_module_matches(modules, observed, model):
    """Retain a decision for each source, including missing/ambiguous RGB support.

    Absence of an observed quad does not label vegetation, healthy equipment or a
    failed panel. Existing overlap/margin/12px limits are unchanged. These are
    geometry correspondences, never semantic acceptance or a measurement mask.
    """
    if (any(not isinstance(m.get('module'), str) or not m['module'] for m in modules)
            or len({m['module'] for m in modules}) != len(modules)):
        raise ValueError('Distinct explicit native module identifiers required')
    if not modules:
        return {'matches': [], 'decisions': [], 'semantic_panel_membership': 'unverified'}
    mapped = [quadratic_points(m['native_polygon'], model) for m in modules]
    if not observed:
        return {'matches': [], 'decisions': [{'module_id': m['module'],
            'status': 'no_observed_rgb_counterpart', 'native_polygon': m['native_polygon'],
            'prior_rgb_polygon': q.tolist(), 'best_observed_rgb_polygon': None}
            for m, q in zip(modules, mapped)], 'semantic_panel_membership': 'unverified'}
    targets = [np.asarray(m['polygon_rgb_xy'], float) for m in observed]
    scores = np.array([[convex_iou(q, p) for p in targets] for q in mapped])
    matches, decisions = [], []
    for i, q in enumerate(mapped):
        order = np.argsort(-scores[i], kind='stable')
        j = int(order[0])
        second = float(scores[i, order[1]]) if len(order) > 1 else 0.
        competitors = np.delete(scores[:, j], i)
        decision = {'module_id': modules[i]['module'], 'native_polygon': modules[i]['native_polygon'],
            'prior_rgb_polygon': q.tolist(), 'best_observed_rgb_index': j,
            'best_observed_rgb_polygon': targets[j].tolist(), 'best_iou': float(scores[i, j]),
            'source_iou_margin': float(scores[i, j]-second),
            'target_iou_margin': float(scores[i, j]-competitors.max()) if len(competitors) else None,
            'status': 'pending'}
        decisions.append(decision)
        if scores[i, j] < .5:
            decision['status'] = 'insufficient_observed_rgb_overlap'
            continue
        if scores[i, j]-second < .2:
            decision['status'] = 'ambiguous_observed_rgb_targets'
            continue
        if len(competitors) and scores[i, j]-competitors.max() < .2:
            decision['status'] = 'competing_native_sources'
            continue
        target = _aligned_corners(q, targets[j])
        native = np.asarray(modules[i]['native_polygon'], float)
        # The fixed prior supplies a pixel scale only, never a temperature conversion.
        centre = native.mean(axis=0)
        step = quadratic_points(centre + [[0, 0], [1, 0], [0, 1]], model)
        jacobian = (step[1:]-step[0]).T
        try:
            residuals = np.linalg.solve(jacobian, (target-q).T).T
        except np.linalg.LinAlgError:
            decision['status'] = 'singular_prior_pixel_scale'
            continue
        maximum = float(np.linalg.norm(residuals, axis=1).max())
        if not np.isfinite(maximum):
            decision['status'] = 'invalid_prior_pixel_scale'
            continue
        decision['max_corner_correction_native_px'] = maximum
        if maximum > 12:
            decision['status'] = 'corner_correction_exceeds_native_bound'
            continue
        decision['status'] = 'bounded_mutual_quad_match'
        matches.append({'module_id': modules[i]['module'], 'native_polygon': native.tolist(),
            'rgb_polygon': target.tolist(), 'prior_rgb_polygon': q.tolist(),
            'iou': float(scores[i, j]), 'rgb_table': observed[j].get('table_index'),
            'prior_errors_native_px': np.linalg.norm(residuals, axis=1).tolist()})
    return {'matches': matches, 'decisions': decisions, 'semantic_panel_membership': 'unverified'}


def match_module_quads(modules, observed, model):
    """Keep mutual, unambiguous overlaps; never force every source into a target."""
    return explain_module_matches(modules, observed, model)['matches']


def fit_module_check(frame, rgb_path, matches, module_id):
    """Fit neighbours only, then check target plus separate spatial neighbours.

    Reuse the existing local registration gate (P95 <=2 native pixels). No
    RANSAC, check-point trimming, threshold relaxation or fallback refit.
    """
    from pvdelta.local_rgb_transfer import _context, fit_local_rgb_registration

    selected = next((m for m in matches if m['module_id'] == module_id), None)
    base = {'status': 'unresolved', 'reasons': [], 'correspondence_verified': False,
        'measurement_authorised': False, 'database_written': False,
        'target_module': selected, 'registration': None}

    def reject(reason):
        base['reasons'].append(reason)
        return base

    if selected is None:
        return reject('Selected native proposal has no unique, bounded full-module RGB match')
    if selected['rgb_table'] is None:
        return reject('Selected RGB module has no observed table envelope')
    centre = np.asarray(selected['native_polygon']).mean(axis=0)
    neighbours = [m for m in matches if m['rgb_table'] == selected['rgb_table']
        and np.linalg.norm(np.asarray(m['native_polygon']).mean(axis=0)-centre) <= 260]
    neighbours.sort(key=lambda m: (float(np.linalg.norm(np.asarray(m['native_polygon']).mean(axis=0)-centre)),
                                   m['module_id']))
    neighbours = neighbours[:16]
    neighbours.sort(key=lambda m: (*np.asarray(m['native_polygon']).mean(axis=0), m['module_id']))
    if selected not in neighbours:
        return reject('Selected module lies outside the bounded neighbour group')
    # Checkerboard modules share almost every corner. Hold out a fixed local
    # block instead: the selected module and its nearest spatial neighbour.
    # Choose roles before fitting, independent of errors or anomaly pixels.
    others = [m for m in neighbours if m['module_id'] != module_id]
    others.sort(key=lambda m: (float(np.linalg.norm(np.asarray(m['native_polygon']).mean(axis=0)-centre)),
                              m['module_id']))
    check = [selected, *others[:1]]
    train = [m for m in neighbours if m not in check]
    base.update(fit_module_ids=[m['module_id'] for m in train],
                check_module_ids=[m['module_id'] for m in check], matched_modules=len(neighbours),
                split_policy='selected_and_nearest_spatial_neighbour_held_out_before_fit')
    if min(len(train), len(check)) < 2:
        return reject('At least two fit and two held-out modules are required')
    raw_points = [{'control_id': f"{m['module_id']}-{j}", 'native_xy': a, 'rgb_xy': b, 'role': role}
            for role, group in (('fit', train), ('check', check)) for m in group
            for j, (a, b) in enumerate(zip(m['native_polygon'], m['rgb_polygon'], strict=True))]

    def shared(a, b):
        return (np.linalg.norm(np.array(a['native_xy'])-b['native_xy']) < 1
                or np.linalg.norm(np.array(a['rgb_xy'])-b['rgb_xy']) < 1)

    # A shared frame corner cannot be both a fit and a held-out observation.
    # Assign such corners to checking, before fitting and without reading residuals.
    checks = []
    for point in (p for p in raw_points if p['role'] == 'check'):
        if not any(shared(point, other) for other in checks):
            checks.append(point)
    fits = []
    for point in (p for p in raw_points if p['role'] == 'fit'):
        if not any(shared(point, other) for other in checks+fits):
            fits.append(point)
    base['shared_corner_policy'] = 'check_ownership_before_fit; all_raw_check_corners_still_scored'
    base['fit_controls_after_shared_corner_exclusion'] = len(fits)
    base['distinct_check_controls'] = len(checks)
    if len(fits) < 8 or len(checks) < 4:
        return reject('Insufficient distinct controls after excluding shared fit/check corners')
    controls = {'schema': 'pvdelta.local-rgb-controls/v1', 'pair': _context(frame, rgb_path),
        'provenance': {'kind': 'automatic_module_edge_correspondence_draft',
            'note': 'Bounded mutual quad overlap from a fixed prior; selected module and spatial '
                    'neighbours excluded from fit. Repeated borders are not independent point truth.'},
        'points': fits+checks}
    try:
        registration = fit_local_rgb_registration(frame, rgb_path, controls)
    except ValueError as exc:
        return reject('Local control geometry rejected: '+str(exc))
    base.update(registration=registration,
        prior_check_p95_native_px=float(np.percentile(
            [e for m in check for e in m['prior_errors_native_px']], 95)))
    if not registration['geometry_gate_passed']:
        base['check_p95_native_px'] = registration['validation']['check_p95_source_px']
        return reject('Held-out module corners exceed the fixed local point gate')
    from pvdelta.alignment import transform_points, validate_homography_checks

    # Excluded duplicates are not dropped from the error assessment. Score every
    # original held-out corner against the frozen fit; never trim difficult points.
    all_checks = validate_homography_checks(np.asarray(registration['native_to_rgb']),
        fit_point_count=len(fits), check_source_xy=np.concatenate([m['native_polygon'] for m in check]),
        check_target_xy=np.concatenate([m['rgb_polygon'] for m in check]),
        max_p95_source_px=2., min_fit_points=8)
    base['check_p95_native_px'] = all_checks.check_p95_source_px
    if not all_checks.accepted:
        return reject('All original held-out corners exceed the fixed local point gate')
    native = np.concatenate([m['native_polygon'] for m in neighbours])
    prior_rgb = np.concatenate([m['prior_rgb_polygon'] for m in neighbours])
    matrix = np.asarray(registration['native_to_rgb'])
    denominators = np.column_stack((native, np.ones(len(native)))) @ matrix[2]
    if np.any(np.abs(denominators) < 1e-9) or np.ptp(np.sign(denominators)):
        return reject('Projective pole crosses the bounded native support')
    corrected_native = transform_points(prior_rgb, np.linalg.inv(matrix))
    base['maximum_correction_native_px'] = float(np.linalg.norm(corrected_native-native, axis=1).max())
    if base['maximum_correction_native_px'] > 12:
        return reject('Local correction exceeds the fixed 12-native-pixel bound')
    # A pass must also improve held-out alignment, not just fall below an absolute threshold.
    if base['check_p95_native_px'] > base['prior_check_p95_native_px']:
        return reject('Local fit worsens the held-out error; original prior retained')
    base['status'] = 'image_consistent_provisional'
    return base


def check_rgb_pair(thermal, rgb, profile, diagnostic, module_id, registration, *, sdk_library=None,
                   use_junctions=False):
    import cv2

    from pvdelta.decode import RJpeg
    from pvdelta.local_rgb_transfer import _context
    from pvdelta.measurement_profile import apply_saved_measurement_profile
    from pvdelta.metadata import parse_camera_profile
    from pvdelta.models import ThermalFrame
    from pvdelta.table_boundaries import refined_module_proposals

    binding = pair_check_binding(thermal, rgb, profile, diagnostic, module_id, registration)
    if binding['thermal_sha256'] != profile['source_sha256']:
        raise ValueError('Saved profile does not match this original thermal source')
    with RJpeg(thermal, sdk_library) as reader:
        values = apply_saved_measurement_profile(reader, thermal, profile)
        native_image = reader.processed_rgb() if use_junctions else None
    frame = ThermalFrame(Path(thermal).stem, str(thermal), binding['thermal_sha256'],
        values, None, values.shape[1], values.shape[0])
    rgb_array = cv2.imdecode(np.frombuffer(Path(rgb).read_bytes(), np.uint8),
        cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if rgb_array is None:
        raise ValueError('Original RGB source could not be decoded')
    if tuple(values.shape) != (512, 640) or rgb_array.shape[:2] != (3024, 4032):
        raise ValueError('Stored M4T prior requires its original native and RGB grids')
    modules = diagnostic.get('modules', [])
    camera = parse_camera_profile(thermal)
    if camera.model != 'M4T':
        raise ValueError('Stored cross-camera prior is limited to the original M4T flight')
    observed = refined_module_proposals(rgb_array, gimbal_pitch_deg=camera.gimbal_pitch)
    if observed['status'] != 'unreviewed_rgb_geometry':
        raise ValueError('Full-module RGB extractor withheld geometry')
    matching = explain_module_matches(modules, observed['modules'], registration['model'])
    matches = matching['matches']
    if use_junctions:
        import base64

        from pvdelta.local_rgb_transfer import propose_observed_junction_transfer

        proposals = [{'proposal_id': f'RGB{i}', 'polygon_rgb_xy': p['polygon_rgb_xy'],
                      'table_index': p.get('table_index')} for i,p in enumerate(observed['modules'])]
        result = propose_observed_junction_transfer(frame, rgb, matches, proposals)
        result['status'] = 'supported_local_geometry' if result['supported_proposals'] else 'unresolved'
        result['reasons'] = ['Source correspondence and independent measurement approval remain unverified.']
        images = {}
        for name, image, polygon_key in (('thermal', cv2.cvtColor(native_image, cv2.COLOR_RGB2BGR), 'polygon_native_xy'),
                                         ('rgb', rgb_array.copy(), 'polygon_rgb_xy')):
            for proposal in result['proposals']:
                cv2.polylines(image, [np.rint(proposal[polygon_key]).astype(np.int32)], True,
                              (80, 255, 80), 2 if name == 'thermal' else 6)
            if name == 'rgb':
                scale = min(1., 1008 / image.shape[1])
                image = cv2.resize(image, (round(image.shape[1]*scale), round(image.shape[0]*scale)),
                                   interpolation=cv2.INTER_AREA)
            ok, encoded = cv2.imencode('.jpg', image)
            if not ok:
                raise ValueError('Could not render source-bound geometry preview')
            images[name] = base64.b64encode(encoded.tobytes()).decode('ascii')
        result['preview_images'] = images
    else:
        result = fit_module_check(frame, rgb, matches, module_id)
    result.update(schema='pvdelta.rgb-pair-check/v1', binding=binding,
        observed_rgb_modules=len(observed['modules']), unique_quad_matches=len(matches),
        matching_diagnostics=matching['decisions'],
        selected_matching_diagnostic=next((d for d in matching['decisions'] if d['module_id'] == module_id), None),
        pair=_context(frame, rgb), scope='Image consistency only; no physical panel identity, glare or fault clearance')
    if pair_check_binding(thermal, rgb, profile, diagnostic, module_id, registration) != binding:
        raise ValueError('Pair inputs changed during the local check')
    result['content_sha256'] = digest(result)
    return result


def rgb_pair_check_view(thermal, rgb, profile, diagnostic, module_id, registration, *, sdk_library, key,
                        automatic_junctions=False):
    import streamlit as st

    if (not thermal or not rgb or profile is None or registration is None
            or diagnostic.get('geometry_method') != 'native_observed_boundaries'
            or not diagnostic.get('modules')):
        if automatic_junctions:
            st.info('Automatic local geometry is unavailable: an original RGB/thermal pair, saved '
                    'radiometric profile, search calibration and native module proposals are required. '
                    'This does not indicate a healthy panel.')
        return None
    try:
        import cv2
    except ImportError:
        st.caption('RGB alignment checks require the optional vision dependencies.')
        return None
    result_key = key+'-result'
    try:
        binding = pair_check_binding(thermal, rgb, profile, diagnostic, module_id, registration)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        st.session_state.pop(result_key, None)
        st.info(f'RGB check unavailable: {exc}')
        return None
    saved = st.session_state.get(result_key)
    if saved and (saved.get('binding') != binding or saved.get('content_sha256') != digest(
            {k: v for k, v in saved.items() if k != 'content_sha256'})):
        st.session_state.pop(result_key, None)
        saved = None
    if st.button('Check RGB alignment', key=key+'-start'):
        st.session_state.pop(result_key, None)
        saved = None
        try:
            with st.spinner('Matching complete RGB module boundaries; checking unused modules…'):
                saved = check_rgb_pair(thermal, rgb, profile, diagnostic, module_id, registration,
                                       sdk_library=sdk_library)
            st.session_state[result_key] = saved
        except (OSError, RuntimeError, ValueError, TypeError, KeyError, cv2.error) as exc:
            st.warning(f'Local RGB check could not finish: {exc}')
    if saved:
        if saved['status'] == 'image_consistent_provisional':
            st.caption(f"Held-out module error: {saved['prior_check_p95_native_px']:.2f} → "
                       f"{saved['check_p95_native_px']:.2f} native px (P95). Physical identity remains unverified.")
        else:
            st.info('Local RGB correction withheld: '+'; '.join(saved['reasons']))
        with st.expander('RGB alignment check details'):
            st.json(saved)
            st.download_button('Download RGB alignment check', json.dumps(saved, indent=2, allow_nan=False),
                file_name='rgb-pair-check.json', mime='application/json', key=key+'-download')
    _junction_transfer_view(thermal, rgb, profile, diagnostic, module_id, registration,
                            sdk_library=sdk_library, key=key, binding=binding, automatic=automatic_junctions)
    return saved


def _junction_transfer_view(thermal, rgb, profile, diagnostic, module_id, registration, *, sdk_library, key, binding,
                             automatic=False):
    """Explicit geometry review shared by Basic/Advanced, isolated from anomaly overlays."""
    import base64

    import cv2
    import streamlit as st

    result_key = key+'-junction-result'
    cache = st.session_state.setdefault('_pvdelta_local_geometry_cache', {})
    if automatic:
        module_id = '__frame_geometry__'
        binding = binding | {'module_id': module_id}
    cache_id = digest({'binding': binding, 'sdk_library': str(sdk_library)})
    cached = cache.get(cache_id, {}) if automatic else {}
    cached_error = cached.get('error')
    result = cached.get('result') if automatic else st.session_state.get(result_key)
    if result and (result.get('binding') != binding or result.get('content_sha256') != digest(
            {k:v for k,v in result.items() if k != 'content_sha256'})):
        st.session_state.pop(result_key, None)
        cache.pop(cache_id, None)
        result = None
        cached_error = None
    with st.expander('Local panel boundary transfer', expanded=automatic):
        st.caption('Green outlines show provisional panel geometry, not anomalies. Each table is checked separately.')
        clicked = st.button('Recheck panel boundary transfer' if automatic else 'Check panel boundary transfer',
                            key=key+'-junction-start')
        if clicked or (automatic and result is None and not cached_error):
            st.session_state.pop(result_key, None)
            result = None
            cached_error = None
            try:
                with st.spinner('Checking observed frame junctions and complete boundary support…'):
                    result = check_rgb_pair(thermal, rgb, profile, diagnostic, module_id, registration,
                                           sdk_library=sdk_library, use_junctions=True)
                st.session_state[result_key] = result
                if automatic:
                    cache[cache_id] = {'result': result}
            except (OSError, RuntimeError, ValueError, TypeError, KeyError, IndexError, cv2.error) as exc:
                cached_error = str(exc)
                if automatic:
                    cache[cache_id] = {'error': cached_error}
        while len(cache) > 8:
            cache.pop(next(iter(cache)))
        if cached_error:
            st.warning(f'Panel boundary transfer unavailable: {cached_error}')
        if result:
            if result['supported_proposals']:
                st.success('Local geometry preview available. Panel identity and anomaly status still require review.')
            else:
                st.info('No local boundary passed all transfer checks. This is an unresolved result, not a healthy-panel decision.')
            st.caption(f"Supported geometry: {result['supported_proposals']} / {result['requested_proposals']} proposals. "
                       'Radiometric masks and findings are unchanged; physical correspondence remains unverified.')
            images = result.get('preview_images', {})
            if images:
                columns = st.columns(3)
                for column, sensor in zip(columns[:2], ('thermal','rgb'), strict=True):
                    with column:
                        st.image(base64.b64decode(images[sensor]), caption=f'{sensor.title()} — local geometry')
                from pvdelta.rgb_preview import original_rgb_image

                with columns[2]:
                    st.image(original_rgb_image(rgb, expected_source_hash=binding['rgb_sha256']),
                             caption='Original RGB · no marks', use_container_width=True)
            st.dataframe(result['decisions'], hide_index=True)
            if result.get('groups'):
                st.dataframe([{'table': g['table_index'],
                    'observed_junctions': sum(n.get('status') == 'matched_provisional'
                        for n in (g.get('junctions') or {}).get('nodes', [])),
                    'reason': '; '.join((g.get('junctions') or {}).get('reasons', [])) or
                        ('No bounded matches' if g.get('junctions') is None else 'See boundary support decisions')}
                    for g in result['groups']], hide_index=True)
            details = {k:v for k,v in result.items() if k != 'preview_images'}
            st.download_button('Download panel boundary transfer', json.dumps(details, indent=2, allow_nan=False),
                               file_name='panel-boundary-transfer.json', mime='application/json', key=key+'-junction-download')
