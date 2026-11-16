"""Source-bound local RGB geometry transfer with held-out point diagnostics."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from pvdelta.alignment import fit_homography, transform_points, validate_homography_checks
from pvdelta.convex_geometry import convex_iou
from pvdelta.models import ThermalFrame


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _context(frame, rgb_path):
    import cv2

    if not isinstance(frame, ThermalFrame) or not np.isfinite(frame.temperature_c).all():
        raise ValueError('ThermalFrame with finite original native values required')
    if _sha(Path(frame.source_path).read_bytes()) != frame.source_hash:
        raise ValueError('Thermal source file identity mismatch')
    path = Path(rgb_path).expanduser().resolve()
    encoded = path.read_bytes()
    rgb = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if rgb is None or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError('Original RGB image could not be decoded')
    context = {'source_sha256': frame.source_hash, 'rgb_sha256': _sha(encoded),
        'native_temperature_sha256': _sha(frame.temperature_c.tobytes()),
        'native_shape': list(frame.temperature_c.shape), 'rgb_shape': list(rgb.shape[:2]),
        'coordinate_spaces': ['thermal_native', 'original_rgb_pixels']}
    return context


def _points(value, shape):
    if (not isinstance(value, (list, tuple)) or any(not isinstance(p, (list, tuple)) or len(p) != 2
            or any(type(n) not in (int, float) for n in p) for p in value)):
        raise ValueError('Explicit numeric pixel coordinates required')
    q = np.asarray(value, dtype=float)
    if (q.ndim != 2 or q.shape[1] != 2 or not np.isfinite(q).all() or (q < 0).any()
            or (q[:, 0] >= shape[1]).any() or (q[:, 1] >= shape[0]).any()):
        raise ValueError('Finite points inside the exact source pixel grid required')
    return q


def _rgb_geometry(context, calibration):
    if calibration is None:
        return {'model':'original_rgb_pinhole'}
    if (not isinstance(calibration,dict) or calibration.get('schema') != 'pvdelta.rgb-brown-calibration/v1'
            or calibration.get('rgb_source_sha256') != context['rgb_sha256']):
        raise ValueError('RGB calibration must bind the exact original RGB source')
    from pvdelta.table_surface import brown_intrinsics

    camera=calibration.get('camera',{})
    names=('focal_x','focal_y','c_x','c_y','k1','k2','p1','p2','k3')
    if (not isinstance(camera,dict) or camera.get('projection_type') != 'brown'
            or any(type(camera.get(n)) not in (int,float) or not np.isfinite(camera[n]) for n in names)
            or any(type(camera.get(n)) is not int for n in ('height','width'))
            or [camera['height'],camera['width']] != context['rgb_shape']
            or min(camera['focal_x'],camera['focal_y']) <= 0
            or not isinstance(calibration.get('provenance'),dict)
            or not isinstance(calibration['provenance'].get('note'),str)
            or not calibration['provenance']['note'].strip()):
        raise ValueError('Finite source-sized Brown camera and provenance required')
    brown_intrinsics(camera)
    return {'model':'brown_undistorted_rgb', 'calibration_sha256':_sha(_canonical(calibration)),
        'calibration':json.loads(_canonical(calibration))}


def _rgb_model_points(points, geometry):
    if geometry['model']=='original_rgb_pinhole':
        return np.asarray(points,float)
    import cv2

    from pvdelta.table_surface import brown_intrinsics

    intrinsic,distortion=brown_intrinsics(geometry['calibration']['camera'])
    criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS,50,1e-12)
    source=np.asarray(points,float).reshape(-1,1,2)
    if hasattr(cv2,'undistortPointsIter'):  # OpenCV 4.x
        result=cv2.undistortPointsIter(source,intrinsic,distortion,None,intrinsic,criteria).reshape(-1,2)
    else:  # OpenCV 5 merges the criteria argument into undistortPoints.
        result=cv2.undistortPoints(source,intrinsic,distortion,P=intrinsic,criteria=criteria).reshape(-1,2)
    if not np.isfinite(result).all():
        raise ValueError('Brown correction produced invalid coordinates')
    if np.linalg.norm(_distort_model_points(result,geometry)-np.asarray(points,float),axis=1).max(initial=0)>.01:
        raise ValueError('Brown correction inverse failed its original-pixel roundtrip check')
    return result


def _distort_model_points(points,geometry):
    if geometry['model']=='original_rgb_pinhole':
        return np.asarray(points,float)
    from pvdelta.table_surface import brown_intrinsics, brown_project

    camera=geometry['calibration']['camera']
    intrinsic,_=brown_intrinsics(camera)
    q=np.asarray(points,float)
    rays=np.column_stack(((q[:,0]-intrinsic[0,2])/intrinsic[0,0],
        (q[:,1]-intrinsic[1,2])/intrinsic[1,1],np.ones(len(q))))
    return brown_project(rays,[0.,0.,0.],[0.,0.,0.],camera)[0]


def _native_matrix(registration):
    return np.asarray(registration['native_to_undistorted_rgb'] if
        registration.get('rgb_geometry',{}).get('model')=='brown_undistorted_rgb' else registration['native_to_rgb'],float)


def project_local_native_to_rgb(points, registration):
    """Render only accepted local geometry on original distorted RGB coordinates."""
    if not registration.get('geometry_gate_passed'):
        raise ValueError('Accepted local point gate required')
    payload={k:v for k,v in registration.items() if k!='registration_id'}
    if registration.get('registration_id')!='sha256:'+_sha(_canonical(payload)):
        raise ValueError('Registration content identity mismatch')
    points=_points(np.asarray(points,float).tolist(),registration['pair']['native_shape'])
    import cv2

    if any(cv2.pointPolygonTest(np.float32(registration[key]),tuple(map(float,p)),True)<0
           for key in ('fit_hull_native_xy','check_hull_native_xy') for p in points):
        raise ValueError('Native points outside local support')
    projected=transform_points(points,_native_matrix(registration))
    geometry=registration.get('rgb_geometry',{'model':'original_rgb_pinhole'})
    return _distort_model_points(projected,geometry)


def _point_gate_policy(max_check_p95_px, tolerance_note):
    if (type(max_check_p95_px) not in (int, float) or not np.isfinite(max_check_p95_px)
            or not 0 < max_check_p95_px <= 2.20):
        raise ValueError('Native check tolerance must be finite, positive and at most2.20px')
    if tolerance_note is not None and (not isinstance(tolerance_note,str) or not tolerance_note.strip()):
        raise ValueError('Tolerance note must be a nonempty string')
    if max_check_p95_px > 2. and tolerance_note is None:
        raise ValueError('An explicit note is required for a development tolerance above2px')
    return {'max_check_p95_native_px':float(max_check_p95_px),
            'note': tolerance_note or ('Default2px local source geometry gate' if max_check_p95_px==2.
                                      else 'Explicit stricter local source geometry gate'),
            'scope':'development_geometry_not_independent_or_IEC_acceptance'}


def fit_local_rgb_registration(frame, rgb_path, controls, *, rgb_calibration=None,
                               max_check_p95_px=2., tolerance_note=None):
    """Fit native→RGB on declared fit points; never fit or trim check points.

    Identity is bound to original source bytes and exact native values/grids.
    The default numerical gate is P95 <=2 native pixels, >=8 fit/4 check
    controls, distinct coordinates and two-dimensional fit/check layouts.
    An explicit development tolerance up to2.20px requires a recorded reason.
    A passed point gate does NOT verify physical correspondence or independence.
    """
    import cv2

    policy = _point_gate_policy(max_check_p95_px,tolerance_note)
    context = _context(frame, rgb_path)
    geometry = _rgb_geometry(context,rgb_calibration)
    if (not isinstance(controls, dict) or controls.get('schema') != 'pvdelta.local-rgb-controls/v1'
            or controls.get('pair') != context or not isinstance(controls.get('points'), list)
            or not 12 <= len(controls['points']) <= 256):
        raise ValueError('Source-bound local RGB controls with 12..256 declared points required')
    provenance = controls.get('provenance')
    if (not isinstance(provenance, dict) or not isinstance(provenance.get('kind'), str)
            or not isinstance(provenance.get('note'), str) or not provenance['note'].strip()):
        raise ValueError('Explicit control provenance and correspondence limitations required')
    _canonical(controls)
    points = controls['points']
    if any(not isinstance(p, dict) for p in points):
        raise ValueError('Control points must be objects')
    ids = [p.get('control_id') for p in points]
    if (any(not isinstance(i, str) or not i.strip() for i in ids) or len(set(ids)) != len(ids)
            or any(p.get('role') not in ('fit', 'check') for p in points)):
        raise ValueError('Unique control IDs and explicit fit/check roles required')
    native = _points([p['native_xy'] for p in points], context['native_shape'])
    rgb = _points([p['rgb_xy'] for p in points], context['rgb_shape'])
    model_rgb = _rgb_model_points(rgb,geometry)
    for q in (native, rgb):
        distance = np.linalg.norm(q[:, None]-q[None, :], axis=2)
        np.fill_diagonal(distance, np.inf)
        if (distance < 1).any():
            raise ValueError('Distinct source observations required; no duplicated fit/check points')
    train = np.array([p['role'] == 'fit' for p in points])
    if int(train.sum()) < 8 or int((~train).sum()) < 4:
        raise ValueError('At least eight fit and four held-out check controls required')
    for q in (native[train], rgb[train], native[~train], rgb[~train]):
        if np.linalg.matrix_rank(q-q.mean(axis=0)) < 2:
            raise ValueError('Fit and check controls each require two-dimensional coverage')
    model = fit_homography(native[train], model_rgb[train])
    if abs(np.linalg.det(model.matrix)) < 1e-12:
        raise ValueError('Singular local pair model')
    validation = validate_homography_checks(model.matrix, fit_point_count=int(train.sum()),
        check_source_xy=native[~train], check_target_xy=model_rgb[~train],
        max_p95_source_px=float(max_check_p95_px), min_fit_points=8)
    predicted_native=transform_points(model_rgb,np.linalg.inv(model.matrix))
    residuals=[{'control_id':identifier,'role':point['role'],'observed_native_xy':n.tolist(),
        'observed_rgb_xy':r.tolist(),'predicted_native_xy':p.tolist(),'error_xy_native_px':(p-n).tolist(),
        'error_native_px':float(np.linalg.norm(p-n))} for identifier,point,n,r,p in zip(ids,points,native,rgb,predicted_native)]
    result = {'schema': 'pvdelta.local-rgb-registration/v1', 'pair': context,
        'point_gate_policy': policy,
        'rgb_geometry':geometry,'control_residuals':residuals,
        'engine_sources': {name: _sha(Path(__file__).with_name(name).read_bytes()) for name in
            ('local_rgb_transfer.py','alignment.py','convex_geometry.py','table_surface.py')},
        'controls_sha256': _sha(_canonical(controls)), 'provenance': provenance,
        'fit_control_ids': [i for i,t in zip(ids,train) if t],
        'check_control_ids': [i for i,t in zip(ids,train) if not t],
        'geometry_gate_passed': validation.accepted, 'validation': asdict(validation),
        'native_to_rgb': model.matrix.tolist() if validation.accepted and geometry['model']=='original_rgb_pinhole' else None,
        'native_to_undistorted_rgb': model.matrix.tolist() if validation.accepted and geometry['model']=='brown_undistorted_rgb' else None,
        'fit_hull_native_xy': cv2.convexHull(np.float32(native[train])).reshape(-1,2).tolist(),
        'check_hull_native_xy': cv2.convexHull(np.float32(native[~train])).reshape(-1,2).tolist(),
        'support_policy': 'all_native_corners_inside_both_fit_and_check_hulls',
        'correspondence_verified': False, 'independent_review_verified': False,
        'measurement_authorised': False, 'main_scan_updated': False, 'database_written': False,
        'scope': 'Local source-pair development geometry; not automatic calibration, panel identity or field acceptance'}
    result = json.loads(_canonical(result))
    result['registration_id'] = 'sha256:'+_sha(_canonical(result))
    return result


def transfer_local_rgb_boundaries(frame, rgb_path, registration, rgb_proposals):
    """Keep only full RGB quads inside the validated local control support.

    All proposals retain a decision, including failed models and out-of-support
    geometry. No clipping, rounding, source-temperature warp, edge-based refit,
    grid filling, stale GPS or extrapolated mask is allowed. Returned quads are
    provisional geometry, never measurement-authorised or scan-activated.
    """
    import cv2

    context = _context(frame, rgb_path)
    if (not isinstance(registration, dict) or registration.get('schema') != 'pvdelta.local-rgb-registration/v1'
            or registration.get('pair') != context):
        raise ValueError('Registration must match these exact source files and native grid')
    payload = {k:v for k,v in registration.items() if k != 'registration_id'}
    if registration.get('registration_id') != 'sha256:'+_sha(_canonical(payload)):
        raise ValueError('Registration content identity mismatch')
    if not isinstance(rgb_proposals, list) or len(rgb_proposals) > 512:
        raise ValueError('At most 512 explicit RGB proposals required')
    if any(not isinstance(p, dict) for p in rgb_proposals):
        raise ValueError('RGB proposals must be objects')
    ids = [p.get('proposal_id') for p in rgb_proposals]
    if any(not isinstance(i,str) or not i.strip() for i in ids) or len(set(ids)) != len(ids):
        raise ValueError('Unique RGB proposal IDs required')
    accepted, decisions = [], []
    for proposal in rgb_proposals:
        identifier = proposal['proposal_id']
        q = _points(proposal['polygon_rgb_xy'], context['rgb_shape'])
        if q.shape != (4,2):
            raise ValueError('Four full RGB corners required')
        convex_iou(q,q)
        reason = None
        moved = None
        if not registration['geometry_gate_passed']:
            reason = 'local_registration_check_failed'
        else:
            matrix = np.linalg.inv(_native_matrix(registration))
            geometry=registration.get('rgb_geometry',{'model':'original_rgb_pinhole'})
            model_q=_rgb_model_points(q,geometry)
            raw_perimeter=np.concatenate([a+np.linspace(0,1,17)[:,None]*(b-a)
                for a,b in zip(q,np.roll(q,-1,axis=0))])
            model_perimeter=_rgb_model_points(raw_perimeter,geometry)
            denominators = np.column_stack((model_perimeter,np.ones(len(model_perimeter)))) @ matrix[2]
            if np.any(np.abs(denominators) < 1e-9) or np.ptp(np.sign(denominators)):
                reason = 'projective_pole_crosses_proposal'
            else:
                moved = transform_points(model_q,matrix)
                perimeter=transform_points(model_perimeter,matrix)
                # Four-corner masks must approximate the nonlinear edge, not just its endpoints.
                deviation=[]
                for i,(a,b) in enumerate(zip(moved,np.roll(moved,-1,axis=0))):
                    v=b-a;d=perimeter[i*17:(i+1)*17]-a
                    deviation.extend(np.abs(d[:,0]*v[1]-d[:,1]*v[0])/np.linalg.norm(v))
                curvature=float(max(deviation))
                h,w = context['native_shape']
                if ((perimeter < 2).any() or (perimeter[:,0] >= w-2).any() or (perimeter[:,1] >= h-2).any()):
                    reason = 'outside_native_frame_margin'
                elif curvature > .5:
                    reason = 'nonlinear_edge_exceeds_quad_tolerance'
                elif any(cv2.pointPolygonTest(np.float32(registration[key]),tuple(map(float,p)),True) < 0
                         for key in ('fit_hull_native_xy','check_hull_native_xy') for p in perimeter):
                    reason = 'outside_local_fit_or_check_support'
        status = reason or 'provisional_local_rgb_geometry'
        decisions.append({'proposal_id':identifier,'status':status,
            'polygon_rgb_xy':q.tolist(), 'polygon_native_xy':moved.tolist() if moved is not None else None})
        if reason is None:
            convex_iou(moved,moved)
            accepted.append({'proposal_id':identifier,'polygon_native_xy':moved.tolist(),
                'polygon_rgb_xy':q.tolist(), 'proposal_origin':'source_bound_local_rgb_transfer',
                'registration_id':registration['registration_id'],
                'semantic_panel_membership':'unverified','measurement_authorised':False})
    if _context(frame,rgb_path) != context:
        raise RuntimeError('Original source/native context changed during RGB transfer')
    return {'schema':'pvdelta.local-rgb-boundaries/v1','pair':context,
        'registration_id':registration['registration_id'],'proposals':accepted,'decisions':decisions,
        'requested_proposals':len(rgb_proposals),'supported_proposals':len(accepted),
        'geometry_gate_passed':registration['geometry_gate_passed'],
        'source_sha256':context['source_sha256'],'native_shape':context['native_shape'],
        'native_temperature_sha256':context['native_temperature_sha256'],
        'radiometric_resampling':False,'correspondence_verified':False,
        'measurement_authorised':False,'database_written':False,'main_scan_updated':False}


def propose_observed_junction_transfer(frame, rgb_path, matched_modules, rgb_proposals, *,
                                      full_module_edge_support=False, max_check_p95_px=2.,
                                      tolerance_note=None):
    """Run guarded local geometry per explicit table on this original source pair.

    Inputs are bounded provisional correspondences, not physical identity labels.
    Every requested proposal retains a decision. No failed table borrows another
    table's transform; no model crosses source pairs. Temperatures remain native.
    """
    import cv2

    from pvdelta.junction_registration import (
        JunctionConfig,
        match_table_junctions,
        radiometric_structure,
    )

    policy = _point_gate_policy(max_check_p95_px,tolerance_note)
    config = JunctionConfig(module_side_only=True, measurement_control_split=True,
        full_module_edge_support=full_module_edge_support,max_check_p95_px=max_check_p95_px)
    context = _context(frame, rgb_path)
    if (not isinstance(matched_modules, list) or not isinstance(rgb_proposals, list)
            or len(matched_modules) > 512 or len(rgb_proposals) > 512):
        raise ValueError('At most512 matched modules and RGB proposals required')
    ids = [p.get('proposal_id') for p in rgb_proposals]
    if any(not isinstance(i, str) or not i.strip() for i in ids) or len(set(ids)) != len(ids):
        raise ValueError('Unique RGB proposal IDs required')

    def table_key(value):
        if isinstance(value, bool) or not isinstance(value, (str, int)) or value == '':
            return None
        return type(value).__name__, value

    encoded = np.frombuffer(Path(rgb_path).read_bytes(), np.uint8)
    gray = (cv2.cvtColor(cv2.imdecode(encoded,cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION),cv2.COLOR_BGR2GRAY)
            if full_module_edge_support else
            cv2.imdecode(encoded,cv2.IMREAD_GRAYSCALE|cv2.IMREAD_IGNORE_ORIENTATION))
    structure, _ = radiometric_structure(frame.temperature_c)
    grouped = {}
    decisions, accepted, groups = [], [], []
    for proposal in rgb_proposals:
        key = table_key(proposal.get('table_index'))
        if key is None:
            decisions.append({'proposal_id': proposal['proposal_id'], 'status': 'explicit_table_unavailable'})
        else:
            grouped.setdefault(key, []).append(proposal)
    for key, proposals in grouped.items():
        matches = [m for m in matched_modules if table_key(m.get('rgb_table')) == key]
        group = {'table_index': key[1], 'registration': None, 'junctions': None}
        groups.append(group)
        reason = 'no_bounded_same_table_matches'
        if matches:
            junctions = match_table_junctions(gray, structure, [m['rgb_polygon'] for m in matches],
                [m['native_polygon'] for m in matches],
                config=config)
            group['junctions'] = junctions
            reason = 'junction_registration_failed'
            if junctions['image_consistent']:
                points = [{'control_id': f'J{i}', 'native_xy': junctions['nodes'][i]['native_xy'],
                           'rgb_xy': junctions['nodes'][i]['rgb_xy'], 'role': role}
                    for role, field in (('fit', 'train_node_indices'), ('check', 'check_node_indices'))
                    for i in junctions[field]]
                controls = {'schema': 'pvdelta.local-rgb-controls/v1', 'pair': context,
                    'points': points, 'provenance': {'kind': 'automatic_development_junctions',
                    'note': 'Fixed2:1 control split; physical identity and independent correspondence unverified.'}}
                registration = fit_local_rgb_registration(frame, rgb_path, controls,
                    max_check_p95_px=max_check_p95_px,tolerance_note=tolerance_note)
                group['registration'] = registration
                transferred = transfer_local_rgb_boundaries(frame, rgb_path, registration, proposals)
                accepted.extend(transferred['proposals'])
                decisions.extend(transferred['decisions'])
                continue
        decisions.extend({'proposal_id': p['proposal_id'], 'status': reason} for p in proposals)
    if _context(frame, rgb_path) != context:
        raise RuntimeError('Original source/native context changed during junction transfer')
    return {'schema': 'pvdelta.observed-junction-transfer/v1', 'pair': context,
        'point_gate_policy':policy,
        'groups': groups, 'proposals': accepted, 'decisions': decisions,
        'requested_proposals': len(rgb_proposals), 'supported_proposals': len(accepted),
        'measurement_authorised': False, 'correspondence_verified': False,
        'radiometric_resampling': False, 'database_written': False, 'main_scan_updated': False}
