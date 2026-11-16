"""Image-supported, provisional panel correspondence for thermal evidence review.

Only RGB features establish correspondence. Native thermal polygons and temperatures
remain those recorded for the target observation. No strongest-finding substitution.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from functools import lru_cache
from itertools import permutations
from pathlib import Path

import numpy as np

from pvdelta.convex_geometry import convex_iou


def repeat_evidence_signature(database, finding_id, site_profile):
    """Invalidate UI evidence when stored measurements, geometry or source files change."""
    import hashlib

    from pvdelta.store import connect_store

    site = Path(site_profile).expanduser().resolve() if site_profile is not None else None
    hasher = hashlib.sha256(site.read_bytes() if site is not None else b'manual-pairs-without-panel-association')
    connection = connect_store(database, read_only=True)
    try:
        source = connection.execute('''SELECT f.analysis_run_id,c.run_id FROM findings f
            JOIN module_observations m USING(observation_id) JOIN captures c ON c.capture_id=m.capture_id
            WHERE f.finding_id=?''', [finding_id]).fetchone()
        if source is None:
            raise ValueError('Finding no longer exists in the selected database')
        run, acquisition = source
        tables = {r[0] for r in connection.execute('SELECT table_name FROM information_schema.tables').fetchall()}
        queries = [
            (('SELECT finding_id,delta_t_k,absolute_temperature_c,pixel_polygon_json,region_pixels_yx_json '
              'FROM findings WHERE analysis_run_id=? ORDER BY finding_id'), [run]),
            (('SELECT observation_id,assessment_status,temperature_median_c,temperature_max_c '
              'FROM module_observations WHERE analysis_run_id=? ORDER BY observation_id'), [run]),
        ]
        if 'measurement_profiles' in tables:
            queries.append((('SELECT capture_id,profile_id,result_fingerprint FROM measurement_profiles '
                             'WHERE analysis_run_id=? ORDER BY capture_id'), [run]))
        if 'flight_screening_frames' in tables:
            queries.append((('SELECT capture_id,md5(CAST(diagnostics_json AS VARCHAR)) FROM flight_screening_frames '
                             'WHERE analysis_run_id=? ORDER BY capture_id'), [run]))
        for sql, params in queries:
            hasher.update(json.dumps(connection.execute(sql, params).fetchall(),
                sort_keys=True, default=str, allow_nan=False).encode())
        captures = connection.execute('''SELECT c.capture_id,c.source_path,r.source_root,c.source_hash,
            c.capture_time_utc,c.paired_capture_id,c.gimbal_pitch_deg,c.gimbal_yaw_deg,c.gimbal_roll_deg,
            c.laser_target_distance_m,c.laser_target_latitude,c.laser_target_longitude,c.width_px,c.height_px
            FROM captures c LEFT JOIN inspection_runs r ON r.run_id=c.run_id
            WHERE c.run_id=? ORDER BY c.capture_id''', [acquisition]).fetchall()
        hasher.update(json.dumps(captures, default=str, allow_nan=False).encode())
        for row in captures:
            path = Path(row[1]) if Path(row[1]).is_absolute() else Path(row[2] or '')/row[1]
            try:
                stat = path.stat()
                state = [str(path.resolve()), stat.st_size, stat.st_mtime_ns]
            except OSError:
                state = [str(path), 'unavailable']
            hasher.update(json.dumps(state).encode())
    finally:
        connection.close()
    for name in ('repeat_views.py', 'measurement_profile.py', 'observation_location.py', 'geolocation.py'):
        hasher.update(Path(__file__).with_name(name).read_bytes())
    return hasher.hexdigest()


def nearby_image_pairs(database, finding_id, *, max_views=8):
    """Expose time-near source pairs for manual review, even without registration/site data.

    No panel association, temperature transfer, overlay, view angle or glare verdict
    is inferred. Selection includes the original frame and up to eight neighbours.
    """
    from pvdelta.store import connect_store

    if type(max_views) is not int or not 1 <= max_views <= 32:
        raise ValueError('max_views must be an integer in 1..32')
    connection = connect_store(database, read_only=True)
    try:
        source = connection.execute('''SELECT c.capture_id,c.run_id,epoch(c.capture_time_utc)
            FROM findings f JOIN module_observations m USING(observation_id)
            JOIN captures c ON c.capture_id=m.capture_id WHERE f.finding_id=?''', [finding_id]).fetchone()
        if source is None:
            raise ValueError('Finding no longer exists in this database')
        cid, acquisition, time_s = source
        result = connection.execute('''SELECT c.capture_id,c.source_path AS thermal_path,
            rgb.source_path AS rgb_path,r.source_root,c.gimbal_yaw_deg,c.gimbal_pitch_deg,
            abs(epoch(c.capture_time_utc)-?) AS time_gap_s
            FROM captures c JOIN captures rgb ON rgb.capture_id=c.paired_capture_id
            LEFT JOIN inspection_runs r ON r.run_id=c.run_id
            WHERE c.run_id=? AND c.image_kind='thermal'
              AND (c.capture_id=? OR abs(epoch(c.capture_time_utc)-?)<=30)
            ORDER BY CASE WHEN c.capture_id=? THEN 0 ELSE 1 END,time_gap_s,c.capture_id LIMIT ?''',
            [time_s, acquisition, cid, time_s, cid, max_views+1])
        columns = [c[0] for c in result.description]
        rows = [dict(zip(columns, row, strict=True)) for row in result.fetchall()]
    finally:
        connection.close()
    for row in rows:
        for key in ('thermal_path', 'rgb_path'):
            relative = Path(row[key])
            root = Path(row['source_root'] or '').expanduser().resolve()
            resolved = (root/relative).resolve()
            if not resolved.is_relative_to(root):
                raise ValueError('Indexed comparison image lies outside its source root')
            row[key] = str(resolved)
        row.update(status='unavailable', reason=('original capture; no cross-frame panel association'
            if row['capture_id'] == cid else 'nearby capture only; panel identity unverified'),
            panel_identity_verified=False, glare_evidence='unresolved')
    return rows


@dataclass(frozen=True)
class PanelMatch:
    status: str
    reason: str
    target_index: int | None = None
    iou: float | None = None
    check_p95_px: float | None = None
    check_count: int = 0
    fit_inliers: int = 0
    target_corner_indices: tuple[int, ...] | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _view_separation(first: dict, second: dict) -> float | None:
    vectors = []
    for row in (first, second):
        vector = np.asarray(row.get("view_direction_enu", []), dtype=float)
        if vector.shape != (3,) or not np.isfinite(vector).all():
            return None
        length = np.linalg.norm(vector)
        if length <= 0:
            return None
        vectors.append(vector / length)
    return math.degrees(math.acos(float(np.clip(np.dot(*vectors), -1, 1))))


def assess_glare_evidence(
    rows: list[dict], *, min_views: int = 2, min_angle_span_deg: float = 20.0,
) -> dict:
    """Summarise provisional component persistence using actual pixel-ray separation.

    Missing analysis is unknown; no recorded candidate is not evidence of a healthy
    panel or glare. The exploratory angle threshold is not a standards requirement.
    """
    if (isinstance(min_views, bool) or not isinstance(min_views, int) or min_views < 2
            or not math.isfinite(min_angle_span_deg) or not 0 < min_angle_span_deg < 180):
        raise ValueError("require min_views >= 2 and a finite angle threshold in (0, 180)")
    # Count each capture once, including the source, rather than duplicated evidence.
    unique = {row["capture_id"]: row for row in rows if row.get("capture_id")}
    matched = [row for row in unique.values()
               if row.get("status") in {"source", "matched_provisional"}]
    assessed = [row for row in matched if row.get("thermal_geometry_accepted") is True
                and (row.get("observation") or {}).get("assessment_status") == "analyzed"]
    def max_angle(population):
        return max((angle for i, a in enumerate(population) for b in population[i + 1:]
                    if (angle := _view_separation(a, b)) is not None), default=0.0)
    angle_span = max_angle(assessed)
    candidates = [row for row in assessed if row.get("component_evidence") == "present"]
    not_detected = [row for row in assessed if row.get("component_evidence") == "not_detected"]
    candidate_span = max_angle(candidates)
    source_present = any(row.get("status") == "source" and row in candidates for row in assessed)
    angles_available = sum(_view_separation(row, row) is not None for row in assessed)
    if len(matched) < min_views:
        status = "insufficient_views"
        reason = "source and another same-panel view are required"
    elif len(assessed) < min_views or angles_available < min_views or not source_present:
        status = "unresolved"
        reason = "source/component, native assessment or pixel-ray geometry is unavailable"
    elif angle_span < min_angle_span_deg:
        status = "angle_diversity_insufficient"
        reason = "actual pixel rays have insufficient angular separation"
    elif len(candidates) >= min_views and candidate_span >= min_angle_span_deg:
        status = "persistent_across_angles"
        reason = "corresponding candidate regions recur in separated provisional panel views"
    else:
        status = "unresolved"
        reason = "candidate persistence across separated views is unproven; glare remains unresolved"
    return {
        "status": status,
        "reason": reason,
        "matched_view_count": len(matched),
        "assessed_view_count": len(assessed),
        "unknown_view_count": len(matched) - len(assessed),
        "candidate_view_count": len(candidates),
        "not_detected_view_count": len(not_detected),
        "angle_span_deg": angle_span,
        "candidate_angle_span_deg": candidate_span,
        "min_angle_span_deg": float(min_angle_span_deg),
        "geometry_status": "provisional",
    }


def component_evidence(source_finding: dict, source_quad, target_findings: list[dict],
                       target_quad, *, min_iou: float = .15) -> dict:
    """Compare like-signed native candidate regions in panel-relative coordinates.

    This provisional spatial overlap does not establish an electrical fault. A missing
    or invalid polygon remains unknown, even when another candidate is present.
    """
    import cv2

    def normalise(polygon, quad):
        points = np.asarray(json.loads(polygon) if isinstance(polygon, str) else polygon,
                            dtype=np.float32)
        panel = np.asarray(quad, dtype=np.float32)
        if (panel.shape != (4, 2) or points.ndim != 2 or points.shape[1] != 2
                or len(points) < 3 or not np.isfinite(points).all()
                or not np.isfinite(panel).all() or not cv2.isContourConvex(panel)
                or abs(cv2.contourArea(panel)) < 1e-6):
            raise ValueError("invalid component or panel geometry")
        transform = cv2.getPerspectiveTransform(panel, np.array(
            [[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32))
        result = cv2.perspectiveTransform(points[:, None], transform)[:, 0]
        if (not np.isfinite(result).all() or not cv2.isContourConvex(result)
                or abs(cv2.contourArea(result)) < 1e-9
                or (result < -.01).any() or (result > 1.01).any()):
            raise ValueError("component outside panel or unsupported component polygon")
        return result

    try:
        source = normalise(source_finding['pixel_polygon_json'], source_quad)
        direction = math.copysign(1, float(source_finding['delta_t_k']))
        if not math.isfinite(float(source_finding['delta_t_k'])):
            raise ValueError("invalid source delta")
        best, best_id = 0.0, None
        for finding in target_findings:
            target = normalise(finding['pixel_polygon_json'], target_quad)
            delta = float(finding['delta_t_k'])
            if not math.isfinite(delta):
                raise ValueError("invalid target delta")
            if math.copysign(1, delta) != direction:
                continue
            overlap = convex_iou(source, target)
            if overlap > best:
                best, best_id = overlap, finding['finding_id']
        return {"component_evidence": "present" if best >= min_iou else "not_detected",
                "component_overlap": best,
                "matching_finding_id": best_id if best >= min_iou else None}
    except (ValueError, TypeError, KeyError, cv2.error):
        return {"component_evidence": "unknown", "component_overlap": None,
                "matching_finding_id": None}


@lru_cache(maxsize=24)
def _features(path: str, mtime_ns: int, size: int, source_sha256: str):
    import cv2
    from PIL import Image

    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"))
    scale = min(1.0, 1600 / max(rgb.shape[:2]))
    rgb = cv2.resize(rgb, (round(rgb.shape[1] * scale), round(rgb.shape[0] * scale)))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    # Repetitive module cells must not dominate the coarse motion estimate.
    # Vegetation texture is useful here as background geometry, never as a hotspot.
    r, g, b = rgb.astype(float).transpose(2, 0, 1)
    background = ((g > r * 1.10) & (g > b * 1.20)).astype(np.uint8) * 255
    points, descriptors = cv2.SIFT_create(nfeatures=8000).detectAndCompute(gray, background)
    return np.array([p.pt for p in points], dtype=np.float32).reshape(-1, 2), descriptors, scale, gray



def _load_features(path: Path):
    import hashlib

    stat = path.stat()
    return _features(str(path.resolve()), stat.st_mtime_ns, stat.st_size,
                     hashlib.sha256(path.read_bytes()).hexdigest())


def location_consistency(source: dict, target: dict, *, max_distance_m: float = 1.5) -> dict:
    """Independent coarse location veto, never a substitute for image matching."""
    values = [source.get('latitude'), source.get('longitude'),
              target.get('latitude'), target.get('longitude')]
    if any(v is None or not math.isfinite(float(v)) for v in values):
        return {"location_consistent": False, "location_distance_m": None}
    lat_a, lon_a, lat_b, lon_b = map(float, values)
    distance = 111_320 * math.hypot(lat_a-lat_b, (lon_a-lon_b)*math.cos(math.radians(lat_a)))
    return {"location_consistent": distance <= max_distance_m, "location_distance_m": distance}


def select_target_panel(
    projected_quad: np.ndarray, target_polygons: list[np.ndarray],
    *, min_iou: float = 0.70, ambiguity_iou: float = 0.30,
) -> tuple[int | None, float]:
    """Reject ambiguous overlap rather than assigning a neighbouring panel."""
    import cv2

    quad = np.asarray(projected_quad, dtype=np.float32)
    if quad.shape != (4, 2) or not np.isfinite(quad).all() or not cv2.isContourConvex(quad):
        return None, 0.0
    scores = []
    for index, polygon in enumerate(target_polygons):
        other = np.asarray(polygon, dtype=np.float32)
        if other.shape != (4, 2) or not np.isfinite(other).all() or not cv2.isContourConvex(other):
            continue
        try:
            scores.append((convex_iou(quad, other), index))
        except ValueError:
            continue
    scores.sort(reverse=True)
    if not scores:
        return None, 0.0
    best, index = scores[0]
    if best < min_iou or (len(scores) > 1 and scores[1][0] > ambiguity_iou):
        return None, best
    return index, best


def _tracking_mask(polygon, support_polygons, shape, scale):
    """Optional observed same-plane patches; no bounding-box or gap filling."""
    import cv2

    mask = np.zeros(shape, np.uint8)
    if support_polygons is None:
        cv2.fillConvexPoly(mask, np.round(polygon * scale).astype(np.int32), 255)
        return mask
    if not isinstance(support_polygons, (list, tuple)) or not 1 <= len(support_polygons) <= 5:
        raise ValueError('One selected and at most four observed neighbour polygons required')
    quads = [np.asarray(p, float) for p in support_polygons]
    vertex_key = lambda p: tuple(sorted(map(tuple, p)))
    for q in quads:
        if (q.shape != (4, 2) or not np.isfinite(q).all()
                or not cv2.isContourConvex(q.astype(np.float32))
                or (q < 0).any() or (q*scale > [shape[1], shape[0]]).any()):
            raise ValueError('Observed support must be convex and inside the original source grid')
    if sum(vertex_key(q) == vertex_key(polygon) for q in quads) != 1:
        raise ValueError('Observed support must contain the selected panel exactly once')
    if any(convex_iou(a, b) > .05 for i, a in enumerate(quads) for b in quads[i+1:]):
        raise ValueError('Observed tracking support polygons conflict or duplicate each other')
    for q in quads:
        cv2.fillConvexPoly(mask, np.round(q*scale).astype(np.int32), 255)
    return mask


def match_panel(
    source_path: Path, target_path: Path,
    source_polygon: np.ndarray, target_polygons: list[np.ndarray],
    *, source_support_polygons=None,
) -> PanelMatch:
    """Fit local RGB geometry; validate held-out matches and unique panel overlap.

    Residuals use the feature image (long side <=1600 px). This establishes a
    provisional correspondence, not physical fault confirmation or glare clearance.
    An explicit same-table support set can supply additional tracking corners.
    Only its observed polygon interiors are used; selected-panel hull, held-out
    residual and unique target-overlap limits remain unchanged. The caller must
    establish the observed source-table association before fitting.
    """
    import cv2

    polygon = np.asarray(source_polygon, dtype=np.float32)
    if polygon.shape != (4, 2) or not np.isfinite(polygon).all():
        return PanelMatch("unavailable", "source panel geometry missing or invalid")
    if not target_polygons:
        return PanelMatch("unavailable", "no target module geometry")
    a, da, sa, ga = _load_features(source_path)
    b, db, sb, gb = _load_features(target_path)
    if da is None or db is None or min(len(a), len(b)) < 24:
        return PanelMatch("unavailable", "insufficient RGB features")
    # Track the selected module plane, not ground/other tilted tables in a disk.
    try:
        panel_mask = _tracking_mask(polygon, source_support_polygons, ga.shape, sa)
    except ValueError as exc:
        return PanelMatch('unavailable', 'Invalid observed table support: ' + str(exc))
    corners = cv2.goodFeaturesToTrack(
        ga, maxCorners=1000, qualityLevel=.005, minDistance=3, mask=panel_mask,
    )
    if corners is None or ga.shape != gb.shape:
        return PanelMatch("unavailable", "compatible RGB tracking grids unavailable")
    if len(corners) < 24:
        return PanelMatch("unavailable", "too few local RGB corners")
    # Global initialization handles large motion; overly coarse local pyramids
    # mix image borders/adjacent tables and can jump to another repetitive row.
    params = {"winSize": (31, 31), "maxLevel": 2,
              "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, .01)}
    # A global feature estimate initializes flow across flight-step displacement.
    # Final acceptance still uses independent local tracking and held-out geometry.
    pairs = cv2.BFMatcher().knnMatch(da, db, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < .70 * p[1].distance]
    initial = None
    if len(good) >= 16:
        coarse, support = cv2.findHomography(
            np.array([a[p.queryIdx] for p in good]),
            np.array([b[p.trainIdx] for p in good]), cv2.RANSAC, 3.0,
        )
        if coarse is not None and support.sum() >= 12:
            initial = cv2.perspectiveTransform(corners, coarse)
    if initial is None:
        return PanelMatch("unavailable", "global RGB displacement could not be initialized")
    aligned = cv2.warpPerspective(gb, np.linalg.inv(coarse), (ga.shape[1], ga.shape[0]))
    overlap = (aligned > 5) & (ga > 5) & (aligned < 250) & (ga < 250)
    if overlap.sum() < 100:
        return PanelMatch("unavailable", "insufficient unsaturated RGB overlap")
    qa = np.percentile(ga[overlap], [25, 50, 75])
    qb = np.percentile(aligned[overlap], [25, 50, 75])
    gain = (qa[2] - qa[0]) / max(1, qb[2] - qb[0])
    tracking_b = np.clip((aligned.astype(float) - qb[1]) * gain + qa[1], 0, 255).astype(np.uint8)
    # A coordinate initializer does not rotate LK's local patches on a return
    # track. Track on a coarse-aligned DISPLAY copy, then restore original RGB
    # coordinates. No thermal measurement array or anomaly region is warped.
    valid_pixels = cv2.warpPerspective(np.full_like(gb, 255), np.linalg.inv(coarse),
        (ga.shape[1], ga.shape[0]), flags=cv2.INTER_NEAREST)
    valid_pixels = cv2.erode(valid_pixels, np.ones((21, 21), np.uint8),
        borderType=cv2.BORDER_CONSTANT, borderValue=0)
    moved, ok, _ = cv2.calcOpticalFlowPyrLK(
        ga, tracking_b, corners, corners.copy(), flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **params,
    )
    if moved is None or ok is None:
        return PanelMatch("unavailable", "forward RGB optical flow unavailable")
    returned, back_ok, _ = cv2.calcOpticalFlowPyrLK(
        tracking_b, ga, moved, corners.copy(), flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **params,
    )
    if returned is None or back_ok is None:
        return PanelMatch("unavailable", "reverse RGB optical flow unavailable")
    valid = ((ok.ravel() == 1) & (back_ok.ravel() == 1)
             & (np.linalg.norm(returned[:, 0] - corners[:, 0], axis=1) < .75)
             & (corners[:, 0, 0] >= 11) & (corners[:, 0, 1] >= 11)
             & (corners[:, 0, 0] < ga.shape[1]-11) & (corners[:, 0, 1] < ga.shape[0]-11))
    # Compare normalized patches, not raw brightness error: camera exposure may vary.
    for i in np.flatnonzero(valid):
        tx, ty = np.rint(moved[i, 0]).astype(int)
        if not (0 <= tx < ga.shape[1] and 0 <= ty < ga.shape[0] and valid_pixels[ty, tx] == 255):
            valid[i] = False
            continue
        source_patch = cv2.getRectSubPix(ga, (21, 21), tuple(corners[i, 0])).astype(float)
        target_patch = cv2.getRectSubPix(aligned, (21, 21), tuple(moved[i, 0])).astype(float)
        source_patch -= source_patch.mean()
        target_patch -= target_patch.mean()
        denom = np.linalg.norm(source_patch) * np.linalg.norm(target_patch)
        valid[i] = denom > 1 and np.sum(source_patch * target_patch) / denom >= .80
    left = corners[valid, 0]
    if len(left) < 24:
        return PanelMatch("unavailable", "too few forward/backward checked RGB tracks")
    right = cv2.perspectiveTransform(moved[valid], coarse)[:, 0]
    # Freeze the split before fitting; held-out points never enter RANSAC/refit.
    check = np.arange(len(left)) % 4 == 0
    cv2.setRNGSeed(0)
    matrix, inliers = cv2.findHomography(left[~check], right[~check], cv2.RANSAC, 2.0)
    if matrix is None or inliers is None or inliers.sum() < 12:
        return PanelMatch("unavailable", "local RGB homography has insufficient support")
    projected_checks = cv2.perspectiveTransform(left[check, None], matrix)[:, 0]
    error = np.linalg.norm(projected_checks - right[check], axis=1)
    p95 = float(np.percentile(error, 95))
    evidence = {"check_p95_px": p95, "check_count": int(check.sum()),
                "fit_inliers": int(inliers.sum())}
    if not np.isfinite(p95) or p95 > 3.0:
        return PanelMatch("unavailable", "held-out RGB correspondence residual too large", **evidence)
    hull = cv2.convexHull(left[~check][inliers.ravel().astype(bool)])
    # Interior texture supports the panel plane; permit bounded extrapolation
    # to the frame edges, but require support around the inner 70% quadrilateral.
    support_quad = (polygon.mean(axis=0) + .70 * (polygon - polygon.mean(axis=0))) * sa
    if any(cv2.pointPolygonTest(hull, tuple(map(float, point)), True) < -2
           for point in support_quad):
        return PanelMatch("unavailable", "panel falls outside supported feature area", **evidence)
    projected = cv2.perspectiveTransform((polygon * sa)[:, None], matrix)[:, 0] / sb
    index, overlap = select_target_panel(projected, target_polygons)
    if index is None:
        return PanelMatch("unavailable", "target panel absent or ambiguous", iou=overlap, **evidence)
    # Keep physical corner correspondence through frame rotation. Independently
    # starting each quad at its image-top-left can compare different hot cells.
    target = np.asarray(target_polygons[index], dtype=np.float32)
    indices = min(permutations(range(4)),
                  key=lambda order: float(np.linalg.norm(projected - target[list(order)], axis=1).sum()))
    shortest_edge = np.linalg.norm(target - np.roll(target, 1, axis=0), axis=1).min()
    corner_error = np.linalg.norm(projected - target[list(indices)], axis=1).max()
    corner_indices = tuple(indices) if corner_error <= .25 * shortest_edge else None
    return PanelMatch("matched_provisional", "RGB geometry and unique panel overlap passed",
                      target_index=index, iou=overlap,
                      target_corner_indices=corner_indices, **evidence)


def build_repeat_evidence(
    database: Path, finding_id: str, *, max_views: int = 8,
    site_profile: Path | None = None,
) -> list[dict]:
    """Inspect nearby RGB pairs, including target modules without findings.

    The bounded search is explicit; this is not an exhaustive whole-flight track.
    Nothing is written to the analysis database or promoted to a confirmed match.
    """
    import cv2

    from pvdelta.store import connect_store

    if isinstance(max_views, bool) or not isinstance(max_views, int) or max_views < 1:
        raise ValueError("max_views must be a positive integer")

    def accepted_geometry(diag, panel):
        error = diag.get('heldout_p90_px')
        return (panel.get('accepted') is True and isinstance(error, (int, float))
                and math.isfinite(error) and error <= 5)

    if site_profile is None:
        return [{"status": "unavailable", "reason": "explicit site geometry profile required for location check"}]
    site = json.loads(site_profile.read_text(encoding='utf-8'))['tables']

    connection = connect_store(database, read_only=True)
    try:
        if not connection.execute("""SELECT count(*) FROM information_schema.tables
            WHERE table_name='flight_screening_frames'""").fetchone()[0]:
            return [{"status": "unavailable", "reason": "screening geometry unavailable in this database"}]
        def records(sql, params):
            result = connection.execute(sql, params)
            names = [col[0] for col in result.description]
            return [dict(zip(names, row, strict=True)) for row in result.fetchall()]

        sources = records("""
            SELECT f.analysis_run_id, m.frame_module_id, c.capture_id, c.run_id,
                   epoch(c.capture_time_utc) AS time_s, rgb.source_path AS rgb_path,
                   c.source_path AS thermal_path, c.width_px, c.height_px,
                   c.gimbal_yaw_deg, c.gimbal_pitch_deg,
                   f.finding_id, f.pixel_polygon_json, f.delta_t_k,
                   r.source_root, s.diagnostics_json
            FROM findings f JOIN module_observations m USING (observation_id)
            JOIN captures c ON c.capture_id=m.capture_id
            LEFT JOIN captures rgb ON rgb.capture_id=c.paired_capture_id
            LEFT JOIN inspection_runs r ON r.run_id=c.run_id
            JOIN flight_screening_frames s ON s.capture_id=c.capture_id
                AND s.analysis_run_id=f.analysis_run_id
            WHERE f.finding_id=?
        """, [finding_id])
        if not sources:
            return [{"status": "unavailable", "reason": "source screening geometry unavailable"}]
        source = sources[0]
        diagnostic = json.loads(source['diagnostics_json'])
        module = next((m for m in diagnostic.get('modules', [])
                       if m['module'] == source['frame_module_id']), {})
        if (not accepted_geometry(diagnostic, module) or not source['rgb_path']
                or source['time_s'] is None):
            return [{"status": "unavailable", "reason": "source RGB/accepted panel/time unavailable"}]

        def path(root, relative):
            return Path(root or '') / relative

        source_path = path(source['source_root'], source['rgb_path'])
        targets = records("""
            SELECT c.capture_id, c.source_path AS thermal_path, rgb.source_path AS rgb_path,
                   c.width_px, c.height_px, c.gimbal_yaw_deg, c.gimbal_pitch_deg,
                   abs(epoch(c.capture_time_utc)-?) AS time_gap_s,
                   r.source_root, s.diagnostics_json
            FROM captures c
            JOIN captures rgb ON rgb.capture_id=c.paired_capture_id
            LEFT JOIN inspection_runs r ON r.run_id=c.run_id
            LEFT JOIN flight_screening_frames s ON s.capture_id=c.capture_id
                AND s.analysis_run_id=?
            WHERE c.run_id=? AND c.image_kind='thermal' AND c.capture_id<>?
                AND abs(epoch(c.capture_time_utc)-?) <= 30
            ORDER BY time_gap_s, c.capture_id LIMIT ?
        """, [source['time_s'], source['analysis_run_id'], source['run_id'],
              source['capture_id'], source['time_s'], max_views])
        # Snapshot metadata before expensive image work; do not hold a DuckDB
        # reader lock while reviewers may need to write their decisions.
        snapshots = {}
        from pvdelta.measurement_profile import read_measurement_profile

        measurement_profiles = {}
        for target in [source, *targets]:
            measurement_profiles[target['capture_id']] = read_measurement_profile(
                connection, source['analysis_run_id'], target['capture_id'])
            observations = records("""
                SELECT frame_module_id, observation_id, assessment_status,
                       temperature_median_c, temperature_max_c FROM module_observations
                WHERE capture_id=? AND analysis_run_id=?
            """, [target['capture_id'], source['analysis_run_id']])
            findings = records("""
                SELECT m.frame_module_id, f.finding_id, f.pixel_polygon_json, f.delta_t_k
                FROM findings f JOIN module_observations m USING(observation_id)
                WHERE m.capture_id=? AND f.analysis_run_id=?
            """, [target['capture_id'], source['analysis_run_id']])
            snapshots[target['capture_id']] = (observations, findings)
        capture_geometry = {}
        for cid in [source['capture_id'], *(t['capture_id'] for t in targets)]:
            capture_geometry[cid] = records("""
                SELECT camera_model, width_px, height_px, laser_target_latitude,
                       laser_target_longitude, laser_target_distance_m,
                       gimbal_pitch_deg, gimbal_yaw_deg, gimbal_roll_deg
                FROM captures WHERE capture_id=?
            """, [cid])[0]
        connection.close()

        def locate(cid, polygon):
            from pvdelta.camera_db import lookup_by_xmp_model
            from pvdelta.observation_location import project_observation_center

            geometry = capture_geometry[cid]
            platform = lookup_by_xmp_model(geometry['camera_model'])
            if platform is None or platform.thermal is None:
                raise ValueError("thermal camera profile unavailable")
            return project_observation_center(
                polygon, target_latitude=geometry['laser_target_latitude'],
                target_longitude=geometry['laser_target_longitude'],
                laser_distance_m=geometry['laser_target_distance_m'],
                gimbal_pitch_deg=geometry['gimbal_pitch_deg'],
                gimbal_yaw_deg=geometry['gimbal_yaw_deg'],
                gimbal_roll_deg=geometry['gimbal_roll_deg'],
                image_width_px=geometry['width_px'], image_height_px=geometry['height_px'],
                diagonal_fov_deg=platform.thermal.dfov_deg,
                surface_tilt_deg=site['surface_tilt_deg'],
                surface_azimuth_deg=site['surface_facing_azimuth_true_deg'],
            )

        source_observations, source_findings = snapshots[source['capture_id']]
        source_observations = [o for o in source_observations
                               if o['frame_module_id'] == module['module']]
        source_row = {
            'status': 'source', 'reason': 'original candidate frame',
            'capture_id': source['capture_id'], 'module': module['module'],
            'measurement_profile': measurement_profiles[source['capture_id']],
            'rgb_path': str(source_path),
            'thermal_path': str(path(source['source_root'], source['thermal_path'])),
            'width_px': source['width_px'], 'height_px': source['height_px'],
            'gimbal_yaw_deg': source['gimbal_yaw_deg'],
            'gimbal_pitch_deg': source['gimbal_pitch_deg'], 'time_gap_s': 0,
            'rgb_polygon': module['rgb_polygon'], 'native_polygon': module['native_polygon'],
            'thermal_geometry_accepted': True,
            'observation': source_observations[0] if len(source_observations) == 1 else None,
            'findings': [f for f in source_findings if f['frame_module_id'] == module['module']],
            **component_evidence(source, module['native_polygon'], [source], module['native_polygon']),
        }
        try:
            source_location = locate(source['capture_id'], module['native_polygon'])
            source_row['view_direction_enu'] = source_location['view_direction_enu']
        except (ValueError, TypeError, KeyError):
            source_location = {}
        output = [source_row]
        for target in targets:
            diag = json.loads(target['diagnostics_json'] or '{}')
            modules = diag.get('modules', [])
            target_path = path(target['source_root'], target['rgb_path'])
            base = {k: v for k, v in target.items() if k != 'diagnostics_json'}
            base.update(rgb_path=str(target_path),
                        thermal_path=str(path(target['source_root'], target['thermal_path'])),
                        measurement_profile=measurement_profiles[target['capture_id']])
            try:
                match = match_panel(source_path, target_path, np.asarray(module['rgb_polygon']),
                                    [np.asarray(m['rgb_polygon']) for m in modules])
                base.update(match.to_dict())
                if match.target_index is not None:
                    matched = modules[match.target_index]
                    target_location = locate(target['capture_id'], matched['native_polygon'])
                    location = location_consistency(
                        source_location, target_location,
                    )
                    base.update(location)
                    if not location['location_consistent']:
                        base.update(status='unavailable', reason='RGB match disagrees with LRF panel location')
                        output.append(base)
                        continue
                    base['module'] = matched['module']
                    base['rgb_polygon'] = matched['rgb_polygon']
                    base['native_polygon'] = matched['native_polygon']
                    base['thermal_geometry_accepted'] = bool(
                        accepted_geometry(diag, matched)
                    )
                    base['view_direction_enu'] = target_location['view_direction_enu']
                    base['view_separation_from_source_deg'] = _view_separation(source_row, base)
                    all_observations, all_findings = snapshots[target['capture_id']]
                    observations = [o for o in all_observations
                                    if o['frame_module_id'] == matched['module']]
                    base['observation'] = observations[0] if len(observations) == 1 else None
                    base['findings'] = [f for f in all_findings
                                        if f['frame_module_id'] == matched['module']]
                    if (base['thermal_geometry_accepted']
                            and (base['observation'] or {}).get('assessment_status') == 'analyzed'
                            and match.target_corner_indices is not None):
                        corresponding_quad = np.asarray(matched['native_polygon'])[
                            list(match.target_corner_indices)]
                        base.update(component_evidence(source, module['native_polygon'],
                                                       base['findings'], corresponding_quad))
                    else:
                        base['component_evidence'] = 'unknown'
            except (OSError, ValueError, KeyError, TypeError, cv2.error) as exc:
                base.update(status='unavailable', reason=str(exc))
            output.append(base)
        glare_summary = assess_glare_evidence(output)
        for row in output:
            row["glare_summary"] = glare_summary
        return output
    finally:
        connection.close()
