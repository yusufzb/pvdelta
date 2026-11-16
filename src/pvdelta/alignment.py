"""Explicit point-based projective mapping between image coordinate systems."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HomographyFit:
    matrix: np.ndarray
    rmse_px: float
    max_error_px: float
    control_point_count: int


@dataclass(frozen=True)
class GridTransform:
    """Reversible mapping between pixel-centre coordinate systems.

    ``source_shape`` and ``target_shape`` use NumPy's ``(height, width)`` order.
    Point arrays use ``(x, y)``.  The optional rotation is applied clockwise
    before scaling, matching a displayed image whose EXIF orientation has
    already been normalized.
    """

    source_shape: tuple[int, int]
    target_shape: tuple[int, int]
    clockwise_quarter_turns: int
    matrix: np.ndarray

    @property
    def inverse_matrix(self) -> np.ndarray:
        return np.linalg.inv(self.matrix)

    def forward(self, points_xy: np.ndarray) -> np.ndarray:
        return transform_points(points_xy, self.matrix)

    def inverse(self, points_xy: np.ndarray) -> np.ndarray:
        return transform_points(points_xy, self.inverse_matrix)


@dataclass(frozen=True)
class RegistrationValidation:
    """Independent-check error for a source-to-target homography."""

    accepted: bool
    fit_point_count: int
    check_point_count: int
    check_rmse_source_px: float
    check_p95_source_px: float
    check_max_source_px: float
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class BoundaryTransferAssessment:
    """Quality gate for a transferred polygon before thermal measurement."""

    accepted: bool
    inside_frame: bool
    edge_score_px: float
    translation_dx_px: float
    translation_dy_px: float
    reasons: tuple[str, ...]


def assess_boundary_transfer(
    polygon_xy: np.ndarray,
    *,
    frame_shape: tuple[int, int],
    edge_score_px: float,
    translation_xy: tuple[float, float],
    max_edge_score_px: float = 8.0,
    max_translation_px: float = 12.0,
    frame_margin_px: float = 2.0,
) -> BoundaryTransferAssessment:
    """Gate a transferred polygon using native-frame and edge-fit constraints.

    The function deliberately does not inspect pixels. Callers obtain ``edge_score_px``
    from their native-image edge matcher, then use this deterministic gate before
    producing any temperature statistic.
    """
    polygon = np.asarray(polygon_xy, dtype=float)
    if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3 or not np.isfinite(polygon).all():
        raise ValueError("polygon_xy must be a finite Nx2 array with at least three points")
    if len(frame_shape) != 2 or min(frame_shape) <= 0:
        raise ValueError("frame_shape must contain positive height and width")
    values = np.asarray((edge_score_px, *translation_xy, max_edge_score_px,
                         max_translation_px, frame_margin_px), dtype=float)
    if not np.isfinite(values).all() or max_edge_score_px < 0 or max_translation_px < 0 or frame_margin_px < 0:
        raise ValueError("transfer thresholds and fit values must be finite and non-negative")
    height, width = frame_shape
    inside = bool(
        (polygon[:, 0] >= frame_margin_px).all()
        and (polygon[:, 0] < width - frame_margin_px).all()
        and (polygon[:, 1] >= frame_margin_px).all()
        and (polygon[:, 1] < height - frame_margin_px).all()
    )
    dx, dy = float(translation_xy[0]), float(translation_xy[1])
    reasons: list[str] = []
    if not inside:
        reasons.append("polygon is outside the native frame margin")
    if edge_score_px > max_edge_score_px:
        reasons.append(f"edge score {edge_score_px:.3f} px exceeds {max_edge_score_px:.3f} px")
    if max(abs(dx), abs(dy)) > max_translation_px:
        reasons.append(f"translation ({dx:.3f}, {dy:.3f}) px exceeds {max_translation_px:.3f} px")
    return BoundaryTransferAssessment(not reasons, inside, float(edge_score_px), dx, dy, tuple(reasons))


def planar_motion_to_image_shift(
    east_m: float,
    north_m: float,
    *,
    gimbal_yaw_deg: float,
    ground_sample_distance_m_px: float,
) -> tuple[float, float]:
    """Convert nadir camera ground motion into an image translation estimate.

    DJI yaw is clockwise from north; image x points right and image y down. The
    scene moves opposite camera motion. This is an initial pose correction only;
    boresight and panel-plane registration still need a bounded image check.
    """
    values = np.asarray((east_m, north_m, gimbal_yaw_deg, ground_sample_distance_m_px), dtype=float)
    if not np.isfinite(values).all() or ground_sample_distance_m_px <= 0:
        raise ValueError("motion, yaw and ground sample distance must be finite; GSD must be positive")
    yaw = np.radians(gimbal_yaw_deg)
    right_east, right_north = -np.sin(yaw), np.cos(yaw)
    down_east, down_north = np.cos(yaw), np.sin(yaw)
    dx = -(east_m * right_east + north_m * right_north) / ground_sample_distance_m_px
    dy = -(east_m * down_east + north_m * down_north) / ground_sample_distance_m_px
    return float(dx), float(dy)


def pixel_grid_transform(
    source_shape: tuple[int, int],
    target_shape: tuple[int, int],
    *,
    clockwise_quarter_turns: int = 0,
    require_uniform_scale: bool = True,
) -> GridTransform:
    """Build a pixel-centre transform with explicit orthogonal orientation.

    A 2× mapping sends native centre ``(0, 0)`` to preview centre
    ``(0.5, 0.5)``.  Pixel-edge coordinates therefore scale by exactly two,
    while pixel-centre coordinates carry the half-pixel offset.
    """
    if (
        len(source_shape) != 2
        or len(target_shape) != 2
        or min(source_shape) <= 0
        or min(target_shape) <= 0
    ):
        raise ValueError("source_shape and target_shape must be positive (height, width) pairs")
    turns = clockwise_quarter_turns % 4
    source_h, source_w = source_shape
    if turns == 0:
        rotation = np.eye(3)
        rotated_h, rotated_w = source_h, source_w
    elif turns == 1:
        rotation = np.array([[0, -1, source_h - 1], [1, 0, 0], [0, 0, 1]], dtype=float)
        rotated_h, rotated_w = source_w, source_h
    elif turns == 2:
        rotation = np.array(
            [[-1, 0, source_w - 1], [0, -1, source_h - 1], [0, 0, 1]], dtype=float
        )
        rotated_h, rotated_w = source_h, source_w
    else:
        rotation = np.array([[0, 1, 0], [-1, 0, source_w - 1], [0, 0, 1]], dtype=float)
        rotated_h, rotated_w = source_w, source_h

    target_h, target_w = target_shape
    scale_x = target_w / rotated_w
    scale_y = target_h / rotated_h
    if require_uniform_scale and not np.isclose(scale_x, scale_y, rtol=0, atol=1e-12):
        raise ValueError("source and target grids do not have the same aspect ratio")
    scale = np.array(
        [[scale_x, 0, (scale_x - 1) / 2], [0, scale_y, (scale_y - 1) / 2], [0, 0, 1]],
        dtype=float,
    )
    matrix = scale @ rotation
    matrix.setflags(write=False)
    return GridTransform(source_shape, target_shape, turns, matrix)


def validate_homography_checks(
    matrix: np.ndarray,
    *,
    fit_point_count: int,
    check_source_xy: np.ndarray,
    check_target_xy: np.ndarray,
    max_p95_source_px: float,
    min_fit_points: int = 4,
    min_check_points: int = 4,
) -> RegistrationValidation:
    """Gate a homography using points excluded from fitting.

    Error is reported back in source pixels so callers can convert a thermal
    preview result to native pixels explicitly.  The function checks geometry,
    not the physical identity of repetitive structures; callers must use
    asymmetric borders, corners or other independently reviewed anchors.
    """
    source = np.asarray(check_source_xy, dtype=float)
    target = np.asarray(check_target_xy, dtype=float)
    reasons = []
    if fit_point_count < min_fit_points:
        reasons.append(f"requires at least {min_fit_points} fit points")
    if source.shape != target.shape or source.ndim != 2 or source.shape[1:] != (2,):
        raise ValueError("check source and target points must have matching Nx2 shape")
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("check points must be finite")
    if len(source) < min_check_points:
        reasons.append(f"requires at least {min_check_points} independent check points")
    try:
        predicted_source = transform_points(target, np.linalg.inv(np.asarray(matrix, dtype=float)))
        errors = np.linalg.norm(predicted_source - source, axis=1)
    except (ValueError, np.linalg.LinAlgError):
        reasons.append("homography is singular or maps a check point to infinity")
        errors = np.full(len(source), np.inf)
    if len(errors):
        rmse = float(np.sqrt(np.mean(errors**2)))
        p95 = float(np.percentile(errors, 95))
        maximum = float(np.max(errors))
    else:
        rmse = p95 = maximum = float("inf")
    if not np.isfinite(p95) or p95 > max_p95_source_px:
        reasons.append(f"check P95 {p95:.3f} px exceeds {max_p95_source_px:.3f} px")
    return RegistrationValidation(
        not reasons, fit_point_count, len(source), rmse, p95, maximum, tuple(reasons)
    )


def fit_homography(source_xy: np.ndarray, target_xy: np.ndarray) -> HomographyFit:
    """Normalized DLT on fit controls only; errors retain original target pixels.

    Centre and isotropically scale each coordinate domain before the homogeneous
    solve. Raw pixel origins/scales must not bias a noisy fit. This is not robust
    outlier removal, a correspondence check or permission to fit check points.
    """
    source = np.asarray(source_xy, dtype=float)
    target = np.asarray(target_xy, dtype=float)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 2:
        raise ValueError("source and target points must have matching Nx2 shape")
    if len(source) < 4 or not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("at least four finite control-point pairs are required")
    def normalize(points):
        centre = points.mean(axis=0)
        centred = points-centre
        distance = float(np.linalg.norm(centred, axis=1).mean())
        if not np.isfinite(distance) or distance <= 0:
            raise ValueError('control points have no finite spatial spread')
        normalized = centred*(np.sqrt(2)/distance)
        if np.linalg.matrix_rank(normalized) < 2:
            raise ValueError('control points have degenerate two-dimensional spread')
        scale = np.sqrt(2)/distance
        transform = np.array([[scale, 0, -scale*centre[0]],
                              [0, scale, -scale*centre[1]], [0, 0, 1]])
        return normalized, transform

    normalized_source, source_transform = normalize(source)
    normalized_target, target_transform = normalize(target)
    rows = []
    for (x, y), (u, v) in zip(normalized_source, normalized_target, strict=True):
        rows.extend(([-x, -y, -1, 0, 0, 0, u*x, u*y, u],
                     [0, 0, 0, -x, -y, -1, v*x, v*y, v]))
    design = np.asarray(rows, dtype=float)
    # 4 pairs produce an 8x9 design: the full basis is needed for its null vector.
    # Larger fits need only 9 vectors, avoiding a quadratic-size left SVD basis.
    _, singular, vh = np.linalg.svd(design, full_matrices=len(rows) < 9)
    tolerance = max(design.shape)*np.finfo(float).eps*singular[0]
    if int((singular > tolerance).sum()) < 8:
        raise ValueError('control points produce a rank-deficient homography')
    matrix = np.linalg.solve(target_transform, vh[-1].reshape(3, 3)) @ source_transform
    if not np.isfinite(matrix).all() or abs(matrix[2, 2]) < 1e-12:
        raise ValueError("control points produce a degenerate homography")
    matrix /= matrix[2, 2]
    if np.linalg.matrix_rank(matrix) < 3:
        raise ValueError('control points produce a degenerate homography')
    predicted = transform_points(source, matrix)
    errors = np.linalg.norm(predicted - target, axis=1)
    return HomographyFit(matrix, float(np.sqrt(np.mean(errors**2))),
                         float(np.max(errors)), len(source))


def transform_points(points_xy: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    points = np.asarray(points_xy, dtype=float)
    homography = np.asarray(matrix, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or homography.shape != (3, 3):
        raise ValueError("points must be Nx2 and matrix must be 3x3")
    homogeneous = np.column_stack((points, np.ones(len(points)))) @ homography.T
    if np.any(np.abs(homogeneous[:, 2]) < 1e-12):
        raise ValueError("point maps to infinity")
    return homogeneous[:, :2] / homogeneous[:, 2, None]
