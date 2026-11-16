"""Exploratory temperature-anomaly candidates, without fault diagnosis."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pvdelta.config import DetectionConfig
from pvdelta.models import Finding, ModuleObservation, ThermalFrame
from pvdelta.normalize import Condition, Extent, normalize
from pvdelta.quality import Capability, assess_region_quality
from pvdelta.reference import erode, robust_median
from pvdelta.regions import polygon_to_mask


def _components(candidate: np.ndarray) -> list[np.ndarray]:
    """Four-connected components as (y, x) coordinate arrays."""
    remaining = {tuple(pos) for pos in np.argwhere(candidate)}
    components = []
    while remaining:
        seed = remaining.pop()
        stack = [seed]
        positions = [seed]
        while stack:
            y, x = stack.pop()
            for neighbour in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if neighbour in remaining:
                    remaining.remove(neighbour)
                    stack.append(neighbour)
                    positions.append(neighbour)
        components.append(np.asarray(positions, dtype=np.int32))
    return components


def _bbox(coords: np.ndarray) -> list[tuple[float, float]]:
    """Half-open pixel bounding box, used as display evidence, not a segmentation polygon."""
    ys, xs = coords[:, 0], coords[:, 1]
    left, right = int(xs.min()), int(xs.max()) + 1
    top, bottom = int(ys.min()), int(ys.max()) + 1
    return [(left, top), (right, top), (right, bottom), (left, bottom)]


def _measurement_grid(frame: ThermalFrame, mask: np.ndarray, module: ModuleObservation) -> np.ndarray:
    if mask.shape != frame.temperature_c.shape or mask.dtype != np.bool_:
        raise ValueError("module mask must be boolean and match the thermal grid")
    envelope = polygon_to_mask(module.pixel_polygon, frame.temperature_c.shape)
    if np.any(mask & ~envelope):
        raise ValueError('measurement mask extends outside module polygon')
    valid = np.isfinite(frame.temperature_c)
    if frame.valid_mask is not None:
        valid &= frame.valid_mask
    return valid


def _robust_background(
    temperature: np.ndarray,
    mask: np.ndarray,
    threshold_k: float,
    iterations: int,
    detrend: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a robust plane to valid module pixels; return background and retained reference."""
    if not detrend:
        retained = mask.copy()
        for _ in range(iterations):
            median, _ = robust_median(temperature, retained)
            updated = retained & (np.abs(temperature - median) < threshold_k)
            if not updated.any() or np.array_equal(updated, retained):
                break
            retained = updated
        median, _ = robust_median(temperature, retained)
        return np.full(temperature.shape, median, dtype=float), retained
    yy, xx = np.indices(temperature.shape)
    positions = np.argwhere(mask)
    design_all = np.column_stack((positions[:, 1], positions[:, 0], np.ones(len(positions))))
    values_all = temperature[mask]
    tolerance = max(0.5, threshold_k / 2)
    rng = np.random.default_rng(0)
    best_inliers = np.ones(len(positions), dtype=bool)
    best_score = (-1, -float("inf"))
    trial_count = min(160, max(24, len(positions) // 20))
    for _ in range(trial_count):
        chosen = rng.choice(len(positions), 3, replace=False)
        sample = design_all[chosen]
        if abs(np.linalg.det(sample)) < 1e-9:
            continue
        coefficients = np.linalg.solve(sample, values_all[chosen])
        errors = np.abs(values_all - design_all @ coefficients)
        inliers = errors < tolerance
        score = (int(inliers.sum()), -float(np.median(errors[inliers])))
        if score > best_score:
            best_score = score
            best_inliers = inliers
    retained = np.zeros(mask.shape, dtype=bool)
    retained[positions[best_inliers, 0], positions[best_inliers, 1]] = True
    if retained.sum() < max(3, mask.sum() // 4):
        retained = mask.copy()
    background = np.full(temperature.shape, np.nan, dtype=float)
    for _ in range(iterations):
        if retained.sum() < 3:
            raise ValueError("too few reference pixels for background fit")
        design = np.column_stack((xx[retained], yy[retained], np.ones(retained.sum())))
        coefficients, *_ = np.linalg.lstsq(design, temperature[retained], rcond=None)
        background = coefficients[0] * xx + coefficients[1] * yy + coefficients[2]
        residual = temperature - background
        centre = float(np.median(residual[retained]))
        mad = float(np.median(np.abs(residual[retained] - centre)))
        cutoff = max(threshold_k, 3.0 * 1.4826 * mad)
        updated = mask & (np.abs(residual - centre) < cutoff)
        if np.array_equal(updated, retained):
            break
        retained = updated
    # The final mask update must also be reflected in the fitted coefficients.
    if retained.sum() < 3:
        raise ValueError("too few reference pixels for background fit")
    design = np.column_stack((xx[retained], yy[retained], np.ones(retained.sum())))
    coefficients, *_ = np.linalg.lstsq(design, temperature[retained], rcond=None)
    background = coefficients[0] * xx + coefficients[1] * yy + coefficients[2]
    return background, retained


@dataclass(frozen=True)
class IntraModuleAssessment:
    findings: tuple[Finding, ...] = ()
    reasons: tuple[str, ...] = ()
    # Opt-in diagnostic evidence; never resampled or used to change acceptance.
    reference_mask: np.ndarray | None = None
    background_c: np.ndarray | None = None


def assess_intra_module_anomalies(
    frame: ThermalFrame,
    module: ModuleObservation,
    module_mask: np.ndarray,
    *,
    config: DetectionConfig | None = None,
    conditions: list[Condition] | None = None,
    extent: Extent | None = None,
    include_background: bool = False,
) -> IntraModuleAssessment:
    """Find distinct hot/cold candidate regions within a manually identified module.

    A candidate is not a diagnosed fault. Normalization needs an explicit extent;
    this function cannot infer point-vs-area physics from the fraction of warm pixels.
    """
    from pvdelta.surface_constraints import apply_configured_surface

    frame=apply_configured_surface(frame)
    if module.frame_id != frame.frame_id:
        raise ValueError("module and frame IDs differ")
    cfg = config or DetectionConfig()
    valid = _measurement_grid(frame, module_mask, module)
    interior = erode(module_mask, cfg.edge_erosion_px)
    if not interior.any():
        return IntraModuleAssessment(reasons=("no measurement pixels after edge erosion",))
    region_quality = assess_region_quality(
        frame.temperature_c, interior, valid_mask=frame.valid_mask,
        min_valid_fraction=cfg.min_valid_fraction, min_reference_pixels=cfg.min_reference_px,
    )
    if region_quality.analysis is Capability.UNAVAILABLE:
        return IntraModuleAssessment(reasons=region_quality.reasons)
    safe = interior & valid
    if safe.sum() < cfg.min_reference_px + cfg.min_region_px:
        return IntraModuleAssessment(reasons=("too few valid native pixels after edge erosion",))
    temp = frame.temperature_c
    if cfg.detrend_gradient:
        positions = np.argwhere(safe)
        design = np.column_stack((positions, np.ones(len(positions))))
        if np.linalg.matrix_rank(design) < 3:
            return IntraModuleAssessment(reasons=("reference support cannot fit a 2D background",))
    try:
        background, reference_seed = _robust_background(
            temp, safe, cfg.delta_threshold_k,
            cfg.max_reference_iterations, cfg.detrend_gradient,
        )
    except ValueError as exc:
        return IntraModuleAssessment(reasons=(f'background unavailable: {exc}',))
    if reference_seed.sum() < cfg.min_reference_px:
        return IntraModuleAssessment(reasons=("too few retained background reference pixels",))
    residual = temp - background
    # Exclude every hot/cold candidate from the measurement reference, including
    # small components that are suppressed only for presentation.
    reference = reference_seed & (np.abs(residual) < cfg.delta_threshold_k)
    if reference.sum() < cfg.min_reference_px:
        return IntraModuleAssessment(reasons=("too few uncontaminated reference pixels",))
    # A refit can expose new hot/cold pixels. Exclude them from every finding's
    # reference and refit to convergence, rather than excluding only its own region.
    for _ in range(cfg.max_reference_iterations):
        if reference.sum() < cfg.min_reference_px:
            return IntraModuleAssessment(reasons=('too few stable reference pixels',))
        if cfg.detrend_gradient:
            yy, xx = np.indices(temp.shape)
            design = np.column_stack((xx[reference], yy[reference], np.ones(reference.sum())))
            if np.linalg.matrix_rank(design) < 3:
                return IntraModuleAssessment(reasons=('degenerate background reference geometry',))
            coefficients, *_ = np.linalg.lstsq(design, temp[reference], rcond=None)
            background = coefficients[0]*xx + coefficients[1]*yy + coefficients[2]
        else:
            reference_median, _ = robust_median(temp, reference)
            background = np.full(temp.shape, reference_median)
        residual = temp-background
        updated = reference & (np.abs(residual) < cfg.delta_threshold_k)
        if np.array_equal(updated, reference):
            break
        reference = updated
    else:
        return IntraModuleAssessment(reasons=('background reference did not converge',))
    findings = []
    for direction, sign in (("hot", 1), ("cold", -1)):
        candidate = safe & (sign * residual >= cfg.delta_threshold_k)
        components = [c for c in _components(candidate) if len(c) >= cfg.min_region_px]
        components.sort(key=lambda c: (int(c[:, 0].min()), int(c[:, 1].min())))
        for index, coords in enumerate(components, 1):
            region = np.zeros(safe.shape, dtype=bool)
            region[coords[:, 0], coords[:, 1]] = True
            measurement_reference = reference & ~region
            if measurement_reference.sum() < cfg.min_reference_px:
                continue
            # Refit if the updated background exposed another candidate in the
            # retained reference. Never label excluded pixels as used reference.
            local_background = background
            if np.any(reference & region):
                if cfg.detrend_gradient:
                    design = np.column_stack((xx[measurement_reference],
                                              yy[measurement_reference],
                                              np.ones(measurement_reference.sum())))
                    coefficients, *_ = np.linalg.lstsq(
                        design, temp[measurement_reference], rcond=None,
                    )
                    local_background = coefficients[0]*xx + coefficients[1]*yy + coefficients[2]
                else:
                    median, _ = robust_median(temp, measurement_reference)
                    local_background = np.full(temp.shape, median)
            n_ref = int(measurement_reference.sum())
            t_region = float(np.median(temp[region]))
            t_ref = float(np.median(local_background[region]))
            region_residual = temp[region] - local_background[region]
            delta = float(np.median(region_residual))
            if sign * delta < cfg.delta_threshold_k:
                continue
            norm_delta = None
            norm_basis = None
            flags = []
            if conditions:
                if extent is None:
                    flags.append("normalization unavailable: anomaly extent not established")
                else:
                    try:
                        result = normalize(delta, conditions, extent)
                    except ValueError as exc:
                        flags.append(f"normalization unavailable: {exc}")
                    else:
                        norm_delta = result.delta_ref_k
                        norm_basis = result.primary
                        flags.extend(result.flags)
                        flags.append(f"normalization formula: {result.formula}; x={result.exponent:g}")
            findings.append(Finding(
                finding_id=f"{module.module_id}:{direction}:{index}",
                module_id=module.module_id,
                frame_id=frame.frame_id,
                anomaly_type=f"{direction}_region",
                delta_t_c=delta,  # numerical Celsius difference equals kelvin
                absolute_t_c=t_region,
                pixel_polygon=_bbox(coords),
                confidence=None,
                is_normalized=norm_delta is not None,
                normalized_delta_t_c=norm_delta,
                normalization_basis=norm_basis,
                normalization_flags=flags,
                reference_t_c=t_ref,
                reference_method=("robust_module_plane_excluding_outliers"
                                  if cfg.detrend_gradient else "module_median_excluding_candidates"),
                n_region_px=len(coords),
                n_reference_px=n_ref,
                region_max_t_c=float(np.max(temp[region])),
                region_p95_t_c=float(np.percentile(temp[region], 95)),
                region_max_residual_k=float(np.max(region_residual)),
                delta_method="median_of_pixel_residuals",
                region_pixels_yx=sorted((int(y), int(x)) for y, x in coords),
            ))
    return IntraModuleAssessment(tuple(findings), reference_mask=reference.copy() if include_background else None,
        background_c=background.copy() if include_background else None)


def detect_intra_module_anomalies(frame, module, module_mask, *, config=None,
                                  conditions=None, extent=None) -> list[Finding]:
    """Compatibility findings-only API; orchestration should use the assessment API."""
    return list(assess_intra_module_anomalies(
        frame, module, module_mask, config=config, conditions=conditions, extent=extent,
    ).findings)


@dataclass(frozen=True)
class InterModuleAssessment:
    findings: tuple[Finding, ...]
    insufficient_groups: tuple[str, ...]
    flags: tuple[str, ...]


def assess_inter_module_anomalies(
    frame: ThermalFrame,
    modules: list[ModuleObservation],
    masks: dict[str, np.ndarray],
    peer_groups: dict[str, str],
    *,
    config: DetectionConfig | None = None,
) -> InterModuleAssessment:
    """Return findings plus peer groups whose common reference is not trustworthy."""
    from pvdelta.surface_constraints import apply_configured_surface

    frame=apply_configured_surface(frame)
    cfg = config or DetectionConfig()
    if len({m.module_id for m in modules}) != len(modules):
        raise ValueError("duplicate module IDs in peer comparison")
    records: dict[str, tuple[float, int, ModuleObservation]] = {}
    flags = []
    for module in modules:
        if module.frame_id != frame.frame_id:
            raise ValueError("module and frame IDs differ")
        if module.module_id not in masks or module.module_id not in peer_groups:
            raise ValueError("each module needs a mask and an explicit peer group")
        valid = _measurement_grid(frame, masks[module.module_id], module)
        interior = erode(masks[module.module_id], cfg.edge_erosion_px)
        if not interior.any():
            flags.append(f'{module.module_id}: no measurement pixels after edge erosion')
            continue
        quality = assess_region_quality(
            frame.temperature_c, interior, valid_mask=frame.valid_mask,
            min_valid_fraction=cfg.min_valid_fraction, min_reference_pixels=cfg.min_reference_px,
        )
        if quality.analysis is Capability.UNAVAILABLE:
            flags.extend(f'{module.module_id}: {reason}' for reason in quality.reasons)
            continue
        safe = interior & valid
        median, n_px = robust_median(frame.temperature_c, safe)
        records[module.module_id] = (median, n_px, module)
    insufficient = []
    usable_groups = set(peer_groups.values())
    for group in sorted(usable_groups):
        members = [value[0] for key, value in records.items() if peer_groups[key] == group]
        if len(members) < 3:
            insufficient.append(group)
            flags.append(f"{group}: fewer than three usable peer modules")
            continue
        centre = float(np.median(members))
        deviating = sum(abs(value - centre) >= cfg.delta_threshold_k for value in members)
        if deviating * 2 >= len(members):
            insufficient.append(group)
            flags.append(f"{group}: widespread or gradient-like group deviation")
    findings = []
    for module_id, (temp_c, n_px, module) in records.items():
        group = peer_groups[module_id]
        if group in insufficient:
            continue
        peers = [value[0] for other_id, value in records.items()
                 if other_id != module_id and peer_groups[other_id] == group]
        reference = float(np.median(peers))
        delta = temp_c - reference
        if not math.isfinite(delta) or abs(delta) < cfg.delta_threshold_k:
            continue
        direction = "hot" if delta > 0 else "cold"
        findings.append(Finding(
            finding_id=f"{module_id}:whole:{direction}", module_id=module_id,
            frame_id=frame.frame_id, anomaly_type=f"whole_module_{direction}_candidate",
            delta_t_c=delta, absolute_t_c=temp_c, pixel_polygon=module.pixel_polygon,
            confidence=None, reference_t_c=reference,
            reference_method="median_of_explicit_peer_modules",
            n_region_px=n_px, n_reference_px=sum(value[1] for other_id, value in records.items()
                                                if other_id != module_id and peer_groups[other_id] == group),
        ))
    return InterModuleAssessment(tuple(findings), tuple(insufficient), tuple(flags))


def detect_inter_module_anomalies(
    frame: ThermalFrame,
    modules: list[ModuleObservation],
    masks: dict[str, np.ndarray],
    peer_groups: dict[str, str],
    *,
    config: DetectionConfig | None = None,
) -> list[Finding]:
    """Compare each module with explicit local peers; do not infer electrical topology."""
    return list(assess_inter_module_anomalies(
        frame, modules, masks, peer_groups, config=config,
    ).findings)
