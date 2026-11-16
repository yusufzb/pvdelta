"""Optional native-temperature peak partitions inside an existing candidate only."""
from __future__ import annotations

import heapq
from dataclasses import asdict, dataclass
from numbers import Integral

import numpy as np


@dataclass(frozen=True)
class PeakPartitionConfig:
    """Exploratory region separation; not IEC limits or camera NETD claims."""

    min_prominence_k: float = 1.25
    min_peak_support_px: int = 2
    min_region_px: int = 4
    max_peaks: int = 16
    max_component_px: int = 4096

    def __post_init__(self):
        if (isinstance(self.min_prominence_k, bool)
                or not np.isfinite(self.min_prominence_k) or self.min_prominence_k <= 0):
            raise ValueError('min_prominence_k must be finite and positive')
        for name in ('min_peak_support_px', 'min_region_px', 'max_peaks', 'max_component_px'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f'{name} must be a positive integer')


def _reachable(seeds, neighbours, allowed):
    reached = set(seeds)
    pending = list(seeds)
    while pending:
        for other in neighbours[pending.pop()]:
            if other in allowed and other not in reached:
                reached.add(other)
                pending.append(other)
    return reached


def _bottlenecks(seeds, neighbours, temperatures):
    """Maximum path-minimum native temperature from a peak plateau."""
    best = np.full(len(temperatures), -np.inf)
    heap = []
    for seed in seeds:
        best[seed] = temperatures[seed]
        heapq.heappush(heap, (-best[seed], seed))
    while heap:
        negative, index = heapq.heappop(heap)
        value = -negative
        if value < best[index]:
            continue
        for other in neighbours[index]:
            capacity = min(value, temperatures[other])
            if capacity > best[other]:
                best[other] = capacity
                heapq.heappush(heap, (-capacity, other))
    return best


def _distances(seeds, neighbours, costs):
    """Shortest node-cost paths, with no traversal outside parent support."""
    best = np.full(len(costs), np.inf)
    heap = []
    for seed in seeds:
        best[seed] = 0.
        heapq.heappush(heap, (0., seed))
    while heap:
        distance, index = heapq.heappop(heap)
        if distance > best[index]:
            continue
        for other in neighbours[index]:
            updated = distance + costs[other]
            if updated < best[other]:
                best[other] = updated
                heapq.heappush(heap, (updated, other))
    return best


def partition_thermal_peaks(temperature_c, region_pixels_yx, *, config=None):
    """Offer child review regions without replacing the parent detector finding.

    Uses exact four-connected native pixels already belonging to one candidate.
    No label, RGB, CAD, inferred grid or new pixel can enter. Regional maxima
    require native saddle prominence and spatial support. Ambiguous path ties
    belong to an explicit bridge set, never an arbitrarily chosen child. A
    failed/unsupported split returns the entire parent unchanged.

    Child temperatures are native statistics only. The detector's fitted plane
    is not available here, so parent delta/normalisation/confidence must NOT be
    inherited by children. This does not establish cell/module identity or cause.
    """
    cfg = config or PeakPartitionConfig()
    if not isinstance(cfg, PeakPartitionConfig):
        raise TypeError('PeakPartitionConfig required')
    grid = np.asarray(temperature_c)
    coords = np.asarray(region_pixels_yx)
    if (grid.ndim != 2 or min(grid.shape, default=0) < 1 or grid.dtype.kind not in 'uif'
            or coords.ndim != 2 or coords.shape[1] != 2 or not len(coords)
            or coords.dtype.kind not in 'ui' or (coords < 0).any()
            or (coords[:, 0] >= grid.shape[0]).any() or (coords[:, 1] >= grid.shape[1]).any()):
        raise ValueError('Nonempty integer native (y,x) coordinates inside a real 2D grid required')
    coords = sorted(tuple(map(int, row)) for row in coords)
    if len(set(coords)) != len(coords):
        raise ValueError('Duplicate parent pixels are not allowed')
    lookup = {p: i for i, p in enumerate(coords)}
    neighbours = [[lookup[q] for q in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)) if q in lookup]
                  for y, x in coords]
    if len(_reachable([0], neighbours, set(range(len(coords))))) != len(coords):
        raise ValueError('Parent candidate must be one four-connected component')
    values = np.array([grid[y, x] for y, x in coords], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('Every parent measurement pixel must be finite')
    result = {'schema': 'pvdelta.thermal-peak-partition/v1', 'status': 'unsplit',
        'config': asdict(cfg), 'parent_pixels_yx': [list(p) for p in coords],
        'regions': [], 'ambiguous_bridge_pixels_yx': [], 'peaks': [],
        'measurement_authorised': False, 'parent_finding_unchanged': True,
        'child_reference_delta_available': False,
        'scope': 'Native temperature topology inside an existing candidate; no semantic module/cell or fault/glare approval'}

    def unsplit(reason):
        return dict(result, reason=reason, regions=[], ambiguous_bridge_pixels_yx=[])

    if len(coords) > cfg.max_component_px:
        return unsplit('component exceeds bounded search size')
    plateaus, seen = [], set()
    for index in range(len(coords)):
        if index in seen:
            continue
        equal = _reachable([index], neighbours, set(np.flatnonzero(values == values[index])))
        seen |= equal
        if not any(values[other] > values[index] for i in equal for other in neighbours[i]):
            plateaus.append(sorted(equal))
    # Fixed spatial ordering makes equal-height peak diagnostics reproducible.
    plateaus.sort(key=lambda p: (-values[p[0]], coords[p[0]]))
    if len(plateaus) > cfg.max_peaks*4:
        return unsplit('too many regional maxima for bounded search')
    retained = []
    for index, plateau in enumerate(plateaus):
        peak = float(values[plateau[0]])
        if index:
            paths = _bottlenecks(plateau, neighbours, values)
            elder_indices = [j for earlier in plateaus[:index] for j in earlier]
            saddle = float(paths[elder_indices].max())
            prominence = peak-saddle
        else:
            saddle = None
            prominence = peak-float(values.min())
        support_level = peak-cfg.min_prominence_k/2
        support = _reachable(plateau, neighbours, set(np.flatnonzero(values >= support_level)))
        accepted = prominence >= cfg.min_prominence_k and len(support) >= cfg.min_peak_support_px
        result['peaks'].append({'peak_pixels_yx': [list(coords[i]) for i in plateau],
            'native_peak_t_c': peak, 'saddle_t_c': saddle, 'prominence_k': prominence,
            'near_peak_support_px': len(support), 'retained': bool(accepted)})
        if accepted:
            retained.append((plateau, result['peaks'][-1]))
    if len(retained) < 2:
        return unsplit('fewer than two supported prominent native peaks')
    if len(retained) > cfg.max_peaks:
        return unsplit('too many peaks for bounded separation')
    # Positive shared node costs favour warm paths, without inventing pixels.
    costs = 1+(float(values.max())-values)/cfg.min_prominence_k
    distances = np.stack([_distances(p, neighbours, costs) for p, _ in retained])
    tied = np.isclose(distances, distances.min(axis=0), atol=1e-9, rtol=0)
    ambiguous = tied.sum(axis=0) != 1
    assigned = np.argmin(distances, axis=0)
    regions = []
    for index, (_, peak) in enumerate(retained):
        indices = np.flatnonzero((assigned == index) & ~ambiguous)
        if len(indices) < cfg.min_region_px:
            return unsplit('a child has too few unambiguous native pixels')
        if len(_reachable([int(indices[0])], neighbours, set(indices))) != len(indices):
            return unsplit('a child is disconnected after withholding ambiguous bridges')
        positions = [coords[i] for i in indices]
        yy, xx = np.asarray(positions).T
        measured = values[indices]
        regions.append({'region_id': f'P{index+1:03d}', 'region_pixels_yx': [list(p) for p in positions],
            'polygon_native_xy': [[int(xx.min()), int(yy.min())], [int(xx.max()+1), int(yy.min())],
                [int(xx.max()+1), int(yy.max()+1)], [int(xx.min()), int(yy.max()+1)]],
            'n_region_px': len(positions), 'temp_min_c': float(measured.min()),
            'temp_max_c': float(measured.max()), 'temp_median_c': float(np.median(measured)),
            'peak': peak})
    bridge = [list(coords[i]) for i in np.flatnonzero(ambiguous)]
    assert sum(r['n_region_px'] for r in regions) + len(bridge) == len(coords)
    return dict(result, status='partitioned_for_review', reason='native prominence and support gates passed',
                regions=regions, ambiguous_bridge_pixels_yx=bridge)
