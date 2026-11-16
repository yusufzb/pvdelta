"""Group repeated geolocated observations into provisional module clusters.

The result is deliberately a *cluster*, not a Panel ID.  A cluster is useful for
coverage and review while CAD, asset anchors, or an independent survey are absent.
Observations from one capture are never merged with one another: adjacent modules
in the same thermal frame must remain separate even when their projected centres
are close.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

_METRES_PER_DEGREE = 111_320.0


@dataclass(frozen=True)
class ObservationPoint:
    """A geolocated frame-local module observation."""

    observation_id: str
    capture_id: str
    latitude: float
    longitude: float


@dataclass(frozen=True)
class ProvisionalModuleCluster:
    """A review-only physical-location hypothesis."""

    cluster_id: str
    observation_ids: tuple[str, ...]
    latitude: float
    longitude: float
    capture_count: int
    spread_m: float


def cluster_observations(
    observations: list[ObservationPoint] | tuple[ObservationPoint, ...],
    *,
    radius_m: float = 0.75,
) -> list[ProvisionalModuleCluster]:
    """Return deterministic proximity clusters for geolocated observations.

    ``radius_m`` is a screening tolerance, not a claimed location accuracy.  The
    implementation uses a local equirectangular projection and a grid index, so it
    has no scientific-computing dependency.  A pair is eligible only when its
    captures differ; this is the guard that prevents one image's neighbouring
    modules from being collapsed into one physical module.
    """
    if radius_m <= 0 or not math.isfinite(radius_m):
        raise ValueError("radius_m must be a finite positive number")
    points = list(observations)
    for point in points:
        if not all(math.isfinite(value) for value in (point.latitude, point.longitude)):
            raise ValueError("observation coordinates must be finite")
    if not points:
        return []

    mean_lat = sum(point.latitude for point in points) / len(points)
    metres_per_longitude_degree = _METRES_PER_DEGREE * math.cos(math.radians(mean_lat))
    xy = [
        (
            point.longitude * metres_per_longitude_degree,
            point.latitude * _METRES_PER_DEGREE,
        )
        for point in points
    ]

    parent = list(range(len(points)))
    component_captures = [{point.capture_id} for point in points]

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            return
        # A transitive chain must not re-introduce two modules from one frame into
        # the same cluster (A~B and B~C is not evidence that A~C is one module).
        if component_captures[left_root].intersection(component_captures[right_root]):
            return
        parent[right_root] = left_root
        component_captures[left_root].update(component_captures[right_root])
        component_captures[right_root].clear()

    cells: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, (x, y) in enumerate(xy):
        cell = (math.floor(x / radius_m), math.floor(y / radius_m))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for other in cells[(cell[0] + dx, cell[1] + dy)]:
                    if points[index].capture_id == points[other].capture_id:
                        continue
                    ox, oy = xy[other]
                    if (x - ox) ** 2 + (y - oy) ** 2 <= radius_m**2:
                        union(index, other)
        cells[cell].append(index)

    grouped: dict[int, list[int]] = defaultdict(list)
    for index in range(len(points)):
        grouped[find(index)].append(index)

    provisional: list[tuple[float, float, list[int], float]] = []
    for indices in grouped.values():
        latitude = sum(points[index].latitude for index in indices) / len(indices)
        longitude = sum(points[index].longitude for index in indices) / len(indices)
        centre_x = longitude * metres_per_longitude_degree
        centre_y = latitude * _METRES_PER_DEGREE
        spread_m = max(
            math.hypot(xy[index][0] - centre_x, xy[index][1] - centre_y)
            for index in indices
        )
        provisional.append((latitude, longitude, indices, spread_m))

    provisional.sort(key=lambda item: (item[0], item[1], item[2][0]))
    result: list[ProvisionalModuleCluster] = []
    for sequence, (latitude, longitude, indices, spread_m) in enumerate(provisional, start=1):
        ordered_ids = tuple(sorted(points[index].observation_id for index in indices))
        captures = {points[index].capture_id for index in indices}
        result.append(
            ProvisionalModuleCluster(
                cluster_id=f"pmc-{sequence:05d}",
                observation_ids=ordered_ids,
                latitude=latitude,
                longitude=longitude,
                capture_count=len(captures),
                spread_m=spread_m,
            )
        )
    return result


__all__ = ["ObservationPoint", "ProvisionalModuleCluster", "cluster_observations"]
