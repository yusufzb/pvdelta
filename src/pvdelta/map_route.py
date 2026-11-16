"""Small, dependency-free helpers for displaying a flight track.

The database contains one GPS point per thermal capture.  A map should not
blindly join every row: an invalid coordinate or a long acquisition gap can
connect two unrelated parts of a flight and make the route look wrong.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any


def _timestamp(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        text = str(value).replace("Z", "+00:00")
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None


def _distance_m(first: tuple[float, float], second: tuple[float, float]) -> float:
    """Approximate distance between two nearby WGS84 coordinates in metres."""
    lat1, lon1 = first
    lat2, lon2 = second
    mean_lat = math.radians((lat1 + lat2) / 2.0)
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    return 6_378_137.0 * math.hypot(d_lat, math.cos(mean_lat) * d_lon)


def build_route_segments(
    features: list[dict[str, Any]],
    *,
    max_gap_s: float = 20.0,
    max_step_m: float = 100.0,
) -> list[list[list[float]]]:
    """Return ordered, valid GPS points split at suspicious gaps or jumps.

    Features are sorted by capture time where available, then capture id.  A
    route segment is kept continuous only when its time and spatial step are
    plausible.  This keeps a missing-GPS row or a second flight from drawing a
    misleading diagonal line across the site.
    """
    valid: list[tuple[datetime | None, str, float, float]] = []
    for item in features:
        try:
            lat = float(item["camera_latitude"])
            lon = float(item["camera_longitude"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (math.isfinite(lat) and math.isfinite(lon) and -90 <= lat <= 90 and -180 <= lon <= 180):
            continue
        valid.append((_timestamp(item.get("capture_time_utc")), str(item.get("capture_id", "")), lat, lon))
    valid.sort(key=lambda row: (row[0] is None, row[0].timestamp() if row[0] else 0.0, row[1]))
    if not valid:
        return []

    segments: list[list[list[float]]] = [[[valid[0][2], valid[0][3]]]]
    previous = valid[0]
    for current in valid[1:]:
        previous_time, _, previous_lat, previous_lon = previous
        current_time = current[0]
        gap = None if previous_time is None or current_time is None else (current_time - previous_time).total_seconds()
        jump = _distance_m((previous_lat, previous_lon), (current[2], current[3]))
        if (gap is not None and gap > max_gap_s) or jump > max_step_m:
            segments.append([])
        segments[-1].append([current[2], current[3]])
        previous = current
    return [segment for segment in segments if segment]


def simplify_route(points: list[list[float]], *, max_points: int = 500) -> list[list[float]]:
    """Reduce dense tracks while always retaining the first and last point."""
    if len(points) <= max_points or max_points < 2:
        return points
    stride = max(1, math.ceil((len(points) - 1) / (max_points - 1)))
    simplified = points[::stride]
    if simplified[-1] != points[-1]:
        simplified.append(points[-1])
    return simplified
