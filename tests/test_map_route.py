from pvdelta.map_route import build_route_segments, simplify_route


def _feature(capture_id, lat, lon, timestamp):
    return {
        "capture_id": capture_id,
        "camera_latitude": lat,
        "camera_longitude": lon,
        "capture_time_utc": timestamp,
    }


def test_route_is_time_ordered_and_keeps_plausible_points_together():
    features = [
        _feature("b", 40.0, 26.0, "2026-01-01T00:00:02+00:00"),
        _feature("a", 40.0, 26.0001, "2026-01-01T00:00:01+00:00"),
        _feature("c", 40.0, 26.0002, "2026-01-01T00:00:03+00:00"),
    ]

    assert build_route_segments(features) == [[
        [40.0, 26.0001], [40.0, 26.0], [40.0, 26.0002]
    ]]


def test_route_splits_on_large_jump_and_ignores_invalid_coordinate():
    features = [
        _feature("a", 40.0, 26.0, "2026-01-01T00:00:00+00:00"),
        _feature("bad", float("nan"), 26.1, "2026-01-01T00:00:01+00:00"),
        _feature("b", 40.0, 26.0001, "2026-01-01T00:00:02+00:00"),
        _feature("c", 40.01, 26.01, "2026-01-01T00:00:03+00:00"),
    ]

    assert build_route_segments(features, max_step_m=50) == [
        [[40.0, 26.0], [40.0, 26.0001]],
        [[40.01, 26.01]],
    ]


def test_simplify_route_preserves_endpoints():
    points = [[40.0, 26.0 + index / 10000] for index in range(20)]

    reduced = simplify_route(points, max_points=5)

    assert len(reduced) <= 5
    assert reduced[0] == points[0]
    assert reduced[-1] == points[-1]
