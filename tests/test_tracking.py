import numpy as np
import pytest

from pvdelta.alignment import transform_points
from pvdelta.tracking import quadrilateral_center


def test_center_commutes_with_perspective_projection():
    quad = np.array([[0, 0], [2, 0], [2, 1], [0, 1]], dtype=float)
    matrix = np.array([[100, 8, 25], [4, 90, 30], [.3, .1, 1]])
    projected = transform_points(quad, matrix)
    expected = transform_points(np.array([[1, .5]]), matrix)[0]
    assert quadrilateral_center(projected) == pytest.approx(expected)
    assert not np.allclose(projected.mean(axis=0), expected)
    assert quadrilateral_center(projected[::-1]) == pytest.approx(expected)


@pytest.mark.parametrize("quad", [
    [[0, 0], [1, 1], [0, 1], [1, 0]],
    [[0, 0], [1, 0], [1, 0], [0, 1]],
    [[0, 0], [1, 0], [.2, .2], [0, 1]],
    [[0, 0], [1, 0], [1, float('nan')], [0, 1]],
    [[0, 0], [1, 0], [0, 1]],
])
def test_invalid_or_incomplete_quad_rejected(quad):
    with pytest.raises(ValueError):
        quadrilateral_center(np.array(quad))
