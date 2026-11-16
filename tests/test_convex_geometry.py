import numpy as np
import pytest

from pvdelta.convex_geometry import convex_iou


def test_nearly_nested_quads_with_shared_edge_keep_the_full_intersection():
    first = np.array([[501.7977905, 226.1936646], [565.2506104, 218.4847412],
                      [581.3344116, 350.8718567], [517.8815918, 358.5807800]])
    second = np.array([[501.2779236, 221.9145966], [564.7307129, 214.2056732],
                       [581.3344116, 350.8718567], [517.8815918, 358.5807800]])
    # Analytic areas of nested parallelograms; some float32 clipping paths returned ~0.326.
    expected = 8524.3245765/8799.8491536
    assert convex_iou(first, second) == pytest.approx(expected, abs=1e-6)
    assert convex_iou(second, first) == pytest.approx(expected, abs=1e-6)
    assert convex_iou(first[::-1], second[::-1]) == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize('offset, expected', [(0, 1), (5, 1/3), (10, 0), (20, 0)])
def test_rectangle_iou_and_shared_edge(offset, expected):
    first = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], float)
    assert convex_iou(first, first+[offset, 0]) == pytest.approx(expected)
    assert convex_iou(first+[1e6, 1e6], first+[offset+1e6, 1e6]) == pytest.approx(expected)


@pytest.mark.parametrize('invalid', [[[0, 0], [0, 0], [0, 0]],
    [[0, 0], [10, 0], [5, 2], [10, 10], [0, 10]], [[0, 0], [1, 0], [0, float('nan')]]])
def test_invalid_geometry_is_not_silently_accepted(invalid):
    with pytest.raises(ValueError):
        convex_iou(invalid, [[0, 0], [1, 0], [0, 1]])
