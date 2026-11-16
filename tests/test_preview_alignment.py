from copy import deepcopy

import cv2
import numpy as np
import pytest

from pvdelta.preview_alignment import check_preview_alignment, preview_regions_to_native


def scene(seed=7):
    rng = np.random.default_rng(seed)
    values = cv2.GaussianBlur(rng.uniform(0, 1, (256, 320)).astype(np.float32), (5, 5), 1.)
    for _ in range(80):
        x, y = rng.integers(8, 220, 2)
        cv2.rectangle(values, (int(x), int(y)), (int(x+20), int(y+15)), float(rng.uniform()), -1)
    return values.astype(float)*30+20


def preview(values, dx=0, dy=0):
    grey = ((values-values.min())/(values.max()-values.min())*240+5).astype(np.uint8)
    moved = cv2.warpAffine(grey, np.array([[1., 0., dx], [0., 1., dy]]), (grey.shape[1], grey.shape[0]))
    return np.repeat(moved[:, :, None], 3, axis=2)


@pytest.mark.parametrize('dx,dy', [(0, 0), (2, 13), (-3, -11), (5, -4)])
def test_same_size_is_not_identity_and_transform_direction_preserves_native_values(dx, dy):
    values = scene()
    rgb = preview(values, dx, dy)
    old_values, old_rgb = values.copy(), rgb.copy()
    values.setflags(write=False)
    rgb.setflags(write=False)
    result = check_preview_alignment(values, rgb)
    assert result['status'] == 'accepted', result
    np.testing.assert_allclose(np.array(result['preview_to_native'])[:, 2], [-dx, -dy], atol=.15)
    assert result['cycle_error_px'] < .15 and not result['measurement_authorised']
    np.testing.assert_array_equal(values, old_values)
    np.testing.assert_array_equal(rgb, old_rgb)


def test_palette_inversion_is_only_geometry_and_cannot_change_temperatures():
    values = scene()
    result = check_preview_alignment(values, 255-preview(values, 1, 8))
    assert result['status'] == 'accepted'
    np.testing.assert_allclose(np.array(result['preview_to_native'])[:, 2], [-1, -8], atol=.2)


def test_correct_centre_but_wrong_outer_frame_is_withheld():
    values = scene()
    rgb = preview(scene(22))
    real = preview(values)
    rgb[45:211, 57:263] = real[45:211, 57:263]
    result = check_preview_alignment(values, rgb)
    assert result['status'] == 'unresolved'
    assert result['preview_to_native'] is None


@pytest.mark.parametrize('case', ['flat', 'periodic', 'different', 'rotation', 'far'])
def test_unidentifiable_or_unsupported_geometry_never_falls_back_to_identity(case):
    values = scene()
    rgb = preview(values)
    if case == 'flat':
        values[:] = 25
        rgb[:] = 100
    elif case == 'periodic':
        values = np.tile(np.sin(np.arange(320)*2*np.pi/16), (256, 1))
        rgb = preview(values, 0, 8)
    elif case == 'different':
        rgb = preview(scene(100))
    elif case == 'rotation':
        rgb = cv2.warpAffine(rgb, cv2.getRotationMatrix2D((160, 128), 4, 1), (320, 256))
    else:
        rgb = preview(values, 40, 0)
    result = check_preview_alignment(values, rgb)
    assert result['status'] == 'unresolved', result
    assert result['preview_to_native'] is None


@pytest.mark.parametrize('case', ['nan', 'inf', 'shape', 'dtype', 'small', 'complex'])
def test_invalid_inputs_are_explicit_errors(case):
    values = scene()
    rgb = preview(values)
    if case in ('nan', 'inf'):
        values[10, 10] = float(case)
    elif case == 'shape':
        rgb = rgb[:-1]
    elif case == 'dtype':
        rgb = rgb.astype(float)
    elif case == 'small':
        values, rgb = values[:32, :32], rgb[:32, :32]
    else:
        values = values.astype(complex)
    with pytest.raises(ValueError):
        check_preview_alignment(values, rgb)


def accepted(dx=2., dy=-13.):
    return {'schema': 'pvdelta.preview-alignment/v1', 'status': 'accepted',
            'native_shape': [256, 320], 'preview_to_native': [[1., 0., dx], [0., 1., dy]]}


def box(left=20, top=30):
    return {'region_id': 'L1', 'published_label': 'F1',
            'polygon_native_xy': [[left, top], [left+10, top], [left+10, top+8], [left, top+8]]}


def test_all_region_ids_and_extents_preserved_without_clipping_mutation_or_resampling():
    regions = [box()]
    old = deepcopy(regions)
    moved = preview_regions_to_native(iter(regions), accepted())
    assert regions == old and moved[0]['region_id'] == 'L1' and moved[0]['published_label'] == 'F1'
    assert moved[0]['polygon_native_xy'] == [[22., 17.], [32., 17.], [32., 25.], [22., 25.]]


def test_one_outside_region_fails_whole_frame_instead_of_dropping_reference():
    with pytest.raises(ValueError, match='inside'):
        preview_regions_to_native([box(), box(top=1) | {'region_id': 'L2'}], accepted())


@pytest.mark.parametrize('case', ['unresolved', 'duplicate', 'nonfinite', 'scale'])
def test_no_unchecked_mapping_or_invalid_reference_set(case):
    alignment = accepted()
    regions = [box()]
    if case == 'unresolved':
        alignment['status'] = 'unresolved'
    elif case == 'duplicate':
        regions *= 2
    elif case == 'nonfinite':
        alignment['preview_to_native'][0][2] = float('nan')
    else:
        alignment['preview_to_native'][0][0] = 1.01
    with pytest.raises(ValueError):
        preview_regions_to_native(regions, alignment)
