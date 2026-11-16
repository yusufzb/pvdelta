import json

import pytest

from pvdelta.labelme_regions import parse_labelme_rectangles


def document():
    return {'imagePath': 'DJI_0001_R.JPG', 'imageWidth': 640, 'imageHeight': 512,
            'shapes': [{'label': 'F1', 'shape_type': 'rectangle', 'points': [[44, 195], [41, 191]]}]}


def parse(data):
    return parse_labelme_rectangles(json.dumps(data), native_shape=(512, 640), expected_image_name='DJI_0001_R.JPG')


def test_reverse_corners_are_sorted_without_changing_region_label_or_inferring_panel_type():
    result = parse(document())
    assert result == [{'region_id': 'L0001', 'published_label': 'F1',
                      'polygon_native_xy': [[41, 191], [44, 191], [44, 195], [41, 195]]}]


def test_only_explicit_existing_empty_shapes_are_empty():
    data = document()
    data['shapes'] = []
    assert parse(data) == []
    del data['shapes']
    with pytest.raises(ValueError):
        parse(data)


def test_author_path_is_checked_as_metadata_and_never_followed():
    data = document()
    data['imagePath'] = 'Z:\\not-a-real-drive\\DJI_0001_R.JPG'
    assert len(parse(data)) == 1
    data['imagePath'] = 'Z:\\not-a-real-drive\\OTHER.JPG'
    with pytest.raises(ValueError, match='basename'):
        parse(data)


@pytest.mark.parametrize('case', ['dimensions', 'unsupported_shape', 'point_count', 'zero_area',
                                'outside', 'nonfinite', 'empty_label'])
def test_bad_labels_fail_as_a_whole_instead_of_dropping_references(case):
    data = document()
    shape = data['shapes'][0]
    if case == 'dimensions':
        data['imageWidth'] = 1280
    elif case == 'unsupported_shape':
        shape['shape_type'] = 'circle'
    elif case == 'point_count':
        shape['points'] = [[1, 1]]
    elif case == 'zero_area':
        shape['points'] = [[1, 1], [1, 5]]
    elif case == 'outside':
        shape['points'][0][0] = 641
    elif case == 'nonfinite':
        shape['points'][0][0] = float('nan')
    else:
        shape['label'] = ''
    with pytest.raises(ValueError):
        parse(data)
