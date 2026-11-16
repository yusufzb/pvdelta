"""Strict published LabelMe rectangle import; never follows embedded image paths."""
from __future__ import annotations

import json

import numpy as np


def parse_labelme_rectangles(text, *, native_shape, expected_image_name):
    """Return supplied region labels, not module polygons or electrical fault types.

    Caller verifies the source/raw grid, orientation and any embedded imageData
    correspondence. This parser never opens imagePath or decodes imageData.
    Missing JSON is an input error; only an explicit empty shapes list is empty.
    Unsupported shapes fail the whole label rather than silently losing boxes.
    """
    if (not isinstance(text, str) or len(native_shape) != 2
            or any(type(n) is not int or n < 1 for n in native_shape)
            or not isinstance(expected_image_name, str) or not expected_image_name):
        raise ValueError('Text, positive native grid and expected source name required')
    data = json.loads(text)
    if not isinstance(data, dict):
        raise TypeError('LabelMe object required')
    if 'shapes' not in data:
        raise ValueError('Explicit LabelMe shapes list required')
    if not isinstance(data['shapes'], list):
        raise TypeError('LabelMe shapes must be a list')
    declared = data.get('imagePath')
    if (not isinstance(declared, str) or declared.replace('\\', '/').split('/')[-1].casefold()
            != expected_image_name.casefold()):
        raise ValueError('Label imagePath basename differs from the verified source')
    height, width = native_shape
    for field, expected in (('imageHeight', height), ('imageWidth', width)):
        value = data.get(field)
        if value is not None and (type(value) is not int or value != expected):
            raise ValueError('Published image dimensions differ from native grid')
    result = []
    for shape in data['shapes']:
        if not isinstance(shape, dict) or shape.get('shape_type') != 'rectangle':
            raise ValueError('Only explicit LabelMe rectangle shapes supported')
        label = shape.get('label')
        if not isinstance(label, str) or not label.strip():
            raise ValueError('Published nonempty region label required')
        points = np.asarray(shape.get('points'), float)
        if (points.shape != (2, 2) or not np.isfinite(points).all() or (points < 0).any()
                or (points[:, 0] > width).any() or (points[:, 1] > height).any()):
            raise ValueError('Finite opposite rectangle corners inside native frame required')
        left, top = points.min(axis=0)
        right, bottom = points.max(axis=0)
        if right <= left or bottom <= top:
            raise ValueError('Published rectangle has zero area')
        result.append({'region_id': f'L{len(result)+1:04d}', 'published_label': label,
            'polygon_native_xy': [[float(left), float(top)], [float(right), float(top)],
                                  [float(right), float(bottom)], [float(left), float(bottom)]]})
    return result
