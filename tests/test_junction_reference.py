import hashlib
import json
from copy import deepcopy

import numpy as np
import pytest
from PIL import Image

from pvdelta.junction_reference import (
    check_junction_references,
    prepare_junction_references,
    validate_junction_references,
)
from pvdelta.models import ThermalFrame


@pytest.fixture
def reference(tmp_path):
    source = tmp_path/'T.bin'
    source.write_bytes(b'original source')
    rgb = tmp_path/'V.png'
    Image.new('RGB', (300, 240), 'gray').save(rgb)
    values = np.arange(120*150, dtype=np.float32).reshape(120, 150)/100
    frame = ThermalFrame('f', str(source), hashlib.sha256(source.read_bytes()).hexdigest(), values, None, 150, 120)
    pairs = [(frame, rgb)]
    pack = prepare_junction_references(pairs, tmp_path/'references')
    locations = [[x, y] for y in (20, 50, 80) for x in (20, 50, 80, 110)]
    work = {'schema': 'pvdelta.junction-reference-work/v1', 'pack_id': pack['pack_id'],
        'reviewer': 'test reviewer', 'frames': [{
            'pair_id': pack['frames'][0]['pair_id'], 'pair': deepcopy(pack['frames'][0]['pair']),
            'points': [{**s, 'native_xy': [x, y], 'rgb_xy': [2*x+3, 2*y+4], 'status': 'observed'}
                       for s, (x, y) in zip(pack['frames'][0]['slots'], locations, strict=True)]}]}
    return pairs, pack, work


def test_source_only_reference_preserves_native_and_fits_known_transform(reference):
    pairs, pack, work = reference
    before = pairs[0][0].temperature_c.copy()
    result = check_junction_references(pairs, pack, work)
    assert result['all_references_complete']
    assert result['pairs'][0]['status'] == 'point_gate_passed_provisional'
    registration = result['pairs'][0]['registration']
    assert len(registration['fit_control_ids']) == 8 and len(registration['check_control_ids']) == 4
    assert registration['validation']['check_p95_source_px'] < 1e-6
    assert not result['physical_correspondence_verified'] and not result['measurement_authorised']
    np.testing.assert_array_equal(before, pairs[0][0].temperature_c)


@pytest.mark.parametrize('bad', ['reviewer', 'pack', 'role', 'source', 'missing', 'bounds', 'duplicate', 'nan'])
def test_reference_validation_rejects_wrong_sources_roles_and_coordinates(reference, bad):
    _, pack, work = reference
    point = work['frames'][0]['points'][0]
    if bad == 'reviewer':
        work['reviewer'] = ' '
    elif bad == 'pack':
        work['pack_id'] = 'wrong'
    elif bad == 'role':
        point['role'] = 'check'
    elif bad == 'source':
        work['frames'][0]['pair']['source_sha256'] = 'wrong'
    elif bad == 'missing':
        work['frames'][0]['points'].pop()
    elif bad == 'bounds':
        point['native_xy'] = [150, 0]
    elif bad == 'duplicate':
        point['native_xy'] = work['frames'][0]['points'][1]['native_xy']
    else:
        point['native_xy'] = [float('nan'), 0]
    with pytest.raises(ValueError):
        validate_junction_references(pack, work)


def test_unclear_reference_remains_pending_without_exposing_a_transform(reference):
    pairs, pack, work = reference
    work['frames'][0]['points'][0].update(status='unclear', native_xy=None, rgb_xy=None)
    result = check_junction_references(pairs, pack, work)
    assert not result['all_references_complete']
    assert result['pairs'][0]['status'] == 'source_reference_pending'
    assert result['pairs'][0]['pending_slots'] == ['J01']
    assert result['pairs'][0]['registration'] is None


def test_complete_but_collinear_checks_stay_explicitly_rejected(reference):
    pairs, pack, work = reference
    for p in work['frames'][0]['points']:
        if p['role'] == 'check':
            p['native_xy'][1] = 55
            p['rgb_xy'][1] = 114
    result = check_junction_references(pairs, pack, work)
    assert result['all_references_complete']
    assert result['pairs'][0]['status'] == 'point_geometry_invalid'
    assert result['pairs'][0]['registration'] is None


def test_actual_source_change_is_not_accepted_as_old_reference(reference):
    pairs, pack, work = reference
    pairs[0][1].write_bytes(b'changed RGB')
    with pytest.raises(ValueError):
        check_junction_references(pairs, pack, work)


def test_preparation_cannot_overwrite_completed_reference_pack(reference, tmp_path):
    pairs, _, _ = reference
    with pytest.raises(FileExistsError):
        prepare_junction_references(pairs, tmp_path/'references')
    page = (tmp_path/'references/review.html').read_text(encoding='utf-8')
    assert '__PACK_JSON__' not in page and '__IMAGES_JSON__' not in page
    assert 'No detector predictions or fitted overlays are shown' in page
    assert 'assets/P01-thermal.png' in page
    assert (tmp_path/'references/assets/P01-rgb.png').is_file()
    assert json.loads((tmp_path/'references/pack.json').read_bytes())['frames'][0]['slots'][1]['role'] == 'check'
