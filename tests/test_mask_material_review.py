"""Protect source binding, explicit decisions and phone navigation without ML/SDK."""
import hashlib
import json
import os
import struct
import zlib
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

import numpy as np
import pytest

from pvdelta.mask_material_review import (
    latest_material_work,
    load_material_pack,
    new_material_work,
    save_material_decision,
    validate_material_work,
)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _write_pack(path, pack):
    identity = {k: v for k, v in pack.items() if k != 'pack_id'}
    pack['pack_id'] = 'sha256:' + hashlib.sha256(_canonical(identity)).hexdigest()
    path.write_bytes(_canonical(pack))


def _png(width, height):
    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))
    scanlines = (b'\0' + b'\x80\x80\x80' * width) * height
    return (b'\x89PNG\r\n\x1a\n'
            + chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0))
            + chunk(b'IDAT', zlib.compress(scanlines)) + chunk(b'IEND', b''))


@pytest.fixture
def prepared_pack(tmp_path):
    for name, width, height in [('native.png', 5, 4), ('crop.png', 24, 24)]:
        (tmp_path / name).write_bytes(_png(width, height))
    coords = [[1, 1], [1, 2]]
    pixel_hash = hashlib.sha256(np.asarray(coords, dtype='<i4').tobytes()).hexdigest()
    items = []
    for index in range(2):
        items.append({
            'item_id': f'C{index + 1:03d}-R01', 'source_name': f'synthetic-{index}_T.JPG',
            'source_sha256': str(index + 1) * 64, 'native_temperature_sha256': 'a' * 64,
            'target_pixels_sha256': pixel_hash, 'target_pixels_yx': coords,
            'target_pixel_count': 2, 'native_shape': [4, 5],
            'temperature_c': {'min': 20., 'median': 21., 'max': 22.},
            'crop': 'crop.png', 'locator': 'native.png', 'unmarked_native': 'native.png',
            'rgb_context': None,
            'asset_sha256': {key: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
                            for key, name in [('crop', 'crop.png'), ('locator', 'native.png'),
                                              ('unmarked_native', 'native.png')]},
        })
    pack = {'schema': 'pvdelta.mask-material-review-pack/v1', 'items': items}
    path = tmp_path / 'material-review-pack.json'
    _write_pack(path, pack)
    return path, pack


def test_open_is_read_only_and_unreviewed_is_not_negative(prepared_pack):
    path, _ = prepared_pack
    before = path.read_bytes()
    pack, root = load_material_pack(path)
    _, checked = latest_material_work(pack, root)
    assert checked['counts']['unreviewed'] == 2
    assert not checked['all_answered'] and not checked['all_material_resolved']
    assert not checked['model_acceptance_changed'] and not checked['database_changed']
    assert path.read_bytes() == before and not (root / 'operator-reviews').exists()


def test_saved_decisions_survive_reload_and_reselecting_an_earlier_version(prepared_pack):
    path, pack = prepared_pack
    first, _, _ = save_material_decision(path, 'C001-R01', 'glass', 'tester', note='synthetic')
    original = first.read_bytes()
    save_material_decision(path, 'C001-R01', 'unclear', 'tester')
    _, work, checked = save_material_decision(path, 'C001-R01', 'glass', 'tester', note='synthetic')
    restored, saved = latest_material_work(pack, path.parent, reviewer='tester')
    assert restored == work and saved == checked
    assert first.read_bytes() == original
    assert saved['counts']['glass'] == 1 and saved['counts']['unreviewed'] == 1
    assert not saved['electrical_fault_or_glare_approval']


def test_different_reviewers_are_not_reattributed(prepared_pack):
    path, pack = prepared_pack
    save_material_decision(path, 'C001-R01', 'glass', 'reviewer-a')
    save_material_decision(path, 'C002-R01', 'unclear', 'reviewer-b')
    _, first = latest_material_work(pack, path.parent, reviewer='reviewer-a')
    _, second = latest_material_work(pack, path.parent, reviewer='reviewer-b')
    assert first['reviewer'] == 'reviewer-a' and first['counts']['glass'] == 1
    assert second['reviewer'] == 'reviewer-b' and second['counts']['glass'] == 0
    assert second['counts']['unclear'] == 1 and not second['all_material_resolved']


def test_two_phone_sessions_merge_decisions_for_same_reviewer(prepared_pack):
    path, pack = prepared_pack
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(save_material_decision, path, case, value, 'same-reviewer')
                   for case, value in [('C001-R01', 'glass'), ('C002-R01', 'frame_or_nonpanel')]]
        for future in futures:
            future.result()
    _, latest = latest_material_work(pack, path.parent, reviewer='same-reviewer')
    assert latest['all_material_resolved'] and latest['counts']['glass'] == 1
    assert latest['counts']['frame_or_nonpanel'] == 1
    assert not latest['model_acceptance_changed']


def test_commit_order_survives_equal_or_backwards_file_times_and_revert(prepared_pack):
    path, pack = prepared_pack
    first, _, _ = save_material_decision(path, 'C001-R01', 'glass', 'tester')
    second, _, _ = save_material_decision(path, 'C002-R01', 'frame_or_nonpanel', 'tester')
    for timestamp in (2_000_000_000, 1):
        os.utime(first, (2_000_000_000, 2_000_000_000))
        os.utime(second, (timestamp, timestamp))
        assert latest_material_work(pack, path.parent)[1]['all_material_resolved']
    restored, _, _ = save_material_decision(path, 'C002-R01', 'unreviewed', 'tester')
    assert restored == first
    os.utime(first, (1, 1))
    os.utime(second, (2_000_000_000, 2_000_000_000))
    assert latest_material_work(pack, path.parent)[1]['counts']['unreviewed'] == 1


@pytest.mark.parametrize('corruption', ['missing-version', 'wrong-reviewer', 'invalid-revision'])
def test_corrupt_commit_head_is_rejected_without_time_fallback(prepared_pack, corruption):
    path, pack = prepared_pack
    save_material_decision(path, 'C001-R01', 'glass', 'tester')
    head_path = next((path.parent / 'operator-reviews').glob('current-*.json'))
    head = json.loads(head_path.read_bytes())
    if corruption == 'missing-version':
        head['latest'] = '0' * 64
    elif corruption == 'wrong-reviewer':
        head['reviewers'] = {'somebody-else': head['latest']}
    else:
        head['revision'] = True
    head_path.write_bytes(_canonical(head))
    with pytest.raises(ValueError, match='commit head'):
        latest_material_work(pack, path.parent)
    with pytest.raises(ValueError, match='commit head'):
        save_material_decision(path, 'C002-R01', 'glass', 'tester')


def test_failed_commit_retains_previous_head_and_does_not_merge_orphan(prepared_pack, monkeypatch):
    path, pack = prepared_pack
    save_material_decision(path, 'C001-R01', 'glass', 'tester')
    replace = os.replace

    def interrupted(*args):
        raise OSError('simulated interrupted commit')

    monkeypatch.setattr(os, 'replace', interrupted)
    with pytest.raises(OSError, match='interrupted'):
        save_material_decision(path, 'C002-R01', 'frame_or_nonpanel', 'tester')
    assert latest_material_work(pack, path.parent)[1]['counts']['unreviewed'] == 1
    assert not list((path.parent / 'operator-reviews').glob('*.tmp'))
    monkeypatch.setattr(os, 'replace', replace)
    save_material_decision(path, 'C001-R01', 'unclear', 'tester')
    checked = latest_material_work(pack, path.parent)[1]
    assert checked['counts']['unclear'] == checked['counts']['unreviewed'] == 1


def test_legacy_versions_migrate_without_losing_other_reviewers(prepared_pack):
    path, pack = prepared_pack
    save_material_decision(path, 'C001-R01', 'glass', 'reviewer-a')
    save_material_decision(path, 'C002-R01', 'unclear', 'reviewer-b')
    next((path.parent / 'operator-reviews').glob('current-*.json')).unlink()
    save_material_decision(path, 'C002-R01', 'glass', 'reviewer-a')
    assert latest_material_work(pack, path.parent, reviewer='reviewer-a')[1]['all_material_resolved']
    assert latest_material_work(pack, path.parent, reviewer='reviewer-b')[1]['counts']['unclear'] == 1


@pytest.mark.parametrize('field,value', [
    ('source_sha256', '0' * 64), ('target_pixels_sha256', '0' * 64),
    ('classification', 'confirmed_fault'),
])
def test_wrong_sources_pixels_and_materials_are_rejected(prepared_pack, field, value):
    _, pack = prepared_pack
    work = new_material_work(pack, 'tester')
    work['items'][0][field] = value
    with pytest.raises(ValueError):
        validate_material_work(pack, _canonical(work))


def test_missing_reviewer_and_stale_pack_cannot_save(prepared_pack):
    path, _ = prepared_pack
    with pytest.raises(ValueError, match='reviewer'):
        save_material_decision(path, 'C001-R01', 'glass', '')
    with pytest.raises(ValueError, match='changed'):
        save_material_decision(path, 'C001-R01', 'glass', 'tester', expected_pack_id='stale')
    assert not (path.parent / 'operator-reviews').exists()


def test_changed_assets_and_escaped_asset_paths_are_rejected(prepared_pack):
    path, pack = prepared_pack
    (path.parent / 'crop.png').write_bytes(b'changed')
    with pytest.raises(ValueError, match='image changed'):
        load_material_pack(path)
    invalid = deepcopy(pack)
    invalid['items'][0]['crop'] = '../outside.png'
    _write_pack(path, invalid)
    with pytest.raises(ValueError, match='outside'):
        load_material_pack(path)


def test_duplicate_and_out_of_grid_pixels_are_rejected(prepared_pack):
    path, pack = prepared_pack
    for coords in ([[1, 1], [1, 1]], [[1, 1], [4, 2]]):
        invalid = deepcopy(pack)
        invalid['items'][0]['target_pixels_yx'] = coords
        _write_pack(path, invalid)
        with pytest.raises(ValueError):
            load_material_pack(path)


def test_streamlit_navigation_and_direct_save_survive_reruns(prepared_pack):
    AppTest = pytest.importorskip('streamlit.testing.v1').AppTest
    path, pack = prepared_pack
    script = path.parent / 'material-review-app.py'
    script.write_text(
        'from pvdelta.mask_material_review_ui import material_review_view\n'
        f'material_review_view({str(path)!r}, key="material-test")\n', encoding='utf-8',
    )
    app = AppTest.from_file(str(script), default_timeout=30).run()
    assert not app.exception and not (path.parent / 'operator-reviews').exists()
    app.checkbox(key='material-test-open').check().run()
    scope = f'material-test-{pack["pack_id"][7:19]}'
    assert not app.exception and app.button(key=f'{scope}-glass').disabled
    app.text_input(key=f'{scope}-reviewer').set_value('synthetic-reviewer').run()
    app.button(key=f'{scope}-glass').click().run()
    assert not app.exception and not app.error
    _, checked = latest_material_work(pack, path.parent, reviewer='synthetic-reviewer')
    assert checked['counts']['glass'] == 1
    app.button(key=f'{scope}-item-next').click().run()
    assert not app.exception and app.selectbox(key=f'{scope}-item').value == 'C002-R01'
    app.button(key=f'{scope}-unclear').click().run()
    assert not app.exception
    _, checked = latest_material_work(pack, path.parent, reviewer='synthetic-reviewer')
    assert checked['all_answered'] and not checked['all_material_resolved']
    app.button(key=f'{scope}-item-previous').click().run()
    assert app.selectbox(key=f'{scope}-item').value == 'C001-R01' and not app.exception
    assert path.read_bytes() == _canonical(pack)
