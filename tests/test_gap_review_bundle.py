import hashlib
import json
import sqlite3
from copy import deepcopy

import numpy as np
import pytest

from pvdelta.gap_review_bundle import (
    evidence_snapshot,
    gap_decisions,
    load_gap_bundle,
    save_gap_decision,
    write_gap_bundle,
)


@pytest.fixture
def bundle_input(tmp_path):
    source = tmp_path / 'source_T.JPG'
    source.write_bytes(b'private raw source')
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    values = np.full((512, 640), 25., dtype=np.float32)
    items = []
    for x in (50, 170, 400):
        pixels = [[100, x], [100, x+1]]
        values[100, x:x+2] = 35
        evidence_id = hashlib.sha256(source_hash.encode() + np.asarray(pixels, dtype='<i4').tobytes()).hexdigest()[:24]
        items.append({'evidence_id': evidence_id, 'source_hash': source_hash,
                      'status': 'unassigned_thermal_evidence_review_required',
                      'confirmed_fault': False, 'measurement_authorised': False,
                      'module_identity': None, 'panel_location': None,
                      'provisional_polygon_native_xy': [[x-10, 90], [x+10, 90], [x+10, 110], [x-10, 110]],
                      'native_finding': {'region_pixels_yx': pixels, 'n_region_px': 2,
                                         'anomaly_type': 'hot_region', 'delta_t_c': 10., 'region_max_t_c': 35.}})
    report = {'schema': 'pvdelta.native-gap-review/v1', 'source_hash': source_hash,
              'native_temperature_sha256': hashlib.sha256(values.tobytes()).hexdigest(),
              'baseline_unchanged': True, 'production_enabled': False, 'evidence': items,
              'counters': {}, 'hypotheses_withheld_by_analysis_budget': 0}
    return report, values, source, 'Synthetic native test; not a field measurement', {'emissivity': .95}


def export(tmp_path, bundle_input):
    folder = tmp_path / 'bundle'
    write_gap_bundle(folder, [bundle_input], engine_source_sha256={'synthetic': '1'*64})
    return folder


def test_exact_native_data_and_identity_roundtrip_without_review_or_inspection_write(tmp_path, bundle_input):
    folder = export(tmp_path, bundle_input)
    frame = load_gap_bundle(folder)[0]
    np.testing.assert_array_equal(frame['native'], bundle_input[1])
    assert not frame['native'].flags.writeable
    assert not (folder/'reviews.sqlite3').exists()
    assert gap_decisions(folder) == {}
    assert frame['report'] == bundle_input[0]
    with pytest.raises(FileExistsError):
        write_gap_bundle(folder, [bundle_input], engine_source_sha256={'synthetic': '1'*64})


@pytest.mark.parametrize('change', ['source', 'native_file', 'snapshot', 'raw_peak', 'fractional', 'duplicate', 'outside', 'fault'])
def test_stale_or_misleading_data_fails_closed(tmp_path, bundle_input, change):
    report, values, source, basis, params = bundle_input
    folder = export(tmp_path, bundle_input)
    if change == 'source':
        source.write_bytes(b'changed')
    elif change == 'native_file':
        np.save(folder/'native-0000.npy', values + 1)
    elif change == 'snapshot':
        manifest = json.loads((folder/'manifest.json').read_text())
        manifest['frames'][0]['basis'] = 'false field calibration claim'
        (folder/'manifest.json').write_text(json.dumps(manifest))
    else:
        report = deepcopy(report)
        item = report['evidence'][0]
        if change == 'raw_peak': item['native_finding']['region_max_t_c'] = 99
        if change == 'fractional': item['native_finding']['region_pixels_yx'][0][0] = 100.5
        if change == 'duplicate': item['native_finding']['region_pixels_yx'][1] = [100, 50]
        if change == 'outside': item['provisional_polygon_native_xy'][0][0] = -1
        if change == 'fault': item['confirmed_fault'] = True
        with pytest.raises((ValueError, TypeError)):
            write_gap_bundle(tmp_path/'invalid', [(report, values, source, basis, params)],
                             engine_source_sha256={'synthetic': '1'*64})
        assert not (tmp_path/'invalid').exists()
        return
    with pytest.raises(ValueError):
        load_gap_bundle(folder)


def test_native_file_cannot_escape_bundle_even_with_updated_manifest_identity(tmp_path, bundle_input):
    folder = export(tmp_path, bundle_input)
    path = folder/'manifest.json'
    manifest = json.loads(path.read_text())
    entry = manifest['frames'][0]
    entry['native_file'] = '../elsewhere.npy'
    entry.pop('record_id')
    entry['record_id'] = hashlib.sha256(json.dumps(entry, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='inside the bundle'):
        load_gap_bundle(folder)


def test_append_only_decisions_bind_whole_snapshot_and_keep_other_evidence_unreviewed(tmp_path, bundle_input):
    folder = export(tmp_path, bundle_input)
    frame = load_gap_bundle(folder)[0]
    item = frame['report']['evidence'][0]
    for status in ('ok', 'not_ok', 'wait'):
        save_gap_decision(folder, frame, item, status, reviewer='Synthetic reviewer', note='boundary uncertain')
    decisions = gap_decisions(folder)
    identity, _ = evidence_snapshot(frame, item)
    assert len(decisions) == 1 and decisions[identity]['status'] == 'wait'
    assert decisions[identity]['reviewer'] == 'Synthetic reviewer'
    with sqlite3.connect(folder/'reviews.sqlite3') as connection:
        assert connection.execute('SELECT count(*) FROM gap_review_events').fetchone()[0] == 3
        assert connection.execute('SELECT status FROM gap_review_events ORDER BY event_id').fetchall() == [('ok',), ('not_ok',), ('wait',)]
    assert all(e['confirmed_fault'] is False for e in load_gap_bundle(folder)[0]['report']['evidence'])


def test_save_rechecks_source_and_stale_snapshot(tmp_path, bundle_input):
    folder = export(tmp_path, bundle_input)
    frame = load_gap_bundle(folder)[0]
    bundle_input[2].write_bytes(b'replaced')
    with pytest.raises(ValueError, match='Source SHA256'):
        save_gap_decision(folder, frame, frame['report']['evidence'][0], 'ok')
    assert not (folder/'reviews.sqlite3').exists()


def test_parameter_or_engine_change_requires_new_review_even_with_same_hot_pixels(tmp_path, bundle_input):
    folder = export(tmp_path, bundle_input)
    frame = load_gap_bundle(folder)[0]
    item = frame['report']['evidence'][0]
    save_gap_decision(folder, frame, item, 'ok')
    old, _ = evidence_snapshot(frame, item)
    changed = tmp_path/'changed-bundle'
    write_gap_bundle(changed, [bundle_input], engine_source_sha256={'synthetic': '2'*64})
    newer = load_gap_bundle(changed)[0]
    new, _ = evidence_snapshot(newer, newer['report']['evidence'][0])
    assert item['evidence_id'] == newer['report']['evidence'][0]['evidence_id']
    assert old != new and new not in gap_decisions(folder)
    with pytest.raises(ValueError, match='snapshot changed'):
        save_gap_decision(changed, frame, item, 'ok')
    assert not (changed/'reviews.sqlite3').exists()


def test_zero_evidence_retains_frame_and_does_not_invent_healthy_count(tmp_path, bundle_input):
    report, values, source, basis, params = bundle_input
    report['evidence'] = []
    folder = export(tmp_path, (report, values, source, basis, params))
    assert load_gap_bundle(folder)[0]['report']['evidence'] == []


def test_display_crop_does_not_modify_native_values_or_fill_hot_pixels(tmp_path, bundle_input):
    pytest.importorskip('PIL')
    from pvdelta.gap_review_ui import evidence_images

    folder = export(tmp_path, bundle_input)
    frame = load_gap_bundle(folder)[0]
    before = frame['native'].copy()
    full, crop = evidence_images(frame, frame['report']['evidence'][0])
    assert full.size == (640, 512) and crop.size == (500, 500)
    np.testing.assert_array_equal(frame['native'], before)
    rgb = np.asarray(full)
    assert rgb[100, 50].tolist() != [255, 255, 255]  # white ring is outside the measured region


def test_review_ui_navigation_decisions_filters_and_missing_source(tmp_path, bundle_input):
    pytest.importorskip('streamlit')
    pytest.importorskip('PIL')
    from streamlit.testing.v1 import AppTest

    folder = export(tmp_path, bundle_input)
    app = AppTest.from_string('from pvdelta.gap_review_ui import native_gap_review\n'
                             f'native_gap_review({str(folder)!r}, key="test-gap")\n').run()
    assert not app.exception
    assert app.metric[1].value == '3' and app.metric[2].value == '3'
    first = app.selectbox(key='test-gap-evidence').value
    app.button(key='test-gap-evidence-next').click().run()
    second = app.selectbox(key='test-gap-evidence').value
    assert first != second and not app.exception
    app.button(key='test-gap-decision-ok').click().run()
    assert not app.exception and app.metric[2].value == '2'
    app.button(key='test-gap-evidence-previous').click().run()
    assert app.selectbox(key='test-gap-evidence').value == first
    app.selectbox(key='test-gap-filter').select('OK').run()
    assert app.selectbox(key='test-gap-evidence').value == second
    assert app.button(key='test-gap-evidence-next').disabled
    app.button(key='test-gap-decision-wait').click().run()
    assert not app.exception and not app.get('image')
    app.selectbox(key='test-gap-filter').select('Wait').run()
    assert app.selectbox(key='test-gap-evidence').value == second
    bundle_input[2].unlink()
    app.run()
    assert not app.exception and app.error and not app.get('image')


def test_empty_bundle_ui_has_no_fault_health_count(tmp_path, bundle_input):
    pytest.importorskip('streamlit')
    from streamlit.testing.v1 import AppTest

    bundle_input[0]['evidence'] = []
    folder = export(tmp_path, bundle_input)
    app = AppTest.from_string('from pvdelta.gap_review_ui import native_gap_review\n'
                             f'native_gap_review({str(folder)!r})\n').run()
    assert not app.exception and app.metric[1].value == '0'
    assert not app.get('image') and not app.button
    assert all('healthy' not in m.label.lower() for m in app.metric)
