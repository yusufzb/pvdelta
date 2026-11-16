import hashlib
from pathlib import Path

import numpy as np
import pytest
from test_gap_review_bundle import bundle_input  # noqa: F401

from pvdelta.gap_context import load_gap_context, write_gap_context
from pvdelta.gap_review_bundle import load_gap_bundle, write_gap_bundle


@pytest.fixture
def context_bundle(tmp_path, request):
    fixture_data = request.getfixturevalue('bundle_input')
    folder = tmp_path/'review'
    write_gap_bundle(folder, [fixture_data], engine_source_sha256={'synthetic': '1'*64})
    parent = load_gap_bundle(folder)[0]
    views = []
    for index in range(2):
        source = tmp_path/f'comparison-{index}.JPG'
        source.write_bytes(f'comparison {index}'.encode())
        native = np.full((512, 640), 26.+index, dtype=np.float32)
        check = {'schema': 'pvdelta.observed-panel-pair-check/v1', 'config': {}, 'matches': [],
                 'heldout_p95_px': None, 'status': 'withheld', 'reasons': ['insufficient_observed_panel_pairs'],
                 'same_panel_verified': False, 'measurement_authorised': False, 'glare_clearance': False}
        views.append({'source_path': str(source), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                      'native': native, 'native_temperature_sha256': hashlib.sha256(native.tobytes()).hexdigest(),
                      'basis': 'Synthetic test only', 'parameters': {}, 'panel_check': check})
    return folder, parent, views


def export(context_bundle):
    folder, parent, views = context_bundle
    return write_gap_context(folder, parent['record_id'], views, engine_source_sha256={'checker': '2'*64})


def test_context_keeps_original_review_and_native_values_unchanged(context_bundle):
    folder, parent, views = context_bundle
    before = (folder/'manifest.json').read_bytes()
    assert load_gap_context(folder, parent['record_id']) is None
    export(context_bundle)
    result = load_gap_context(folder, parent['record_id'])
    assert len(result['views']) == 2 and (folder/'manifest.json').read_bytes() == before
    for row, original in zip(result['views'], views):
        np.testing.assert_array_equal(row['native'], original['native'])
        assert not row['native'].flags.writeable
        assert row['panel_check']['same_panel_verified'] is False
    assert not (folder/'reviews.sqlite3').exists()
    with pytest.raises(FileExistsError):
        export(context_bundle)


@pytest.mark.parametrize('change', ['source', 'native', 'manifest', 'parent'])
def test_stale_context_is_withheld(context_bundle, change):
    folder, parent, views = context_bundle
    context = export(context_bundle)
    if change == 'source': Path(views[0]['source_path']).write_bytes(b'replaced')
    if change == 'native': np.save(context/'native-0000.npy', views[0]['native']+1)
    if change == 'manifest':
        path = context/'manifest.json'
        path.write_text(path.read_text().replace('Synthetic test only', 'false surface correction'))
    if change == 'parent': Path(parent['source_path']).write_bytes(b'replaced-parent')
    with pytest.raises(ValueError):
        load_gap_context(folder, parent['record_id'])


@pytest.mark.parametrize('flag', ['same_panel_verified', 'measurement_authorised', 'glare_clearance'])
def test_cannot_serialize_context_as_accepted_match(context_bundle, flag):
    folder, parent, views = context_bundle
    views[0]['panel_check'][flag] = True
    with pytest.raises(ValueError, match='unverified'):
        export(context_bundle)
    assert not (folder/'context'/parent['record_id']).exists()


def test_repeated_frame_is_not_an_extra_view(context_bundle):
    context_bundle[2].append(context_bundle[2][0])
    with pytest.raises(ValueError, match='repeats'):
        export(context_bundle)


def test_path_traversal_parent_is_rejected(context_bundle):
    with pytest.raises(ValueError, match='SHA256'):
        load_gap_context(context_bundle[0], '../elsewhere')


def test_gallery_navigation_and_stale_context_never_project_a_box(context_bundle):
    pytest.importorskip('streamlit')
    pytest.importorskip('PIL')
    from streamlit.testing.v1 import AppTest

    folder, _, views = context_bundle
    export(context_bundle)
    app = AppTest.from_string('from pvdelta.gap_review_ui import native_gap_review\n'
                             f'native_gap_review({str(folder)!r}, key="context-ui")\n').run()
    assert not app.exception
    navigation = next(s for s in app.selectbox if s.label == 'Comparison frame')
    first = navigation.value
    key = navigation.key
    app.button(key=f'{key}-next').click().run()
    assert not app.exception and app.selectbox(key=key).value != first
    app.button(key=f'{key}-previous').click().run()
    assert app.selectbox(key=key).value == first
    assert len(app.get('image')) == 4  # primary crop/full + source/comparison; no transformed measurement
    assert any('Same-panel match: withheld' in m.value for m in app.markdown)
    Path(views[0]['source_path']).unlink()
    app.run()
    assert not app.exception and len(app.get('image')) == 2  # valid primary evidence stays visible
    assert any('Comparison frames withheld' in e.value for e in app.error)
