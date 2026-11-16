import inspect

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest


def navigation_app():
    import streamlit as st

    from pvdelta.review_ui import finding_navigation

    scope = st.selectbox("Scope", ["all", "last", "empty"])
    ids = {"all": ["A", "B", "C"], "last": ["C"], "empty": []}[scope]
    for key in ("stage1-finding-id", "review-finding-id"):
        finding_navigation(ids, key=key, label=key)


@pytest.mark.parametrize("key", ["stage1-finding-id", "review-finding-id"])
def test_navigation_clicks_boundaries_and_filter_reset(key, tmp_path):
    script = tmp_path / "navigation.py"
    script.write_text(inspect.getsource(navigation_app) + "\nnavigation_app()\n", encoding="utf-8")
    app = AppTest.from_file(str(script)).run()
    assert not app.exception
    assert app.button(key=f"{key}-previous").disabled
    app.button(key=f"{key}-next").click().run()
    assert not app.exception
    assert app.selectbox(key=key).value == "B"
    app.button(key=f"{key}-next").click().run()
    assert app.selectbox(key=key).value == "C"
    assert app.button(key=f"{key}-next").disabled
    app.button(key=f"{key}-previous").click().run()
    assert app.selectbox(key=key).value == "B"
    app.selectbox[0].select("last").run()
    assert not app.exception
    assert app.selectbox(key=key).value == "C"
    app.selectbox[0].select("empty").run()
    assert not app.exception
    assert len(app.button) == 0
    app.selectbox[0].select("all").run()
    assert not app.exception
    assert app.selectbox(key=key).value == "A"


def repeat_app():
    from pathlib import Path

    from pvdelta.review_ui import repeat_view_review

    repeat_view_review(Path('test.duckdb'), 'finding-1', key='test')


def test_manual_pair_browser_works_without_site_profile_or_registration(tmp_path, monkeypatch):
    from PIL import Image

    from pvdelta import repeat_views

    def signature(db, fid, profile):
        if profile is not None:
            raise FileNotFoundError('No site geometry profile')
        return 'manual-state'

    rows = []
    for index in range(2):
        thermal, rgb = tmp_path/f'{index}_T.JPG', tmp_path/f'{index}_V.JPG'
        Image.new('RGB', (64, 51), 'gray').save(thermal)
        Image.new('RGB', (128, 102), 'blue').save(rgb)
        rows.append({'capture_id': str(index), 'status': 'unavailable', 'reason': 'unverified nearby capture',
            'thermal_path': str(thermal), 'rgb_path': str(rgb), 'time_gap_s': index})
    monkeypatch.setattr(repeat_views, 'repeat_evidence_signature', signature)
    monkeypatch.setattr(repeat_views, 'nearby_image_pairs', lambda *a, **k: rows)
    script = tmp_path/'manual.py'
    script.write_text(inspect.getsource(repeat_app)+'\nrepeat_app()\n', encoding='utf-8')
    # Cold dataframe/image imports on Windows can exceed the framework's3s default.
    # This test checks review behavior, not a cold-start performance limit.
    app = AppTest.from_file(str(script), default_timeout=30).run()
    assert app.button(key='test-find-views').disabled
    app.button(key='test-nearby-pairs').click().run()
    assert not app.exception and len(app.get('image')) == 2
    assert any('2 nearby image pairs · panel matching not requested' in x.value for x in app.caption)
    assert not any('Automatic panel association did not pass' in x.value for x in app.info)
    manual = next(x for x in app.selectbox if x.label == 'Nearby thermal frame')
    app.button(key=f'{manual.key}-next').click().run()
    assert not app.exception and app.selectbox(key=manual.key).value == '1'


def test_failed_matching_still_shows_navigable_manual_image_pairs(tmp_path, monkeypatch):
    from PIL import Image

    from pvdelta import repeat_views

    rows = []
    for index in range(2):
        rgb, thermal = tmp_path / f'{index}_V.jpg', tmp_path / f'{index}_T.jpg'
        Image.new('RGB', (400, 300), 'blue').save(rgb)
        Image.new('RGB', (64, 51), 'gray').save(thermal)
        rows.append({'status': 'unavailable', 'reason': 'target panel absent or ambiguous',
                     'capture_id': str(index), 'rgb_path': str(rgb), 'thermal_path': str(thermal),
                     'time_gap_s': index + 1})
    monkeypatch.setattr(repeat_views, 'build_repeat_evidence', lambda *a, **kw: rows)
    monkeypatch.setattr(repeat_views, 'repeat_evidence_signature', lambda *a: 'synthetic-test-evidence')
    script = tmp_path / 'repeat.py'
    script.write_text(inspect.getsource(repeat_app) + '\nrepeat_app()\n', encoding='utf-8')
    app = AppTest.from_file(str(script)).run()
    app.button(key='test-find-views').click().run()
    assert not app.exception
    assert any('0 provisional panel matches / 2 inspected views' in x.value for x in app.caption)
    assert len(app.get('image')) == 2
    manual = next(x for x in app.selectbox if x.label == 'Nearby thermal frame')
    assert manual.value == '0'
    app.button(key=f'{manual.key}-next').click().run()
    assert not app.exception
    assert app.selectbox(key=manual.key).value == '1'
    assert len(app.get('image')) == 2
    assert not app.warning


@pytest.mark.parametrize('sdk_available', [True, False])
def test_source_frame_is_navigable_with_native_render_or_safe_fallback(tmp_path, monkeypatch, sdk_available):
    import numpy as np
    from PIL import Image

    from pvdelta import decode, repeat_views

    monkeypatch.setattr(repeat_views, 'repeat_evidence_signature', lambda *a: 'synthetic-test-evidence')

    rgb, thermal = tmp_path / 'source_V.jpg', tmp_path / 'source_T.jpg'
    Image.new('RGB', (400, 300), 'blue').save(rgb)
    Image.new('RGB', (128, 102), 'gray').save(thermal)
    rows = [{'status': 'source', 'reason': 'original candidate frame',
             'capture_id': 'source', 'rgb_path': str(rgb), 'thermal_path': str(thermal),
             'module': 'M001', 'width_px': 64, 'height_px': 51,
             'rgb_polygon': [[10, 10], [50, 10], [50, 40], [10, 40]],
             'native_polygon': [[10, 10], [30, 10], [30, 40], [10, 40]],
             'thermal_geometry_accepted': True, 'findings': [], 'observation': {},
             'gimbal_yaw_deg': 0, 'gimbal_pitch_deg': -90}]
    calls = []

    class FakeSDK:
        def __init__(self, *args):
            calls.append('decode')
            if not sdk_available:
                raise RuntimeError('SDK unavailable')

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def processed_rgb(self):
            return np.zeros((51, 64, 3), dtype=np.uint8)

    monkeypatch.setattr(decode, 'RJpeg', FakeSDK)
    monkeypatch.setattr(repeat_views, 'build_repeat_evidence', lambda *a, **kw: rows)
    script = tmp_path / 'source_review.py'
    script.write_text(inspect.getsource(repeat_app) + '\nrepeat_app()\n', encoding='utf-8')
    app = AppTest.from_file(str(script)).run()
    app.button(key='test-find-views').click().run()
    assert not app.exception
    assert calls == ['decode']
    assert len(app.get('image')) == 2
    assert any(x.label == 'Panel evidence frame (including source)' for x in app.selectbox)
