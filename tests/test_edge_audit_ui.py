import pytest

pytest.importorskip('streamlit')
from streamlit.testing.v1 import AppTest
from test_gap_review_bundle import bundle_input  # noqa: F401

from pvdelta.gap_review_bundle import write_gap_bundle


def test_diagnostic_is_on_demand_and_frame_navigation_does_not_copy_results(tmp_path, request):
    folder = tmp_path/'bundle'
    write_gap_bundle(folder, [request.getfixturevalue('bundle_input')],
                     engine_source_sha256={'test': '1'*64})
    app = AppTest.from_string('from pvdelta.gap_review_ui import native_gap_review\n'
                             f'native_gap_review({str(folder)!r}, key="audit-ui")\n').run()
    assert not app.exception and len(app.get('image')) == 2
    app.button(key='audit-ui-inspect-edges').click().run()
    assert not app.exception and len(app.get('image')) == 3
    assert app.dataframe[0].value['Native ridge check'].tolist() == ['Unresolved']*4
    assert app.metric[1].value == '3'
    app.button(key='audit-ui-evidence-next').click().run()
    assert not app.exception and len(app.get('image')) == 2
    assert not app.dataframe
    assert not (folder/'reviews.sqlite3').exists()


def test_changed_audit_engine_invalidates_cached_overlay(tmp_path, request, monkeypatch):
    from pvdelta import native_edge_audit

    folder = tmp_path/'bundle'
    write_gap_bundle(folder, [request.getfixturevalue('bundle_input')],
                     engine_source_sha256={'test': '1'*64})
    app = AppTest.from_string('from pvdelta.gap_review_ui import native_gap_review\n'
                             f'native_gap_review({str(folder)!r}, key="audit-ui")\n').run()
    app.button(key='audit-ui-inspect-edges').click().run()
    original = native_edge_audit.edge_audit_identity
    def changed(*args, **kwargs):
        token, payload = original(*args, **kwargs)
        return 'changed-'+token, payload
    monkeypatch.setattr(native_edge_audit, 'edge_audit_identity', changed)
    app.run()
    assert not app.exception and len(app.get('image')) == 2 and not app.dataframe
