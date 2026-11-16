import hashlib
import json
from contextlib import nullcontext

import pytest
from test_native_flight import _proposals, native_source  # noqa: F401

from pvdelta import reference_evidence
from pvdelta.native_flight import scan_native_flight
from pvdelta.store import connect_store, upsert_finding_review
from pvdelta.temperature_evidence_store import (
    build_temperature_checks,
    check_and_save_temperature_evidence,
    evidence_path,
    load_temperature_checks,
    temperature_groups,
)


@pytest.fixture
def evidence_store(request, tmp_path, monkeypatch):
    source, decoder, _ = request.getfixturevalue('native_source')
    out = tmp_path/'native'
    scan_native_flight(source, 'flight', out, 'native-1', decoder=decoder, proposal_builder=_proposals)
    database = out/'inspection.duckdb'
    sdk = tmp_path/'sdk'
    sdk.mkdir()
    (sdk/'libdirp.dll').write_bytes(b'test SDK identity only')
    monkeypatch.setattr(reference_evidence, 'RJpeg', lambda *a: nullcontext(None))
    monkeypatch.setattr(reference_evidence, 'apply_saved_measurement_profile',
        lambda reader, path, profile: decoder(path, None).temperature_c)
    c = connect_store(database, read_only=True)
    records = c.execute('SELECT observation_id,finding_id FROM findings ORDER BY 1').fetchall()
    c.close()
    return database, sdk, records


def test_read_only_open_then_check_reuse_and_review_independence(evidence_store, monkeypatch):
    database, sdk, records = evidence_store
    before = hashlib.sha256(database.read_bytes()).hexdigest()
    assert {v['status'] for v in load_temperature_checks(database, 'native-1').values()} == {'missing'}
    assert not evidence_path(database, 'native-1', records[0][0]).parent.exists()
    result = build_temperature_checks(database, 'native-1', sdk_library=sdk)
    assert result == {'checked': 2, 'reused': 0, 'failed': []}
    assert hashlib.sha256(database.read_bytes()).hexdigest() == before
    monkeypatch.setattr(reference_evidence, 'check_temperature_evidence',
        lambda *a, **k: pytest.fail('Opening or reusing evidence must not decode'))
    assert {v['status'] for v in load_temperature_checks(database, 'native-1', sdk_library=sdk).values()} == {'available'}
    assert build_temperature_checks(database, 'native-1', sdk_library=sdk) == {'checked': 0, 'reused': 2, 'failed': []}
    c = connect_store(database)
    upsert_finding_review(c, finding_id=records[0][1], review_status='not_ok', note='Human decision')
    c.close()
    assert {v['status'] for v in load_temperature_checks(database, 'native-1', sdk_library=sdk).values()} == {'available'}


@pytest.mark.parametrize('change', ['findings', 'source', 'settings', 'sdk', 'engine', 'corrupt', 'profile', 'geometry'])
def test_changes_are_stale_never_current(evidence_store, monkeypatch, change):
    from pvdelta import temperature_evidence_store as store

    database, sdk, records = evidence_store
    oid, fid = records[0]
    check_and_save_temperature_evidence(database, 'native-1', oid, sdk_library=sdk)
    if change == 'findings':
        c = connect_store(database)
        c.execute('UPDATE findings SET delta_t_k=delta_t_k+1 WHERE finding_id=?', [fid])
        c.close()
    elif change == 'source':
        # Source files are test-owned placeholders, not user originals.
        path = database.parent.parent/'flight/A_T.JPG'
        path.write_bytes(b'changed')
        # Hash ordering need not put A first: mutate both test sources.
        (path.parent/'C_T.JPG').write_bytes(b'changed')
    elif change == 'settings':
        path = database.with_name('summary.json')
        summary = json.loads(path.read_bytes())
        summary['settings']['edge_erosion_px'] += 1
        path.write_text(json.dumps(summary))
    elif change == 'sdk':
        (sdk/'libdirp.dll').write_bytes(b'updated SDK binary')
    elif change == 'engine':
        original = store._runtime
        monkeypatch.setattr(store, '_runtime', lambda sdk: original(sdk) | {'engine': {'changed': 'yes'}})
    elif change == 'profile':
        c = connect_store(database)
        c.execute("UPDATE measurement_profiles SET profile_json='{}'")
        c.close()
    elif change == 'geometry':
        c = connect_store(database)
        c.execute("UPDATE module_observations SET pixel_polygon_json='[[0,0],[3,0],[3,3],[0,3]]' WHERE observation_id=?", [oid])
        c.close()
    else:
        path = evidence_path(database, 'native-1', oid)
        saved = json.loads(path.read_bytes())
        saved['report']['regions'][0]['minimum_pixel_support'] = True
        path.write_text(json.dumps(saved))
    state = load_temperature_checks(database, 'native-1', sdk_library=sdk)[oid]
    assert state['status'] == 'stale' and 'report' not in state


def test_copied_other_observation_evidence_cannot_be_reused(evidence_store):
    database, sdk, records = evidence_store
    oid = records[0][0]
    other = records[1][0]
    check_and_save_temperature_evidence(database, 'native-1', oid, sdk_library=sdk)
    evidence_path(database, 'native-1', other).write_bytes(evidence_path(database, 'native-1', oid).read_bytes())
    checks = load_temperature_checks(database, 'native-1', sdk_library=sdk)
    assert checks[oid]['status'] == 'available' and checks[other]['status'] == 'stale'
    assert load_temperature_checks(database, 'other-run', sdk_library=sdk) == {}


def test_change_during_computation_prevents_saving(evidence_store, monkeypatch):
    database, sdk, records = evidence_store
    oid, fid = records[0]
    original = reference_evidence.check_temperature_evidence

    def changed(*args, **kwargs):
        report = original(*args, **kwargs)
        c = connect_store(database)
        c.execute('UPDATE findings SET n_reference_px=n_reference_px+1 WHERE finding_id=?', [fid])
        c.close()
        return report

    monkeypatch.setattr(reference_evidence, 'check_temperature_evidence', changed)
    with pytest.raises(ValueError, match='changed during'):
        check_and_save_temperature_evidence(database, 'native-1', oid, sdk_library=sdk)
    assert not evidence_path(database, 'native-1', oid).exists()


def test_queue_preserves_mixed_card_regions_and_unknown_checks():
    groups = [{'observation_id': 'a', 'regions': ['r1', 'r2']}, {'observation_id': 'b', 'regions': ['r3']}]
    checks = {'a': {'status': 'available', 'report': {
        'regions': [{'minimum_pixel_support': False}, {'minimum_pixel_support': True}],
        'sensitivity': {'regions': {'r1': {'reference_test': 'crosses_threshold'},
                                    'r2': {'reference_test': 'above_threshold_in_all_folds'}}}}}}
    assert temperature_groups(groups, checks, 'Reference-sensitive') == [groups[0]]
    assert temperature_groups(groups, checks, 'Minimum pixel support') == [groups[0]]
    assert temperature_groups(groups, checks, 'Checks missing / stale') == [groups[1]]
