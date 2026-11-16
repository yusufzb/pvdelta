import hashlib
import json

import numpy as np
import pytest
from test_native_flight import _proposals, native_source  # noqa: F401

from pvdelta.candidate_triage import candidate_support_audit, region_support, support_groups
from pvdelta.native_flight import scan_native_flight
from pvdelta.store import connect_store


def test_exact_pixels_not_display_box_and_internal_exclusion_edges():
    mask = np.zeros((20, 20), bool)
    mask[2:18, 2:18] = True
    assert region_support(mask, [[10, 10]], 1)['support'] == 'interior'
    assert region_support(mask, [[2, 10]], 1)['support'] == 'boundary'
    mask[9, 10] = False  # ambiguous overlapping ownership is a measurement edge too
    assert region_support(mask, [[10, 10]], 1)['support'] == 'boundary'
    with pytest.raises(ValueError, match='outside'):
        region_support(mask, [[9, 10]], 1)


@pytest.mark.parametrize('points,count', [([[3, 3], [3, 3]], 2), ([[3.5, 3]], 1),
    ([[3, 3]], 2), ([[-1, 3]], 1), ([[30, 3]], 1), (None, 1)])
def test_malformed_support_is_not_promoted(points, count):
    with pytest.raises(ValueError):
        region_support(np.ones((20, 20), bool), points, count)


def test_saved_masks_bound_to_run_fail_closed_on_changed_hash_or_settings(request, tmp_path):
    source, decoder, _ = request.getfixturevalue('native_source')
    out = tmp_path/'native'
    scan_native_flight(source, 'flight', out, 'native-1', decoder=decoder, proposal_builder=_proposals)
    db = out/'inspection.duckdb'
    before = hashlib.sha256(db.read_bytes()).hexdigest()
    c = connect_store(db, read_only=True)
    audit = candidate_support_audit(c, db, 'native-1')
    assert len(audit) == 2 and {v['support'] for v in audit.values()} == {'interior'}
    c.close()
    assert hashlib.sha256(db.read_bytes()).hexdigest() == before
    c = connect_store(db)
    cid, raw = c.execute("SELECT capture_id,diagnostics_json FROM flight_screening_frames WHERE measured > 0 LIMIT 1").fetchone()
    diagnostic = json.loads(raw)
    diagnostic['modules'][0]['mask_sha256'] = 'wrong'
    c.execute('UPDATE flight_screening_frames SET diagnostics_json=? WHERE capture_id=?', [json.dumps(diagnostic), cid])
    audit = candidate_support_audit(c, db, 'native-1')
    assert sorted(v['support'] for v in audit.values()) == ['interior', 'unavailable']
    summary = json.loads((out/'summary.json').read_bytes())
    summary['analysis_run_id'] = 'other-run'
    (out/'summary.json').write_text(json.dumps(summary))
    assert {v['support'] for v in candidate_support_audit(c, db, 'native-1').values()} == {'unavailable'}
    c.close()


def test_queue_preserves_mixed_card_and_all_evidence():
    groups = [{'regions': [{'finding_id': 'a'}, {'finding_id': 'b'}]},
              {'regions': [{'finding_id': 'c'}]}]
    audit = {'a': {'support': 'interior'}, 'b': {'support': 'boundary'}, 'c': {'support': 'unavailable'}}
    assert support_groups(groups, audit, 'Interior candidates') == [groups[0]]
    assert support_groups(groups, audit, 'Boundary review') == [groups[0]]
    assert support_groups(groups, audit, 'Support unavailable') == [groups[1]]
    assert support_groups(groups, audit, 'All candidates') is groups
