import json
from copy import deepcopy

import pytest
from test_region_annotation import prepared as prepared  # noqa: PLC0414 -- re-export pytest fixture

from pvdelta.surface_reference import audit_surface_predictions, import_surface_work


def poly(identifier, x):
    return {'region_id': identifier, 'polygon_native_xy': [[x,5],[x+4,5],[x+4,9],[x,9]]}


def test_import_preserves_overlap_as_unknown_and_requires_confirmed_meaning(prepared):
    _, pack, work = prepared
    work['frames'][0].update(regions=[poly('R1',5)], ignore=[poly('U1',7)],
                             exhaustive=True, reviewer='synthetic')
    original = deepcopy(work)
    with pytest.raises(ValueError, match='confirmation'):
        import_surface_work(pack, json.dumps(work), confirmed_meaning='hotspot')
    ref, masks = import_surface_work(pack, json.dumps(work),
                                     confirmed_meaning='panel_surface_and_non_panel')
    assert work == original
    counts = ref['frames'][0]['pixel_counts']
    assert counts['1'] == counts['2'] == counts['3'] == 8
    assert int((masks['a'*64] == 0).sum()) == 4800-24
    assert not ref['independent_review_verified']


def test_candidates_use_exact_pixels_and_unlabelled_is_not_nonpanel(prepared):
    _, pack, work = prepared
    work['frames'][0].update(regions=[poly('R1',5)], ignore=[poly('U1',20)],
                             exhaustive=True, reviewer='synthetic')
    ref, masks = import_surface_work(pack, json.dumps(work),
                                     confirmed_meaning='panel_surface_and_non_panel')
    p = {'source_sha256':'a'*64,'native_shape':[60,80],
         'proposals':[{'proposal_id':'M1','polygon_native_xy':[[5,5],[24,5],[24,9],[5,9]]}],
         'findings':[{'finding_id':'hot','region_pixels_yx':[[5,5],[5,6]],
                      'polygon_native_xy':[[0,0],[30,0],[30,30],[0,30]]},
                     {'finding_id':'unknown','region_pixels_yx':[[40,40]]}]}
    original = deepcopy(p)
    report = audit_surface_predictions(ref['frames'][0], masks['a'*64], p)
    assert p == original
    assert report['findings'][0]['support'] == 'panel_only'
    assert report['findings'][1]['support'] == 'unresolved'
    assert report['modules'][0]['support'] == 'mixed'
    assert report['panel_pixels'] == report['panel_pixels_in_proposed_envelopes'] == 16


def test_unfinished_surface_remains_pending(prepared):
    _, pack, work = prepared
    ref, masks = import_surface_work(pack, json.dumps(work),
                                     confirmed_meaning='panel_surface_and_non_panel')
    report = audit_surface_predictions(ref['frames'][0], masks['a'*64],
                                       {'source_sha256':'a'*64,'native_shape':[60,80]})
    assert report['status'] == 'pending' and report['findings'] is None


@pytest.mark.parametrize('pixels', [[[5,5],[5,5]], [[60,5]], [[True,5]], [], [[5.2,5]]])
def test_bad_native_candidate_coordinates_rejected(prepared, pixels):
    _, pack, work = prepared
    work['frames'][0].update(exhaustive=True, reviewer='synthetic', empty_reason='Empty case')
    ref, masks = import_surface_work(pack, json.dumps(work),
                                     confirmed_meaning='panel_surface_and_non_panel')
    with pytest.raises(ValueError, match='candidate pixel'):
        audit_surface_predictions(ref['frames'][0], masks['a'*64],
            {'source_sha256':'a'*64,'native_shape':[60,80],'proposals':[],
             'findings':[{'finding_id':'f','region_pixels_yx':pixels}]})
