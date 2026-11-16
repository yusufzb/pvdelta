import hashlib
from copy import deepcopy

import numpy as np
import pytest

from pvdelta.models import ThermalFrame
from pvdelta.native_flight import analyze_native_frame


@pytest.fixture
def scene(tmp_path):
    source=tmp_path/'source_T.bin';source.write_bytes(b'original source')
    values=np.full((96,192),25.,np.float32)
    values[20:28,20:28]=35.
    values[20:28,80:88]=60.  # Hot background in a wrongly proposed module.
    values[20:28,158:166]=45.  # Unlabelled heat must not be silently discarded.
    frame=ThermalFrame('f',str(source),hashlib.sha256(source.read_bytes()).hexdigest(),values,None,192,96)
    proposals=[{'proposal_id':name,'polygon_native_xy':[[x,5],[x+50,5],[x+50,85],[x,85]]}
               for name,x in [('panel',5),('background',65),('partial',125)]]
    geometry={'proposals':proposals}
    reference={'review_complete':True,'source_sha256':frame.source_hash,
        'native_temperature_sha256':hashlib.sha256(values.tobytes()).hexdigest(),
        'native_height':96,'native_width':192,
        'panel_polygons':[{'polygon_native_xy':[[0,0],[60,0],[60,95],[0,95]]},
                          {'polygon_native_xy':[[120,0],[150,0],[150,95],[120,95]]}],
        'non_panel_polygons':[{'polygon_native_xy':[[60,0],[120,0],[120,95],[60,95]]}]}
    return frame,geometry,reference


def analyze(frame,geometry,reference):
    return analyze_native_frame(frame,proposal_builder=lambda *a,**k:geometry,
        surface_reference=reference,surface_reference_meaning='panel_surface_and_non_panel')


def test_real_hotspot_and_unlabelled_heat_retained_background_not_assessed(scene):
    frame,geometry,reference=scene
    saved=deepcopy((geometry,reference));values=frame.temperature_c.copy()
    baseline,old=analyze_native_frame(frame,proposal_builder=lambda *a,**k:geometry)
    assert baseline[1].findings  # Proves the background failure exists before the guard.
    reports,detail=analyze(frame,geometry,reference)
    assert reports[0].findings and reports[0].assessment_status=='analyzed'
    assert reports[1].assessment_status=='not_assessed' and not reports[1].findings
    assert reports[2].assessment_status=='analyzed' and reports[2].findings
    assert all(f.assessment_status=='review_required' for f in reports[2].findings)
    assert 'unlabelled_surface_review_required' in reports[2].module.quality_flags
    assert detail['modules'][0]['mask_sha256']==old['modules'][0]['mask_sha256']
    assert detail['modules'][1]['interior_pixels']==0
    assert detail['modules'][2]['mask_sha256']==old['modules'][2]['mask_sha256']
    assert detail['declared_surface_policy']['withheld_observations']==1
    assert not detail['measurement_authorised'] and detail['physical_panel_count'] is None
    assert (geometry,reference)==saved
    np.testing.assert_array_equal(frame.temperature_c,values)


@pytest.mark.parametrize('field,value',[
    ('source_sha256','wrong'),('native_temperature_sha256','wrong'),
    ('native_height',95),('review_complete',False),('panel_polygons',None)])
def test_wrong_or_incomplete_surface_identity_stops_before_geometry(scene,field,value):
    frame,_geometry,reference=scene;reference[field]=value
    with pytest.raises(ValueError,match='source-bound'):
        analyze_native_frame(frame,proposal_builder=lambda *a,**k:pytest.fail('must validate first'),
            surface_reference=reference,surface_reference_meaning='panel_surface_and_non_panel')


def test_surface_meaning_and_original_bytes_are_required(scene):
    frame,geometry,reference=scene
    with pytest.raises(ValueError,match='source-bound'):
        analyze_native_frame(frame,surface_reference=reference,surface_reference_meaning='hotspots')
    from pathlib import Path
    Path(frame.source_path).write_bytes(b'changed')
    with pytest.raises(ValueError,match='Original source'):
        analyze(frame,geometry,reference)


def test_conflicting_label_cannot_authorize_hot_panel_measurement(scene):
    frame,geometry,reference=scene
    reference['non_panel_polygons'].append({'polygon_native_xy':[[18,18],[30,18],[30,30],[18,30]]})
    reports,detail=analyze(frame,geometry,reference)
    assert 'mixed_declared_surface_review_required' in reports[0].module.quality_flags
    assert detail['modules'][0]['declared_surface_support']['vetoed_interior_pixels']>0
    assert not detail['measurement_authorised']
