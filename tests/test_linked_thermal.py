import json
from copy import deepcopy

import numpy as np
import pytest

from pvdelta import linked_thermal as subject

MODEL={'centre':[0,0],'scale':[1,1],
    'coefficients':[[0,0],[1,0],[0,1],[0,0],[0,0],[0,0]]}
QUAD=np.array([[10,10],[30,10],[30,50],[10,50]],float)


@pytest.mark.parametrize('roll',[0,1,2,3])
def test_tracked_corners_survive_return_rotation_and_arbitrary_vertex_origin(roll):
    # Target camera has rotated 180 degrees; physical source TL is target BR.
    raw=np.roll(QUAD,roll,axis=0)
    expected=np.roll(QUAD,2,axis=0)
    indices=[int(np.flatnonzero(np.all(raw==p,axis=1))[0]) for p in expected]
    a,b=subject.aligned_native_quads(QUAD,QUAD,QUAD,raw,indices,MODEL)
    np.testing.assert_array_equal(a,QUAD)
    np.testing.assert_array_equal(b,expected)
    source=subject.normalised_region([[20,15]],a)
    target=subject.normalised_region([[40,25]],b)
    np.testing.assert_allclose(source['centre_uv'],target['centre_uv'])


@pytest.mark.parametrize('indices',[None,[0,1,1,3],[0,2,1,3],[True,1,2,3]])
def test_invalid_or_crossing_corner_correspondence_is_not_guessed(indices):
    with pytest.raises(ValueError):
        subject.aligned_native_quads(QUAD,QUAD,QUAD,QUAD,indices,MODEL)


def test_source_anchor_order_must_match_native_not_image_top_left():
    with pytest.raises(ValueError,match='Source RGB corner order'):
        subject.aligned_native_quads(QUAD,QUAD,np.roll(QUAD,1,axis=0),QUAD,[0,1,2,3],MODEL)


@pytest.mark.parametrize('pixels',[[[0,0]],[[20,20],[20,20]],[[20.1,20]],[]])
def test_missing_or_outside_native_support_remains_unavailable(pixels):
    with pytest.raises(ValueError):
        subject.normalised_region(pixels,QUAD)


def synthetic(monkeypatch,*,changed=False,replay_error=False):
    values=np.arange(64*64,dtype=np.float32).reshape(64,64)/100
    rows=[{'observation_id':name,'path':name,'profile':{'profile_id':name},
        'findings':[(name+'-hot',json.dumps([[20,15],[21,15],[22,15],[23,15]]),
            float(np.median(values[20:24,15])),7.,100)]} for name in ('source','target')]
    calls=[]
    def context(*args):
        calls.append('context')
        return deepcopy(rows),(QUAD.copy(),QUAD.copy()),'changed' if changed and len(calls)>1 else 'same'
    monkeypatch.setattr(subject,'_pair_context',context)
    monkeypatch.setattr(subject,'candidate_links_signature',lambda *a,**k:'link')
    def replay(*a,**k):
        if replay_error:raise ValueError('population does not reproduce')
        return {'native_temperature_sha256':'native'}
    monkeypatch.setattr(subject,'check_temperature_evidence',replay)
    class Reader:
        def __init__(self,*a):pass
        def __enter__(self):return self
        def __exit__(self,*a):pass
    monkeypatch.setattr(subject,'RJpeg',Reader)
    monkeypatch.setattr(subject,'apply_saved_measurement_profile',lambda *a:values)
    return {'finding_id':'source-hot','binding':'link','max_targets':4,'content_sha256':'report'},values


def test_sdk_pair_preserves_all_regions_temperatures_and_never_confirms_identity(monkeypatch):
    link,values=synthetic(monkeypatch)
    original=values.copy()
    report,images=subject.compare_linked_thermal('db','run',link,'target',MODEL)
    assert len(images)==2 and len(report['cards'])==2 and len(report['all_region_pairs'])==1
    assert report['all_region_pairs'][0]['centre_distance_uv']==0
    assert all(card['regions'][0]['pixel_count']==4 for card in report['cards'])
    for name in ('component_correspondence_verified','panel_identity_verified','glare_clearance',
                 'physical_panels_merged','database_written'):
        assert report[name] is False
    np.testing.assert_array_equal(values,original)


def test_changed_inputs_prevent_publishing_pair(monkeypatch):
    link,_=synthetic(monkeypatch,changed=True)
    with pytest.raises(ValueError,match='changed during'):
        subject.compare_linked_thermal('db','run',link,'target',MODEL)


def test_nonreproducing_saved_candidates_cannot_show_new_measurements(monkeypatch):
    link,_=synthetic(monkeypatch,replay_error=True)
    with pytest.raises(ValueError,match='does not reproduce'):
        subject.compare_linked_thermal('db','run',link,'target',MODEL)


def test_stale_rgb_link_is_rejected_before_reading_sdk(monkeypatch):
    link,_=synthetic(monkeypatch)
    monkeypatch.setattr(subject,'candidate_links_signature',lambda *a,**k:'changed')
    with pytest.raises(ValueError,match='stale'):
        subject.compare_linked_thermal('db','run',link,'target',MODEL)


def test_altered_report_and_unlinked_target_cannot_read_native_data():
    report={'schema':'pvdelta.candidate-card-links/v1','physical_panels_merged':False,'rows':[]}
    report['content_sha256']=subject.digest(report)
    with pytest.raises(ValueError,match='bidirectionally linked'):
        subject.thermal_pair_signature('missing','run',report,'target',MODEL)
    report['physical_panels_merged']=True
    with pytest.raises(ValueError,match='unmerged'):
        subject.thermal_pair_signature('missing','run',report,'target',MODEL)


def test_thermal_ui_is_explicit_and_clears_on_target_and_binding_change(monkeypatch):
    from PIL import Image
    from streamlit.testing.v1 import AppTest

    calls=[];binding=['current']
    region={'label':'A1','median_c':30.,'delta_t_k':7.,'pixel_count':4,'centre_uv':[.2,.4]}
    def check(*a,**k):
        calls.append(a[3])
        report={'binding':binding[0],'cards':[{'source_file':n,'regions':[region]} for n in ('a','b')],
            'display_scale_c':[20.,40.],'all_region_pairs':[]}
        report['content_sha256']=subject.digest(report)
        return report,[Image.new('RGB',(64,64))]*2
    monkeypatch.setattr(subject,'compare_linked_thermal',check)
    monkeypatch.setattr(subject,'thermal_pair_signature',lambda *a,**k:binding[0])
    app=AppTest.from_string('''import streamlit as st
from pvdelta.candidate_links_ui import _linked_thermal_review
target=st.selectbox('Target',['a','b'],key='target')
_linked_thermal_review('db','run',{'content_sha256':'saved'},target,{},'sdk','pair')
st.button('Navigation remains available',key='nav')
''',default_timeout=30).run()
    assert not calls and not app.exception
    app.button(key='pair-compare-related-native').click().run()
    assert not app.exception and calls==['a'] and len(app.get('image'))==2
    app.selectbox(key='target').select('b').run()
    assert not app.exception and len(app.get('image'))==0 and 'pair-related-native-pair' not in app.session_state
    app.button(key='pair-compare-related-native').click().run()
    assert calls==['a','b'] and len(app.get('image'))==2
    binding[0]='changed';app.run()
    assert not app.exception and len(app.get('image'))==0
    assert any('changed; check again' in w.value for w in app.warning)


def test_thermal_failure_does_not_remove_candidate_navigation(monkeypatch):
    from streamlit.testing.v1 import AppTest

    def failed(*a,**k):raise ValueError('Source unavailable')
    monkeypatch.setattr(subject,'compare_linked_thermal',failed)
    app=AppTest.from_string('''import streamlit as st
from pvdelta.candidate_links_ui import _linked_thermal_review
_linked_thermal_review('db','run',{'content_sha256':'saved'},'target',{},'sdk','pair')
st.button('Open related candidate card',key='nav')
''',default_timeout=30).run()
    app.button(key='pair-compare-related-native').click().run()
    assert not app.exception and not app.button(key='nav').disabled
    assert any('Source unavailable' in w.value for w in app.warning)


def test_old_thermal_pair_cleared_even_when_next_source_has_no_anchor(monkeypatch):
    from streamlit.testing.v1 import AppTest

    from pvdelta import candidate_links_ui

    monkeypatch.setattr(candidate_links_ui,'load_candidate_comparison',lambda *a,**k:
        {'status':'unavailable','reason':'No source anchor'})
    app=AppTest.from_string('''import streamlit as st
from pvdelta.candidate_links_ui import related_candidate_cards
oid=st.selectbox('Source',['first','second'],key='source')
related_candidate_cards('db',oid,'f',{},'sdk',[],key='pair',run_id='run')
''',default_timeout=30).run()
    app.session_state['pair-related-native-pair']={'source_observation_id':'first'}
    app.selectbox(key='source').select('second').run()
    assert not app.exception and 'pair-related-native-pair' not in app.session_state
