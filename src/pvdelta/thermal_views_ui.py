"""Candidate-independent nearby thermal evidence, with explicit unknown cases."""
from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from pvdelta import rgb_repeat
from pvdelta.review_ui import finding_navigation
from pvdelta.rgb_pair_check import digest
from pvdelta.thermal_view_scan import (
    ERRORS,
    load_view_scan,
    save_view_scan,
    scan_thermal_views,
    thermal_view_images,
)


def thermal_views_review(database, observation_id, finding_id, registration, sdk_library, *, key):
    with st.expander('All nearby thermal views · provisional'):
        st.caption('Search nearby paired captures whether or not they contain a saved candidate. '
            'Both RGB directions and each matched panel’s own native SDK assessment must pass. '
            'A candidate-free assessment is not a healthy-panel or glare verdict. '
            'Camera proximity and the bounded search do not enumerate every view of a physical panel.')
        limit=st.selectbox('Nearby capture limit',[32,16,8],key=key+'-thermal-view-limit')
        requested=st.button('Check nearby thermal views',key=key+'-thermal-view-check',disabled=registration is None)
        if registration is None:
            st.caption('A compatible RGB search calibration is required for provisional panel association.')
            return
        try:
            if requested:
                progress=st.progress(0.)
                with st.spinner('Checking nearby views including captures with no recorded candidates…'):
                    report=scan_thermal_views(database,finding_id,registration,sdk_library=sdk_library,max_views=limit,
                        progress=lambda i,n,row:progress.progress(i/n,text=f'{i}/{n} · {row["status"]}'))
                if report['source_observation_id']!=observation_id:
                    raise ValueError('Source observation differs from selected card')
                save_view_scan(database,report)
            else:
                saved=load_view_scan(database,finding_id,registration,sdk_library=sdk_library,max_views=limit)
                if saved['status']=='missing':
                    return
                if saved['status']=='stale':
                    st.warning('Saved nearby-view evidence needs rechecking: '+saved['reason'])
                    return
                report=saved['report']
            if report['source_observation_id']!=observation_id:
                raise ValueError('Saved nearby views belong to another observation')
            rows=report['rows']
            matched=[r for r in rows if r['status']=='matched_provisional']
            no_candidates=sum(r['thermal_evidence']['status']=='analyzed_no_candidates' for r in matched)
            st.caption(f'{len(rows)} retrieved · {len(matched)} provisional panel matches · '
                f'{no_candidates} matched observations with no saved candidate. '
                'Unavailable and unattempted cases remain in the table.')
            if report['status']=='unavailable':
                st.info('Automatic thermal comparison unavailable: '+report['reason'])
            st.dataframe([{'Thermal frame':Path(r['thermal_path']).name,'Match':r['status'],
                'Native assessment':(r['thermal_evidence'] or {}).get('status','unknown'),
                'Target regions':len(r['thermal_evidence']['regions']) if r['thermal_evidence'] else None,
                'Capture candidate records':r['capture_candidate_count'],
                'Camera separation (m)':r.get('camera_distance_m'),'Time gap (s)':r.get('time_gap_s'),
                'Reason':r['reason']} for r in rows],hide_index=True,use_container_width=True)
            if rows:
                ids=[r['capture_id'] for r in rows]
                selected=finding_navigation(ids,key=key+'-thermal-view-'+finding_id+f'-{limit}',
                    label='Thermal comparison capture',item_label='View',
                    names={r['capture_id']:Path(r['thermal_path']).name+' · '+r['status'] for r in rows})
                row=next(r for r in rows if r['capture_id']==selected)
                if st.checkbox('Show selected thermal/RGB pair',value=report['status']=='unavailable',
                               key=key+'-thermal-view-show'):
                    converted={'schema':'pvdelta.observed-rgb-repeat/v1','source':report['source'],
                        'source_rgb_polygon':report['source_rgb_polygon'],'rows':rows}
                    converted['content_sha256']=digest(converted)
                    source_thermal,source_rgb=rgb_repeat.repeat_pair_images(converted,report['source']['capture_id'])
                    target_thermal,target_rgb=rgb_repeat.repeat_pair_images(converted,selected)
                    measured=row['status']=='matched_provisional' and row['thermal_evidence'] is not None
                    if measured:
                        (source_thermal,target_thermal),scale=thermal_view_images(report,selected,sdk_library=sdk_library)
                        st.caption(f'Own-frame SDK evidence · common scale {scale[0]:.2f}–{scale[1]:.2f} °C. '
                            'Green: each capture’s own saved hot pixels. Cyan: provisional panel. '
                            'An empty green overlay means no saved candidate at these settings; glare remains unresolved.')
                    else:
                        st.caption('Original unmarked thermal photos; target panel correspondence or native assessment unavailable.')
                    for a,b,label in ((source_thermal,target_thermal,'thermal_path'),(source_rgb,target_rgb,'rgb_path')):
                        left,right=st.columns(2)
                        left.image(a,caption=Path(report['source'][label]).name,use_container_width=True)
                        right.image(b,caption=Path(row[label]).name,use_container_width=True)
            st.download_button('Download all nearby-view evidence',json.dumps(report,indent=2,allow_nan=False),
                file_name='pvdelta-nearby-thermal-views.json',mime='application/json',key=key+'-thermal-view-download')
        except ERRORS as exc:
            st.warning(f'Nearby thermal-view check unavailable: {exc}')
