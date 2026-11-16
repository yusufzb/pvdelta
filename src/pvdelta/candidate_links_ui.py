"""Explicit, provisional candidate-card comparisons with safe navigation."""
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import streamlit as st

from pvdelta.candidate_links import (
    candidate_link_images,
    candidate_links_signature,
)
from pvdelta.candidate_links_store import (
    build_candidate_comparisons,
    check_and_save_comparison,
    load_candidate_comparison,
)
from pvdelta.review_ui import finding_navigation
from pvdelta.rgb_pair_check import digest


def related_candidate_cards(database, observation_id, finding_id, registration, sdk_library,
                            allowed_observations, *, key, run_id=None):
    previous_pair=st.session_state.get(key+'-related-native-pair')
    if previous_pair and previous_pair.get('source_observation_id')!=observation_id:
        st.session_state.pop(key+'-related-native-pair',None)
    with st.expander('Related candidate cards · provisional'):
        st.caption('Check up to4 nearby captures that contain candidates. Both RGB tracking directions '
            'must pass and return to the same observed panel. The native-to-RGB anchors remain approximate; '
            'a link does not merge panels, transfer review decisions or resolve glare.')
        requested = st.button('Check related candidate cards', key=key+'-find-related', disabled=registration is None)
        if run_id is not None:
            left,right=st.columns(2)
            missing=left.button('Check all missing / stale comparisons',key=key+'-build-related',disabled=registration is None)
            force=right.button('Recheck all comparisons',key=key+'-rebuild-related',disabled=registration is None)
            if missing or force:
                try:
                    bar=st.progress(0.)
                    result=build_candidate_comparisons(database,run_id,registration,sdk_library=sdk_library,
                        force=force,progress=lambda i,n,r:bar.progress(i/n,text=f"{i}/{n} · {r['status']}"))
                    st.session_state[key+'-related-batch']=(str(database),run_id,result)
                    st.rerun()
                except (OSError,ValueError,RuntimeError,TypeError,KeyError,duckdb.Error) as exc:
                    st.warning(f'Batch comparisons interrupted: {exc}. Completed records are preserved.')
            previous=st.session_state.get(key+'-related-batch')
            if previous and previous[:2]==(str(database),run_id):
                st.caption(f"Last batch: {len(previous[2]['results'])}/{previous[2]['population']} records processed. "
                    'Unavailable sources and failed checks remain explicit; no candidates are removed.')
        result_key = key+'-related-result'
        saved = st.session_state.get(result_key)
        if saved and (saved.get('finding_id') != finding_id or saved.get('source_observation_id') != observation_id):
            st.session_state.pop(result_key, None)
            saved = None
        persistent_valid=False
        if not requested and run_id is not None:
            record=load_candidate_comparison(database,run_id,observation_id,finding_id,registration,sdk_library=sdk_library)
            if record['status']=='available':
                saved=record['report'];persistent_valid=True
                st.caption('Saved comparison · '+record['checked_at_utc']+' · current bindings verified')
            elif record['status']=='unavailable':
                st.session_state.pop(result_key,None)
                st.info('Saved source attempt unavailable: '+record['reason'])
                return
            elif record['status']=='stale':
                st.caption('Saved comparison is outdated: '+record['reason'])
        if not requested and saved is None:
            return
        try:
            if saved is not None and not persistent_valid:
                binding = candidate_links_signature(database, finding_id, registration, sdk_library=sdk_library)
                if (saved['binding'] != binding
                        or saved['content_sha256'] != digest({k:v for k,v in saved.items() if k!='content_sha256'})):
                    st.session_state.pop(result_key, None)
                    saved = None
                    if not requested:
                        st.warning('Related-card evidence changed; run the check again.')
                        return
            if requested:
                st.session_state.pop(result_key, None)
                if run_id is None:
                    raise ValueError('Analysis run is required to save a comparison')
                with st.spinner('Checking both original RGB directions and saved native profiles…'):
                    record=check_and_save_comparison(database,run_id,observation_id,finding_id,registration,sdk_library=sdk_library)
                if record['status']=='unavailable':
                    st.info('Source comparison unavailable: '+record['reason'])
                    return
                saved=record['report']
                if saved['source_observation_id'] != observation_id:
                    raise ValueError('Comparison source differs from the selected card')
                st.session_state[result_key] = saved
            rows = saved['rows']
            st.caption(f"{sum(r['status']=='provisional_candidate_link' for r in rows)}/{len(rows)} "
                'attempted captures have a provisional link to another candidate card. '
                'No unique physical-panel count is inferred.')
            st.dataframe([{'Thermal frame':Path(r['thermal_path']).name, 'Status':r['status'],
                'Reason':r['reason'], 'Camera separation (m)':r.get('camera_distance_m'),
                'Time gap (s)':r.get('time_gap_s'),
                'Forward P95 (feature px)':(r.get('forward') or {}).get('check_p95_px'),
                'Reverse P95 (feature px)':(r.get('reverse') or {}).get('check_p95_px'),
                'Target candidate regions':len(r['target_finding_ids'])} for r in rows],
                hide_index=True, use_container_width=True)
            if not rows:
                st.info('No other candidate-containing capture in this bounded search. Other views may still exist.')
                return
            selected = finding_navigation([r['capture_id'] for r in rows], key=key+'-related-view-'+finding_id,
                label='Related capture', item_label='Comparison',
                names={r['capture_id']:Path(r['thermal_path']).name+' · '+r['status'] for r in rows})
            row = next(r for r in rows if r['capture_id']==selected)
            source_thermal, source_rgb = candidate_link_images(saved, saved['source']['capture_id'])
            target_thermal, target_rgb = candidate_link_images(saved, selected)
            left,right = st.columns(2)
            left.image(source_rgb, caption=Path(saved['source']['rgb_path']).name+' · approximate source anchor', use_container_width=True)
            right.image(target_rgb, caption=Path(row['rgb_path']).name+' · cyan panel only when both RGB directions pass', use_container_width=True)
            if st.checkbox('Show original thermal comparison', key=key+'-related-show-thermal'):
                left,right=st.columns(2)
                left.image(source_thermal, caption=Path(saved['source']['thermal_path']).name+' · unmarked embedded photo', use_container_width=True)
                right.image(target_thermal, caption=Path(row['thermal_path']).name+' · unmarked embedded photo', use_container_width=True)
            if row['status']=='provisional_candidate_link' and run_id is not None:
                _linked_thermal_review(database,run_id,saved,selected,registration,sdk_library,key)
            can_open = (row['status']=='provisional_candidate_link'
                and row['target_observation_id'] in allowed_observations)
            if st.button('Open related candidate card', key=key+'-open-related-card', disabled=not can_open):
                st.session_state[key+'-open-related'] = row['target_observation_id']
                st.rerun()
            st.caption('Opening a target loads its own saved regions and SDK evidence. '
                'No hotspot outline, temperature or decision is transferred between frames.')
            st.download_button('Download related-card evidence', json.dumps(saved,indent=2,allow_nan=False),
                file_name='pvdelta-related-candidates.json', mime='application/json', key=key+'-related-download')
        except (OSError,ValueError,RuntimeError,TypeError,KeyError,ImportError,duckdb.Error) as exc:
            st.session_state.pop(result_key,None)
            st.warning(f'Related-card comparison unavailable: {exc}')


def _linked_thermal_review(database, run_id, saved, selected, registration, sdk_library, key):
    try:
        from pvdelta.linked_thermal import compare_linked_thermal, thermal_pair_signature

        thermal_key=key+'-related-native-pair'
        pair=st.session_state.get(thermal_key)
        identity=(saved['content_sha256'],selected)
        if pair and pair['identity']!=identity:
            st.session_state.pop(thermal_key,None)
            pair=None
        if st.button('Compare own-frame SDK hotspots',key=key+'-compare-related-native'):
            st.session_state.pop(thermal_key,None)
            with st.spinner('Reproducing both saved candidate populations from original SDK grids…'):
                evidence,images=compare_linked_thermal(database,run_id,saved,selected,registration,
                    sdk_library=sdk_library)
            pair={'identity':identity,'source_observation_id':saved.get('source_observation_id'),
                'report':evidence,'images':images}
            st.session_state[thermal_key]=pair
        if pair:
            evidence=pair['report']
            if (thermal_pair_signature(database,run_id,saved,selected,registration,sdk_library=sdk_library)
                    !=evidence['binding'] or evidence['content_sha256']!=digest(
                        {k:v for k,v in evidence.items() if k!='content_sha256'})):
                st.session_state.pop(thermal_key,None)
                st.warning('Own-frame thermal evidence changed; check again.')
            else:
                left,right=st.columns(2)
                for column,image,card in zip((left,right),pair['images'],evidence['cards'],strict=True):
                    column.image(image,caption=card['source_file']+' · exact saved hotspots in green',
                        use_container_width=True)
                    column.dataframe([{'Region':r['label'],'Median °C':r['median_c'],
                        'ΔT K':r['delta_t_k'],'Pixels':r['pixel_count'],
                        'Panel u':r['centre_uv'][0],'Panel v':r['centre_uv'][1]}
                        for r in card['regions']],hide_index=True)
                low,high=evidence['display_scale_c']
                st.caption(f'Common SDK temperature scale: {low:.2f}–{high:.2f} °C. '
                    'Each image retains its own orientation. Green marks its own saved pixels; '
                    'cyan is the provisional panel. u/v use tracked corner order, including return flights. '
                    'Native/RGB anchors are approximate: positions cannot confirm the same hot cell or resolve glare.')
                st.dataframe(evidence['all_region_pairs'],hide_index=True)
                st.download_button('Download own-frame thermal evidence',json.dumps(evidence,indent=2,allow_nan=False),
                    file_name='pvdelta-linked-thermal.json',mime='application/json',key=key+'-linked-thermal-download')
    except (OSError, ValueError, RuntimeError, TypeError, KeyError, ImportError, duckdb.Error) as exc:
        st.session_state.pop(key+"-related-native-pair", None)
        st.warning(f"Own-frame SDK comparison unavailable: {exc}")
