"""Optional observed RGB module correspondence, separate from native anomaly review."""
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import streamlit as st

from pvdelta.review_ui import finding_navigation
from pvdelta.rgb_pair_check import digest
from pvdelta.rgb_repeat import observed_rgb_repeat, repeat_pair_images, rgb_repeat_signature


def observed_rgb_review(database, finding_id, registration, sdk_library, *, key):
    table_context = st.checkbox('Use observed table neighbours for RGB tracking',
        value=False, key=key+'-rgb-table-context',
        help='Optional geometry-only support from observed adjacent modules in the same table. '
             'Temperature masks, target selection and acceptance limits are unchanged.')
    options = {'table_context': True} if table_context else {}
    requested = st.button('Find observed RGB panel views', key=key+'-rgb-find', disabled=registration is None)
    result_key = key+'-rgb-result'
    saved = st.session_state.get(result_key)
    if not requested and saved is None:
        return
    try:
        binding = rgb_repeat_signature(database, finding_id, registration, sdk_library=sdk_library, **options)
        if saved is not None and saved['binding'] != binding:
            st.session_state.pop(result_key, None)
            saved = None
            if not requested:
                st.info('RGB comparison changed; run the check for the current region and source files.')
                return
        if requested:
            st.session_state.pop(result_key, None)
            with st.spinner('Checking original sources and tracking observed RGB panels…'):
                saved = observed_rgb_repeat(database, finding_id, registration, sdk_library=sdk_library, **options)
            st.session_state[result_key] = saved
        if saved is None:
            return
        if saved.get('content_sha256') != digest({k: v for k, v in saved.items() if k != 'content_sha256'}):
            raise ValueError('RGB comparison content changed; recompute before review')
        with st.expander('Observed RGB panel views · provisional', expanded=True):
            matched = sum(r['status'] == 'matched_provisional' for r in saved['rows'])
            st.caption(f"{matched}/{len(saved['rows'])} target views passed reciprocal RGB image checks. "
                'The initial native-to-RGB panel association remains approximate. '
                'This is not verified physical identity, thermal component correspondence or glare clearance.')
            if saved.get('tracking_support_mode') == 'observed_table_neighbours':
                st.caption('Tracking uses observed adjacent module interiors in the same indexed table, '
                    'chosen before fitting. Table coplanarity remains an assumption; no missing module is filled.')
            st.dataframe([{'Frame': Path(r['rgb_path']).name, 'Status': r['status'], 'Reason': r['reason'],
                'Return-to-source check': r.get('reciprocal_check_passed', False),
                'Reverse P95 (feature-grid px)': (r.get('reverse_result') or {}).get('check_p95_px'),
                'Source neighbour patches': max(0, len(r.get('source_tracking_polygons') or [])-1),
                'Panel overlap (IoU)': r.get('iou'), 'P95 (feature-grid px)': r.get('check_p95_px'),
                'Held-out points': r.get('check_count'), 'Horizontal camera separation (m)': r.get('camera_distance_m'),
                'Time gap (s)': r.get('time_gap_s')} for r in saved['rows']], hide_index=True, use_container_width=True)
            source = saved['source']
            ids = [source['capture_id'], *[r['capture_id'] for r in saved['rows']]]
            names = {source['capture_id']: Path(source['rgb_path']).name+' · source anchor (approximate)'}
            names.update({r['capture_id']: Path(r['rgb_path']).name+' · '+r['status'] for r in saved['rows']})
            selected = finding_navigation(ids, key=key+'-rgb-frame-'+finding_id,
                label='Observed RGB comparison frame', names=names, item_label='View')
            row = source if selected == source['capture_id'] else next(r for r in saved['rows'] if r['capture_id'] == selected)
            st.caption(f"Thermal: {Path(row['thermal_path']).name} · RGB: {Path(row['rgb_path']).name}")
            try:
                thermal, rgb = repeat_pair_images(saved, selected)
                left, right = st.columns(2)
                left.image(thermal, caption='Unmarked embedded thermal photo · no transferred measurement', use_container_width=True)
                right.image(rgb, caption='Cyan: observed RGB panel used for provisional tracking; context crop when available',
                            use_container_width=True)
            except (OSError, ValueError) as exc:
                st.warning(f'Selected comparison photos unavailable: {exc}')
            st.caption('No target hotspot outline, temperature or review decision is borrowed from the source. '
                'Both directions must return to the same observed source panel; adjacent source panels compete. '
                'Yaw change alone does not establish a different viewing incidence angle.')
            st.download_button('Download observed RGB comparison', json.dumps(saved, indent=2, allow_nan=False),
                file_name='pvdelta-observed-rgb-repeat.json', mime='application/json', key=key+'-rgb-download')
    except (OSError, RuntimeError, ValueError, TypeError, KeyError, ImportError, duckdb.Error) as exc:
        st.session_state.pop(result_key, None)
        st.warning(f'Observed RGB comparison unavailable: {exc}')
