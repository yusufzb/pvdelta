"""Panel-card review V2; every region retains its own radiometry, ID and decision."""
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import streamlit as st

from pvdelta.measurement_profile import read_measurement_profile
from pvdelta.panel_preview import rgb_panel_preview, thermal_panel_preview
from pvdelta.panel_review import filter_panel_groups, panel_groups, save_panel_review
from pvdelta.review_ui import finding_navigation, repeat_view_review
from pvdelta.rgb_pair_check import rgb_pair_check_view
from pvdelta.rgb_preview import original_rgb_image
from pvdelta.store import connect_store

_ERRORS = (OSError, RuntimeError, ValueError, TypeError, KeyError, duckdb.Error)


def _path(root, relative):
    if not root or not relative:
        return None
    root = Path(root).expanduser().resolve()
    path = (root/relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Indexed image lies outside the source folder')
    return path


def labelled_regions(rows):
    """Use spatial, stable A labels in both sensors; never rename persistent finding IDs."""
    ordered = sorted(rows, key=lambda r: (*np.asarray(json.loads(r['pixel_polygon_json'])).mean(axis=0)[::-1],
                                        r['finding_id']))
    return [{'label': f'A{i+1}', 'finding_id': row['finding_id'],
        'polygon': json.loads(row['pixel_polygon_json']),
        'pixels_yx': json.loads(row['region_pixels_yx_json'] or 'null'), 'record': row}
        for i, row in enumerate(ordered)]


def _alternate_views(database, observation_id, regions, sdk_library, *, key, registration=None):
    """Keep cross-frame evidence region-scoped and opt-in within a grouped card."""
    if not st.checkbox('Compare other photos', key=key+'-compare'):
        return
    st.caption('Choose a region to investigate across photos. Evidence for one region '
               'does not resolve the other regions on this card. Nearby photos alone '
               'do not establish the same panel, a different viewing angle or glare clearance.')
    selected = st.selectbox('Region for comparison', [r['finding_id'] for r in regions],
        format_func=lambda value: next(r['label'] for r in regions if r['finding_id'] == value),
        key=f'{key}-comparison-region-{observation_id}')
    from pvdelta.rgb_repeat_ui import observed_rgb_review

    observed_rgb_review(database, selected, registration, sdk_library, key=key)
    _gps_pairs(database, selected, key=key)
    _native_comparison(database, selected, sdk_library, key=key)
    try:
        repeat_view_review(database, selected, key=key+'-comparison', sdk_library=sdk_library)
    except (*_ERRORS, ImportError) as exc:
        st.warning(f'Other-photo evidence unavailable: {exc}')


def _gps_pairs(database, finding_id, *, key):
    from PIL import Image

    from pvdelta.repeat_candidates import select_repeat_pairs

    enabled_key = key+'-gps-browser-enabled'
    if st.button('Browse GPS-nearby image pairs', key=key+'-gps-browse'):
        st.session_state[enabled_key] = True
    if not st.session_state.get(enabled_key):
        return
    with st.expander('GPS-nearby photos · panel identity unverified', expanded=True):
        st.caption('Camera proximity retrieves photos, including return tracks; it does not establish '
            'the same panel or viewing-angle diversity. Photos are unmarked embedded images, '
            'not native temperature measurements. No anomaly is transferred.')
        try:
            rows = select_repeat_pairs(database, finding_id)
            if not rows:
                st.info('No paired captures available.')
                return
            method = rows[0]['search_method']
            if method != 'camera_gps':
                st.info('Source camera GPS unavailable: using the ±30 second time search.')
            st.caption('Source: '+Path(rows[0]['thermal_path']).name)
            selected = finding_navigation([r['capture_id'] for r in rows],
                key=f'{key}-gps-frame-{finding_id}', label='GPS comparison frame', item_label='View',
                names={r['capture_id']: Path(r['thermal_path']).name for r in rows})
            row = next(r for r in rows if r['capture_id'] == selected)
            distance = row.get('camera_distance_m')
            gap = row.get('time_gap_s')
            st.caption(f"Horizontal camera separation: {distance:.2f} m" if distance is not None else 'Camera separation unavailable')
            st.caption((f'Time gap: {gap:.1f} s' if gap is not None else 'Time gap unavailable')+
                f" · Gimbal yaw/pitch: {row.get('gimbal_yaw_deg')} / {row.get('gimbal_pitch_deg')}°. "
                'Yaw change alone does not establish a different viewing incidence angle.')
            for column, name in zip(st.columns(2), ('thermal_path', 'rgb_path'), strict=True):
                try:
                    with Image.open(row[name]) as image:
                        column.image(image.convert('RGB'), caption=Path(row[name]).name+' · unmarked photo',
                                     use_container_width=True)
                except OSError as exc:
                    column.warning(f'Comparison photo unavailable: {exc}')
        except _ERRORS as exc:
            st.warning(f'GPS-nearby photos unavailable: {exc}')


def _native_comparison(database, finding_id, sdk_library, *, key):
    from pvdelta.native_repeat import native_repeat_checks, native_repeat_signature

    st.caption('Native frame geometry check works with the current thermal analysis. '
        'It searches up to 8 paired captures within 20 m of the camera GPS, including return tracks. '
        'Missing source GPS uses the 30-second time search. '
        'It checks image motion and independently held-out structural controls; '
        'a pass alone does not verify this physical panel or clear glare.')
    requested = st.button('Check native frame geometry', key=key+'-native-check')
    result_key = key+'-native-result'
    saved = st.session_state.get(result_key)
    if not requested and saved is None:
        return
    try:
        binding = native_repeat_signature(database, finding_id, sdk_library)
        if saved is not None and saved['binding'] != binding:
            st.session_state.pop(result_key, None)
            saved = None
            if not requested:
                st.info('Native comparison changed; run the check for the current region and source files.')
                return
        if requested:
            st.session_state.pop(result_key, None)
            with st.spinner('Checking original SDK grids and held-out panel structure…'):
                saved = native_repeat_checks(database, finding_id, sdk_library=sdk_library)
            st.session_state[result_key] = saved
        if saved is None:
            return
        st.caption(f"Source: {saved['source_file']} · {len(saved['rows'])} nearby thermal frames checked. "
            'No anomaly outline or temperature is transferred. No review decision is changed.')
        method = saved.get('search_method', 'time')
        st.caption({'camera_gps': 'Search: camera GPS distance; camera positions are not panel coordinates.',
            'time': 'Search: capture time within 30 seconds.',
            'time_fallback_missing_source_gps': 'Search: time fallback because source camera GPS is unavailable.'}.get(method, method))
        st.dataframe([{'Frame': r['thermal_file'], 'Geometry status': r['status'],
            'Horizontal camera separation (m)': r.get('camera_distance_m'), 'Time gap (s)': r.get('time_gap_s'),
            'Reason': '; '.join(r['reasons']), 'Feature check P95 (px)': r.get('feature_check_p95_px'),
            'Observed control pairs': len((r.get('panel_controls') or {}).get('matches', [])),
            'Panel control P95 (px)': (r.get('panel_controls') or {}).get('heldout_p95_px'),
            'Selected control target': r['selected_control_target']} for r in saved['rows']],
            hide_index=True, use_container_width=True)
        if not saved['rows']:
            st.info('No additional paired captures in the bounded search window.')
        st.download_button('Download native geometry check', json.dumps(saved, indent=2, allow_nan=False),
            file_name='pvdelta-native-frame-check.json', mime='application/json', key=key+'-native-download')
    except (*_ERRORS, ImportError) as exc:
        st.session_state.pop(result_key, None)
        st.warning(f'Native frame comparison unavailable: {exc}')


def panel_review_view(database, run_id, sdk_library, rgb_registration, *, key='panels-v2'):
    from collections import Counter

    from pvdelta.candidate_triage import LABELS, support_groups
    from pvdelta.candidate_triage_ui import support_audit
    from pvdelta.temperature_evidence_store import (
        build_temperature_checks,
        load_temperature_checks,
        temperature_groups,
    )

    connection = connect_store(database, read_only=True)
    try:
        groups = panel_groups(connection, run_id)
    finally:
        connection.close()
    from pvdelta.identity_ui import identity_browser

    identity_browser(database,groups,key=key)
    pending_card = st.session_state.pop(key+'-open-related', None)
    if pending_card in {g['observation_id'] for g in groups}:
        # Apply navigation before either radio/selectbox is instantiated.
        st.session_state[key+'-evidence-queue'] = 'All candidates'
        st.session_state[key+'-filter'] = 'All'
        st.session_state[key+'-card__pending'] = pending_card
    st.subheader('Panel review · v2')
    st.caption(f'{len(groups)} panel observation cards · {sum(g["region_count"] for g in groups)} candidate regions. '
        'Each card groups one saved panel observation in one frame. Physical panels may repeat across photos.')
    if not groups:
        st.info('No panel observations with candidates in this run.')
        return
    audit = support_audit(database, run_id)
    counts = Counter(r['support'] for r in audit.values())
    st.caption(f"Support audit: {counts['interior']} interior candidates · "
        f"{counts['boundary']} boundary-sensitive regions · {counts['unavailable']} unavailable. "
        'These are not confirmed faults or unique panels. Boundary candidates are retained for review, not rejected.')
    checks = load_temperature_checks(database, run_id, sdk_library=sdk_library)
    available = [c['report'] for c in checks.values() if c['status'] == 'available']
    sensitive_count = sum(r['reference_test'] in ('crosses_threshold', 'below_threshold_in_all_folds')
        for report in available for r in report['sensitivity']['regions'].values())
    minimum_count = sum(r['minimum_pixel_support'] for report in available for r in report['regions'])
    st.caption(f'Saved temperature checks: {len(available)}/{len(groups)} cards current · '
        f'{sensitive_count} reference-sensitive regions · {minimum_count} at minimum pixel support. '
        'Checks are invalidated when source files, measurements, geometry, SDK or engine change.')
    with st.expander('Manage saved temperature checks'):
        st.caption('Opening this screen validates saved evidence without decoding again. '
            'Checks are stored locally beside the inspection; no finding or review decision is changed.')
        missing_button, all_button = st.columns(2)
        requested_missing = missing_button.button('Check missing / stale', key=key+'-build-checks')
        requested_all = all_button.button('Recheck all', key=key+'-rebuild-checks')
        if requested_missing or requested_all:
            try:
                bar = st.progress(0.)
                result = build_temperature_checks(database, run_id, sdk_library=sdk_library,
                    force=requested_all, progress=lambda i, n: bar.progress(i/n))
                st.session_state[key+'-checks-result'] = (str(database), run_id, result)
                st.rerun()
            except _ERRORS as exc:
                st.warning(f'Temperature checks could not finish: {exc}')
        last = st.session_state.get(key+'-checks-result')
        if last and last[:2] == (str(database), run_id):
            st.caption(f"Last check: {last[2]['checked']} calculated · {last[2]['reused']} reused · "
                f"{len(last[2]['failed'])} failed. Failed/missing checks do not clear candidates.")
        unavailable = [{'Observation': oid, 'Status': item['status'], 'Reason': item.get('reason', '')}
            for oid, item in checks.items() if item['status'] != 'available']
        if unavailable and st.checkbox('Show missing / stale check details', key=key+'-check-details'):
            st.dataframe(unavailable, hide_index=True, use_container_width=True)
    lanes = ('Interior candidates', 'Boundary review', 'Support unavailable',
             'Reference-sensitive', 'Minimum pixel support', 'Checks missing / stale', 'All candidates')
    lane = st.radio('Evidence queue', lanes,
        index=0 if counts['interior'] else lanes.index('All candidates'), horizontal=True, key=key+'-evidence-queue')
    queued = (temperature_groups(groups, checks, lane) if lane in
        ('Reference-sensitive', 'Minimum pixel support', 'Checks missing / stale')
        else support_groups(groups, audit, lane))
    st.caption('Cards retain all sibling regions, including those in other queues. '
        'An interior location alone does not verify panel membership or rule out glare.')
    selection = st.radio('Panel review filter',
        ('All', 'Contains unreviewed', 'All OK', 'All Not OK', 'Contains Wait', 'Mixed decisions'),
        horizontal=True, key=key+'-filter')
    filtered = filter_panel_groups(queued, selection)
    selected = finding_navigation([g['observation_id'] for g in filtered], key=key+'-card',
        label='Panel observation', item_label='Panel observation',
        names={g['observation_id']: f'{Path(g["thermal_path"]).name} · {g["module_id"]} · '
                                  f'{g["region_count"]} region(s)' for g in filtered})
    if not selected:
        st.info('No cards in this evidence queue and review filter. All candidates remain available.')
        return
    group = next(g for g in filtered if g['observation_id'] == selected)
    from pvdelta.identity_ui import identity_assignment

    identity_assignment(database,group,key=key)
    regions = labelled_regions(group['regions'])
    row = regions[0]['record']
    if any(audit[r['finding_id']]['support'] != 'interior' for r in regions):
        st.warning('This card includes boundary-sensitive or unavailable support. '
            'Check the cyan panel boundary before accepting a candidate; ground, edge effects and real edge faults can look similar.')
    connection = connect_store(database, read_only=True)
    try:
        try:
            profile = read_measurement_profile(connection, run_id, group['capture_id'])
            profile_error = None
        except _ERRORS as exc:
            profile, profile_error = None, str(exc)
        has_screening = connection.execute("SELECT 1 FROM information_schema.tables "
            "WHERE table_name='flight_screening_frames'").fetchone()
        raw = connection.execute('SELECT diagnostics_json FROM flight_screening_frames '
            'WHERE analysis_run_id=? AND capture_id=?', [run_id, group['capture_id']]).fetchone() if has_screening else None
        diagnostic = json.loads(raw[0]) if raw and raw[0] else {}
    finally:
        connection.close()
    thermal, rgb = _path(row['source_root'], row['thermal_path']), _path(row['source_root'], row['rgb_path'])
    columns = st.columns(4)
    columns[0].metric('Regions on this panel', len(regions))
    columns[1].metric('Largest |ΔT|', f"{max(abs(r['record']['delta_t_k']) for r in regions):.2f} K")
    columns[2].metric('Highest region temperature', f"{max(r['record']['absolute_temperature_c'] for r in regions):.2f} °C")
    columns[3].metric('Review', {'ok': 'OK', 'not_ok': 'Not OK', 'wait': 'Wait',
        'unreviewed': 'Unreviewed', 'mixed': 'Mixed decisions'}[group['review_status']])
    st.caption(f'Thermal: {Path(row["thermal_path"]).name} · RGB: '
        f'{Path(row["rgb_path"]).name if row["rgb_path"] else "Unavailable"} · Module: {group["module_id"]}')
    module = json.loads(row['module_polygon'])
    native_size = (row['width_px'], row['height_px'])
    left, right, original = st.columns(3)
    native_verified = False
    with left:
        st.markdown('**Thermal · all regions**')
        try:
            if profile_error:
                raise ValueError('Saved native profile invalid; overlay withheld: '+profile_error)
            if thermal:
                preview = thermal_panel_preview(thermal, regions, sdk_library=sdk_library,
                    native_size=native_size, module_polygon=module, measurement_profile=profile,
                    expected_source_hash=row['thermal_hash'])
                st.image(preview.image, caption=preview.message, use_container_width=True)
                native_verified = preview.overlay_applied
                if preview.overlay_applied and not preview.radiometry_verified:
                    st.warning('Legacy analysis: showing saved candidate locations. The historical '
                        'temperature values above have not been reproduced; no saved measurement profile exists.')
            else:
                st.info('Thermal source photo unavailable.')
        except _ERRORS as exc:
            st.warning(f'Panel thermal preview unavailable: {exc}')
    rgb_states = {}
    with right:
        st.markdown('**RGB · all regions**')
        try:
            if rgb:
                image_slot = st.container()
                pair_check = rgb_pair_check_view(thermal, rgb, profile, diagnostic, group['module_id'],
                    rgb_registration, sdk_library=sdk_library, key=key+'-rgb-check',
                    automatic_junctions=True) if native_verified else None
                preview = rgb_panel_preview(rgb, regions, native_size=native_size,
                    registration=rgb_registration, camera_model=row['camera_model'], source_root=row['source_root'],
                    expected_rgb_hash=row['rgb_hash'], native_verified=native_verified, module_polygon=module,
                    pair_check=pair_check, measurement_profile=profile)
                rgb_states = preview.region_states
                with image_slot:
                    st.image(preview.image, caption=preview.message, use_container_width=True)
                    if any(v == 'approximate' for v in rgb_states.values()):
                        st.warning('Approximate RGB projection. Local physical correspondence remains unverified.')
                    if any(v == 'unavailable' for v in rgb_states.values()):
                        st.info('Some RGB outlines are unavailable; all saved regions remain in the table.')
            else:
                st.info('Paired RGB source photo unavailable.')
        except _ERRORS as exc:
            st.warning(f'Panel RGB preview unavailable: {exc}')
    with original:
        st.markdown('**Original RGB · no marks**')
        try:
            if rgb:
                st.image(original_rgb_image(rgb, expected_source_hash=row['rgb_hash']),
                         caption=rgb.name, use_container_width=True)
            else:
                st.info('Paired RGB source photo unavailable.')
        except _ERRORS as exc:
            st.warning(f'Original RGB unavailable: {exc}')
    table = [{'Region': r['label'], 'Type': r['record']['anomaly_type'], 'ΔT (K)': r['record']['delta_t_k'],
        'Temperature (°C)': r['record']['absolute_temperature_c'],
        'Reference (°C)': r['record']['reference_temperature_c'], 'Pixels': r['record']['n_region_px'],
        'Support': LABELS[audit[r['finding_id']]['support']],
        'Support reason': audit[r['finding_id']]['reason'],
        'Review': r['record']['review_status'], 'RGB projection': rgb_states.get(r['label'], 'unavailable'),
        'Latitude': r['record']['latitude'], 'Longitude': r['record']['longitude']}
        for r in regions]
    st.markdown('**Regions on this panel**')
    st.dataframe(table, hide_index=True, use_container_width=True)
    with st.expander('Temperature reference evidence'):
        st.caption('Re-read the original SDK temperatures and reproduce every saved region on this card. '
            'Check whether the reference plane is extrapolated and how much native-pixel support is available. '
            'This does not confirm a fault or rule out glare.')
        requested = st.button('Check temperature evidence', key=key+'-temperature-evidence')
        saved = checks.get(selected, {'status': 'missing'})
        if requested or saved['status'] == 'available':
            from pvdelta.temperature_evidence_store import check_and_save_temperature_evidence

            try:
                if requested:
                    with st.spinner('Reproducing saved native measurements…'):
                        evidence = check_and_save_temperature_evidence(database, run_id, selected, sdk_library=sdk_library)
                    st.caption('Fresh SDK check saved locally. Queue counts refresh on the next interaction.')
                else:
                    evidence = saved['report']
                    st.caption('Saved check: '+saved['checked_at_utc']+' · current source and analysis bindings verified')
                labels = {r['finding_id']: r['label'] for r in regions}
                st.dataframe([{'Region': labels[r['finding_id']],
                    'Reference': r['reference_status'], 'Extrapolated region (%)': 100*r['extrapolated_fraction'],
                    'Reference noise MAD (K)': r['reference_mad_k'],
                    'Reference pixels': r['reference_pixels'], 'Candidate pixels': r['candidate_pixels'],
                    'At minimum pixel support': r['minimum_pixel_support'],
                    'Median residual (K)': r['median_residual_k'], 'Peak residual (K)': r['peak_residual_k']}
                    for r in evidence['regions']], hide_index=True, use_container_width=True)
                if any(r['extrapolated_pixels'] for r in evidence['regions']):
                    st.warning('Some candidate temperatures are compared with a plane extended beyond '
                        'the reference pixels. Inspect the mask and thermal gradient; this is not a false-positive verdict.')
                if any(r['minimum_pixel_support'] for r in evidence['regions']):
                    st.warning('A region is at the detector’s minimum pixel count. Its small support needs '
                        'repeat-image or closer-resolution evidence; it is not automatically noise.')
                sensitivity = evidence['sensitivity']
                names = {'above_threshold_in_all_folds': 'Above threshold in all 4 checks',
                    'below_threshold_in_all_folds': 'Below threshold in all 4 checks',
                    'crosses_threshold': 'Changes across threshold',
                    'insufficient_reference_blocks': 'Insufficient reference coverage'}
                st.markdown('**Reference and boundary sensitivity**')
                st.dataframe([{'Region': labels[fid], 'Reference test': names[s['reference_test']],
                    'Lowest ΔT (K)': s['delta_min_k'], 'Highest ΔT (K)': s['delta_max_k'],
                    'Valid checks': s['valid_folds'],
                    'Pixels retained with +1 px margin (%)': 100*s['extra_margin_retained_fraction']['1'],
                    'Pixels retained with +2 px margin (%)': 100*s['extra_margin_retained_fraction']['2']}
                    for fid, s in sensitivity['regions'].items()], hide_index=True, use_container_width=True)
                if any(s['reference_test'] != 'above_threshold_in_all_folds'
                       for s in sensitivity['regions'].values()):
                    st.warning('Reference sensitivity needs review: a different reference subset changes '
                        'the threshold result, or reference coverage is insufficient. The candidate is retained.')
                st.caption('Four checks each omit one spatial part of the frozen reference. '
                    'Margin percentages describe remaining pixels, not a new detection or a calibrated alignment error. '
                    'Passing these same-frame stress tests does not confirm a panel fault or resolve glare.')
                st.caption('All saved regions and temperatures reproduced. MAD describes reference residuals, '
                    'not camera accuracy or fault probability. Use Check temperature evidence for a fresh SDK calculation.')
                st.download_button('Download temperature evidence', json.dumps(evidence, indent=2, allow_nan=False),
                    file_name='pvdelta-temperature-evidence.json', mime='application/json', key=key+'-temperature-download')
            except (*_ERRORS, ImportError) as exc:
                st.warning(f'Temperature evidence unavailable: {exc}')
        elif saved['status'] == 'stale':
            st.warning('Saved temperature check withheld: '+saved.get('reason', 'Inputs changed'))
    with st.expander('User review'):
        scope = st.selectbox('Apply decision to', ['all', *[r['finding_id'] for r in regions]],
            format_func=lambda value: f'All {len(regions)} regions on this card' if value == 'all' else
                next(r['label'] for r in regions if r['finding_id'] == value),
            key=f'{key}-scope-{selected}')
        st.caption('OK: accept candidate · Not OK: reject · Wait: needs more evidence. '
                   'Empty note preserves saved notes. Decisions do not certify faults or IEC compliance.')
        existing_note = '' if scope == 'all' else next(r['record']['review_note'] for r in regions if r['finding_id'] == scope)
        note = st.text_input('Note (optional)', value=existing_note, key=f'{key}-note-{selected}-{scope}')
        for column, label, value in zip(st.columns(3), ('OK', 'Not OK', 'Wait'), ('ok', 'not_ok', 'wait')):
            if column.button(label, key=key+'-mark-'+value, use_container_width=True):
                connection = connect_store(database)
                try:
                    save_panel_review(connection, run_id, selected, [r['finding_id'] for r in regions], value,
                        note=note if note else None, target_finding_id=None if scope == 'all' else scope)
                finally:
                    connection.close()
                st.rerun()
    with st.expander('Stored region records'):
        st.dataframe([{'Region': r['label'], 'Finding ID': r['finding_id'],
            'Observation ID': selected, 'Note': r['record']['review_note']} for r in regions], hide_index=True)
        export = pd.DataFrame([dict(t, finding_id=r['finding_id'], observation_id=selected,
            thermal_file=row['thermal_path'], rgb_file=row['rgb_path']) for t, r in zip(table, regions, strict=True)])
        st.download_button('Download panel regions', export.to_csv(index=False), file_name='pvdelta-panel-regions.csv',
                           mime='text/csv', key=key+'-download')
    from pvdelta.candidate_links_ui import related_candidate_cards
    from pvdelta.thermal_views_ui import thermal_views_review

    thermal_views_review(database, selected, min(r['finding_id'] for r in regions), rgb_registration,
        sdk_library, key=key)

    related_candidate_cards(database, selected, min(r['finding_id'] for r in regions), rgb_registration,
        sdk_library, {g['observation_id'] for g in groups}, key=key, run_id=run_id)
    _alternate_views(database, selected, regions, sdk_library, key=key, registration=rgb_registration)
