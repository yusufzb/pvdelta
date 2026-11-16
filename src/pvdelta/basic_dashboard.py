"""A short local inspection workflow using the existing native engine and review store."""
from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path

import duckdb
import streamlit as st
from PIL import Image

from pvdelta.coverage_audit import build_coverage_audit
from pvdelta.finding_preview import thermal_finding_preview
from pvdelta.inspection_library import load_inspection_library
from pvdelta.measurement_profile import read_measurement_profile
from pvdelta.review_ui import finding_navigation
from pvdelta.store import connect_store, upsert_finding_review

_ERRORS = (OSError, RuntimeError, ValueError, TypeError, KeyError, duckdb.Error)


def _flight_geometry_snapshot(database, run):
    """Display an optional completed local batch for this inspection."""
    sidecar = Path(database).with_suffix('.geometry-review.json')
    if not sidecar.is_file():
        return
    try:
        snapshot = json.loads(sidecar.read_bytes())
        if snapshot.get('schema') != 'pvdelta.flight-geometry-snapshot/v1':
            return
        if snapshot['run'] != run or Path(snapshot['database']).resolve() != Path(database).resolve():
            return
        folder = Path(snapshot['report_folder'])
        with st.expander('Flight-wide panel boundaries'):
            metrics = st.columns(3)
            metrics[0].metric('Frames checked', snapshot['frames'])
            metrics[1].metric('Frames with supported boundaries', snapshot['outcomes'].get('supported', 0))
            metrics[2].metric('Supported boundary observations', snapshot['supported_observations'])
            st.caption('Saved batch results. A physical panel can appear in several frames. '
                       'Green outlines show panel geometry. Fault decisions require separate review.')
            st.write('Frames without native panel geometry:', snapshot['outcomes'].get('no_native_geometry', 0))
            st.write('Frames with withheld transfer:', snapshot['outcomes'].get('withheld', 0))
            st.write('Frames with unavailable inputs or incompatible view:', snapshot['outcomes'].get('unavailable', 0)
                     + snapshot['outcomes'].get('missing_pair_or_profile', 0))
            repeat = snapshot.get('repeat_review')
            if repeat:
                st.caption(f"{repeat['view_checks']:,} region-to-view checks across {repeat['findings']} candidates. "
                           f"Potential repeat matches in {repeat['regions_with_provisional_target']} regions. "
                           'Glare requires review.')
                st.download_button('Download repeat-view checks', (folder/'repeat-views.csv').read_bytes(),
                                   file_name='repeat-views.csv', mime='text/csv', key='basic-geometry-repeats')
            gallery = snapshot['gallery']
            if gallery:
                index = st.selectbox('Frame with supported geometry', range(len(gallery)),
                                     format_func=lambda i: gallery[i]['thermal'], key='basic-geometry-gallery')
                frame = gallery[index]
                columns = st.columns(3 if frame.get('original_rgb') else 2)
                for column, sensor in zip(columns[:2], ('thermal', 'rgb'), strict=True):
                    image_path = folder/'frames'/f"{int(frame['index']):04d}-{sensor}.jpg"
                    with column:
                        st.image(str(image_path), caption=frame[sensor], use_container_width=True)
                if frame.get('original_rgb'):
                    from pvdelta.rgb_preview import original_rgb_image

                    with columns[2]:
                        st.image(original_rgb_image(frame['original_rgb'],
                                 expected_source_hash=frame['original_rgb_hash']),
                                 caption='Original RGB · no marks', use_container_width=True)
            for name, label in (('frame-inventory.csv', 'Download frame inventory'),
                                ('candidate-support.csv', 'Download candidate overlap')):
                st.download_button(label, (folder/name).read_bytes(), file_name=name, mime='text/csv',
                                   key='basic-flight-geometry-'+name)
    except (OSError, ValueError, TypeError, KeyError, IndexError) as exc:
        st.warning(f'Saved flight geometry could not be displayed: {exc}')


def _select_library_inspection(entries):
    selected = st.session_state['basic-saved-inspection']
    entry = next(r for r in entries if r['id'] == selected)
    # Even copied stores can reuse observation IDs. Never carry unsaved review
    # notes, filters or comparison state into a different selected inspection.
    for name in list(st.session_state):
        if name.startswith(('basic-panels-v2', 'basic-candidate', 'basic-note-',
                            'basic-rgb-check', 'basic-nearby', 'basic-compare', 'basic-review-filter')):
            st.session_state.pop(name, None)
    # Explicit widget values also update an already mounted browser control;
    # deleting a radio key alone can leave its previous frontend selection visible.
    st.session_state['basic-review-filter'] = 'All'
    st.session_state['basic-panels-v2-filter'] = 'All'
    st.session_state['basic-panels-v2-evidence-queue'] = 'All candidates'
    st.session_state['basic-database'] = entry['database_path']
    st.session_state['active-inspection-database'] = entry['database_path']
    if entry.get('analysis_run_id'):
        st.session_state['basic-analysis__pending'] = entry['analysis_run_id']


def _inspection_shortcuts(library_path):
    try:
        entries = load_inspection_library(library_path)
        if not entries:
            return
        current = str(Path(st.session_state['basic-database']).expanduser().resolve())
        matching = [r for r in entries if r['database_path'] == current]
        if not matching:
            entries.insert(0, {'id':'current','label':'Current inspection','database_path':current})
            selected = 'current'
        else:
            desired_run = st.session_state.get('basic-analysis__pending', st.session_state.get('basic-analysis'))
            selected = next((r['id'] for r in matching
                if r['analysis_run_id'] == desired_run), matching[0]['id'])
        if st.session_state.get('basic-saved-inspection') not in {
                r['id'] for r in entries if r['database_path'] == current} or (
                    matching and any(r['analysis_run_id'] == desired_run for r in matching)
                    and st.session_state.get('basic-saved-inspection') != selected):
            st.session_state['basic-saved-inspection'] = selected
        st.selectbox('Saved inspection', [r['id'] for r in entries],
            format_func=lambda value:next(r['label'] for r in entries if r['id']==value),
            key='basic-saved-inspection', on_change=_select_library_inspection, args=(entries,))
        st.caption('Each inspection keeps its own candidates and review decisions. '
                   'Switching does not merge panels, transfer decisions or start a scan.')
    except _ERRORS as exc:
        st.warning(f'Saved inspection shortcuts unavailable: {exc}')


def _source_path(root, relative):
    if not relative or not root:
        return None
    base = Path(root).expanduser().resolve()
    path = (base / relative).resolve()
    if not path.is_relative_to(base):
        raise ValueError('Indexed image lies outside its source folder')
    return path


def _new_scan(output_root, sdk):
    from pvdelta.decode import load_library
    from pvdelta.ingest import index_flight_folder
    from pvdelta.native_flight import scan_native_flight

    st.subheader('Start a new inspection')
    st.write('Choose the local folder containing the original DJI thermal and RGB photos.')
    folder = st.text_input('Flight photo folder', key='basic-flight-folder',
                           placeholder='C:\\Flights\\My flight')
    st.caption('Photos stay on this computer. A new inspection is saved for each scan. '
        'The scan uses the original thermal files and their embedded measurement settings. '
        'Use Advanced to adjust measurement settings or inspect calibration.')
    if st.button('Start inspection', type='primary', key='basic-start', disabled=not folder.strip()):
        try:
            # Fail before indexing if the SDK cannot be loaded. Never guess its path from the flight folder.
            load_library(sdk)
            run = 'inspection-' + dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d-%H%M%S-%f')
            workspace = Path(output_root) / run
            workspace.mkdir(parents=True, exist_ok=False)
            bar, status = st.progress(0.), st.empty()

            def index_progress(index, total, path):
                bar.progress(index / total)
                status.caption(f'Indexing photos · {index}/{total} · {path.name}')

            indexed = index_flight_folder(folder, workspace / 'index.duckdb',
                run_id=run + '-photos', progress=index_progress)
            if not indexed.thermal_captures:
                raise ValueError('No original DJI thermal photos were found in this folder')

            def scan_progress(index, total, name):
                bar.progress(index / total)
                status.caption(f'Inspecting thermal frames · {index}/{total} · {name}')

            result = scan_native_flight(workspace / 'index.duckdb', run + '-photos',
                workspace / 'analysis', run, sdk_library=sdk, progress=scan_progress)
            st.session_state['basic-scan-result'] = result
            st.session_state['basic-pending-database'] = str(workspace / 'analysis' / 'inspection.duckdb')
            st.rerun()
        except _ERRORS as exc:
            st.error(f'Inspection could not finish: {exc}')
            st.caption('Existing inspections are preserved. Check the folder and SDK settings in Advanced.')


def _nearby_pairs(database, finding_id):
    from pvdelta.repeat_views import nearby_image_pairs

    if not st.checkbox('Compare nearby image pairs', key='basic-compare'):
        return
    st.caption('Manual comparison only: nearby photos may show different panels. '
        'These views do not establish panel identity, a different viewing angle or a glare verdict.')
    try:
        rows = nearby_image_pairs(database, finding_id)
        selected = finding_navigation([r['capture_id'] for r in rows], key='basic-nearby',
            label='Comparison frame', item_label='View',
            names={r['capture_id']: Path(r['thermal_path']).name for r in rows})
        row = next((r for r in rows if r['capture_id'] == selected), None)
        if row:
            for column, kind in zip(st.columns(2), ('thermal', 'rgb')):
                with Image.open(row[f'{kind}_path']) as source:
                    column.image(source.convert('RGB'), use_container_width=True,
                        caption=f"{Path(row[f'{kind}_path']).name} · unmarked photo")
    except _ERRORS as exc:
        st.warning(f'Comparison images unavailable: {exc}')


def _review(database, run, findings, sdk, rgb_registration):
    if findings.empty:
        st.info('No candidates were recorded for this analysis. This does not confirm healthy panels.')
        return
    status = st.radio('Review filter', ('All', 'Unreviewed', 'OK', 'Not OK', 'Wait'),
                      horizontal=True, key='basic-review-filter')
    values = {'Unreviewed': 'unreviewed', 'OK': 'ok', 'Not OK': 'not_ok', 'Wait': 'wait'}
    filtered = findings if status == 'All' else findings[findings.review_status == values[status]]
    selected = finding_navigation(filtered.finding_id.tolist(), key='basic-candidate', label='Candidate',
        names={r.finding_id: f'{Path(r.thermal_file).name} · {r.module} · ΔT {r.delta_t_k:.1f} K'
               for r in filtered.itertuples()})
    if not selected:
        return
    connection = connect_store(database, read_only=True)
    try:
        has_screening = connection.execute("SELECT 1 FROM information_schema.tables "
            "WHERE table_name='flight_screening_frames'").fetchone()
        screening_select = ', s.diagnostics_json AS diagnostics' if has_screening else ', NULL AS diagnostics'
        screening_join = (' LEFT JOIN flight_screening_frames s ON s.capture_id=c.capture_id '
            'AND s.analysis_run_id=f.analysis_run_id ' if has_screening else '')
        result = connection.execute('''SELECT f.*, m.frame_module_id,
            m.pixel_polygon_json AS module_polygon, c.capture_id,c.width_px,c.height_px,
            c.source_path AS thermal_path, rgb.source_path AS rgb_path,r.source_root,c.camera_model,
            rgb.source_hash AS rgb_hash, c.source_hash AS thermal_hash,
            coalesce(f.estimated_latitude,m.estimated_latitude) AS latitude,
            coalesce(f.estimated_longitude,m.estimated_longitude) AS longitude'''+screening_select+'''
            FROM findings f JOIN module_observations m USING(observation_id)
            JOIN captures c ON c.capture_id=m.capture_id
            LEFT JOIN captures rgb ON rgb.capture_id=c.paired_capture_id
            LEFT JOIN inspection_runs r ON r.run_id=c.run_id
            '''+screening_join+'''WHERE f.analysis_run_id=? AND f.finding_id=?''', [run, selected])
        row = dict(zip([c[0] for c in result.description], result.fetchone(), strict=True))
        try:
            profile = read_measurement_profile(connection, run, row['capture_id'])
            profile_error = None
        except _ERRORS as exc:
            profile, profile_error = None, str(exc)
    finally:
        connection.close()
    thermal_path = _source_path(row['source_root'], row['thermal_path'])
    rgb_path = _source_path(row['source_root'], row['rgb_path'])
    left, middle, right = st.columns(3)
    left.metric('Temperature difference', f"{row['delta_t_k']:.2f} K")
    middle.metric('Candidate temperature', f"{row['absolute_temperature_c']:.2f} °C")
    review_status = findings.loc[findings.finding_id == selected, 'review_status'].iloc[0]
    right.metric('Your review', {'ok': 'OK', 'not_ok': 'Not OK', 'wait': 'Wait'}.get(review_status, 'Unreviewed'))
    st.caption(f"Thermal: {Path(row['thermal_path']).name} · RGB: "
               f"{Path(row['rgb_path']).name if row['rgb_path'] else 'Unavailable'} · "
               f"Frame-local module: {row['frame_module_id']}")
    if row['latitude'] is not None and row['longitude'] is not None:
        st.caption(f"Estimated candidate position: {row['latitude']:.7f}, {row['longitude']:.7f} · "
                   'provisional location; no CAD/panel identity verification')
    else:
        st.caption('Panel coordinates unavailable. Camera GPS is not the panel location.')
    left, right, original = st.columns(3)
    native_verified = False
    with left:
        st.markdown('**Thermal anomaly crop**')
        try:
            if profile_error:
                raise ValueError('Saved measurement profile invalid; thermal overlay withheld: '+profile_error)
            if thermal_path:
                preview = thermal_finding_preview(thermal_path, json.loads(row['pixel_polygon_json']),
                    sdk_library=sdk, native_size=(row['width_px'], row['height_px']),
                    region_pixels_yx=json.loads(row['region_pixels_yx_json'] or 'null'),
                    module_polygon=json.loads(row['module_polygon']), measurement_profile=profile,
                    expected_source_hash=row['thermal_hash'])
                st.image(preview.image, caption=preview.message, use_container_width=True)
                native_verified = preview.overlay_applied
                if not preview.overlay_applied:
                    st.warning('Thermal overlay withheld: '+preview.message)
                elif not preview.radiometry_verified:
                    st.warning('Legacy analysis: showing saved candidate locations. Historical '
                               'temperatures have not been reproduced; no saved measurement profile exists.')
            else:
                st.info('Thermal source image unavailable.')
        except _ERRORS as exc:
            st.warning(f'Thermal preview unavailable: {exc}')
    with right:
        st.markdown('**Paired RGB frame**')
        try:
            if rgb_path:
                from pvdelta.rgb_pair_check import rgb_pair_check_view
                from pvdelta.rgb_preview import rgb_finding_preview

                image_slot = st.container()
                pair_check = rgb_pair_check_view(thermal_path, rgb_path, profile,
                    json.loads(row['diagnostics'] or '{}'), row['frame_module_id'], rgb_registration,
                    sdk_library=sdk, key='basic-rgb-check', automatic_junctions=True) if native_verified else None
                rgb_preview = rgb_finding_preview(rgb_path, json.loads(row['pixel_polygon_json']),
                    registration=rgb_registration, camera_model=row['camera_model'],
                    native_size=(row['width_px'], row['height_px']), source_root=row['source_root'],
                    expected_rgb_hash=row['rgb_hash'], native_verified=native_verified,
                    module_polygon=json.loads(row['module_polygon']), pair_check=pair_check,
                    measurement_profile=profile)
                with image_slot:
                    st.image(rgb_preview.image, caption=rgb_preview.message, use_container_width=True)
                    if rgb_preview.registration_status == 'approximate':
                        st.warning('Approximate RGB location. Local cross-camera alignment is not verified.')
                    elif rgb_preview.registration_status == 'image_checked_provisional':
                        st.caption('Module-edge checks passed. This is provisional image correspondence, not independent panel validation.')
                    elif not rgb_preview.overlay_applied:
                        st.info(rgb_preview.message)
            else:
                st.info('No paired RGB photo available.')
        except _ERRORS as exc:
            st.warning(f'RGB preview unavailable: {exc}')
    with original:
        st.markdown('**Original RGB · no marks**')
        try:
            if rgb_path:
                from pvdelta.rgb_preview import original_rgb_image

                st.image(original_rgb_image(rgb_path, expected_source_hash=row['rgb_hash']),
                         caption=Path(rgb_path).name, use_container_width=True)
            else:
                st.info('Paired RGB source photo unavailable.')
        except _ERRORS as exc:
            st.warning(f'Original RGB unavailable: {exc}')
    with st.expander('User review'):
        st.caption('OK: accept this candidate · Not OK: reject · Wait: needs more evidence. '
                   'A review does not certify a fault or IEC compliance.')
        saved_note = findings.loc[findings.finding_id == selected, 'review_note'].iloc[0]
        note = st.text_input('Note (optional)', value=saved_note, key=f'basic-note-{database}-{selected}')
        for column, label, value in zip(st.columns(3), ('OK', 'Not OK', 'Wait'), ('ok', 'not_ok', 'wait')):
            if column.button(label, key=f'basic-mark-{value}', use_container_width=True):
                try:
                    connection = connect_store(database)
                    try:
                        upsert_finding_review(connection, finding_id=selected, review_status=value, note=note)
                    finally:
                        connection.close()
                    st.rerun()
                except _ERRORS as exc:
                    st.error(f'Review could not be saved: {exc}')
    _nearby_pairs(database, selected)


def _results(database, sdk, rgb_registration, review_version, *, surface_manifest=None):
    connection = connect_store(database, read_only=True)
    try:
        screening = connection.execute("SELECT 1 FROM information_schema.tables "
            "WHERE table_name='flight_screening_frames'").fetchone()
        extra = (' OR EXISTS (SELECT 1 FROM flight_screening_frames s WHERE s.analysis_run_id=r.run_id)'
                 if screening else '')
        runs = [r[0] for r in connection.execute('''SELECT r.run_id FROM inspection_runs r
            WHERE EXISTS (SELECT 1 FROM module_observations m WHERE m.analysis_run_id=r.run_id)'''
            + extra + ' ORDER BY r.created_at_utc DESC,r.run_id DESC').fetchall()]
        if not runs:
            st.info('Photos are indexed, but no inspection has been run. Use Start a scan below.')
            return
        pending_run = st.session_state.pop('basic-analysis__pending', None)
        if pending_run is not None:
            if pending_run not in runs:
                st.error('The selected shortcut does not match an analysis in this database. '
                         'Check the saved inspection entry; another run has not been substituted.')
                return
            st.session_state['basic-analysis'] = pending_run
        elif st.session_state.get('basic-analysis') not in runs:
            st.session_state.pop('basic-analysis', None)
        run = st.selectbox('Inspection', runs, key='basic-analysis')
        has_profiles = connection.execute("SELECT 1 FROM information_schema.tables "
            "WHERE table_name='measurement_profiles'").fetchone()
        profiles = connection.execute('SELECT count(*) FROM measurement_profiles WHERE analysis_run_id=?',
            [run]).fetchone()[0] if has_profiles else 0
        if profiles:
            st.caption(f'Recorded measurement settings: {profiles:,} thermal frames. '
                       'Original SDK temperatures are checked when displaying native evidence.')
        else:
            st.info('Legacy inspection: saved candidate locations are available, but historical '
                    'temperatures cannot be reproduced without their recorded measurement settings. '
                    'Choose a radiometric inspection in Saved inspection when available.')
        audit = build_coverage_audit(connection, run)
        findings = connection.execute('''SELECT f.finding_id,f.anomaly_type,f.delta_t_k,
            f.absolute_temperature_c,m.frame_module_id AS module,c.source_path AS thermal_file,
            coalesce(r.review_status,'unreviewed') AS review_status, coalesce(r.note,'') AS review_note
            FROM findings f JOIN module_observations m USING(observation_id)
            JOIN captures c ON c.capture_id=m.capture_id
            LEFT JOIN finding_reviews r USING(finding_id)
            WHERE f.analysis_run_id=? ORDER BY c.source_path,m.frame_module_id,f.finding_id''', [run]).fetchdf()
    finally:
        connection.close()
    if review_version is None:
        layout = st.radio('Review version', ('Panel cards (v2)', 'Individual anomalies (v1)'),
                          horizontal=True, key='basic-review-version')
        review_version = 'panel' if layout == 'Panel cards (v2)' else 'individual'
    if review_version not in ('panel', 'individual'):
        raise ValueError('Review version must be panel or individual')
    surface_available = surface_manifest is not None and Path(surface_manifest).is_file()
    tabs = st.tabs(['Overview', 'Review candidates'] + (['Surface-assisted scan'] if surface_available else []))
    overview, review = tabs[:2]
    if surface_available:
        from pvdelta.surface_recovery_ui import surface_recovery_view

        with tabs[2]:
            surface_recovery_view(surface_manifest, sdk,
                output_root=Path(surface_manifest).parent/'runs', key='basic-surface-recovery')
    with overview:
        counts, scope = audit['counts'], audit['scope']
        columns = st.columns(5 if review_version == 'panel' else 4)
        columns[0].metric('Thermal frames in flight', scope['indexed_thermal_captures']
                          if scope['indexed_thermal_captures'] is not None else 'Unknown')
        columns[1].metric('Frames processed', counts['screened_distinct_captures'])
        columns[2].metric('Analyzed observations', counts['analyzed_observations'])
        columns[3].metric('Candidate regions' if review_version == 'panel' else 'Candidates to review', counts['finding_records'])
        if review_version == 'panel':
            columns[4].metric('Panel cards', counts['observations_with_findings'])
        st.caption('Observations can repeat the same panel across photos. The unique panel count is '
            'not established. A candidate is not a confirmed fault; no candidate does not mean healthy.')
        from collections import Counter

        from pvdelta.candidate_triage_ui import support_audit

        _flight_geometry_snapshot(database, run)
        support = support_audit(database, run)
        support_counts = Counter(r['support'] for r in support.values())
        cols = st.columns(3)
        cols[0].metric('Interior candidates', support_counts['interior'])
        cols[1].metric('Boundary review', support_counts['boundary'])
        cols[2].metric('Support unavailable', support_counts['unavailable'])
        st.info('Confirmed physical fault count: not established. Boundary-sensitive regions '
            'need a separate check and are not counted as confirmed anomalies. Review candidates → '
            'Evidence queue lets you inspect each group without deleting detections.')
        st.write(f"**{counts['analyzed_without_finding']:,}** analyzed observations without a candidate · "
                 f"**{counts['module_observations'] - counts['analyzed_observations']:,}** unassessed observations")
        if audit['frame_stages'].get('processing_error'):
            st.warning(f"{audit['frame_stages']['processing_error']} frames failed and remain unassessed.")
        missing = counts['scoped_captures_without_screening_record']
        if missing:
            st.warning(f'{missing} indexed thermal frames have no screening record in this inspection.')
        stages = audit['frame_stages']
        no_geometry = stages.get('no_native_structural_proposals', 0)
        if no_geometry:
            st.info(f'{no_geometry} processed frames have no usable structural proposals. '
                    'They are not cleared as healthy. Use Advanced for boundary and unassigned evidence review.')
        if not findings.empty:
            left, right = st.columns(2)
            left.markdown('**Candidate types**')
            left.bar_chart(findings.groupby('anomaly_type').size())
            right.markdown('**Review progress**')
            right.bar_chart(findings.groupby('review_status').size())
        st.download_button('Download candidate list', findings.to_csv(index=False),
            file_name='pvdelta-candidates.csv', mime='text/csv', key='basic-download')
    with review:
        if review_version == 'individual':
            _review(database, run, findings, sdk, rgb_registration)
        else:
            from pvdelta.panel_review_ui import panel_review_view

            panel_review_view(database, run, sdk, rgb_registration, key='basic-panels-v2')


def basic_dashboard(default_database, default_sdk, *, output_root=None, rgb_registration=None, review_version=None,
                    surface_manifest=None, library_path=None):
    """Default to existing local evidence; processing and review writes require explicit buttons."""
    st.title('pvdelta · Thermal inspection')
    st.caption('Open your inspection, review the evidence, or scan a new flight. Everything stays local.')
    if os.environ.get('PVDELTA_SURFACE_MANIFEST'):
        st.caption('Saved non-panel exclusions are active for matching source frames. '
                   'Unlabelled frames and pixels remain unverified; no masks or healthy labels are inferred.')
    pending = st.session_state.pop('basic-pending-database', None)
    if pending:
        st.session_state['basic-database'] = pending
    if 'basic-database' not in st.session_state:
        st.session_state['basic-database'] = st.session_state.get('active-inspection-database', str(default_database))
    _inspection_shortcuts(library_path or os.environ.get('PVDELTA_INSPECTION_LIBRARY') or
        Path(__file__).resolve().parents[2]/'data'/'inspection-library.json')
    with st.sidebar.expander('Open an existing inspection'):
        database_text = st.text_input('Inspection database', key='basic-database')
        st.caption('Use Advanced for SDK configuration, flight planning, maps and detailed analysis.')
    st.session_state['active-inspection-database'] = database_text
    database = Path(database_text).expanduser()
    sdk = os.environ.get('PVDELTA_DIRP_LIB') or default_sdk
    with st.expander('Start a scan', expanded=not database.is_file()):
        _new_scan(output_root or Path(__file__).resolve().parents[2] / 'data' / 'inspections', sdk)
    if database_text.strip() and database.is_file():
        try:
            _results(database, sdk, rgb_registration, review_version, surface_manifest=surface_manifest)
        except _ERRORS as exc:
            st.error(f'Inspection could not be opened: {exc}')
            st.caption('Select an inspection database, or use Advanced for older database migration.')
    else:
        st.info('No inspection is open. Start with a flight folder below, or open an existing inspection in the sidebar.')
    saved = st.session_state.get('basic-scan-result')
    if saved:
        st.caption(f"Last scan: {saved['totals'].get('completed_frames', 0)} completed frames · "
                   f"{saved['totals'].get('failed_frames', 0)} failed frames. Results saved locally.")
