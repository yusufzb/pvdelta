"""Operator entry point for locally configured surface-assisted recovery."""
import json
import os
from pathlib import Path

from pvdelta.review_ui import finding_navigation
from pvdelta.surface_recovery import (
    available_surface_runs,
    load_surface_manifest,
    load_surface_run,
    recovery_engine_sources,
    run_surface_recovery,
)


def surface_recovery_view(manifest_path, sdk_library, *, output_root, key='surface-recovery'):
    import streamlit as st

    with st.expander('Surface-assisted scan', expanded=True):
        st.caption('Use saved panel/table areas to recover missing boundaries and recompute native '
                   'thermal measurements. Runs are saved separately from the main inspection.')
        if not Path(manifest_path).is_file():
            st.info('No surface-assisted scan configuration is available for this workspace.')
            return
        try:
            manifest, _ = load_surface_manifest(manifest_path)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            st.error(f'Surface configuration unavailable: {exc}')
            return
        total = manifest.get('flight_thermal_frames')
        st.write(f"Configured frames: {len(manifest['frames'])}" + (f' / {total} indexed thermal frames' if total else ''))
        st.caption('Only the configured frames have eligible saved surface areas. This is partial '
                   'flight coverage; counts below are frame observations, not unique physical modules.')
        allow_isolated = st.checkbox('Recover isolated tables inside saved panel areas',
            value=False, key=f'{key}-allow-isolated',
            help='Allows recovery without an adjacent baseline panel. Four observed edges and two '
                 'observed scale references are still required. Only saved manual panel areas are eligible.')
        exclude_declared_non_panel=bool(os.environ.get('PVDELTA_SURFACE_MANIFEST'))
        if exclude_declared_non_panel:
            st.caption('Configured source-bound non-panel exclusions are active throughout this app. '
                       'Unlabelled frames/pixels remain unverified; earlier runs are preserved.')
        else:
            exclude_declared_non_panel=st.checkbox('Exclude saved non-panel areas from measurements',
                value=False,key=f'{key}-require-surface',
                help='Exclude explicitly saved non-panel and conflicting pixels from source-bound measurements. '
                     'Labelled panel pixels are preserved and unlabelled pixels remain uncertain. '
                     'Original geometry and earlier runs are preserved; this is not automatic whole-flight validation.')
        if st.button('Run surface-assisted scan', key=f'{key}-run', type='primary'):
            bar = st.progress(0., text='Preparing native SDK measurements…')
            try:
                attempted, _ = run_surface_recovery(manifest_path, sdk_library, output_root,
                    progress=lambda done, count: bar.progress(done/count, text=f'Frame {done} / {count}'),
                    **({'allow_isolated': True} if allow_isolated else {}),
                    **({'exclude_declared_non_panel':True} if exclude_declared_non_panel else {}))
                st.session_state[f'{key}-saved-run'] = attempted['run_id']
                if attempted['failed_frames']:
                    st.warning('This attempt has failed frames. The previous complete run remains the default; '
                               'use Saved scan run to switch between attempts.')
            except (OSError, ValueError, TypeError, KeyError, RuntimeError, ImportError) as exc:
                st.error(f'Surface scan could not finish: {exc}')
            finally:
                bar.empty()
        try:
            run_ids = available_surface_runs(output_root)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            st.error(f'Saved scan history unavailable: {exc}')
            return
        if not run_ids:
            st.info('Run the scan to display boundaries, temperatures and candidate regions here.')
            return
        run_key = f'{key}-saved-run'
        if st.session_state.get(run_key) not in run_ids:
            st.session_state[run_key] = run_ids[0]
        selected_run = st.selectbox('Saved scan run', run_ids, key=run_key,
            help='Saved attempts include partial and failed scans. Selecting a run never starts a scan.')
        try:
            summary, directory = load_surface_run(output_root, manifest_path, run_id=selected_run)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            st.error(f'Saved surface scan unavailable: {exc}')
            return
        if summary.get('engine_source_sha256') != recovery_engine_sources():
            st.warning('Saved run uses an earlier engine. Run again to use the current implementation.')
        if summary.get('started_at_utc'):
            st.caption(f"Started: {summary['started_at_utc']} (UTC)")
        st.caption('Saved run mode: '+('isolated-table recovery with manual panel support'
            if summary.get('allow_isolated', False) else 'adjacent baseline required'))
        if summary.get('exclude_declared_non_panel',False):
            st.caption('Explicit saved non-panel/conflicting pixels are excluded; unlabelled pixels remain '
                       'uncertain. Grey outlines are not assessed, not healthy. Earlier inspections are preserved.')
        rows = sorted((r for r in summary['frames'] if r['status']=='completed'),
                      key=lambda r: (-r['added_observations'], r['case_id']))
        metrics = st.columns(4)
        for col, name, value in zip(metrics,
            ('Completed frames', 'Failed frames', 'Module observations', 'Added observations'),
            (len(rows), summary['failed_frames'], sum(r['observation_count'] for r in rows),
             sum(r['added_observations'] for r in rows))):
            col.metric(name, value)
        if summary['failed_frames']:
            st.warning('Some frames failed and are excluded from measurement totals. They are not normal or empty frames.')
            st.dataframe([r for r in summary['frames'] if r['status']=='failed'], hide_index=True)
        st.caption(f"Saved run: {summary['run_id']} · {sum(r['candidate_regions'] for r in rows)} local candidate regions. "
                   'No candidate does not establish a healthy module. Field accuracy and whole-module peer comparison remain separate.')
        if not rows:
            return
        st.caption('Frames with recovered observations appear first. This saved surface run is separate '
                   'from the inspection selected elsewhere in the dashboard.')
        selected = finding_navigation([r['case_id'] for r in rows], key=f'{key}-frame',
            label='Surface scan frame', names={r['case_id']:f"{r['case_id']} · {r['source_file']} · +{r['added_observations']}" for r in rows},
            item_label='Frame')
        record = json.loads((directory/f'{selected}.json').read_bytes())
        left, right = st.columns(2)
        left.image(str(directory/f'{selected}-before.png'), caption='Before · retained boundaries', width='stretch')
        right.image(str(directory/f'{selected}-after.png'), caption='After · cyan retained, yellow added, red candidate regions', width='stretch')
        st.caption(record['source_file']+' · Native thermal coordinates. Colours are display only; measurements use original SDK pixels.')
        table = []
        for report in record['module_reports']:
            m = report['module']
            table.append({'Observation':m['module_id'], 'Status':report['assessment_status'],
                'Min °C':m['temp_min'], 'Median °C':m['temp_median'], 'Max °C':m['temp_max'],
                'Candidate regions':len(report['findings'])})
        st.dataframe(table, hide_index=True, width='stretch')
        with st.expander('Measurement profile and source identity'):
            st.json(record['measurement_profile'])
            st.caption('Existing measurement pixels and results were checked for exact preservation. '
                       'This saved run does not modify findings or user reviews in the main database.')
        st.download_button('Download surface scan frame', (directory/f'{selected}.json').read_bytes(),
            file_name=f'{selected}-surface-scan.json', mime='application/json', key=f'{key}-download')
