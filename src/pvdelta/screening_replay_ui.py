"""Run source-bound reanalysis from the inspection dashboard."""
from pathlib import Path

from pvdelta.screening_replay import attach_site_geometry, replay_screening, screening_runs


def screening_replay_view(database, sdk_library, *, output_root, key='replay'):
    import datetime as dt
    import json

    import duckdb
    import streamlit as st

    with st.expander('Run anomaly analysis on stored panel boundaries'):
        source_text = st.text_input('Source inspection database', str(database), key=f'{key}-source')
        try:
            runs = screening_runs(Path(source_text).expanduser())
        except (OSError, RuntimeError, ValueError, duckdb.Error) as exc:
            st.info(f'No screening geometry available: {exc}')
            return
        if not runs:
            st.info('No compatible legacy RGB-derived screening geometry is available. '
                    'For native-only runs, use Native thermal boundaries & anomaly scan.')
            return
        baseline = st.selectbox('Geometry source run', runs, key=f'{key}-baseline')
        run_id = st.text_input('New analysis run ID', 'stage1-'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d-%H%M%S'),
                               key=f'{key}-run')
        output = st.text_input('New private analysis folder', str(Path(output_root)/run_id), key=f'{key}-output')
        profile_text = st.text_input('Analysis measurement overrides JSON file (optional)', key=f'{key}-profile')
        per_frame_text = st.text_input('Per-frame measurement overrides JSON file (optional)', key=f'{key}-per-frame')
        proposed_site = Path(source_text).expanduser().parent/'site-geometry.json'
        site_text = st.text_input('Site geometry JSON for repeat-view checks (optional)',
            str(proposed_site) if proposed_site.is_file() else '', key=f'{key}-site')
        st.caption('Every source is decoded again. Rejected geometry stays unassessed. Accepted boundaries '
            'remain provisional; this reanalysis does not verify panel membership. Results go to a new '
            'inspection.duckdb with per-analysis SDK profiles. No physical panel count is inferred.')
        result_key = f'{key}-result'
        if st.button('Run radiometric anomaly analysis', key=f'{key}-start', type='primary'):
            st.session_state.pop(result_key, None)
            try:
                fixed = json.loads(Path(profile_text).expanduser().read_bytes()) if profile_text.strip() else None
                per_frame = json.loads(Path(per_frame_text).expanduser().read_bytes()) if per_frame_text.strip() else None
                if site_text.strip():
                    json.loads(Path(site_text).expanduser().read_bytes())
                bar, label = st.progress(0.), st.empty()

                def progress(index, total, name):
                    bar.progress(index/total)
                    label.caption(f'{index}/{total} · {name}')

                result = replay_screening(Path(source_text).expanduser(), baseline, Path(output).expanduser(),
                    run_id, sdk_library=sdk_library or None, measurement_overrides=fixed,
                    per_frame_overrides=per_frame, progress=progress)
                if site_text.strip():
                    result = attach_site_geometry(Path(output).expanduser(), Path(site_text).expanduser())
                st.session_state[result_key] = (str(Path(output).expanduser().resolve()), result)
            except (ImportError, OSError, RuntimeError, ValueError, TypeError, KeyError, duckdb.Error) as exc:
                st.error(f'Anomaly reanalysis failed: {exc}')
        saved = st.session_state.get(result_key)
        if saved:
            folder, result = saved
            totals = result['totals']
            st.write({'Run': result['run_id'], 'Status': result['status'],
                'SDK decoded frames': totals.get('decoded_frames', 0),
                'Analyzed observations': totals.get('analyzed_observations', 0),
                'Unassessed observations': totals.get('not_assessed_observations', 0),
                'Candidate regions': totals.get('finding_records', 0),
                'Failed frames': totals.get('failed_frames', 0)})
            st.caption('Open the following database in Local data & map to inspect the new run. '
                       'Candidate regions and repeated observations are not physical panel or confirmed fault counts.')
            st.code(str(Path(folder)/'inspection.duckdb'), language='text')
            failed = [r for r in result['frames'] if r['status'] == 'failed']
            if failed:
                st.warning(f'{len(failed)} frames failed and remain unassessed.')
                st.dataframe(failed, hide_index=True, width='stretch')
            st.download_button('Download anomaly analysis summary', json.dumps(result, indent=2, allow_nan=False),
                file_name='anomaly-analysis-summary.json', mime='application/json', key=f'{key}-download')
