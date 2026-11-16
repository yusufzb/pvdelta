"""Explicit local native-grid flight processing in the inspection dashboard."""
from __future__ import annotations

from pathlib import Path


def native_flight_view(database, sdk_library, *, output_root, key='native-flight'):
    import datetime as dt
    import json

    import duckdb
    import streamlit as st

    from pvdelta.native_flight import native_acquisitions, scan_native_flight

    with st.expander('Native thermal boundaries & anomaly scan'):
        source = st.text_input('Native scan source database', str(database), key=f'{key}-source')
        try:
            acquisitions = native_acquisitions(Path(source).expanduser())
        except (OSError, RuntimeError, ValueError, duckdb.Error) as exc:
            st.info(f'Index an original flight first: {exc}')
            return
        if not acquisitions:
            st.info('No indexed thermal acquisition is available.')
            return
        acquisition = st.selectbox('Native scan acquisition', acquisitions, key=f'{key}-acquisition')
        run_id = st.text_input('Native analysis run ID', 'native-'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d-%H%M%S'),
                               key=f'{key}-run')
        output = st.text_input('New native analysis folder', str(Path(output_root)/run_id), key=f'{key}-output')
        parameters = st.text_input('Native analysis SDK overrides JSON file (optional)', key=f'{key}-profile')
        workers = st.selectbox('Native processing workers', [1, 2, 4], key=f'{key}-workers')
        st.caption('Extract observed boundaries directly on original SDK temperature pixels. '
            'No RGB registration or CAD is required. Shared envelope pixels and a 2-pixel edge margin '
            'are excluded. These are provisional structural proposals; panel membership, faults and '
            'unique physical panel counts remain unverified. RGB hotspot transfer is withheld.')
        result_key = f'{key}-result'
        if st.button('Run native-grid inspection', key=f'{key}-start', type='primary'):
            st.session_state.pop(result_key, None)
            try:
                overrides = json.loads(Path(parameters).expanduser().read_bytes()) if parameters.strip() else None
                bar, label = st.progress(0.), st.empty()

                def progress(index, total, name):
                    bar.progress(index/total)
                    label.caption(f'{index}/{total} · {name}')

                result = scan_native_flight(Path(source).expanduser(), acquisition, Path(output).expanduser(),
                    run_id, sdk_library=sdk_library or None, measurement_overrides=overrides,
                    workers=workers, progress=progress)
                st.session_state[result_key] = (str(Path(output).expanduser().resolve()), result)
            except (ImportError, OSError, RuntimeError, ValueError, TypeError, KeyError, duckdb.Error) as exc:
                st.error(f'Native inspection failed: {exc}')
        saved = st.session_state.get(result_key)
        if saved:
            folder, result = saved
            if result.get('surface_constraints'):
                support=result['surface_constraints']
                st.write({'Source-matched surface frames':support['matching_source_frames'],
                          'Frames without saved surface reference':support['unreferenced_source_frames']})
                st.caption('Frames without a saved reference retain provisional geometry. '
                           'They are not treated as masked, empty or healthy.')
            totals = result['totals']
            st.write({'Analysis run': result['analysis_run_id'], 'Status': result['status'],
                'SDK decoded frames': totals.get('decoded_frames', 0),
                'Completed frames': totals.get('completed_frames', 0),
                'Failed frames': totals.get('failed_frames', 0),
                'Provisional module observations': totals.get('module_observations', 0),
                'Analyzed observations': totals.get('analyzed_observations', 0),
                'Unassessed observations': totals.get('not_assessed_observations', 0),
                'Candidate regions requiring review': totals.get('finding_records', 0)})
            st.caption('Last completed attempt. Open this database in Local data & map. '
                'An analyzed observation without a finding is not a healthy-panel confirmation.')
            st.code(str(Path(folder)/'inspection.duckdb'), language='text')
            failed = [r for r in result['frames'] if r['status'] == 'failed']
            if failed:
                st.warning(f'{len(failed)} frames failed and remain unassessed.')
                st.dataframe(failed, hide_index=True, width='stretch')
            st.download_button('Download native inspection summary', json.dumps(result, indent=2, allow_nan=False),
                file_name='native-inspection-summary.json', mime='application/json', key=f'{key}-download')
