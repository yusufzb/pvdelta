"""On-demand Streamlit review of provisional native panel boundaries."""
from __future__ import annotations

import io
import json
from pathlib import Path

from pvdelta.boundary_review_queue import (
    REVIEW_STATES,
    boundary_issue_counts,
    boundary_review_queue,
    filter_boundary_queue,
)
from pvdelta.boundary_review_store import (
    BOUNDARY_REASONS,
    boundary_engine_signature,
    boundary_result_identity,
    boundary_review_for_result,
    boundary_review_history,
    latest_boundary_reviews,
    save_boundary_review,
)
from pvdelta.boundary_validation_ui import boundary_reference_validation
from pvdelta.native_review import (
    boundary_engine_sources,
    boundary_preview_token,
    indexed_native_frames,
    native_boundary_preview,
)
from pvdelta.review_ui import finding_navigation


def _boundary_decision(report, database, key):
    import duckdb
    import streamlit as st

    result_id = boundary_result_identity(report)
    try:
        current = boundary_review_for_result(database, result_id)
        history = boundary_review_history(database, report['source_sha256'])
    except (OSError, RuntimeError, ValueError, duckdb.Error) as exc:
        st.warning(f'Saved boundary review unavailable: {exc}')
        current, history = None, []
    with st.expander('Boundary review', expanded=True):
        st.caption('OK accepts the displayed frame boundaries and coverage for visual review. '
                   'It does not confirm an anomaly or authorise measurement. Decisions are saved '
                   'for this exact source, native data, engine and result; previous decisions remain in history.')
        status = {'ok':'OK','not_ok':'Not OK','wait':'Wait'}
        st.write('This comparison: '+(status[current['review_status']] if current else 'No saved decision'))
        if current:
            st.caption(f"{current['reviewer'] or 'Reviewer not specified'} · {current['reviewed_at_utc']}")
        reviewer_column, note_column = st.columns([1,3])
        with reviewer_column:
            reviewer = st.text_input('Boundary reviewer (optional)',
                (current or {}).get('reviewer') or '', key=f'{key}-{result_id}-reviewer')
        with note_column:
            note = st.text_input('Boundary note (optional)',
                (current or {}).get('note') or '', key=f'{key}-{result_id}-note')
        reasons = st.multiselect('Boundary issues (optional)', list(BOUNDARY_REASONS),
            default=(current or {}).get('reason_codes', []), format_func=BOUNDARY_REASONS.get,
            key=f"{key}-{result_id}-{(current or {}).get('event_id', 0)}-reasons")
        st.caption('Reasons describe boundary errors, not electrical fault causes. '
                   'Not OK / Wait saves selected issues. OK clears active issues; earlier records remain in history.')
        for column, (value,label) in zip(st.columns(3), status.items()):
            if column.button(label, key=f'{key}-decision-{value}', use_container_width=True):
                try:
                    save_boundary_review(database,report,value,reviewer=reviewer,note=note,
                                         reason_codes=reasons if value!='ok' else ())
                except (OSError, RuntimeError, ValueError, duckdb.Error) as exc:
                    st.error(f'Boundary decision was not saved: {exc}')
                else:
                    st.session_state[f'{key}-saved-message'] = f"Boundary review saved: {label} · {report['source_name']}"
                    st.rerun()
        if history:
            with st.expander('Decision history (latest 50 for this source)'):
                st.dataframe(history, hide_index=True, width='stretch')


def native_boundary_review(database, sdk_library, *, key='native-boundaries', revision_output=None):
    import duckdb
    import streamlit as st

    st.subheader('Panel boundary comparison')
    st.caption('Compare native thermal geometry on one frame. Cyan: existing boundaries; '
               'amber: supplemental boundaries. This preview does not change stored findings.')
    message = st.session_state.pop(f'{key}-saved-message', None)
    if message:
        st.success(message)
    result_key = f'{key}-result'
    with st.expander('Source settings'):
        database_text = st.text_input('Inspection database', str(database), key=f'{key}-database')
        sdk_text = st.text_input('DJI Thermal SDK library', str(sdk_library or ''), key=f'{key}-sdk')
    try:
        saved = latest_boundary_reviews(Path(database_text).expanduser())
    except (OSError, RuntimeError, ValueError, duckdb.Error) as exc:
        st.warning(f'Boundary review queue unavailable: {exc}')
        saved = {}
    engine = boundary_engine_signature(boundary_engine_sources())
    mode = st.radio('Frame source', ['Indexed frames', 'Local thermal file'], horizontal=True,
                    key=f'{key}-source-mode')
    expected_hash = None
    if mode == 'Indexed frames':
        try:
            frames = indexed_native_frames(Path(database_text).expanduser())
        except (OSError, ValueError, RuntimeError, duckdb.Error) as exc:
            st.session_state.pop(result_key, None)
            st.info(f'Indexed frames unavailable: {exc}. Index a flight in Local data & map, '
                    'or choose Local thermal file.')
            return
        if not frames:
            st.session_state.pop(result_key, None)
            st.info('No indexed thermal frames. Index a flight or choose Local thermal file.')
            return
        flights = list(dict.fromkeys(f['run_id'] for f in frames))
        flight = st.selectbox('Flight / acquisition', flights,
            format_func=lambda value: value or 'Unassigned flight', key=f'{key}-flight')
        scoped = [f for f in frames if f['run_id']==flight]
        queue = boundary_review_queue(scoped, saved, engine)
        issue_counts = boundary_issue_counts(queue)
        with st.expander('Boundary review queue'):
            for column, (state,label) in zip(st.columns(len(REVIEW_STATES)), REVIEW_STATES.items()):
                column.metric(label, sum(r['saved_state']==label for r in queue))
            st.caption('Latest saved visual decisions for this acquisition. These are historical '
                       'review records; compare a frame to check the exact current result. '
                       'An engine change needs a new review. No saved review is not an accepted frame.')
            st.dataframe([{k:r[k] for k in ('frame','saved_state','reason_codes','note','reviewer','reviewed_at')}
                          for r in queue], hide_index=True, width='stretch')
            st.caption('Issue counts: frame records with a current-engine Not OK / Wait decision. '
                       'One frame can have several issues; counts overlap and are not panel totals. '
                       'Older-engine issues remain in history and the export.')
            if issue_counts:
                st.dataframe(issue_counts, hide_index=True, width='stretch')
            else:
                st.info('No tagged Not OK / Wait decisions for this engine and acquisition.')
            export = {'schema':'pvdelta.boundary-review-queue/v1', 'acquisition_id':flight,
                      'scope':'Historical frame review records; exact current results have not been revalidated.',
                      'frame_records':len(queue), 'issue_counts':issue_counts, 'queue':queue}
            st.download_button('Download boundary review queue (JSON)', json.dumps(export,indent=2),
                file_name='boundary-review-queue.json', mime='application/json', key=f'{key}-queue-download')
        filter_name = st.selectbox('Boundary review filter',
            ['All frames','Needs review','Saved OK','Saved Not OK','Saved Wait','Other engine version'],
            key=f'{key}-review-filter')
        reason_filter = st.selectbox('Boundary issue filter',
            ['All issues','No recorded reason',*BOUNDARY_REASONS],
            format_func=lambda value: BOUNDARY_REASONS.get(value,value), key=f'{key}-issue-filter')
        selected_ids = {r['capture_id'] for r in filter_boundary_queue(queue,filter_name,reason_filter)}
        scoped = [f for f in scoped if f['capture_id'] in selected_ids]
        if not scoped:
            st.session_state.pop(result_key, None)
            st.info('No frames match these boundary review filters.')
            return
        selected = finding_navigation([f['capture_id'] for f in scoped], key=f'{key}-frame',
            label='Thermal frame', names={f['capture_id']:f.get('source_name') or Path(f['source_path']).name for f in scoped},
            item_label='Frame')
        record = next(f for f in scoped if f['capture_id']==selected)
        if not record['source_path']:
            st.session_state.pop(result_key, None)
            st.info('Acquisition source root unavailable or relative path leaves it. Choose Local thermal file.')
            return
        path_text = record['source_path']
        expected_hash = record['source_hash']
        st.caption(f"{Path(path_text).name} · {record['captured_at'] or 'Capture time unavailable'}")
    else:
        path_text = st.text_input('DJI thermal R-JPEG path', '', key=f'{key}-path')
    if not path_text.strip():
        st.session_state.pop(result_key, None)
        return
    path = Path(path_text).expanduser()
    try:
        token = boundary_preview_token(path, sdk_text or None, expected_sha256=expected_hash)
    except (OSError, ValueError, RuntimeError) as exc:
        st.session_state.pop(result_key, None)
        st.error(f'Boundary preview unavailable: {exc}')
        return
    previous = st.session_state.get(result_key)
    if previous is not None and previous.report['preview_token'] != token:
        st.session_state.pop(result_key, None)
    if st.button('Compare panel boundaries', key=f'{key}-compare', type='primary'):
        st.session_state.pop(result_key, None)
        try:
            with st.spinner('Decoding native SDK temperatures and comparing boundaries…'):
                st.session_state[result_key] = native_boundary_preview(
                    path, sdk_text or None, expected_sha256=expected_hash)
        except (OSError, ValueError, RuntimeError, TypeError, ImportError) as exc:
            st.error(f'Boundary comparison failed: {exc}')
    preview = st.session_state.get(result_key)
    if preview is None:
        st.caption('Choose a frame and press Compare panel boundaries. Computation runs only on request.')
        return
    report = preview.report
    metrics = st.columns(4)
    for column, label, value in zip(metrics,
        ('Baseline candidates', 'Proposed candidates', 'Supplemental candidates', 'Empty interiors'),
        (report['baseline_proposals'], report['proposals'], report['raw_additions'], report['empty_interiors'])):
        column.metric(label, value)
    st.caption('These are unreviewed observations in this frame, not unique site panels or anomaly counts.')
    left, right = st.columns(2)
    left.image(preview.baseline_image, caption=f"{report['source_name']} · Baseline boundaries",
               use_container_width=True)
    right.image(preview.proposed_image, caption=f"{report['source_name']} · Supplemental search",
                use_container_width=True)
    if report['lost_baseline_proposal_ids'] or report['old_pixels_lost_outside_shared_envelopes']:
        st.warning('Geometry regression detected. Review the comparison and diagnostic report.')
    _boundary_decision(report,Path(database_text).expanduser(),key)
    boundary_reference_validation(preview,key=key)
    from pvdelta.geometry_revision_ui import geometry_revision_view

    geometry_revision_view(report, path, sdk_text or None,
        output=revision_output or Path(database_text).expanduser().parent / 'geometry-revisions', key=f'{key}-revisions')
    from pvdelta.local_rgb_transfer_ui import local_rgb_transfer_view

    local_rgb_transfer_view(report, path, sdk_text or None, key=f'{key}-local-rgb')
    if st.checkbox('Show pixel support', key=f'{key}-show-support'):
        st.image(preview.support_image, caption='Cyan tint: unique 2px-inset interior. '
                 'Magenta: shared envelope pixels, excluded from every ROI.', use_container_width=True)
        st.write({'Unique interior pixels':report['unique_interior_pixels'],
                  'Shared envelope pixels':report['shared_envelope_pixels'],
                  'Previously assigned pixels now excluded':report['old_interior_pixels_excluded']})
    with st.expander('Native temperature statistics and provenance'):
        st.caption('Descriptive values from original SDK pixels in provisional ROIs. '
                   'No finding means neither a healthy panel nor an IEC acceptance decision.')
        st.dataframe(report['descriptive_statistics'], hide_index=True, width='stretch')
        st.json({k:report[k] for k in ('source_name','source_sha256','native_temperature_sha256',
                                      'native_shape','radiometric_parameters','native_values_unchanged')})
    report_json = json.dumps(report, indent=2, allow_nan=False)
    st.download_button('Download boundary review (JSON)', report_json,
        file_name=f'{path.stem}-boundary-review.json', mime='application/json', key=f'{key}-download-json')
    image_data = io.BytesIO()
    preview.proposed_image.save(image_data, format='PNG')
    st.download_button('Download marked thermal image', image_data.getvalue(),
        file_name=f'{path.stem}-boundaries.png', mime='image/png', key=f'{key}-download-image')
