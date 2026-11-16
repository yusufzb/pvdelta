"""Optional local full-panel reference scoring inside the native preview."""
from __future__ import annotations

import io
import json
from pathlib import Path

from pvdelta.boundary_validation import compare_boundary_reference, reference_validation_token


def boundary_reference_validation(preview, *, key):
    import streamlit as st

    result_key = f'{key}-reference-result'
    with st.expander('Reference validation (optional)'):
        st.caption('Use a local full-frame panel reference JSON, or a bundle containing frames. '
                   'Source SHA256 and native grid must match. This scores existing geometry '
                   'without another SDK decode or saving a review decision.')
        path_key = f'{key}-reference-path'
        # Streamlit removes widget state while a frame has no preview. Keep the
        # chosen bundle path separately so navigating does not require retyping it.
        if path_key not in st.session_state:
            st.session_state[path_key] = st.session_state.get(f'{key}-reference-file','')
        path = st.text_input('Panel reference JSON path',key=path_key)
        st.session_state[f'{key}-reference-file'] = path
        if not path.strip():
            st.session_state.pop(result_key,None)
            return
        try:
            content = Path(path).expanduser().read_bytes()
            token = reference_validation_token(preview.report,content)
        except (OSError, ValueError, TypeError) as exc:
            st.session_state.pop(result_key,None)
            st.error(f'Reference unavailable: {exc}')
            return
        previous = st.session_state.get(result_key)
        if previous is not None and previous.report['validation_token']!=token:
            st.session_state.pop(result_key,None)
        if st.button('Validate against reference',key=f'{key}-validate-reference'):
            st.session_state.pop(result_key,None)
            try:
                st.session_state[result_key] = compare_boundary_reference(preview,content)
            except (ValueError, TypeError, KeyError, OSError) as exc:
                st.error(f'Reference validation failed: {exc}')
        result = st.session_state.get(result_key)
        if result is None:
            return
        report = result.report
        st.write('Declared reference status: '+report['declared_reference_review_status'])
        if report.get('reference_native_temperature_binding') != 'matched':
            st.caption('This reference has no native temperature hash. Source and grid matching '
                       'do not verify that the measurement profile matches.')
        st.caption('Label status is supplied metadata; this screen does not verify independence. '
                   'Developer drafts and data used for tuning remain development evidence.')
        st.warning('Reference placement has not been verified here. An unmatched draft may be '
                   'misplaced rather than a missed panel. Check the original image before '
                   'changing detector boundaries; do not tune to displaced labels.')
        rows = []
        for mode,thresholds in report['scores'].items():
            for threshold,score in thresholds.items():
                rows.append({'mode':mode,'IoU':float(threshold),
                    'Reference panels':score['reference_instances'],
                    'Matched (TP)':score['true_positives'],'Unmatched candidates (FP)':score['false_positives'],
                    'Unmatched references (FN)':score['false_negatives'],
                    'Ignored candidates':len(score['ignored_proposal_ids']),
                    'Precision':score['precision'],'Recall':score['recall']})
        st.dataframe(rows,hide_index=True,width='stretch')
        st.caption('Overlays at IoU 0.75: green TP = matched; amber FP = unmatched candidate; '
                   'dashed red FN = unmatched reference; grey IGN = explicit ignore region. '
                   'Colours describe geometry matching, not thermal anomaly severity.')
        left,right = st.columns(2)
        left.image(result.baseline_image,caption='Baseline vs supplied references',use_container_width=True)
        right.image(result.proposed_image,caption='Proposed vs supplied references',use_container_width=True)
        with st.expander('Matched and unmatched IDs / provenance'):
            st.json(report)
        st.download_button('Download reference comparison (JSON)',json.dumps(report,indent=2,allow_nan=False),
            file_name=f"{Path(preview.report['source_name']).stem}-reference-comparison.json",
            mime='application/json',key=f'{key}-reference-download')
        image_data = io.BytesIO()
        result.proposed_image.save(image_data,format='PNG')
        st.download_button('Download reference error overlay',image_data.getvalue(),
            file_name=f"{Path(preview.report['source_name']).stem}-reference-errors.png",
            mime='image/png',key=f'{key}-reference-image')
