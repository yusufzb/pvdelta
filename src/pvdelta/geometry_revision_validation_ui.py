"""Compare active revision to its session baseline against explicit panel labels."""
from __future__ import annotations

import json
from pathlib import Path

from pvdelta.boundary_validation import _load_reference, _overlay
from pvdelta.geometry_revision_validation import (
    compare_geometry_versions,
    revision_validation_token,
)


def revision_reference_validation(before, after, picture, *, key):
    import streamlit as st

    with st.expander('Evaluate revision against panel reference'):
        st.caption('Optional exhaustive full-frame panel reference JSON or bundle. '
            'Compare the active version to the session baseline without changing either. '
            'AI drafts and edits using their corners remain assisted development evidence.')
        path = st.text_input('Revision panel reference JSON path', key=f'{key}-path')
        result_key = f'{key}-result'
        if not path.strip():
            st.session_state.pop(result_key, None)
            return
        try:
            content = Path(path).expanduser().read_bytes()
            token = revision_validation_token(before, after, content)
        except (OSError, ValueError, TypeError) as exc:
            st.session_state.pop(result_key, None)
            st.error(f'Revision reference unavailable: {exc}')
            return
        previous = st.session_state.get(result_key)
        if previous is not None and previous['validation_token'] != token:
            st.session_state.pop(result_key, None)
        if st.button('Evaluate active revision', key=f'{key}-evaluate'):
            st.session_state.pop(result_key, None)
            try:
                if picture.size != tuple(reversed(before['source']['native_shape'])):
                    raise ValueError('Evaluation image must use the exact native grid')
                st.session_state[result_key] = compare_geometry_versions(before, after, content)
            except (ValueError, TypeError, KeyError, OSError) as exc:
                st.error(f'Revision evaluation failed: {exc}')
        result = st.session_state.get(result_key)
        if result is None:
            return
        st.write('Declared reference status: '+result['declared_reference_review_status'])
        st.caption('Declared status does not verify independent review. Gains/losses below are '
            'geometry matches against the supplied labels, not verified anomaly or healthy-panel counts.')
        st.dataframe([{'Version': label, 'IoU': float(t), 'Reference panels': s['reference_instances'],
            'Matched': s['true_positives'], 'Unmatched candidates': s['false_positives'],
            'Missed': s['false_negatives'], 'Ignored': len(s['ignored_proposal_ids']),
            'Precision': s['precision'], 'Recall': s['recall']}
            for label, thresholds in result['scores'].items() for t, s in thresholds.items()],
            hide_index=True, width='stretch')
        st.write('Gains and losses at IoU 0.75')
        st.json(result['transitions']['0.75'], expanded=False)
        st.caption('Green: matched geometry; amber: unmatched proposal; dashed red: missed reference; '
            'grey: explicit ignore. These colours do not show thermal severity.')
        reference, _ = _load_reference(content, before['source']['source_sha256'])
        for col, label, version in zip(st.columns(2), ('before', 'after'), (before, after)):
            col.image(_overlay(picture, reference, version['geometry']['proposals'],
                result['scores'][label]['0.75']), caption=f'{label.title()} reference errors', width='stretch')
        st.write('Measured hot/cold pixels relative to the reference panels')
        rows = [{'Version': label, **row} for label, placement in result['component_placement'].items()
                for row in placement['components']]
        if rows:
            st.dataframe(rows, hide_index=True, width='stretch')
        else:
            st.caption('No measured candidate components in these versions. This is not a healthy-panel confirmation.')
        st.caption('Placement uses exact measured component pixels, not the enlarged display rectangle. '
            'Pixels outside supplied panel labels remain visible; the labels themselves may need correction. '
            'This evaluation does not re-decode temperatures or activate scan findings.')
        st.download_button('Download revision evaluation (JSON)', json.dumps(result, indent=2, allow_nan=False),
            file_name='geometry-revision-evaluation.json', mime='application/json', key=f'{key}-download')
