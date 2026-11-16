"""Display a read-only coverage audit without certifying provisional geometry."""
from __future__ import annotations

import json


def coverage_audit_view(audit):
    import pandas as pd
    import streamlit as st

    with st.expander("Coverage audit & geometry rejection reasons"):
        scope, counts = audit['scope'], audit['counts']
        denominator = scope['indexed_thermal_captures']
        indexed = f'{denominator:,}' if denominator is not None else 'Unknown'
        columns = st.columns(4)
        columns[0].metric('Indexed thermal captures (scoped)', indexed)
        columns[1].metric('Captures with screening records', f"{counts['screened_distinct_captures']:,}")
        columns[2].metric('Analyzed observation records', f"{counts['analyzed_with_finite_statistics']:,}")
        columns[3].metric('Unique physical panels', 'Not established')
        st.caption(
            'Selected analysis run only. Observations repeat across overlapping images. '
            'Analyzed without a finding does not mean healthy. Registration verification and '
            'SDK decode coverage are not revalidated by this read-only audit.'
        )
        if not scope['scope_complete']:
            st.warning('Acquisition scope is incomplete; the indexed-flight denominator is not established.')
        st.markdown('**Where frames leave the screening pipeline**')
        stages = pd.DataFrame([{'stage': k.replace('_', ' '), 'frames': v}
                               for k, v in audit['frame_stages'].items()])
        if not stages.empty:
            st.bar_chart(stages.set_index('stage')['frames'])
        else:
            st.info('No frame-screening records are stored for this run.')
        reasons = pd.DataFrame([{'reason': k.replace('_', ' '), 'observations': v}
                               for k, v in audit['overlapping_module_rejection_reasons'].items()])
        st.markdown('**Legacy module geometry rejection reasons**')
        if not reasons.empty:
            st.bar_chart(reasons.set_index('reason')['observations'])
        st.caption(
            'A rejected observation can have multiple reasons; bars must not be summed. '
            'These explain stored legacy gates, not ground-truth error. RGB proposals outside '
            'the smaller thermal field of view are not proven missed panels.'
        )
        blockers = pd.DataFrame([{'blocking gate': k.replace('_', ' '), 'observations': v}
                                for k, v in audit['exclusive_module_gate_states'].items()])
        st.markdown('**Gate combinations (one state per observation)**')
        st.dataframe(blockers, hide_index=True, width='stretch')
        priority = pd.DataFrame(audit['priority_review_frames'])
        if not priority.empty:
            st.markdown('**Review queue: edges passed, shared frame registration rejected**')
            st.dataframe(priority.drop(columns='capture_id'), hide_index=True, width='stretch')
            st.caption('Review priority only. Passing legacy edges alone does not establish a valid panel mask.')
        assessment = pd.DataFrame([{'state': k.replace('_', ' '), 'observations': v}
                                  for k, v in audit['module_assessment_statuses'].items()])
        st.dataframe(assessment, hide_index=True, width='stretch')
        st.write({
            'Analyzed without finding (not confirmed healthy)': counts['analyzed_without_finding'],
            'Observation records with findings': counts['observations_with_findings'],
            'Finding records': counts['finding_records'],
            'Independent mask verification': 'Not established by this audit',
        })
        if audit['data_issues']:
            st.warning('Stored counters or diagnostic records need review.')
            st.json(audit['data_issues'])
        else:
            st.caption('No counter inconsistencies found by this audit. This does not validate geometry.')
        st.download_button('Download coverage audit (JSON)',
            json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False),
            file_name='coverage-audit.json', mime='application/json', key='coverage-audit-download')
