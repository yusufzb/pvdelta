"""Thermal reference review, isolated from operational findings."""
from __future__ import annotations

import json
from pathlib import Path

from pvdelta.region_annotation import (
    load_region_pack,
    region_editor_html,
    save_region_export,
    validate_region_export,
)


def region_annotation_view(default_pack='', *, output, key='thermal-reference'):
    import streamlit as st
    import streamlit.components.v1 as components

    with st.expander('Thermal reference review · mark missed and uncertain regions'):
        st.caption('Review the entire unmarked frame, including areas outside detected panels. '
                   'Paired RGB is unregistered context. This does not change inspection findings.')
        path = st.text_input('Thermal reference pack JSON', str(default_pack), key=f'{key}-pack')
        if not path.strip():
            st.info('Choose a prepared native reference pack.')
            return
        if not st.checkbox('Load thermal reference pack', key=f'{key}-load'):
            return
        try:
            pack = load_region_pack(path)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            st.error(f'Cannot load thermal reference pack: {exc}')
            return
        st.write(f'{len(pack["frames"])} full native frames · predictions hidden')
        if st.checkbox('Open thermal reference editor', key=f'{key}-open'):
            components.html(region_editor_html(pack), height=1050, scrolling=True)
        st.download_button('Download thermal reference editor (HTML)', region_editor_html(pack),
                           file_name='thermal-reference-review.html', mime='text/html',
                           key=f'{key}-html')
        st.caption('Export JSON before leaving the editor. Resume that export inside the editor; '
                   'upload it below to validate and save. An empty frame needs a reviewer and reason.')
        upload = st.file_uploader('Thermal reference JSON export', type=['json'], key=f'{key}-upload')
        if upload is None:
            return
        try:
            content = upload.getvalue()
            result = validate_region_export(pack, content)
        except (ValueError, TypeError, KeyError) as exc:
            st.error(f'Thermal reference validation failed: {exc}')
            return
        st.dataframe([{'Frame': f['case_id'],
                       'Regions': len(f['regions'] if f['review_complete'] else f['draft_regions']),
                       'Uncertain': len(f['ignore']),
                       'Status': 'Declared reviewed' if f['review_complete'] else 'Incomplete',
                       'Reviewer': f['reviewer'] or ''} for f in result['frames']],
                     hide_index=True, width='stretch')
        st.caption('Review completion does not certify independence, fault cause or glare rejection. '
                   'Draw a polygon around each region; inward corners are supported. '
                   'A frozen baseline comparison is available only with the matching protocol file.')
        protocol_path = Path(path).expanduser().resolve().with_name('evaluation-protocol.json')
        polygon_protocol = protocol_path.with_name('evaluation-protocol-v2.json')
        if polygon_protocol.is_file():
            protocol_path = polygon_protocol
        if protocol_path.is_file() and st.checkbox('Compare with frozen baseline', key=f'{key}-score'):
            from pvdelta.region_reference_benchmark import evaluate_region_work

            try:
                scores = evaluate_region_work(pack, content, json.loads(protocol_path.read_bytes()))
            except (OSError, ValueError, TypeError, KeyError) as exc:
                st.error(f'Frozen comparison unavailable: {exc}')
            else:
                st.write(f'{scores["scored_frames"]} scored frames · {scores["pending_frames"]} pending')
                if scores['totals'] is None:
                    st.info('Awaiting full-frame reference review. No accuracy score is available.')
                else:
                    st.dataframe([{'Protocol': name, **values}
                                  for name, values in scores['totals'].items()],
                                 hide_index=True, width='stretch')
                    st.caption(f'Geometry: {scores["geometry"]}. Primary IoU ≥ 0.5; auxiliary IoU ≥ 0.1. '
                               'These are development geometry scores for the frozen baseline, '
                               'not electrical fault accuracy or IEC acceptance.')
                st.download_button('Download baseline comparison and missed-region IDs',
                                   json.dumps(scores, indent=2), file_name='region-baseline-comparison.json',
                                   mime='application/json', key=f'{key}-scores-json')
        if st.button('Save thermal reference version', key=f'{key}-save'):
            try:
                saved, result = save_region_export(path, content, output)
            except (OSError, ValueError, TypeError, KeyError) as exc:
                st.error(f'Thermal reference save failed: {exc}')
                return
            st.success(f'Saved {result["completed_frames"]}/{len(result["frames"])} declared reviews.')
            st.code(str(saved))
            st.download_button('Download validated thermal references', saved.read_bytes(),
                               file_name=saved.name, mime='application/json', key=f'{key}-saved')
