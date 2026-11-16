"""Local drawing and versioned manual references, separate from inspection findings."""
from __future__ import annotations

import hashlib
from pathlib import Path

from pvdelta.panel_annotation import (
    annotation_editor_html,
    load_annotation_pack,
    save_annotation_export,
    validate_annotation_export,
)


def panel_annotation_view(default_pack='', *, output, key='panel-reference-editor'):
    import streamlit as st
    import streamlit.components.v1 as components

    with st.expander('Draw full-panel references (development validation)'):
        st.caption('Draw on unmarked native thermal images. This creates manual development references, '
                   'not anomaly findings, automatic panel identification or independent validation.')
        path = st.text_input('Unmarked panel reference pack JSON', value=str(default_pack), key=f'{key}-pack')
        if not path.strip():
            st.info('Choose a pending panel reference pack to begin.')
            return
        try:
            pack = load_annotation_pack(path)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            st.error(f'Cannot load annotation pack: {exc}')
            return
        st.write(f'{len(pack["frames"])} native frames · no detector overlays')
        if st.checkbox('Open drawing editor', key=f'{key}-open'):
            components.html(annotation_editor_html(pack), height=1000, scrolling=True)
        st.download_button('Download standalone editor (HTML)', annotation_editor_html(pack),
                           file_name='native-panel-editor.html', mime='text/html', key=f'{key}-html')
        st.caption('Export the editor JSON before changing Streamlit controls. Resume saved drafts '
                   'inside the editor. Upload the exported JSON below to check and save it.')
        uploaded = st.file_uploader('Editor JSON export', type=['json'], key=f'{key}-upload')
        if uploaded is None:
            return
        try:
            content = uploaded.getvalue()
            validated = validate_annotation_export(pack, content)
        except (ValueError, TypeError, KeyError) as exc:
            st.error(f'Annotation validation failed: {exc}')
            return
        st.dataframe([{'Frame': f['case_id'],
            'Modules': len(f['panels'] if f['ready_to_score'] else f['draft_panels']),
            'Ignore_regions': len(f['ignore']), 'Completion': 'Declared complete' if f['ready_to_score'] else 'Incomplete',
            'Reviewer': f['reviewer'] or ''} for f in validated['frames']], hide_index=True, width='stretch')
        st.caption('Incomplete frames remain panels:null and cannot be scored. Completed frames are '
                   'developer drafts; completion does not verify reviewer independence or field faults.')
        if st.button('Save validated reference version', key=f'{key}-save'):
            try:
                saved, result = save_annotation_export(path, content, output)
            except (OSError, ValueError, TypeError, KeyError) as exc:
                st.error(f'Reference save failed: {exc}')
                return
            st.session_state[f'{key}-saved-version'] = {
                'path':str(saved), 'pack_token':pack['pack_token'], 'completed_frames':result['completed_frames'],
                'upload_sha256':hashlib.sha256(content).hexdigest()}
        stored = st.session_state.get(f'{key}-saved-version')
        if (stored and stored['pack_token'] == pack['pack_token']
                and stored['upload_sha256'] == hashlib.sha256(content).hexdigest()):
            saved = Path(stored['path'])
            if not saved.is_file():
                st.warning('The previously saved version is no longer available.')
                return
            st.success(f'Saved {stored["completed_frames"]}/{len(validated["frames"])} completed development frames.')
            st.code(str(saved))
            st.caption('Use this JSON path in Reference validation (optional) after preparing a native '
                       'preview of the same source. The inspection database and findings were not changed.')
            st.download_button('Download saved reference bundle', Path(saved).read_bytes(),
                file_name=Path(saved).name, mime='application/json', key=f'{key}-saved-download')
