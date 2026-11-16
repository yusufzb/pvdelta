"""Explicit local geometry edits with separate, versioned native reanalysis."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from pvdelta.geometry_revision import (
    analyze_geometry_version,
    apply_geometry_patch,
    geometry_version_impact,
    revision_engine_sources,
    save_geometry_version,
)


def _decode_revision(report, path, sdk_library, *, parent=None, patch=None, actor='local edit'):
    import numpy as np
    from PIL import Image

    from pvdelta.decode import RJpeg
    from pvdelta.models import ThermalFrame

    source = Path(path).expanduser().resolve()
    if hashlib.sha256(source.read_bytes()).hexdigest() != report['source_sha256']:
        raise ValueError('Source changed after boundary preview')
    with RJpeg(source, sdk_library) as image:
        values = image.temperature()
        pixels = image.processed_rgb()
        parameters = asdict(image.measurement_params)
    if (hashlib.sha256(values.tobytes()).hexdigest() != report['native_temperature_sha256']
            or list(values.shape) != report['native_shape'] or parameters != report['radiometric_parameters']):
        raise ValueError('Native radiometry or parameters changed; prepare a new boundary preview')
    if pixels.dtype != np.uint8 or pixels.shape != (*values.shape, 3):
        raise ValueError('SDK display must match the exact native grid')
    geometry = apply_geometry_patch(parent, patch) if parent else report['proposed']
    frame = ThermalFrame(source.stem, str(source), report['source_sha256'], values, None,
                         values.shape[1], values.shape[0])
    version = analyze_geometry_version(frame, geometry, measurement_parameters=parameters,
        provenance={'kind': 'user_edit' if parent else 'automatic', 'actor': actor,
            'note': 'Local geometry revision preview; no independent review or scan activation'},
        parent=parent, patch=patch)
    return version, Image.fromarray(pixels)


def _marked(version, picture, colour):
    from PIL import ImageDraw

    image = picture.copy()
    draw = ImageDraw.Draw(image)
    for p in version['geometry']['proposals']:
        q = list(map(tuple, p['polygon_native_xy']))
        draw.line(q+q[:1], fill=colour, width=1)
        draw.text(q[0], p['proposal_id'].encode('ascii', 'replace').decode(),
                  fill=colour, stroke_width=1, stroke_fill='#000000')
    for report in version['module_reports']:
        for finding in report['findings']:
            q = list(map(tuple, finding['pixel_polygon']))
            draw.line(q+q[:1], fill='#ff9b31', width=1)
    return image


def geometry_revision_view(report, path, sdk_library, *, output, key='geometry-revision'):
    import streamlit as st

    with st.expander('Revise boundaries and reanalyze (separate versions)'):
        st.caption('Add, replace or remove native panel regions. The entire frame is reanalyzed because '
                   'neighbouring measurement masks can change. Saved inspection findings stay unchanged. '
                   'An edit does not establish panel identity, a healthy panel, fault or independent review.')
        token = hashlib.sha256(json.dumps({
            'source': report['source_sha256'], 'native': report['native_temperature_sha256'],
            'parameters': report['radiometric_parameters'], 'engine': revision_engine_sources()},
            sort_keys=True).encode()).hexdigest()
        session_key = f'{key}-{token}-session'
        session = st.session_state.get(session_key)
        if session is None and st.button('Start geometry revision session', key=f'{key}-{token}-start'):
            try:
                with st.spinner('Reading original SDK pixels and saving a baseline in this session…'):
                    root, picture = _decode_revision(report, path, sdk_library, actor='baseline boundary extractor')
                session = {'root': root['version_id'], 'active': root['version_id'],
                    'versions': {root['version_id']: root}, 'picture': picture}
                st.session_state[session_key] = session
            except (OSError, ValueError, TypeError, RuntimeError, ImportError) as exc:
                st.error(f'Cannot prepare geometry revision: {exc}')
                return
        if session is None:
            st.caption('Start a session after comparing a source frame. Nothing is changed automatically.')
            return
        current = session['versions'][session['active']]
        root = session['versions'][session['root']]
        left, right = st.columns(2)
        left.image(_marked(root, session['picture'], '#69cfff'), caption='Source version · cyan boundaries',
                   width='stretch')
        right.image(_marked(current, session['picture'], '#77ff77'), caption='Active revision · green boundaries; orange hot-region candidates',
                    width='stretch')
        height, width = current['source']['native_shape']
        st.caption(f'Native coordinates: {width}×{height}; x increases right, y down. '
                   'Preview uses the recorded SDK parameters, 2px ROI margin and default exploratory Stage 1 settings '
                   '(ΔT 2.5 K). Environmental normalisation and explicit whole-module peers may be unavailable.')
        active_key = f'{key}-{token}-{current["version_id"]}'
        ids = [p['proposal_id'] for p in current['geometry']['proposals']]
        operations = ['Replace boundary', 'Remove region', 'Add region'] if ids else ['Add region']
        operation = st.selectbox('Geometry operation', operations, key=f'{active_key}-operation')
        if operation == 'Add region':
            identifier = st.text_input('New frame-local region ID', value='R0001', key=f'{active_key}-new-id')
            x1, y1 = min(20, width*.2), min(20, height*.2)
            x2, y2 = min(width-3, 60), min(height-3, 100)
            original = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        else:
            identifier = st.selectbox('Region ID', ids, key=f'{active_key}-region')
            original = next(p['polygon_native_xy'] for p in current['geometry']['proposals'] if p['proposal_id'] == identifier)
        with st.form(f'{active_key}-{operation}-{identifier}-edit'):
            corners = None
            if operation != 'Remove region':
                corners = st.text_area('Four native corners [[x,y], ...]', value=json.dumps(original),
                    key=f'{active_key}-{operation}-{identifier}-corners')
            reason = st.text_input('Reason for this geometry change', key=f'{active_key}-{operation}-{identifier}-reason')
            actor = st.text_input('Declared editor (optional)', key=f'{active_key}-{operation}-{identifier}-actor')
            submitted = st.form_submit_button('Preview revised analysis')
        if submitted:
            try:
                op = {'op': {'Replace boundary': 'replace', 'Remove region': 'remove', 'Add region': 'add'}[operation],
                      'proposal_id': identifier, 'reason': reason}
                if corners is not None:
                    op['polygon_native_xy'] = json.loads(corners)
                patch = {'schema': 'pvdelta.native-geometry-patch/v1', 'parent_version_id': current['version_id'],
                    'source_sha256': current['source']['source_sha256'],
                    'native_temperature_sha256': current['source']['native_temperature_sha256'], 'operations': [op]}
                with st.spinner('Reanalyzing all ROIs from original SDK pixels…'):
                    revised, picture = _decode_revision(report, path, sdk_library, parent=current, patch=patch,
                                                        actor=actor.strip() or 'unnamed local edit')
                session['versions'][revised['version_id']] = revised
                session['active'] = revised['version_id']
                session['picture'] = picture
                st.rerun()
            except (OSError, ValueError, TypeError, RuntimeError, ImportError) as exc:
                st.error(f'Revision was not applied: {exc}')
        if current['parent_version_id']:
            parent = session['versions'][current['parent_version_id']]
            impact = geometry_version_impact(parent, current)
            st.json(impact, expanded=False)
            if st.button('Return to parent (keep this version in history)', key=f'{active_key}-parent'):
                session['active'] = parent['version_id']
                st.rerun()
        st.dataframe([{'Region': r['module']['module_id'], 'Assessment': r['assessment_status'],
            'Peak °C': r['module']['temp_max'], 'Hot/cold regions': len(r['findings']),
            'Limitations': '; '.join(r['reasons'])} for r in current['module_reports']],
            hide_index=True, width='stretch')
        st.caption('Zero candidates is not a healthy-panel confirmation. Physical panel GPS and RGB overlays '
                   'are not inherited from a replaced polygon. Existing database results are not activated or replaced.')
        from pvdelta.geometry_revision_validation_ui import revision_reference_validation

        revision_reference_validation(root, current, session['picture'], key=f'{key}-{token}-validation')
        st.download_button('Download active geometry version (JSON)', json.dumps(current, indent=2, allow_nan=False),
            file_name=f'geometry-version-{current["version_id"][7:]}.json', mime='application/json', key=f'{active_key}-download')
        if st.button('Save revision history (local JSON)', key=f'{active_key}-save'):
            try:
                paths = [save_geometry_version(v, output) for v in session['versions'].values()]
            except (OSError, ValueError, TypeError) as exc:
                st.error(f'Versions could not be saved: {exc}')
            else:
                st.success(f'{len(paths)} immutable versions saved. Main inspection database unchanged.')
                st.code(str(Path(output).expanduser().resolve()))
        with st.expander('Version history (includes returned and branched edits)'):
            st.dataframe([{'Version': v['version_id'], 'Parent': v['parent_version_id'],
                'Regions': len(v['geometry']['proposals']), 'Active': v['version_id'] == session['active'],
                'Provenance': v['provenance']['kind']} for v in session['versions'].values()],
                hide_index=True, width='stretch')
