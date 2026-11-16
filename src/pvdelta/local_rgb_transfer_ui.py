"""Separate local RGB transfer preview; no inspection database writes."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path

from pvdelta.local_rgb_transfer import fit_local_rgb_registration, transfer_local_rgb_boundaries


def _engine_sources():
    from pvdelta.geometry_revision import revision_engine_sources

    sources = revision_engine_sources()
    for name in ('local_rgb_transfer_ui.py', 'local_rgb_transfer.py', 'alignment.py',
                 'table_boundaries.py', 'table_surface.py', 'boundaries.py', 'decode.py', 'metadata.py'):
        sources[name] = hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
    return sources


def _prepare(report, thermal, rgb_path, sdk_library, content, rgb_calibration_content=None, *,
             max_check_p95_px=2., tolerance_note=None):
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw

    from pvdelta.decode import RJpeg
    from pvdelta.geometry_revision import analyze_geometry_version
    from pvdelta.metadata import parse_camera_profile
    from pvdelta.models import ThermalFrame
    from pvdelta.table_boundaries import refined_module_proposals

    path = Path(thermal).expanduser().resolve()
    with RJpeg(path, sdk_library) as source:
        values, pixels = source.temperature(), source.processed_rgb()
        parameters = asdict(source.measurement_params)
    source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    if (source_hash != report['source_sha256'] or list(values.shape) != report['native_shape']
            or hashlib.sha256(values.tobytes()).hexdigest() != report['native_temperature_sha256']
            or parameters != report['radiometric_parameters']):
        raise ValueError('Source/native values or parameters changed after the boundary preview')
    if pixels.dtype != np.uint8 or pixels.shape != (*values.shape, 3):
        raise ValueError('SDK display must match the exact native grid')
    rgb_encoded = Path(rgb_path).expanduser().read_bytes()
    rgb_hash = hashlib.sha256(rgb_encoded).hexdigest()
    controls = json.loads(content)
    if isinstance(controls, dict) and 'frames' in controls:
        matches = [r for r in controls['frames'] if isinstance(r, dict)
                   and r.get('pair', {}).get('source_sha256') == source_hash
                   and r.get('pair', {}).get('rgb_sha256') == rgb_hash]
        if len(matches) != 1:
            raise ValueError('Control bundle requires exactly one matching source pair')
        controls = matches[0]
    frame = ThermalFrame(path.stem, str(path), source_hash, values, None, values.shape[1], values.shape[0])
    calibration=None
    if rgb_calibration_content is not None:
        calibration=json.loads(rgb_calibration_content)
        if isinstance(calibration,dict) and 'frames' in calibration:
            matches=[c for c in calibration['frames'] if isinstance(c,dict) and c.get('rgb_source_sha256')==rgb_hash]
            if len(matches)!=1:
                raise ValueError('Calibration bundle requires exactly one matching original RGB source')
            calibration=matches[0]
    registration = fit_local_rgb_registration(frame, rgb_path, controls, rgb_calibration=calibration,
        max_check_p95_px=max_check_p95_px,tolerance_note=tolerance_note)
    rgb = cv2.imdecode(np.frombuffer(rgb_encoded, np.uint8), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    observed = refined_module_proposals(rgb, gimbal_pitch_deg=parse_camera_profile(path).gimbal_pitch)
    if observed['status'] != 'unreviewed_rgb_geometry':
        raise ValueError('RGB panel extractor withheld geometry: '+observed['status'])
    proposals = [{'proposal_id': f'RGB{i+1:04d}', 'polygon_rgb_xy': m['polygon_rgb_xy']}
                 for i, m in enumerate(observed['modules'])]
    transfer = transfer_local_rgb_boundaries(frame, rgb_path, registration, proposals)
    version = None
    if transfer['proposals']:
        version = analyze_geometry_version(frame, transfer, measurement_parameters=parameters,
            margin_px=math.ceil(max_check_p95_px),
            provenance={'kind': 'AI_developer_draft', 'actor': 'control-assisted local RGB preview',
                'reference_assisted': True, 'note': 'Declared control-assisted geometry; correspondence not independently verified'})
    thermal_image = Image.fromarray(pixels)
    rgb_image = Image.fromarray(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
    for image, space in ((thermal_image, 'polygon_native_xy'), (rgb_image, 'polygon_rgb_xy')):
        draw = ImageDraw.Draw(image)
        for proposal in transfer['proposals']:
            q = list(map(tuple, proposal[space]))
            draw.line(q+q[:1], fill='#77ff77', width=1 if space == 'polygon_native_xy' else 3)
            draw.text(q[0], proposal['proposal_id'], fill='#77ff77', stroke_width=1, stroke_fill='black')
    payload = {'registration': registration, 'transfer': transfer, 'native_version': version,
        'preview_engine_sources': _engine_sources(),
        'rgb_extractor_status': observed['status'], 'radiometric_parameters': parameters,
        'database_written': False, 'main_scan_updated': False, 'measurement_authorised': False}
    return payload, thermal_image, rgb_image


def local_rgb_transfer_view(report, thermal, sdk_library, *, key):
    import streamlit as st

    from pvdelta.metadata import visible_pair_candidate

    with st.expander('Local RGB → thermal boundary preview (control-assisted)'):
        st.caption('Use declared fit/check controls for this exact pair. Every transferred corner must be '
            'inside both control hulls. A numerical pass is not independent correspondence verification. '
            'Native statistics remain provisional and do not activate inspection findings.')
        guessed = visible_pair_candidate(thermal)
        rgb = st.text_input('Original paired RGB file', value=str(guessed or ''),
            key=f'{key}-{report["source_sha256"]}-rgb')
        controls_path = st.text_input('Local pair controls JSON or bundle', key=f'{key}-controls')
        calibration_path=st.text_input('RGB Brown calibration JSON (optional)',
            key=f'{key}-{report["source_sha256"]}-brown-calibration')
        tolerance=st.selectbox('Development geometry tolerance (native px)',[2.,2.20],
            format_func=lambda v:f'{v:.2f}',key=f'{key}-point-tolerance')
        tolerance_note=('User-selected2.20px development geometry tolerance; physical correspondence '
                        'and independent/IEC acceptance remain unverified.') if tolerance>2. else None
        if tolerance_note:
            st.caption('2.20px applies only to this preview. The saved model records this choice; '
                       'native edge margin is3px. Inspection findings are unchanged.')
        result_key = f'{key}-result'
        if not rgb.strip() or not controls_path.strip():
            st.session_state.pop(result_key, None)
            return
        try:
            content = Path(controls_path).expanduser().read_bytes()
            calibration_content=Path(calibration_path).expanduser().read_bytes() if calibration_path.strip() else None
            token = hashlib.sha256(json.dumps({'report':report, 'engine_sources':_engine_sources(),
                'sdk_library':str(sdk_library),'max_check_p95_px':tolerance,
                'tolerance_note':tolerance_note}, sort_keys=True).encode()+content
                +(calibration_content or b'')+Path(rgb).expanduser().read_bytes()+Path(thermal).expanduser().read_bytes()
                ).hexdigest()
        except (OSError, ValueError, TypeError) as exc:
            st.session_state.pop(result_key, None)
            st.error(f'Local pair input unavailable: {exc}')
            return
        old = st.session_state.get(result_key)
        if old and old[0] != token:
            st.session_state.pop(result_key, None)
        if st.button('Preview local RGB boundaries', key=f'{key}-preview'):
            st.session_state.pop(result_key, None)
            try:
                with st.spinner('Reading original SDK data, RGB panels and local control checks…'):
                    result = _prepare(report, thermal, rgb, sdk_library, content, calibration_content,
                        max_check_p95_px=tolerance,tolerance_note=tolerance_note)
                st.session_state[result_key] = (token, result)
            except (OSError, ValueError, TypeError, RuntimeError, KeyError, ImportError) as exc:
                st.error(f'Local RGB transfer failed: {exc}')
        saved = st.session_state.get(result_key)
        if not saved:
            return
        payload, thermal_image, rgb_image = saved[1]
        registration, transfer = payload['registration'], payload['transfer']
        validation = registration['validation']
        st.write(f"Check P95: {validation['check_p95_source_px']:.3f} native px · "
            f"{transfer['supported_proposals']} / {transfer['requested_proposals']} RGB regions inside local support")
        st.caption('RGB coordinate model: '+registration.get('rgb_geometry',{}).get('model','original_rgb_pinhole')+
            '. Lens coefficients, when supplied, are fixed source-bound inputs; check points never refit them.')
        if not registration['geometry_gate_passed']:
            st.warning('Local point gate failed: '+'; '.join(validation['reasons']))
        else:
            st.info('Local point gate passed. Source controls and panel semantics remain unverified development evidence.')
        for col, image, caption in zip(st.columns(2), (thermal_image, rgb_image),
                ('Native SDK grid · supported local RGB regions', 'Original RGB grid · same region IDs')):
            col.image(image, caption=caption, width='stretch')
        st.dataframe([{'Region': r['proposal_id'], 'Decision': r['status']} for r in transfer['decisions']],
            hide_index=True, width='stretch')
        version = payload['native_version']
        if version:
            st.dataframe([{'Region': r['module']['module_id'], 'Peak °C': r['module']['temp_max'],
                'Candidate regions': len(r['findings']), 'Assessment': r['assessment_status']}
                for r in version['module_reports']], hide_index=True, width='stretch')
        st.caption('Provisional native measurements use recorded SDK parameters, 2px ROI margin and default '
            'exploratory settings (ΔT 2.5 K). No candidate is not a healthy-panel confirmation. '
            'No panel GPS, physical inventory, glare clearance, fault or IEC acceptance is inferred.')
        with st.expander('Recorded radiometric parameters / control provenance'):
            st.json({'parameters': payload['radiometric_parameters'], 'provenance': registration['provenance']})
        with st.expander('Control residuals (no refit)'):
            st.dataframe(registration.get('control_residuals',[]),hide_index=True,width='stretch')
            st.caption('Observed versus predicted native positions from the fixed fit-only model. '
                'Residuals are diagnostics; points are not moved, dropped or relabelled to pass the gate.')
        st.download_button('Download local RGB transfer (JSON)', json.dumps(payload, indent=2, allow_nan=False),
            file_name='local-rgb-boundary-preview.json', mime='application/json', key=f'{key}-download')
