"""Streamlit triage of frozen, unassigned native thermal evidence."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np

from pvdelta.gap_review_bundle import (
    evidence_snapshot,
    gap_decisions,
    load_gap_bundle,
    save_gap_decision,
)
from pvdelta.review_ui import finding_navigation


def evidence_images(frame, item):
    """Render native values; white ring leaves measured pixels visible, red is 5x context."""
    from PIL import Image, ImageDraw

    image = native_context_image(frame['native'])
    rgb = np.asarray(image).copy()
    values = frame['native']
    region = np.zeros(values.shape, dtype=bool)
    pixels = np.asarray(item['native_finding']['region_pixels_yx'])
    region[pixels[:, 0], pixels[:, 1]] = True
    padded = np.pad(region, 1)
    ring = (padded[:-2, 1:-1] | padded[2:, 1:-1] | padded[1:-1, :-2]
            | padded[1:-1, 2:]) & ~region
    rgb[ring] = 255
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    quad = list(map(tuple, item['provisional_polygon_native_xy']))
    draw.line(quad + quad[:1], fill='#ffb347', width=1)
    y0, x0 = pixels.min(axis=0) - .5
    y1, x1 = pixels.max(axis=0) + .5
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    draw.rectangle((cx - 2.5*(x1-x0), cy - 2.5*(y1-y0),
                    cx + 2.5*(x1-x0), cy + 2.5*(y1-y0)), outline='#ff2020', width=1)
    width, height = min(500, image.width), min(500, image.height)
    left = int(np.clip(cx - width/2, 0, image.width - width))
    top = int(np.clip(cy - height/2, 0, image.height - height))
    return image, image.crop((left, top, left + width, top + height))


def native_context_image(values):
    """Display native data without annotations; no registration or measurement."""
    from PIL import Image

    low, high = np.percentile(values, [2, 98])
    scaled = np.clip((values - low) / max(float(high - low), 1e-9), 0, 1)
    stops = np.array([[18, 22, 77], [17, 115, 156], [58, 183, 118],
                      [239, 224, 69], [222, 65, 35]], dtype=float)
    rgb = np.stack([np.interp(scaled, np.linspace(0, 1, len(stops)), stops[:, c])
                    for c in range(3)], axis=-1).astype(np.uint8)
    return Image.fromarray(rgb)


def _context_gallery(folder, frame, item, *, key):
    import streamlit as st

    from pvdelta.gap_context import load_gap_context

    with st.expander('Nearby thermal frames — unverified comparison', expanded=False):
        try:
            context = load_gap_context(folder, frame['record_id'])
        except (OSError, ValueError, TypeError, KeyError) as exc:
            st.error(f'Comparison frames withheld: {exc}')
            return
        if context is None:
            st.info('No comparison bundle for this source. Same-panel identity and glare remain unresolved.')
            return
        views = {v['view_id']: v for v in context['views']}
        st.warning('Nearby filenames are not verified same-panel views or distinct viewing angles. '
                   'No anomaly box or temperature is transferred to these frames.')
        selected = finding_navigation(list(views), key=f'{key}-{context["context_id"]}-view',
            label='Comparison frame', item_label='Comparison frame',
            names={i: Path(v['source_path']).name for i, v in views.items()})
        view = views[selected]
        check = view['panel_check']
        st.write('**Same-panel match: withheld** · ' + ', '.join(check['reasons']))
        a, b = st.columns(2)
        with a:
            full, _ = evidence_images(frame, item)
            st.image(full, caption=f"Source · {Path(frame['source_path']).name}", width='stretch')
        with b:
            st.image(native_context_image(view['native']),
                caption=f"Unmarked comparison · {Path(view['source_path']).name}", width='stretch')
        st.caption(f"Comparison basis: {view['basis']}. Frame-local colour scales differ; "
                   'colours cannot establish temperature persistence.')
        with st.expander('Observed-panel control audit'):
            st.json({'matched_panel_controls': len(check['matches']),
                     'heldout_p95_px': check['heldout_p95_px'], 'config': check['config'],
                     'source_sha256': view['source_sha256'],
                     'native_temperature_sha256': view['native_temperature_sha256'],
                     'parameters': view['parameters'], 'same_panel_verified': False,
                     'glare_clearance': False})


def _edge_audit_view(frame, item, *, key):
    import streamlit as st
    from PIL import ImageDraw

    from pvdelta.native_edge_audit import audit_native_edges, edge_audit_identity

    with st.expander('Panel boundary support — diagnostic', expanded=False):
        st.caption('Inspect the existing boundary on the verified native grid. This is the ridge '
                   'channel, not the full hybrid detector. Missing ridge evidence is unresolved; '
                   'it does not remove the hot evidence or establish a healthy panel.')
        state_key = f'{key}-{frame["record_id"]}-{item["evidence_id"]}-edge-audit'
        quad = item['provisional_polygon_native_xy']
        arguments = {'source_sha256': frame['report']['source_hash'],
                     'native_temperature_sha256': frame['report']['native_temperature_sha256']}
        try:
            token, _ = edge_audit_identity(frame['native'], quad, **arguments)
            previous = st.session_state.get(state_key)
            if previous and previous.get('audit_token') != token:
                st.session_state.pop(state_key, None)
            if st.button('Inspect native boundary evidence', key=f'{key}-inspect-edges'):
                st.session_state.pop(state_key, None)
                st.session_state[state_key] = audit_native_edges(frame['native'], quad, **arguments)
        except (OSError, ValueError, TypeError, RuntimeError) as exc:
            st.session_state.pop(state_key, None)
            st.error(f'Boundary diagnostic withheld: {exc}')
            return
        report = st.session_state.get(state_key)
        if report is None:
            return
        historical = item.get('geometry_evidence', {}).get('support', [])
        rows = []
        image = native_context_image(frame['native'])
        draw = ImageDraw.Draw(image)
        for index, side in enumerate(report['sides']):
            a, b = np.asarray(side['start_native_xy']), np.asarray(side['end_native_xy'])
            for position, supported in zip(np.linspace(0, 1, side['sample_count']), side['supported']):
                point = a + position * (b-a)
                draw.point(tuple(np.rint(point).astype(int)), fill='#00ffff' if supported else '#ff00ff')
            mid = (a+b)/2
            draw.text(tuple(mid), side['side'], fill='white', stroke_width=1, stroke_fill='black')
            rows.append({'Edge': side['side'], 'Native ridge coverage': round(side['coverage'], 3),
                'Largest native gap': round(side['largest_gap_fraction'], 3),
                'Native ridge check': 'Supported' if side['native_ridge_supported'] else 'Unresolved',
                'Historical hybrid coverage': round(historical[index][1], 3) if len(historical) == 4 else None})
        st.dataframe(rows, hide_index=True, width='stretch')
        st.caption('Project diagnostic limits: coverage ≥0.80 and gap ≤0.18; not IEC acceptance. '
                   'Cyan: observed ridge samples. Magenta: unresolved samples. No edge is filled or moved. '
                   'Historical hybrid coverage also includes observed line segments and can differ.')
        st.image(image, caption='Native boundary evidence only — no new anomaly overlay', width='stretch')
        st.download_button('Download boundary diagnostic', json.dumps(report, indent=2),
            file_name=f'edge-audit-{item["evidence_id"]}.json', mime='application/json',
            key=f'{key}-edge-audit-download')


def native_gap_review(directory, *, key='native-gap'):
    import streamlit as st

    st.subheader('Unassigned thermal evidence')
    st.warning('Unresolved panel boundaries: possible panel / ground / reflection. '
               'These records are excluded from module, fault and healthy totals.')
    with st.expander('Review source'):
        directory_key = f'{key}-directory'
        if directory_key not in st.session_state:
            st.session_state[directory_key] = str(directory)
        text = st.text_input('Local evidence bundle', key=directory_key)
    folder = Path(text).expanduser()
    if not (folder / 'manifest.json').is_file():
        st.info('No local evidence bundle loaded. Export a native-gap review bundle first. '
                'The regular inspection database is unaffected.')
        return
    try:
        frames = load_gap_bundle(folder)
        decisions = gap_decisions(folder)
    except (OSError, ValueError, TypeError, KeyError, sqlite3.Error) as exc:
        st.error(f'Evidence unavailable; overlays withheld: {exc}')
        return
    rows = {}
    for frame in frames:
        for item in frame['report']['evidence']:
            identity, _ = evidence_snapshot(frame, item)
            rows[identity] = (frame, item)
    columns = st.columns(3)
    columns[0].metric('Frames in this review bundle', len(frames))
    columns[1].metric('Unassigned evidence records', len(rows))
    columns[2].metric('Not yet reviewed', sum(i not in decisions for i in rows))
    st.caption('Evidence records can refer to the same panel. Zero records does not mean healthy '
               'or complete coverage. This bundle is a frozen experiment, not a new flight scan.')
    choice = st.selectbox('Evidence review filter', ['All', 'Not reviewed', 'OK', 'Not OK', 'Wait'],
                          key=f'{key}-filter')
    states = {'OK': 'ok', 'Not OK': 'not_ok', 'Wait': 'wait'}
    ids = [i for i in rows if choice == 'All' or (choice == 'Not reviewed' and i not in decisions)
           or decisions.get(i, {}).get('status') == states.get(choice, '__none__')]
    if not ids:
        st.info('No unassigned evidence matches this review filter.')
        return
    names = {i: f"{Path(f['source_path']).name} · {e['evidence_id']}" for i, (f, e) in rows.items()}
    identity = finding_navigation(ids, key=f'{key}-evidence', names=names,
                                  label='Evidence to review', item_label='Evidence')
    frame, item = rows[identity]
    finding = item['native_finding']
    st.write(f"**Frame:** {Path(frame['source_path']).name} · **Evidence:** {item['evidence_id']}")
    st.caption(f"Measurement basis: {frame['basis']}")
    st.caption('Panel ID / panel GPS: unavailable. RGB overlay and same-panel multi-angle '
               'confirmation: unavailable. No glare clearance or IEC conformity is claimed.')
    cols = st.columns(3)
    cols[0].metric('Native hot pixels', finding['n_region_px'])
    cols[1].metric('Local residual ΔT', f"{finding['delta_t_c']:.3f} K")
    cols[2].metric('Native region peak', f"{finding['region_max_t_c']:.2f} °C")
    full, crop = evidence_images(frame, item)
    st.image(crop, caption='Native thermal evidence crop · up to 500 × 500 native pixels',
             width='stretch')
    st.caption('White: exterior of exact hot pixels. Amber: unresolved proposed boundary. '
               'Red: 5× visual outline, not measured anomaly extent. Colour scale: frame 2–98 percentiles.')
    with st.expander('Full native thermal frame'):
        st.image(full, width='stretch')
    _edge_audit_view(frame, item, key=key)
    _context_gallery(folder, frame, item, key=key)
    current = decisions.get(identity, {})
    with st.expander('Evidence user review', expanded=True):
        st.caption('OK = retain useful thermal evidence for follow-up; Not OK = reject this '
                   'evidence; Wait = unresolved. No choice confirms a panel or electrical fault.')
        st.write('Saved decision: ' + current.get('status', 'Not reviewed'))
        if current:
            st.caption(f"{current['reviewer'] or 'Unnamed reviewer'} · {current['reviewed_at_utc']}")
        a, b = st.columns([1, 3])
        reviewer = a.text_input('Reviewer', current.get('reviewer', ''), key=f'{key}-{identity}-reviewer')
        note = b.text_input('Evidence review note', current.get('note', ''), key=f'{key}-{identity}-note')
        for col, (label, status) in zip(st.columns(3), states.items()):
            if col.button(label, key=f'{key}-decision-{status}', use_container_width=True):
                try:
                    save_gap_decision(folder, frame, item, status, reviewer=reviewer, note=note)
                except (OSError, ValueError, TypeError, KeyError, sqlite3.Error) as exc:
                    st.error(f'Review was not saved: {exc}')
                else:
                    st.rerun()
    with st.expander('Source, native pixels and radiometric parameters'):
        st.json({'source_path': frame['source_path'], 'source_sha256': frame['report']['source_hash'],
                 'native_shape': list(frame['native'].shape),
                 'native_temperature_sha256': frame['report']['native_temperature_sha256'],
                 'snapshot_id': frame['record_id'], 'parameters': frame['parameters'],
                 'frozen_engine_source_sha256': frame['engine_source_sha256'],
                 'budget_counters': frame['report']['counters'],
                 'analyses_withheld': frame['report']['hypotheses_withheld_by_analysis_budget']})
        st.download_button('Download this evidence record', json.dumps(item, indent=2),
                           file_name=f"evidence-{item['evidence_id']}.json", mime='application/json',
                           key=f'{key}-download')
