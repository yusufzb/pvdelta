"""Opt-in paired-photo review without silently merging panel identities."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import duckdb
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw

from pvdelta.panel_pair_review import load_pair_proposals, pair_decisions, save_pair_decision
from pvdelta.review_ui import finding_navigation


def _image(group, sensor):
    row = group['regions'][0]
    data = (Path(row['source_root'])/row[sensor+'_path']).read_bytes()
    if hashlib.sha256(data).hexdigest() != row[sensor+'_hash']:
        raise ValueError('Photo differs from indexed original')
    with Image.open(io.BytesIO(data)) as source:
        picture = source.convert('RGB')
    if sensor == 'thermal':
        scale = np.array(picture.size)/[row['width_px'], row['height_px']]
        draw = ImageDraw.Draw(picture)
        q = [tuple(p) for p in np.array(json.loads(row['module_polygon']))*scale]
        draw.line(q+[q[0]], fill='#00ffff', width=2)
        for region in group['regions']:
            q = [tuple(p) for p in np.array(json.loads(region['pixel_polygon_json']))*scale]
            draw.line(q+[q[0]], fill='#ff3030', width=2)
    return picture


def panel_pair_review(database, groups, *, key):
    if not Path(database).with_suffix('.panel-pairs.json').is_file():
        return
    with st.expander('Review same-panel photo proposals'):
        if not st.checkbox('Open paired-photo review', key=key+'-pairs-open'):
            return
        try:
            package = load_pair_proposals(database, groups)
            if package is None:
                st.info('No prepared pair proposals for this inspection run.')
                return
            pairs = package['pairs']
            decisions = pair_decisions(database, pairs)
            metrics = st.columns(3)
            for column, status in zip(metrics, ('same_panel', 'different_panels', 'wait')):
                column.metric(status.replace('_', ' ').title(), sum(d['status'] == status for d in decisions.values()))
            st.caption(f'{len(pairs)} pair proposals; {len(decisions)} reviewed. '
                       f"{package['stale']} stale proposals withheld. Counts are not unique panels or faults.")
            st.caption('Camera GPS retrieves nearby frames. Cyan: saved panel proposal; red: existing regions. '
                       'RGB is original context. Check the panel and its neighbours before deciding. '
                       'Decisions are stored separately; identities and anomaly verdicts are unchanged.')
            queue = st.selectbox('Pair review queue', ('All pairs', 'Unreviewed', 'Wait'),
                                 key=key+'-pair-queue')
            rgb_only = st.checkbox('Show only pairs with RGB overlap or repeat support',
                                   key=key+'-pair-rgb-only')
            rgb_priority = {'provisional_repeat_support': 0, 'overlap_only_review_required': 1}
            visible = [p for p in pairs
                       if (queue == 'All pairs'
                           or (queue == 'Unreviewed' and p['pair_id'] not in decisions)
                           or (queue == 'Wait' and decisions.get(p['pair_id'], {}).get('status') == 'wait'))
                       and (not rgb_only or p['rgb_status'] in rgb_priority)]
            visible.sort(key=lambda p: (p['original_rgb_texture'] is None,
                                        rgb_priority.get(p['rgb_status'], 2),
                                        -min(p['iou'], p['reverse_iou']), p['pair_id']))
            st.caption(f'{len(visible)} pairs in this queue. RGB evidence and bidirectional overlap set the order.')
            if not visible:
                st.session_state.pop(key+'-pair-id', None)
                st.info('No panel pairs match the current filters.')
                return
            names = {p['pair_id']: p['first']['thermal_path']+' / '+p['first']['module_id']+' ↔ '
                     +p['second']['thermal_path']+' / '+p['second']['module_id'] for p in visible}
            selected = finding_navigation(list(names), key=key+'-pair-id',
                label='Panel photo pair', names=names, item_label='Pair')
            if selected is None:
                return
            proposal = next(p for p in visible if p['pair_id'] == selected)
            st.caption(f"Camera separation {proposal['camera_distance_m']:.1f} m · "
                       f"panel overlap {proposal['iou']:.2f}/{proposal['reverse_iou']:.2f} · "
                       'physical panel identity unverified')
            st.caption('RGB follow-up: '+proposal['rgb_status']+'. This is image evidence, not an identity verdict.')
            texture = proposal['original_rgb_texture']
            if texture:
                st.caption(f"Original RGB texture: {texture['fit_points']} fit points and "
                           f"{texture['check_points']} check points. "
                           f"{texture['neighbour_controls']} neighbouring panel controls were rejected. "
                           f"Estimated viewing ray difference: {texture['nominal_ray_span_deg']:.1f}°.")
            for column, group in zip(st.columns(2), (proposal['first'], proposal['second'])):
                with column:
                    st.image(_image(group, 'thermal'), caption=group['thermal_path'], width='stretch')
                    st.image(_image(group, 'rgb'), caption=group['regions'][0]['rgb_path']+' · original RGB',
                             width='stretch')
            st.caption('JPEG previews support geometry review only. Existing native SDK measurements remain unchanged.')
            if st.button('Recheck both thermal measurements', key=key+'-pair-replay'):
                from pvdelta.thermal_view_scan import ERRORS, replay_review_pair

                try:
                    evidence = replay_review_pair(database, proposal)
                except ERRORS as exc:
                    st.warning(f'Thermal measurements could not be reproduced: {exc}')
                else:
                    st.dataframe([
                        {'Thermal file': item['source_name'], 'Module': item['module_id'],
                         'Measured pixels': item['measured_pixels'], 'Region median (°C)': region['median_c'],
                         'Local ΔT (K)': region['delta_t_k'],
                         'Distance (m)': item['profile']['effective_parameters']['distance_m']}
                        for item in evidence for region in item['regions']
                    ], hide_index=True, width='stretch')
                    st.caption('Both sources reproduced their saved measurements. '
                               'Panel identity, viewing angle and reflection require separate evidence.')
            previous = decisions.get(selected, {})
            if previous:
                st.info('Saved review: '+previous['status']+' · '+previous['reviewer'])
            with st.form(key+'-pair-form-'+selected):
                reviewer = st.text_input('Reviewer', key=key+'-pair-reviewer')
                note = st.text_input('Pair review note', value=previous.get('note', ''),
                                     key=key+'-pair-note-'+selected)
                columns = st.columns(3)
                same = columns[0].form_submit_button('Same panel')
                different = columns[1].form_submit_button('Different panels')
                wait = columns[2].form_submit_button('Wait')
            status = 'same_panel' if same else 'different_panels' if different else 'wait' if wait else None
            if status:
                save_pair_decision(database, proposal, status, reviewer, note)
                st.rerun()
        except (OSError, ValueError, TypeError, KeyError, duckdb.Error) as exc:
            st.error(f'Paired-photo review unavailable: {exc}')
