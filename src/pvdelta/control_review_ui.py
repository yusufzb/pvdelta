"""Original-source point review; no fitted or predicted point is displayed."""
from __future__ import annotations

from pathlib import Path

from pvdelta.control_review import (
    digest,
    export_reviewed_controls,
    latest_control_decisions,
    load_review_pack,
    review_points,
    save_control_decision,
)


def _crop(path, xy, radius):
    from PIL import Image, ImageDraw, ImageOps

    with Image.open(path) as source:
        image = source.convert('RGB')  # Original encoded pixel orientation, no EXIF rotation.
    x, y = xy
    box = (max(0, int(x-radius)), max(0, int(y-radius)),
           min(image.width, int(x+radius+1)), min(image.height, int(y+radius+1)))
    image = image.crop(box)
    draw = ImageDraw.Draw(image)
    px, py = x-box[0], y-box[1]
    draw.line((px-3, py, px+3, py), fill='magenta', width=1)
    draw.line((px, py-3, px, py+3), fill='magenta', width=1)
    # Edge-clipped crops must not stretch one axis and distort the frame geometry.
    return ImageOps.pad(image, (410, 410), method=Image.Resampling.NEAREST,
                        color='#202020'), box


def _overview(path, xy):
    """Display-only source context; never draw a fitted correspondence."""
    from PIL import Image, ImageDraw

    with Image.open(path) as source:
        image = source.convert('RGB')
    original_size = image.size
    image.thumbnail((1000, 750), Image.Resampling.LANCZOS)
    x, y = xy[0]*image.width/original_size[0], xy[1]*image.height/original_size[1]
    draw = ImageDraw.Draw(image)
    draw.ellipse((x-9, y-9, x+9, y+9), outline='magenta', width=2)
    draw.line((x-13, y, x+13, y), fill='magenta', width=1)
    draw.line((x, y-13, x, y+13), fill='magenta', width=1)
    return image


def _export_controls(pack, decisions, output, token, complete, total, *, frame_index=None):
    import streamlit as st

    label = 'Export revised controls' if frame_index is None else 'Export selected photo pair'
    if st.button(label, disabled=complete != total, key=f'{token}-export'):
        try:
            exported = export_reviewed_controls(pack, output,
                **({'frame_index': frame_index} if frame_index is not None else {}))
            st.session_state[f'{token}-exported'] = (digest(decisions), str(exported))
        except (OSError, ValueError, TypeError, KeyError) as exc:
            st.error(f'Control export failed: {exc}')
    exported = st.session_state.get(f'{token}-exported')
    if exported and exported[0] == digest(decisions) and complete == total:
        path = Path(exported[1])
        try:
            content = path.read_bytes()
        except OSError as exc:
            st.error(f'Exported controls unavailable: {exc}')
        else:
            st.success('New control bundle saved. Use it in Local RGB → thermal boundary preview.')
            st.code(str(path), language='text')
            st.download_button('Download revised controls JSON', content,
                file_name=path.name, mime='application/json', key=f'{token}-download')


def _export_options(pack, decisions, output, token, complete, total, frame_index):
    import streamlit as st

    if frame_index >= 0:
        points = pack['frames'][frame_index]['controls']['points']
        approved = sum(decisions.get(p['control_id'], {}).get('status') == 'OK' for p in points)
        st.caption(f'Selected photo pair: {approved}/{len(points)} OK. Every original fit/check '
                   'control in this pair is required; other photo pairs may remain pending.')
        _export_controls(pack, decisions, output, f'{token}-pair-{frame_index}',
                         approved, len(points), frame_index=frame_index)
    _export_controls(pack, decisions, output, token, complete, total)


def control_review_view(default_pack='', *, output, key='control-review', pack_options=None):
    import streamlit as st

    from pvdelta.review_ui import finding_navigation

    with st.expander('RGB–thermal source control review'):
        if pack_options:
            dataset = st.selectbox('Source review dataset', list(pack_options), key=f'{key}-dataset')
            # A distinct path widget per dataset prevents a previous pack's text
            # from silently overriding the newly selected source set.
            path_key = f'{key}-pack-{digest(dataset)[:16]}'
            pack_text = st.text_input('Source control review pack', str(pack_options[dataset]), key=path_key)
        else:
            pack_text = st.text_input('Source control review pack', str(default_pack), key=f'{key}-pack')
        if not pack_text.strip():
            st.caption('Prepare a source-bound control pack with prepare-control-review.')
            return
        try:
            path = Path(pack_text).expanduser().resolve()
            pack = load_review_pack(path)
            decisions = latest_control_decisions(pack, output)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            st.error(f'Control review unavailable: {exc}')
            return
        points = {p['control_id']: (entry, p) for entry, p in review_points(pack)}
        token = f"{key}-{pack['pack_id']}"
        complete = sum(d.get('status') == 'OK' for d in decisions.values())
        st.write(f'OK: {complete}/{len(points)} · '
                 f'Wait / Not OK / unreviewed: {len(points)-complete}')
        st.caption(pack['centre_contract'])
        counts = {status: sum(decisions.get(p, {}).get('status', 'Unreviewed') == status
                             for p in points) for status in ('Unreviewed', 'Wait', 'Not OK', 'OK')}
        st.caption(' · '.join(f'{status}: {count}' for status, count in counts.items()))
        frame_labels = {i: f'{i+1}: ' + ' ↔ '.join(e['source_names'])
                        for i, e in enumerate(pack['frames'])}
        frame_filter = st.selectbox('Source photo pair', [-1, *frame_labels],
            format_func=lambda i: 'All photo pairs' if i == -1 else frame_labels[i],
            key=f'{token}-frame-filter')
        status_filter = st.selectbox('Control review status',
            ['All', 'Needs review', 'Unreviewed', 'Wait', 'Not OK', 'OK'], key=f'{token}-status-filter')
        visible = []
        for i, entry in enumerate(pack['frames']):
            if frame_filter != -1 and i != frame_filter:
                continue
            for point in entry['controls']['points']:
                identifier = point['control_id']
                status = decisions.get(identifier, {}).get('status', 'Unreviewed')
                if (status_filter == 'All' or status_filter == status
                        or status_filter == 'Needs review' and status != 'OK'):
                    visible.append(identifier)
        st.caption(f'Showing {len(visible)} / {len(points)} controls. '
                   'Status filters only change the review queue. Pair export requires every '
                   'original control in that pair; whole-pack export requires all controls.')
        if not visible:
            st.info('No source controls match these filters.')
            _export_options(pack, decisions, output, token, complete, len(points), frame_filter)
            return
        selected = finding_navigation(visible, key=f'{token}-selected',
            label='Source control', item_label='Control')
        entry, point = points[selected]
        decision = decisions.get(selected, {})
        if decision.get('review_basis') == 'AI_source_visual_review':
            st.info('AI source review: development observations; independent correspondence is unverified.')
        current = {field: decision.get(field, point[field]) for field in ('native_xy', 'rgb_xy')}
        st.caption(f"Fixed role: {point['role']} · " + ' ↔ '.join(entry['source_names']))
        if st.checkbox('Show full source frames', key=f'{token}-show-context'):
            for col, field, file_key, name in zip(st.columns(2), ('native_xy', 'rgb_xy'),
                    ('native_preview', 'original_rgb'), entry['source_names']):
                col.image(_overview(path.parent/entry[file_key], current[field]),
                    caption=name+' · selected source observation', width='stretch')
            st.caption('Display thumbnails preserve orientation and aspect ratio. '
                       'Magenta marks the saved observation, not a validated match or anomaly.')
        reviewer = st.text_input('Control reviewer', key=f'{key}-reviewer')
        fields = {}
        # Decision identity in widget keys prevents stale coordinates after an external edit.
        version = decision.get('decision_id', 'original')
        form_key = f'{token}-{selected}-{version}'
        with st.form(form_key):
            for column, field, shape_key, file_key, label, radius in zip(st.columns(2),
                    ('native_xy', 'rgb_xy'), ('native_shape', 'rgb_shape'),
                    ('native_preview', 'original_rgb'), ('Native thermal', 'Original RGB'), (20, 60)):
                shape = entry['controls']['pair'][shape_key]
                picture, box = _crop(path.parent/entry[file_key], current[field], radius)
                column.image(picture, caption=f'{label} · crop origin ({box[0]}, {box[1]})', width='stretch')
                x = column.number_input(f'{label} X', min_value=0., max_value=float(shape[1]-1),
                    value=float(current[field][0]), step=.25, key=f'{form_key}-{field}-x')
                y = column.number_input(f'{label} Y', min_value=0., max_value=float(shape[0]-1),
                    value=float(current[field][1]), step=.25, key=f'{form_key}-{field}-y')
                fields[field] = [x, y]
            status = st.radio('Source correspondence', ['Wait', 'OK', 'Not OK'], horizontal=True,
                index=['Wait', 'OK', 'Not OK'].index(decision.get('status', 'Wait')))
            note = st.text_input('Control note', decision.get('note', ''))
            save = st.form_submit_button('Save source control')
            save_next = st.form_submit_button('Save and next')
        st.caption('Magenta cross: selected observation only. No model prediction or residual is '
            'shown. Save to refresh the crop. OK records your visual assessment, not independent '
            'calibration, glare clearance or field acceptance.')
        if save or save_next:
            try:
                save_control_decision(pack, output, selected, native_xy=fields['native_xy'],
                    rgb_xy=fields['rgb_xy'], status=status, reviewer=reviewer, note=note)
                if save_next:
                    index = visible.index(selected)
                    if index+1 < len(visible):
                        st.session_state[f'{token}-selected__pending'] = visible[index+1]
                st.rerun()
            except (OSError, ValueError, TypeError) as exc:
                st.error(f'Control decision not saved: {exc}')
        _export_options(pack, decisions, output, token, complete, len(points), frame_filter)
