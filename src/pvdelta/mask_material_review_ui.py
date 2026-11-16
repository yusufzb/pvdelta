"""Direct local material review in Basic/Advanced, including phone sessions."""
from __future__ import annotations

import hashlib
import json

from pvdelta.mask_material_review import (
    asset_path,
    latest_material_work,
    load_material_pack,
    save_material_decision,
)
from pvdelta.review_ui import finding_navigation


def material_review_view(default_pack='', *, advanced=False, key='mask-material'):
    import streamlit as st

    if not default_pack and not advanced:
        return
    with st.expander('Mask material review · glass, frame or gap', expanded=False):
        path = (st.text_input('Material review pack JSON', str(default_pack), key=f'{key}-pack')
                if advanced else str(default_pack))
        if not path.strip():
            st.info('Choose a prepared source-bound material review pack.')
            return
        if not st.checkbox('Open material review', key=f'{key}-open'):
            return
        try:
            pack, root = load_material_pack(path)
            latest, _ = latest_material_work(pack, root)
            scope = f'{key}-{pack["pack_id"][7:19]}'
            st.caption('Classify the pink native pixels. Decisions save locally; no JSON export is '
                       'required. This does not change masks, temperatures, findings or model acceptance.')
            reviewer = st.text_input('Reviewer', latest['reviewer'], key=f'{scope}-reviewer')
            work, checked = latest_material_work(pack, root, reviewer=reviewer.strip())
            identifier = finding_navigation(
                [r['item_id'] for r in pack['items']], key=f'{scope}-item',
                label='Review region', item_label='Region',
                names={r['item_id']: f'{r["item_id"]} · {r["source_name"]}' for r in pack['items']},
            )
            item = next(r for r in pack['items'] if r['item_id'] == identifier)
            current = next(r for r in work['items'] if r['item_id'] == identifier)
            st.code(item['source_name'])
            temperatures = item['temperature_c']
            st.caption(f'{item["target_pixel_count"]} native pixels · recorded-profile SDK '
                       f'{temperatures["min"]:.3f} / {temperatures["median"]:.3f} / '
                       f'{temperatures["max"]:.3f} °C (min / median / max). '
                       'These are not independent field measurements.')
            crop, locator = st.columns(2)
            crop.image(str(asset_path(root, item['crop'])), use_container_width=True,
                       caption='Pink pixels to classify · display enlargement only')
            locator.image(str(asset_path(root, item['locator'])), use_container_width=True,
                          caption='Native frame · yellow rectangle locates the crop')
            reviewer_key = hashlib.sha256(reviewer.strip().encode('utf-8')).hexdigest()[:12]
            note = st.text_input('Material note', current['note'],
                                 key=f'{scope}-{identifier}-{reviewer_key}-note')
            st.caption('Select a material button to save the decision and note. '
                       'Unclear stays unresolved; unreviewed never means nonpanel.')
            choices = st.columns(3)
            selected = None
            for column, value, label in zip(choices,
                    ('glass', 'frame_or_nonpanel', 'unclear'),
                    ('Panel glass', 'Frame / gap', 'Unclear')):
                if column.button(label, key=f'{scope}-{value}',
                                 help='Frame / gap also includes clamps and nonpanel ground.',
                                 disabled=not reviewer.strip(), use_container_width=True):
                    selected = value
            if st.button('Reset to unreviewed', key=f'{scope}-reset', disabled=not reviewer.strip()):
                selected = 'unreviewed'
            if selected is not None:
                _, work, checked = save_material_decision(
                    path, identifier, selected, reviewer, note=note,
                    expected_pack_id=pack['pack_id'],
                )
                current = next(r for r in work['items'] if r['item_id'] == identifier)
                st.success(f'Saved: {current["classification"]}')
            st.write(f'Current material: **{current["classification"]}**')
            counts = checked['counts']
            st.caption(f'{len(pack["items"])-counts["unreviewed"]} / {len(pack["items"])} answered '
                       f'for this reviewer · {counts["unclear"]} unclear. '
                       'Material descriptions do not approve a hotspot, glare verdict or Stage 1 release.')
            with st.expander('Unmarked thermal and RGB context'):
                st.image(str(asset_path(root, item['unmarked_native'])), use_container_width=True,
                         caption=item['source_name'])
                if item.get('rgb_context'):
                    st.image(str(asset_path(root, item['rgb_context'])), use_container_width=True,
                             caption=f'{item.get("rgb_source_name", "RGB")} · unregistered context; '
                                     'pixel positions do not correspond directly')
                st.json(item.get('effective_parameters', {}))
            st.download_button('Download material review JSON', json.dumps(work, indent=2),
                               file_name='pvdelta-mask-material-review.json', mime='application/json',
                               key=f'{scope}-download')
        except (OSError, ValueError, TypeError, KeyError) as exc:
            st.error(f'Material review unavailable: {exc}')
