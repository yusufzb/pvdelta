"""Optional Streamlit controls shared by the two finding review screens."""

from __future__ import annotations


def finding_navigation(
    ids: list[str], *, key: str, label: str,
    names: dict[str, str] | None = None, item_label: str = "Candidate",
) -> str | None:
    """Navigate the current filtered population with pre-rerun callbacks."""
    import streamlit as st

    if not ids:
        st.session_state.pop(key, None)
        st.info("No findings match the current filter.")
        return None
    pending_key = f"{key}__pending"
    pending = st.session_state.pop(pending_key, None)
    if pending in ids:
        # This runs before the selectbox is instantiated on the new rerun.
        st.session_state[key] = pending
    elif st.session_state.get(key) not in ids:
        st.session_state[key] = ids[0]

    def select(value: str) -> None:
        # Do not mutate a selectbox's own key from its button callback.  The
        # assignment is applied at the start of the following rerun instead;
        # this avoids Streamlit's WidgetAlreadyInstantiatedError.
        st.session_state[pending_key] = value

    selected = st.selectbox(label, ids, key=key, format_func=lambda value: (names or {}).get(value, value))
    index = ids.index(selected)
    previous, following, position = st.columns([1, 1, 2])
    previous.button(
        "← Previous", key=f"{key}-previous", disabled=index == 0,
        on_click=select, args=(ids[max(0, index - 1)],), use_container_width=True,
    )
    following.button(
        "Next →", key=f"{key}-next", disabled=index == len(ids) - 1,
        on_click=select, args=(ids[min(len(ids) - 1, index + 1)],), use_container_width=True,
    )
    position.caption(f"{item_label} {index + 1} of {len(ids)}")
    return selected


def repeat_view_review(database, finding_id: str, *, key: str, sdk_library=None) -> None:
    """Show image-supported panel views without borrowing unrelated findings."""
    import json
    from pathlib import Path

    import duckdb
    import streamlit as st
    from PIL import Image, ImageDraw

    from pvdelta.decode import RJpeg
    from pvdelta.repeat_views import (
        build_repeat_evidence,
        nearby_image_pairs,
        repeat_evidence_signature,
    )

    with st.expander("Same-panel views / glare evidence", expanded=False):
        st.caption(
            "Check up to 8 nearby captures within ±30 seconds using motion inside the selected RGB panel, "
            "held-out geometry and unique panel overlap. This is a bounded search, not "
            "all observations of this panel. Matching does not confirm a fault or rule out glare."
        )
        profile = st.text_input("Site geometry profile", value=str(Path(database).parent / 'site-geometry.json'),
                                key=f"{key}-site-profile")
        st.caption("Use this flight's site profile. LRF position consistency is required (1.5 m coarse tolerance, not accuracy).")
        match_allowed = True
        try:
            signature = repeat_evidence_signature(database, finding_id, Path(profile))
        except (OSError, ValueError, RuntimeError, duckdb.Error) as exc:
            st.info(f'Repeat-view evidence unavailable: {exc}')
            match_allowed = False
            try:
                signature = repeat_evidence_signature(database, finding_id, None)
            except (OSError, ValueError, RuntimeError, duckdb.Error) as manual_exc:
                st.info(f'Nearby image pairs unavailable: {manual_exc}')
                return
        cache_key = f"{key}-{database}-{finding_id}-{signature}-repeat-v6-measurement-profile"
        if st.button("Find matching panel views", key=f"{key}-find-views", disabled=not match_allowed):
            st.session_state.pop(cache_key, None)
            st.session_state[cache_key+'__mode'] = 'matching'
            try:
                with st.spinner("Checking RGB correspondence and recorded thermal evidence…"):
                    st.session_state[cache_key] = build_repeat_evidence(
                        Path(database), finding_id, site_profile=Path(profile),
                    )
            except (OSError, ValueError, RuntimeError, ImportError) as exc:
                st.error(f"Repeat-view inspection unavailable: {exc}")
        if st.button('Browse nearby image pairs (unverified)', key=f'{key}-nearby-pairs'):
            st.session_state.pop(cache_key, None)
            st.session_state[cache_key+'__mode'] = 'manual'
            try:
                st.session_state[cache_key] = nearby_image_pairs(Path(database), finding_id)
            except (OSError, ValueError, RuntimeError, duckdb.Error) as exc:
                st.error(f'Nearby image pairs unavailable: {exc}')
        rows = st.session_state.get(cache_key)
        if rows is None:
            return
        if not rows:
            st.info("No paired captures exist in the search window.")
            return
        glare_summary = rows[0].get("glare_summary", {})
        if glare_summary:
            st.info(
                f"Multi-angle evidence: **{glare_summary['status']}** — "
                f"{glare_summary['reason']} "
                f"({glare_summary['assessed_view_count']} assessed views, "
                f"{glare_summary['unknown_view_count']} unassessed, "
                f"{glare_summary['angle_span_deg']:.1f}° pixel-ray separation). "
                "Geometry and component correspondence remain provisional."
            )
        matched = [r for r in rows if r['status'] == 'matched_provisional']
        nearby = [r for r in rows if r['status'] != 'source']
        manual_only = st.session_state.get(cache_key+'__mode') == 'manual'
        if manual_only:
            st.caption(f'{len(rows)} nearby image pairs · panel matching not requested')
        else:
            st.caption(f"{len(matched)} provisional panel matches / {len(nearby)} inspected views")
        st.dataframe([
            {"frame": Path(r.get('thermal_path', '')).name, "status": r['status'],
             "reason": r['reason'], "check_p95_px": r.get('check_p95_px'),
             "panel_overlap": r.get('iou'), "time_gap_s": r.get('time_gap_s'),
             "location_distance_m": r.get('location_distance_m'),
             "pixel_ray_separation_deg": r.get('view_separation_from_source_deg'),
             "component_evidence": r.get('component_evidence', 'unknown'),
             "multi_angle": r.get('glare_summary', {}).get('status')}
            for r in rows
        ], hide_index=True)
        if not matched and not manual_only:
            st.info(
                "Automatic panel association did not pass. Available nearby image pairs "
                "can still be compared manually below; this result is not evidence of glare."
            )
        evidence_views = [r for r in rows if r['status'] in {'source', 'matched_provisional'}]
        selected = finding_navigation(
            [r['capture_id'] for r in evidence_views], key=f"{cache_key}-matched",
            label="Panel evidence frame (including source)",
            names={r['capture_id']: Path(r['thermal_path']).name for r in evidence_views}, item_label="View",
        ) if evidence_views else None
        for row in (r for r in evidence_views if r['capture_id'] == selected):
            st.markdown(f"**{Path(row['thermal_path']).name} · {row['module']}**")
            left, right = st.columns(2)
            try:
                with Image.open(row['rgb_path']) as source:
                    rgb = source.convert('RGB')
                with Image.open(row['thermal_path']) as source:
                    thermal = source.convert('RGB')
            except OSError as exc:
                st.error(f"Image pair could not be opened: {exc}")
                continue
            native_render = False
            try:
                with RJpeg(row['thermal_path'], sdk_library) as image:
                    if row.get('measurement_profile') is not None:
                        from pvdelta.measurement_profile import apply_saved_measurement_profile

                        apply_saved_measurement_profile(image, row['thermal_path'], row['measurement_profile'])
                    thermal = Image.fromarray(image.processed_rgb())
                native_render = thermal.size == (row['width_px'], row['height_px'])
            except (OSError, RuntimeError, ValueError):
                # The embedded JPEG is not a validated scaled copy of native pixels.
                # Keep it available for manual inspection, with no thermal overlay.
                pass
            ImageDraw.Draw(rgb).polygon([tuple(p) for p in row['rgb_polygon']],
                                        outline='#00ffff', width=max(3, rgb.width // 600))
            right.image(rgb, caption="Cyan: corresponding panel in RGB", use_container_width=True)
            if row['thermal_geometry_accepted'] and native_render:
                sx, sy = thermal.width / row['width_px'], thermal.height / row['height_px']
                draw = ImageDraw.Draw(thermal)
                draw.polygon([(x * sx, y * sy) for x, y in row['native_polygon']],
                             outline='#00ffff', width=2)
                for finding in row['findings']:
                    polygon = json.loads(finding['pixel_polygon_json'])
                    cx = sum(p[0] for p in polygon) / len(polygon)
                    cy = sum(p[1] for p in polygon) / len(polygon)
                    draw.polygon([((cx + 5 * (x - cx)) * sx, (cy + 5 * (y - cy)) * sy)
                                  for x, y in polygon], outline='#ff2020', width=2)
                left.image(thermal, caption="Native SDK thermal · cyan: panel; red: findings (5× outline)",
                           use_container_width=True)
            else:
                left.image(thermal, caption="Native geometry or SDK rendering unavailable; overlay withheld",
                           use_container_width=True)
            observation = row.get('observation') or {}
            st.caption(
                f"Assessment: {observation.get('assessment_status', 'unavailable')} · "
                f"Median: {observation.get('temperature_median_c')} °C · "
                f"Max: {observation.get('temperature_max_c')} °C · "
                f"Recorded regions: {len(row['findings'])} · "
                f"Gimbal yaw/pitch: {row['gimbal_yaw_deg']} / {row['gimbal_pitch_deg']}°"
            )
            st.caption("No recorded region does not establish a healthy panel. Camera yaw is not the viewing incidence angle.")
        unmatched = [r for r in rows if r['status'] not in {'source', 'matched_provisional'}
                     and r.get('rgb_path') and r.get('thermal_path')]
        if unmatched:
            with st.expander("Nearby pairs — manual comparison (panel identity unverified)",
                             expanded=not matched):
                st.caption(
                    "These are nearby captures, not verified observations of this panel. "
                    "The panel may be outside the frame. No transferred outline or temperature "
                    "is assigned to the selected candidate."
                )
                selected_manual = finding_navigation(
                    [r['capture_id'] for r in unmatched], key=f"{cache_key}-manual",
                    label="Nearby thermal frame",
                    names={r['capture_id']: Path(r['thermal_path']).name for r in unmatched},
                    item_label="View",
                )
                row = next(r for r in unmatched if r['capture_id'] == selected_manual)
                st.caption(f"Automatic association: {row['reason']}")
                st.caption(
                    f"Time gap: {row.get('time_gap_s')} s · "
                    f"Gimbal yaw/pitch: {row.get('gimbal_yaw_deg')} / {row.get('gimbal_pitch_deg')}°"
                )
                for col, image_key in zip(st.columns(2), ('thermal_path', 'rgb_path'), strict=True):
                    try:
                        with Image.open(row[image_key]) as source:
                            col.image(source.convert('RGB'), caption=Path(row[image_key]).name,
                                      use_container_width=True)
                    except OSError as exc:
                        col.error(f"Image could not be opened: {exc}")
