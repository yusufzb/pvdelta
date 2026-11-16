"""Explore the synthetic inspection in Streamlit without a camera or SDK."""

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw
from synthetic_inspection import synthetic_frame

from pvdelta.pipeline import analyze_annotations


def thermal_preview(values, modules, findings):
    """Color the demo grid and outline the exact regions returned by the pipeline."""
    low, high = np.array([39, 113, 124]), np.array([255, 182, 100])
    weights = np.clip((values - 30) / 10, 0, 1)[..., None]
    pixels = low + weights * (high - low)
    scale = 6
    height, width = values.shape
    image = Image.fromarray(pixels.astype(np.uint8)).resize(
        (width * scale, height * scale), Image.Resampling.NEAREST
    )
    draw = ImageDraw.Draw(image)
    for polygon, color, line_width in (
        *[(module.polygon, '#b1e1dd', 2) for module in modules],
        *[(finding.pixel_polygon, '#fff4dd', 3) for finding in findings],
    ):
        points = [(round(x * scale), round(y * scale)) for x, y in polygon]
        draw.line([*points, points[0]], fill=color, width=line_width)
    return image


st.set_page_config(page_title='PV Delta synthetic inspection', layout='wide')
st.title('PV Delta · Thermal inspection')
st.caption('Synthetic inspection · Generated temperature grid and supplied module regions')

frame, modules = synthetic_frame()
reports = analyze_annotations(frame, modules, direction='hot')
findings = [finding for report in reports for finding in report.findings]
values = frame.temperature_c
overview, review = st.tabs(['Overview', 'Review candidates'])

with overview:
    analyzed, candidates, peak = st.columns(3)
    analyzed.metric('Modules analyzed', sum(r.assessment_status == 'analyzed' for r in reports))
    candidates.metric('Hot region candidates', len(findings))
    peak.metric('Peak grid temperature', f'{values.max():.0f} °C')
    row = st.slider('Native row', 0, values.shape[0] - 1, value=29)
    st.line_chart({'Temperature (°C)': values[row]}, height=260)

with review:
    st.image(thermal_preview(values, modules, findings),
             caption='Synthetic thermal grid with module and candidate outlines')
    for finding in findings:
        with st.container(border=True):
            st.subheader(f'{finding.module_id} · Hot region')
            local, observed, reference = st.columns(3)
            local.metric('Local ΔT', f'{finding.delta_t_c:+.1f} K')
            observed.metric('Region temperature', f'{finding.absolute_t_c:.1f} °C')
            reference.metric('Local reference', f'{finding.reference_t_c:.1f} °C')
            st.caption(f'{finding.n_region_px} native pixels in the candidate region')
