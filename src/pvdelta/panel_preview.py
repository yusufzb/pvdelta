"""V2 multi-region displays: one native SDK decode, one panel card, unchanged region records."""
from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pvdelta.finding_preview import DISPLAY_OUTLINE_SCALE


@dataclass(frozen=True)
class PanelPreview:
    image: Image.Image
    overlay_applied: bool
    message: str
    region_states: dict[str, str]
    radiometry_verified: bool = False


def _regions(regions, native_size):
    regions = list(regions)
    if not regions or len(regions) > 256:
        raise ValueError('One to 256 explicit regions from one observation required')
    labels = [r['label'] for r in regions]
    if any(not isinstance(s, str) or not s for s in labels) or len(set(labels)) != len(labels):
        raise ValueError('Distinct, nonempty display labels required')
    width, height = native_size
    for region in regions:
        q = np.asarray(region['polygon'], float)
        if (q.shape != (4, 2) or not np.isfinite(q).all() or (q < 0).any()
                or (q[:, 0] >= width).any() or (q[:, 1] >= height).any()):
            raise ValueError('Every region must use four finite corners on the common native grid')
    return regions


def _label(draw, label, polygon, size, used, *, scale=1):
    low = np.asarray(polygon).min(axis=0)
    x = int(np.clip(low[0], 0, size[0]-25*scale))
    y = int(np.clip(low[1]-12*scale, 0, size[1]-12*scale))
    while any(abs(x-a) < 24*scale and abs(y-b) < 12*scale for a, b in used) and y < size[1]-24*scale:
        y += 12*scale
    used.append((x, y))
    font = None
    for name in ('arial.ttf', 'DejaVuSans.ttf'):
        try:
            font = ImageFont.truetype(name, 12*scale)
            break
        except OSError:
            continue
    draw.text((x, y), label, font=font, fill='white', stroke_width=max(1, scale//2), stroke_fill='black')


def thermal_panel_preview(path, regions, *, sdk_library=None, native_size=(640, 512),
                          module_polygon=None, measurement_profile=None, expected_source_hash=None):
    from pvdelta.finding_preview import _native_candidate_display
    from pvdelta.mask_review import review_region_support
    from pvdelta.reference import erode
    from pvdelta.regions import polygon_to_mask

    regions = _regions(regions, native_size)
    try:
        pixels = _native_candidate_display(path, sdk_library, native_size,
                                           measurement_profile, expected_source_hash)
    except (OSError, RuntimeError, ValueError) as exc:
        with Image.open(path) as source:
            image = source.convert('RGB')
        return PanelPreview(image, False, f'Unmarked thermal photo; all native overlays withheld: {exc}',
                            {r['label']: 'unavailable' for r in regions})
    marked = pixels.copy()
    if module_polygon is not None:
        mask = polygon_to_mask(module_polygon, pixels.shape[:2])
        marked[mask & ~erode(mask, 1)] = [0, 255, 255]
    for region in regions:
        points = region.get('pixels_yx')
        if points is None or not len(points):
            continue
        review_region_support(np.ones(pixels.shape[:2], bool), points)
        mask = np.zeros(pixels.shape[:2], bool)
        y, x = np.asarray(points).T
        mask[y, x] = True
        padded = np.pad(mask, 1)
        ring = (padded[:-2, 1:-1] | padded[2:, 1:-1] | padded[1:-1, :-2] | padded[1:-1, 2:]) & ~mask
        marked[ring] = [255, 255, 255]
    image = Image.fromarray(marked)
    draw = ImageDraw.Draw(image)
    outlines, used = [], []
    for region in regions:
        q = np.asarray(region['polygon'], float)
        outline = q.mean(axis=0) + DISPLAY_OUTLINE_SCALE*(q-q.mean(axis=0))
        outlines.append(outline)
        draw.line([*map(tuple, outline), tuple(outline[0])], fill='#ff2020', width=1)
        _label(draw, region['label'], outline, image.size, used)
    all_points = np.concatenate(outlines)
    low = np.maximum(0, np.floor(all_points.min(axis=0)-500)).astype(int)
    high = np.minimum(image.size, np.ceil(all_points.max(axis=0)+500)).astype(int)
    verified = measurement_profile is not None
    identity = ('native SDK profile/hash checked' if verified else
                'source/grid checked; saved candidate locations only; historical temperatures not reproduced')
    return PanelPreview(image.crop((*low, *high)), True,
        f"{len(regions)} regions together · {identity} · red: {DISPLAY_OUTLINE_SCALE}× display outlines · "
        'white: exact-pixel exterior rings when available · cyan: provisional module · A labels match the region table',
        {r['label']: 'native_verified' if verified else 'legacy_location' for r in regions}, verified)


def rgb_panel_preview(path, regions, *, native_size=(640, 512), **kwargs):
    from pvdelta.rgb_preview import rgb_finding_preview

    regions = _regions(regions, native_size)
    encoded = Path(path).read_bytes()
    with Image.open(io.BytesIO(encoded)) as source:
        image = source.convert('RGB')
    expected_hash = kwargs.get('expected_rgb_hash')
    if expected_hash is None or hashlib.sha256(encoded).hexdigest() != expected_hash:
        return PanelPreview(image, False, 'RGB source identity unavailable or changed; all region overlays withheld',
                            {r['label']: 'unavailable' for r in regions})
    mapped, states, reasons = [], {}, []
    module = None
    for region in regions:
        preview = rgb_finding_preview(path, region['polygon'], native_size=native_size, **kwargs)
        states[region['label']] = preview.registration_status
        if not preview.overlay_applied or preview.outline_rgb is None:
            reasons.append(region['label']+': '+preview.message)
            continue
        mapped.append((region['label'], np.asarray(preview.outline_rgb, float)))
        if module is None or preview.registration_status == 'image_checked_provisional':
            module = preview.module_outline_rgb
    if not mapped:
        return PanelPreview(image, False, 'No RGB region overlays available · '+'; '.join(reasons), states)
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected_hash:
        return PanelPreview(image, False, 'RGB source changed during preview; all region overlays withheld',
                            {r['label']: 'unavailable' for r in regions})
    draw = ImageDraw.Draw(image)
    if module is not None:
        draw.line([*module, module[0]], fill='#00ffff', width=4)
    used = []
    for label, outline in mapped:
        draw.line([*map(tuple, outline), tuple(outline[0])], fill='#ff2020', width=5)
        _label(draw, label, outline, image.size, used, scale=3)
    points = np.concatenate([q for _, q in mapped])
    centres = np.array([q.mean(axis=0) for _, q in mapped])
    low = np.minimum(points.min(axis=0), centres.min(axis=0)-500)
    high = np.maximum(points.max(axis=0), centres.max(axis=0)+500)
    bounds = (*np.maximum(0, np.floor(low-180)).astype(int),
              *np.minimum(image.size, np.ceil(high+180)).astype(int))
    approximate = sum(s == 'approximate' for s in states.values())
    checked = sum(s == 'image_checked_provisional' for s in states.values())
    message = (f'{len(mapped)}/{len(regions)} RGB regions together · red: {DISPLAY_OUTLINE_SCALE}× display outlines · '
        f'{approximate} approximate, {checked} module-edge checked (provisional) · '
        'A labels correspond to the thermal view; physical identity remains unverified')
    if reasons:
        message += ' · Unmapped: '+'; '.join(reasons)
    return PanelPreview(image.crop(bounds), True, message, states)
