"""Render native findings only on the SDK grid; embedded JPEG fallback stays unmarked."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DISPLAY_OUTLINE_SCALE = 6


@dataclass(frozen=True)
class FindingPreview:
    image: object
    overlay_applied: bool
    message: str
    radiometry_verified: bool = False


def _native_candidate_display(path, sdk_library, native_size, measurement_profile, expected_source_hash):
    """Pixel locations need source/grid identity, separately from historic radiometry.

    A missing legacy profile permits a source-bound location preview only. An
    existing but invalid profile still fails; it never falls back to this mode.
    Embedded JPEG pixels are never used as the native coordinate grid.
    """
    import hashlib
    from pathlib import Path

    from pvdelta.decode import RJpeg
    from pvdelta.measurement_profile import apply_saved_measurement_profile

    def check_source():
        if expected_source_hash is not None:
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected_source_hash:
                raise ValueError('Indexed thermal source identity mismatch')
        elif measurement_profile is None:
            raise ValueError('Indexed thermal source hash required for legacy candidate locations')

    check_source()
    with RJpeg(path, sdk_library) as source:
        if measurement_profile is not None:
            values = apply_saved_measurement_profile(source, path, measurement_profile)
            if values.shape != (native_size[1], native_size[0]):
                raise ValueError('Saved measurement grid differs from the candidate grid')
        pixels = source.processed_rgb()
    check_source()
    if (pixels.dtype != np.uint8 or pixels.shape != (native_size[1], native_size[0], 3)):
        raise ValueError('SDK rendering dimensions do not match stored native grid')
    return pixels


def native_frame_preview(path, diagnostic, measurement_profile, *, sdk_library=None):
    """Show proposed envelopes and their actual disjoint support, including zero findings."""
    from PIL import Image, ImageDraw

    from pvdelta.decode import RJpeg
    from pvdelta.measurement_profile import apply_saved_measurement_profile
    from pvdelta.regions import polygon_to_mask
    from pvdelta.support_partition import partition_measurement_support

    try:
        if diagnostic.get('geometry_method') != 'native_observed_boundaries':
            raise ValueError('Native structural geometry required')
        with RJpeg(path, sdk_library) as source:
            values = apply_saved_measurement_profile(source, path, measurement_profile)
            display = source.processed_rgb()
        if display.dtype != np.uint8 or display.shape != (*values.shape, 3):
            raise ValueError('SDK display does not match the saved native grid')
        modules = diagnostic['modules']
        masks = [polygon_to_mask(m['native_polygon'], values.shape) for m in modules]
        supports = partition_measurement_support(masks, margin_px=2)
        union = np.zeros(values.shape, bool)
        occupancy = np.zeros(values.shape, np.int32)
        for mask, support, module in zip(masks, supports, modules):
            import hashlib

            if hashlib.sha256(support.interior.tobytes()).hexdigest() != module['mask_sha256']:
                raise ValueError('Stored native measurement support differs from the displayed geometry')
            union |= support.interior
            occupancy += mask
        marked = display.copy()
        marked[union] = (.8*marked[union]+.2*np.array([0, 200, 255])).astype(np.uint8)
        marked[occupancy > 1] = [255, 0, 255]
        image = Image.fromarray(marked)
        draw = ImageDraw.Draw(image)
        for module in modules:
            points = [tuple(p) for p in module['native_polygon']]
            draw.line(points+points[:1], fill='#00ffff', width=1)
        return FindingPreview(image, True,
            f'{len(modules)} provisional native proposals · cyan: envelope · shaded: measurement support · '
            'magenta: shared pixels excluded · saved native hash checked · panel membership unverified')
    except (OSError, RuntimeError, ValueError, TypeError, KeyError) as exc:
        with Image.open(path) as source:
            image = source.convert('RGB')
        return FindingPreview(image, False, f'Unmarked embedded JPEG; native support preview withheld: {exc}')


def thermal_finding_preview(
    path, polygon, *, sdk_library=None, native_size=(640, 512), region_pixels_yx=None,
    module_polygon=None,
    measurement_profile=None,
    expected_source_hash=None,
) -> FindingPreview:
    from PIL import Image, ImageDraw

    from pvdelta.regions import polygon_to_mask

    points = np.asarray(polygon, dtype=float)
    if (points.shape != (4, 2) or not np.isfinite(points).all()
            or len(native_size) != 2 or any(
                isinstance(n, bool) or not isinstance(n, int) or n <= 0 for n in native_size)):
        raise ValueError("finding needs four finite native corners and positive integer dimensions")
    try:
        pixels = _native_candidate_display(path, sdk_library, native_size,
                                           measurement_profile, expected_source_hash)
    except (OSError, RuntimeError, ValueError) as exc:
        with Image.open(path) as source:
            image = source.convert("RGB")
        return FindingPreview(image, False,
                              f"Unmarked embedded JPEG; native SDK overlay unavailable: {exc}")
    image = Image.fromarray(pixels)
    draw = ImageDraw.Draw(image)
    module_drawn = False
    if module_polygon is not None:
        envelope = polygon_to_mask(module_polygon, pixels.shape[:2])
        # A one-pixel raster outline uses the very same pixel-centre convention as measurement.
        from pvdelta.reference import erode
        border = envelope & ~erode(envelope, 1)
        array = np.asarray(image).copy()
        array[border] = [0, 255, 255]
        image = Image.fromarray(array)
        draw = ImageDraw.Draw(image)
        module_drawn = True
    exact_drawn = False
    if region_pixels_yx is not None and len(region_pixels_yx):
        from pvdelta.mask_review import review_region_support
        # Validate indices even without a panel polygon. Never cast fractions to pixels.
        review_region_support(np.ones(pixels.shape[:2], dtype=bool), region_pixels_yx)
        region = np.zeros(pixels.shape[:2], dtype=bool)
        y, x = np.asarray(region_pixels_yx).T
        region[y, x] = True
        from pvdelta.reference import erode
        # Leave hot pixels visible: mark only the exterior, 4-neighbour ring.
        padded = np.pad(region, 1)
        ring = (padded[:-2, 1:-1] | padded[2:, 1:-1]
                | padded[1:-1, :-2] | padded[1:-1, 2:]) & ~region
        array = np.asarray(image).copy()
        array[ring] = [255, 255, 255]
        image = Image.fromarray(array)
        draw = ImageDraw.Draw(image)
        exact_drawn = True
    centre = points.mean(axis=0)
    expanded = centre + DISPLAY_OUTLINE_SCALE * (points - centre)
    outline = [tuple(point) for point in expanded]
    draw.line(outline + [outline[0]], fill="#ff2020", width=1)
    # Preserve the requested 500px context at native resolution; no geometry scaling.
    low = np.maximum(0, np.floor(expanded.min(axis=0) - 500)).astype(int)
    high = np.minimum(image.size, np.ceil(expanded.max(axis=0) + 500)).astype(int)
    if np.any(high <= low):
        raise ValueError("finding display outline does not intersect native frame")
    image = image.crop((*low, *high))
    legend = f"Native SDK grid · red: {DISPLAY_OUTLINE_SCALE}× display outline"
    if measurement_profile is not None:
        legend += ' · saved measurement profile reconstructed and native hash checked'
    else:
        legend += ' · source/grid checked · saved candidate locations only; historical temperatures not reproduced'
    if exact_drawn:
        legend += " · white: exterior of exact finding pixels"
    else:
        legend += " · exact finding pixels unavailable"
    if module_drawn:
        legend += " · cyan: stored provisional module boundary"
    return FindingPreview(image, True, legend, measurement_profile is not None)
