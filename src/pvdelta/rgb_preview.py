"""Display-only cross-camera projections, explicitly separate from measured native pixels."""
from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from pvdelta.finding_preview import DISPLAY_OUTLINE_SCALE


@dataclass(frozen=True)
class RGBPreview:
    image: Image.Image
    overlay_applied: bool
    message: str
    registration_status: str = 'unavailable'
    outline_rgb: tuple | None = None
    module_outline_rgb: tuple | None = None


def original_rgb_image(path, *, expected_source_hash):
    """Load the indexed encoded RGB grid directly, without drawing overlays."""
    encoded = Path(path).read_bytes()
    if not expected_source_hash or hashlib.sha256(encoded).hexdigest() != expected_source_hash:
        raise ValueError('Original RGB differs from the indexed source')
    with Image.open(io.BytesIO(encoded)) as source:
        return source.convert('RGB')


def quadratic_points(points, model):
    q = np.asarray(points, dtype=float)
    centre, scale = np.asarray(model['centre'], float), np.asarray(model['scale'], float)
    coefficients = np.asarray(model['coefficients'], float)
    if (q.ndim != 2 or q.shape[1] != 2 or centre.shape != (2,) or scale.shape != (2,)
            or coefficients.shape != (6, 2) or not np.isfinite(q).all()
            or not np.isfinite(centre).all() or not np.isfinite(scale).all()
            or not np.isfinite(coefficients).all() or np.any(scale == 0)):
        raise ValueError('Finite native points and a valid quadratic model required')
    x, y = ((q-centre)/scale).T
    result = np.column_stack((np.ones(len(q)), x, y, x*y, x*x, y*y)) @ coefficients
    if not np.isfinite(result).all():
        raise ValueError('Non-finite RGB projection')
    return result


def rgb_finding_preview(path, polygon, *, registration=None, camera_model=None,
        native_size=None, source_root=None, expected_rgb_hash=None, native_verified=False,
        module_polygon=None, pair_check=None, measurement_profile=None):
    """Show a same-flight M4T prior as APPROXIMATE, never as validated correspondence.

    The historical quadratic diagnostic has no cross-frame acceptance. Its use
    here is a labelled visual search aid only, confined to its original flight
    folder/camera/size. No displacement is fitted to anomaly locations, and no
    temperature, panel identity or database record is changed.
    """
    path = Path(path)
    encoded = path.read_bytes()
    with Image.open(path) as source:
        image = source.convert('RGB')

    def unavailable(reason):
        return RGBPreview(image, False, 'Unmarked RGB · ' + reason)

    if expected_rgb_hash is None or hashlib.sha256(encoded).hexdigest() != expected_rgb_hash:
        return unavailable('source identity check unavailable or failed')
    if not native_verified:
        return unavailable('thermal source/native coordinate grid has not been verified')
    if registration is None:
        return unavailable('no compatible cross-camera model is configured')
    calibration_name = registration.get('calibration_frame')
    if (camera_model != 'M4T' or tuple(native_size or ()) != (640, 512)
            or image.size != (4032, 3024)):
        return unavailable('the stored M4T model does not cover this camera or image size')
    if (not source_root or not isinstance(calibration_name, str)
            or Path(calibration_name).name != calibration_name
            or not (Path(source_root)/calibration_name).is_file()):
        return unavailable('the model calibration frame is not in this flight folder')
    q = np.asarray(polygon, float)
    if (q.shape != (4, 2) or not np.isfinite(q).all() or (q < 0).any()
            or (q[:, 0] >= 640).any() or (q[:, 1] >= 512).any()):
        raise ValueError('Four native finding corners inside the saved grid required')
    centre = q.mean(axis=0)
    model = registration['model']
    mapped = quadratic_points(q, model)
    checked = False
    local_note = ''
    observed_module = None
    if pair_check is not None and pair_check.get('status') == 'image_consistent_provisional':
        from pvdelta.local_rgb_transfer import project_local_native_to_rgb
        from pvdelta.rgb_pair_check import digest

        try:
            binding = pair_check['binding']
            if (pair_check.get('content_sha256') != digest({k: v for k, v in pair_check.items()
                                                           if k != 'content_sha256'})
                    or measurement_profile is None
                    or binding['rgb_sha256'] != expected_rgb_hash
                    or binding['thermal_sha256'] != measurement_profile['source_sha256']
                    or binding['profile_id'] != measurement_profile['profile_id']
                    or binding['native_sha256'] != measurement_profile['native_temperature_sha256']
                    or binding['prior_sha256'] != digest(registration)
                    or digest(pair_check['target_module']['native_polygon']) != digest(module_polygon)):
                raise ValueError('source, profile, prior or module differs from the saved local check')
            mapped = project_local_native_to_rgb(q, pair_check['registration'])
            observed_module = np.asarray(pair_check['target_module']['rgb_polygon'], float)
            checked = True
        except (ValueError, TypeError, KeyError) as exc:
            local_note = ' · local correction withheld: '+str(exc)
    if ((mapped < 0).any() or (mapped[:, 0] >= image.width).any()
            or (mapped[:, 1] >= image.height).any()):
        return unavailable('projected candidate is outside the RGB frame')
    draw = ImageDraw.Draw(image)
    module = None
    if module_polygon is not None:
        module = observed_module if checked else quadratic_points(module_polygon, model)
        draw.line([*map(tuple, module), tuple(module[0])], fill='#00ffff', width=4)
    outline = (mapped.mean(axis=0)+DISPLAY_OUTLINE_SCALE*(mapped-mapped.mean(axis=0)) if checked
               else quadratic_points(centre + DISPLAY_OUTLINE_SCALE*(q-centre), model))
    draw.line([*map(tuple, outline), tuple(outline[0])], fill='#ff2020', width=5)
    # A conspicuous amber label makes the provisional visual prior distinguishable
    # from a source-pair registration that actually passed local correspondence checks.
    low, high = outline.min(axis=0), outline.max(axis=0)
    anchor = (int(np.clip(low[0], 0, image.width-220)), int(np.clip(low[1]-30, 0, image.height-24)))
    draw.text(anchor, 'MODULE-EDGE CHECKED (PROVISIONAL)' if checked else 'APPROXIMATE RGB LOCATION',
              fill='#ffcc33', stroke_width=2, stroke_fill='black')
    # Keep at least a whole-module-sized view, instead of hiding context in a tiny crop.
    low = np.minimum(low, mapped.mean(axis=0)-[500, 500])
    high = np.maximum(high, mapped.mean(axis=0)+[500, 500])
    bounds = (*np.maximum(0, np.floor(low-180)).astype(int),
              *np.minimum(image.size, np.ceil(high+180)).astype(int))
    if checked:
        return RGBPreview(image.crop(bounds), True,
            f'Red: module-edge-checked projection ({DISPLAY_OUTLINE_SCALE}× display outline) · cyan: observed RGB module · '
            'held-out image checks passed; physical correspondence and fault status remain unverified',
            'image_checked_provisional', tuple(map(tuple, outline)),
            tuple(map(tuple, module)) if module is not None else None)
    return RGBPreview(image.crop(bounds), True,
        f'Red: approximate projected anomaly area ({DISPLAY_OUTLINE_SCALE}× display outline) · '
        'cyan: projected provisional module · local alignment unverified; not a panel-ID or temperature measurement'+local_note,
        'approximate', tuple(map(tuple, outline)),
        tuple(map(tuple, module)) if module is not None else None)
