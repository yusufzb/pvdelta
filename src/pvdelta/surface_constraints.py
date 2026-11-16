"""Explicit application-wide, source-bound native measurement exclusions."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import numpy as np

from pvdelta.models import ThermalFrame
from pvdelta.region_polygon import simple_polygon_mask


def declared_surface_masks(frame, reference, meaning, native_hash):
    if (meaning != 'panel_surface_and_non_panel' or not isinstance(reference,dict)
            or reference.get('review_complete') is not True
            or reference.get('source_sha256') != frame.source_hash
            or reference.get('native_temperature_sha256') != native_hash
            or [reference.get('native_height'),reference.get('native_width')] != [frame.height,frame.width]
            or not isinstance(reference.get('panel_polygons'),list)
            or not isinstance(reference.get('non_panel_polygons'),list)):
        raise ValueError('Complete source-bound declared panel/non-panel surface reference required')
    if hashlib.sha256(Path(frame.source_path).read_bytes()).hexdigest()!=frame.source_hash:
        raise ValueError('Original source differs from declared surface reference')
    panel=np.zeros(frame.temperature_c.shape,bool);outside=np.zeros_like(panel)
    for destination,polygons in ((panel,reference['panel_polygons']),(outside,reference['non_panel_polygons'])):
        for polygon in polygons:
            destination |= simple_polygon_mask(polygon['polygon_native_xy'],panel.shape)
    return panel & ~outside,outside,~(panel|outside)


@lru_cache(maxsize=4)
def _load(path, modified_ns, size):
    # File stat is part of the cache key. Re-read/hash again for each new revision.
    from pvdelta.surface_recovery import load_surface_manifest

    manifest,digest=load_surface_manifest(path)
    return digest,{item['surface_reference']['source_sha256']:item['surface_reference']
                   for item in manifest['frames']}


def configured_surface_index(manifest_path=None):
    configured=manifest_path or os.environ.get('PVDELTA_SURFACE_MANIFEST')
    if not configured:return None,{}
    path=Path(configured).expanduser().resolve()
    stat=path.stat()
    return _load(str(path),stat.st_mtime_ns,stat.st_size)


def surface_engine_binding():
    digest,_=configured_surface_index()
    return {'configured_surface_manifest':digest} if digest else {}


def apply_surface_reference(frame,reference,manifest_sha256):
    native_hash=hashlib.sha256(frame.temperature_c.tobytes()).hexdigest()
    reference_hash=hashlib.sha256(json.dumps(reference,sort_keys=True,allow_nan=False).encode()).hexdigest()
    old=frame.raw_metadata.get('pvdelta_surface_constraint')
    if old is not None:
        if (old.get('manifest_sha256')!=manifest_sha256 or old.get('source_sha256')!=frame.source_hash
                or old.get('native_temperature_sha256')!=native_hash
                or old.get('reference_sha256')!=reference_hash
                or frame.valid_mask is None
                or old.get('valid_mask_sha256')!=hashlib.sha256(frame.valid_mask.tobytes()).hexdigest()):
            raise ValueError('Surface constraint changed; decode the original source again')
        return frame
    _,outside,unknown=declared_surface_masks(frame,reference,'panel_surface_and_non_panel',native_hash)
    permitted=~outside
    if frame.valid_mask is not None:permitted &= frame.valid_mask
    permitted.setflags(write=False)
    binding={'manifest_sha256':manifest_sha256,'source_sha256':frame.source_hash,
        'native_temperature_sha256':native_hash,'valid_mask_sha256':hashlib.sha256(permitted.tobytes()).hexdigest(),
        'reference_sha256':reference_hash,
        'method':'explicit_non_panel_and_conflict_veto','unlabelled_pixels_preserved':True,
        'unlabelled_pixels':int(unknown.sum()),'independent_accuracy_verified':False}
    return replace(frame,valid_mask=permitted,raw_metadata={**frame.raw_metadata,
        'pvdelta_surface_constraint':binding,
        'pvdelta_surface_reference':json.loads(json.dumps(reference,allow_nan=False))})


def apply_configured_surface(frame,manifest_path=None):
    if not isinstance(frame,ThermalFrame):
        return frame
    attached=(frame.raw_metadata or {}).get('pvdelta_surface_constraint')
    if manifest_path is None and attached is not None:
        return apply_surface_reference(frame,frame.raw_metadata['pvdelta_surface_reference'],
                                       attached['manifest_sha256'])
    digest,index=configured_surface_index(manifest_path)
    if digest is None:return frame
    reference=index.get(frame.source_hash)
    if reference is None:
        if frame.raw_metadata.get('pvdelta_surface_constraint'):
            raise ValueError('Surface source reference changed; decode the original source again')
        return frame
    return apply_surface_reference(frame,reference,digest)
