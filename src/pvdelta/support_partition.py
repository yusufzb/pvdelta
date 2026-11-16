"""Partition observed native envelopes without assigning ambiguous shared pixels."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pvdelta.masking import MeasurementSupport
from pvdelta.reference import erode


@dataclass(frozen=True)
class PartitionedSupport(MeasurementSupport):
    """Excluded support includes geometric edges and ownership uncertainty.

    ``excluded_ambiguous`` is the subset of excluded_edge that would otherwise
    belong to the eroded measurement interior. Native envelopes stay unchanged.
    This establishes unique pixel ownership only, never semantic panel identity.
    """

    excluded_ambiguous: np.ndarray


def partition_measurement_support(native_envelopes, *, margin_px: int):
    """Exclude every shared envelope pixel from every measurement mask.

    The strict disjoint-envelope builder is unchanged. This explicit alternative
    preserves all source envelopes and empty masks for review. No temperature,
    nearest-centre rule or arbitrary instance priority determines ownership.
    """
    if isinstance(margin_px, bool) or not isinstance(margin_px, int) or margin_px < 0:
        raise ValueError('margin_px must be a nonnegative integer')
    if not native_envelopes:
        return ()
    envelopes = [np.asarray(mask) for mask in native_envelopes]
    shape = envelopes[0].shape
    if len(shape) != 2 or min(shape) < 1:
        raise ValueError('native envelopes must have nonempty 2D shape')
    if any(mask.shape != shape or mask.dtype != np.bool_ or not mask.any() for mask in envelopes):
        raise ValueError('native envelopes must be nonempty boolean masks with matching shapes')
    envelopes = [mask.copy() for mask in envelopes]
    shared = np.sum(envelopes, axis=0, dtype=np.int32) > 1
    result = []
    for full in envelopes:
        eroded = erode(full, margin_px)
        interior = eroded & ~shared
        ambiguous = eroded & shared
        excluded = full & ~interior
        for mask in (full, interior, ambiguous, excluded):
            mask.setflags(write=False)
        result.append(PartitionedSupport(full, interior, excluded, margin_px, ambiguous))
    return tuple(result)
