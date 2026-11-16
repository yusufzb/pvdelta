"""Opt-in streamed diagnostics for actual native line-search decisions.

The sink receives detached records, never arrays or proposals owned by the
extractor. References do not enter the search. Consumers must bound their own
storage; exceptions abort the audited run rather than silently losing evidence.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

import numpy as np

_active = ContextVar('native_search_trace', default=None)


@dataclass
class _Trace:
    root: np.ndarray
    sink: object
    count: int = 0

    def begin(self, values, *, support_mode, local_contrast, pair_budget):
        array = np.asarray(values)
        if not np.shares_memory(array, self.root) or array.strides != self.root.strides:
            raise ValueError('Audited search must use native root or an unresampled window')
        offset = array.ctypes.data-self.root.ctypes.data
        y, remainder = divmod(offset, self.root.strides[0])
        x, remainder = divmod(remainder, self.root.strides[1])
        if remainder or min(x, y) < 0 or y+array.shape[0] > self.root.shape[0] or x+array.shape[1] > self.root.shape[1]:
            raise ValueError('Invalid native trace window')
        self.count += 1
        info = {'pass_id': self.count, 'window_xyxy': [int(x), int(y), int(x+array.shape[1]), int(y+array.shape[0])],
                'support_mode': support_mode, 'local_contrast': local_contrast, 'pair_budget': pair_budget}
        self.sink(dict(event='pass_begin', **dict(info, window_xyxy=list(info['window_xyxy']))))
        return _Pass(self.sink, info)


@dataclass
class _Pass:
    sink: object
    info: dict

    def candidate(self, stage, quad, support, **extra):
        offset = np.asarray(self.info['window_xyxy'][:2])
        self.sink(dict(event='candidate', pass_id=self.info['pass_id'], stage=stage,
            polygon_native_xy=(np.asarray(quad)+offset).tolist(),
            support=[[bool(s[0]), float(s[1]), float(s[2])] for s in support], **extra))

    def geometry_batch(self, gates):
        """Count actual pre-support tests without storing millions of polygons.

        Failures overlap. ``first_failure`` partitions rejected intersections in
        test order; ``sole_failure`` is only a pre-support counterfactual, never
        evidence that a panel would survive the later support/selection gates.
        """
        masks = [(name, np.asarray(mask, bool)) for name, mask in gates]
        remaining = np.ones(masks[0][1].shape, bool)
        failures = sum((~mask).astype(np.int8) for _, mask in masks)
        counts = {}
        for name, mask in masks:
            counts[name] = {'failed': int((~mask).sum()),
                           'first_failure': int((remaining & ~mask).sum()),
                           'sole_failure': int(((failures == 1) & ~mask).sum())}
            remaining &= mask
        self.sink({'event': 'geometry_batch', 'pass_id': self.info['pass_id'],
                   'pair_intersections': int(remaining.size),
                   'passed_geometry': int(remaining.sum()), 'gates': counts})

    def contour(self, stage, quad, **metrics):
        offset = np.asarray(self.info['window_xyxy'][:2])
        self.sink({'event': 'contour_candidate', 'pass_id': self.info['pass_id'],
                   'stage': stage, 'polygon_native_xy': (np.asarray(quad)+offset).tolist(),
                   **metrics})

    def contour_summary(self, counts):
        self.sink({'event': 'contour_summary', 'pass_id': self.info['pass_id'],
                   'counts': dict(counts)})

    def finish(self, result):
        self.sink({'event': 'pass_end', 'pass_id': self.info['pass_id'],
            'pair_search': dict(result['pair_search']), 'line_status': result['line_status'],
            'counters': dict(result['line_diagnostics'])})


def begin_trace(values, **info):
    active = _active.get()
    return None if active is None else active.begin(values, **info)


@contextmanager
def trace_native_search(temperature_c, sink):
    """Stream exact gate decisions; no effect on search limits or selection.

    A C-contiguous, finite native numeric root and callable sink are required.
    Nested tracing is rejected. Source changes fail the run on context exit.
    The callback may retain/modify its detached dictionaries, not source data.
    Pair-budget skips are explicit pass events. Geometry batches count early
    filters; contour events cover area-admissible four-corner observations.
    These diagnostics do not establish semantic panel membership.
    """
    import hashlib

    root = np.asarray(temperature_c)
    if (root.ndim != 2 or min(root.shape, default=0) < 1 or root.dtype.kind not in 'fiu'
            or not root.flags.c_contiguous or not np.isfinite(root).all()):
        raise ValueError('Finite contiguous native 2D numeric root required')
    if not callable(sink):
        raise TypeError('A callable trace sink is required')
    if _active.get() is not None:
        raise RuntimeError('Nested native search tracing is not supported')
    digest = hashlib.sha256(root.tobytes()).digest()
    token = _active.set(_Trace(root, sink))
    try:
        yield
    finally:
        _active.reset(token)
        if hashlib.sha256(root.tobytes()).digest() != digest:
            raise RuntimeError('Native grid changed during traced search')
