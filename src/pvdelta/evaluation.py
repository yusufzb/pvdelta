"""Evaluation metrics for labelled module and region candidates.

The evaluator is intentionally label-driven. A review-required thermal candidate is not a
ground-truth fault label and must be passed to this module only after a held-out reference set
has been frozen.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass


@dataclass(frozen=True)
class DetectionEvaluation:
    """Counts and derived rates for one frozen evaluation population."""

    evaluated_count: int
    reference_count: int
    predicted_count: int
    true_positive: int
    false_positive: int
    false_negative: int
    precision: float | None
    recall: float | None
    false_alarms_per_100: float | None
    coverage: float


def evaluate_keyed_detections(
    predicted_keys: Collection[str],
    reference_keys: Collection[str],
    *,
    evaluated_keys: Collection[str] | None = None,
) -> DetectionEvaluation:
    """Evaluate exact keyed detections against frozen reference labels.

    ``evaluated_keys`` identifies the module/region population for which the detector was
    actually allowed to make a decision. It is separate from reference labels so unsupported
    or rejected-transfer cells do not silently become true negatives.
    """
    predicted = _keys(predicted_keys, "predicted_keys")
    reference = _keys(reference_keys, "reference_keys")
    evaluated = _keys(evaluated_keys, "evaluated_keys") if evaluated_keys is not None else predicted | reference
    outside = predicted - evaluated
    if outside:
        raise ValueError(f"predicted keys are outside evaluated population: {sorted(outside)!r}")
    assessed_reference = reference & evaluated
    true_positive = len(predicted & assessed_reference)
    false_positive = len(predicted - reference)
    false_negative = len(assessed_reference - predicted)
    evaluated_count = len(evaluated)
    return DetectionEvaluation(
        evaluated_count=evaluated_count,
        reference_count=len(reference),
        predicted_count=len(predicted),
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=false_negative,
        precision=(true_positive / len(predicted)) if predicted else None,
        recall=(true_positive / len(assessed_reference)) if assessed_reference else None,
        false_alarms_per_100=(false_positive * 100 / evaluated_count) if evaluated_count else None,
        coverage=(len(assessed_reference) / len(reference)) if reference else 0.0,
    )


def _keys(values: Collection[str] | None, name: str) -> set[str]:
    if values is None:
        raise ValueError(f"{name} must not be None")
    result = set(values)
    if any(not isinstance(value, str) or not value.strip() for value in result):
        raise ValueError(f"{name} must contain non-empty strings")
    return result
