"""Time- and scope-aware environmental/electrical condition matching."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from pvdelta.normalize import Basis, Condition


@dataclass(frozen=True)
class ConditionObservation:
    condition: Condition
    observed_at: dt.datetime
    source: str
    scope_id: str
    usable_for_normalization: bool = True
    rejection_reason: str | None = None

    def __post_init__(self) -> None:
        if self.observed_at.tzinfo is None:
            raise ValueError("condition time must include a timezone")
        if not self.source.strip() or not self.scope_id.strip():
            raise ValueError("condition source and scope must not be empty")
        if not self.usable_for_normalization and not self.rejection_reason:
            raise ValueError("unusable conditions require a rejection reason")


@dataclass(frozen=True)
class ConditionMatch:
    conditions: tuple[Condition, ...]
    sources: tuple[str, ...]
    flags: tuple[str, ...]


def match_conditions(
    capture_time: dt.datetime | None,
    observations: list[ConditionObservation],
    *,
    scope_id: str,
    max_age: dt.timedelta,
) -> ConditionMatch:
    if max_age.total_seconds() < 0:
        raise ValueError("max_age must be non-negative")
    if capture_time is None or capture_time.tzinfo is None:
        return ConditionMatch((), (), ("normalization unavailable: capture time is missing",))
    candidates: dict[Basis, list[tuple[dt.timedelta, ConditionObservation]]] = {}
    flags = []
    for observation in observations:
        if observation.scope_id != scope_id:
            continue
        age = abs(observation.observed_at - capture_time)
        if age > max_age:
            flags.append(f"{observation.condition.basis.value}: condition is stale")
            continue
        if not observation.usable_for_normalization:
            flags.append(
                f"{observation.condition.basis.value}: {observation.rejection_reason}"
            )
            continue
        candidates.setdefault(observation.condition.basis, []).append((age, observation))
    accepted: dict[Basis, ConditionObservation] = {}
    for basis, choices in candidates.items():
        nearest_age = min(age for age, _ in choices)
        nearest = [item for age, item in choices if age == nearest_age]
        if len(nearest) == 1:
            accepted[basis] = nearest[0]
        else:
            flags.append(f"{basis.value}: equally close duplicate condition")
    ordered = tuple(accepted[basis] for basis in Basis if basis in accepted)
    if not ordered:
        flags.append("normalization unavailable: no usable time- and scope-matched condition")
    return ConditionMatch(
        tuple(item.condition for item in ordered),
        tuple(item.source for item in ordered),
        tuple(flags),
    )
