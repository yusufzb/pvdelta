"""Strict JSON helpers for the public data contracts."""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any

import numpy as np

from pvdelta.models import (
    AssetMatch,
    CameraProfile,
    LocationEstimate,
    LocationStatus,
    MatchStatus,
    ThermalFrame,
)


def _json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dt.datetime):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if is_dataclass(value):
        return {key: _json_value(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("NaN and Inf are not valid contract values")
    return value


def dumps_strict(value: Any) -> str:
    """Serialize without silently emitting non-standard NaN/Infinity tokens."""
    return json.dumps(_json_value(value), allow_nan=False, ensure_ascii=False, sort_keys=True)


def thermal_frame_to_dict(frame: ThermalFrame) -> dict[str, Any]:
    return _json_value(frame)


def thermal_frame_from_dict(data: dict[str, Any]) -> ThermalFrame:
    profile = data.get("camera_profile")
    capture = data.get("capture_time")
    mask = data.get("valid_mask")
    return ThermalFrame(
        frame_id=data["frame_id"],
        source_path=data["source_path"],
        source_hash=data["source_hash"],
        temperature_c=np.asarray(data["temperature_c"], dtype=np.float32),
        valid_mask=None if mask is None else np.asarray(mask, dtype=np.bool_),
        width=int(data["width"]),
        height=int(data["height"]),
        capture_time=None if capture is None else dt.datetime.fromisoformat(capture),
        camera_profile=None if profile is None else CameraProfile(**profile),
        raw_metadata=dict(data.get("raw_metadata", {})),
    )


def location_from_dict(data: dict[str, Any]) -> LocationEstimate:
    return LocationEstimate(
        observation_id=data["observation_id"],
        status=LocationStatus(data.get("status", LocationStatus.NOT_ATTEMPTED.value)),
        latitude=data.get("latitude"), longitude=data.get("longitude"),
        horizontal_accuracy_m=data.get("horizontal_accuracy_m"),
        method=data.get("method"),
    )


def asset_match_from_dict(data: dict[str, Any]) -> AssetMatch:
    return AssetMatch(
        observation_id=data["observation_id"],
        status=MatchStatus(data.get("status", MatchStatus.NOT_ATTEMPTED.value)),
        asset_id=data.get("asset_id"), ranking_score=data.get("ranking_score"),
        candidate_asset_ids=tuple(data.get("candidate_asset_ids", ())),
        evidence=data.get("evidence"),
    )
