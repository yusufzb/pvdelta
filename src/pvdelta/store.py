"""Local DuckDB storage for private inspection metadata and derived results.

The database is an analysis index. Original images and dense temperature arrays stay as
external local files referenced by URI and hash; they are deliberately not packed into the
database or public repository.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from pvdelta.inventory import PanelAsset, validate_panel_assets
from pvdelta.measurement_profile import verify_measurement_profile
from pvdelta.models import AssetMatch
from pvdelta.report import ModuleReport
from pvdelta.view_selection import RankedView

SCHEMA_VERSION = "8"


@dataclass(frozen=True)
class CaptureRecord:
    """One image-level record suitable for the local inspection index."""

    capture_id: str
    source_path: str
    source_hash: str
    image_kind: str
    run_id: str | None = None
    capture_time_utc: dt.datetime | None = None
    paired_capture_id: str | None = None
    camera_model: str | None = None
    width_px: int | None = None
    height_px: int | None = None
    gps_status: str | None = None
    rtk_flag: str | None = None
    camera_latitude: float | None = None
    camera_longitude: float | None = None
    absolute_altitude_m: float | None = None
    relative_altitude_m: float | None = None
    gimbal_pitch_deg: float | None = None
    gimbal_yaw_deg: float | None = None
    gimbal_roll_deg: float | None = None
    laser_target_distance_m: float | None = None
    laser_status: str | None = None
    laser_target_latitude: float | None = None
    laser_target_longitude: float | None = None
    laser_target_absolute_altitude_m: float | None = None
    focal_length_mm: float | None = None
    gain_mode: str | None = None
    emissivity: float | None = None
    atmospheric_temperature_c: float | None = None
    reflected_apparent_temperature_c: float | None = None
    relative_humidity_pct: float | None = None
    object_distance_m: float | None = None
    temperature_array_uri: str | None = None
    temperature_min_c: float | None = None
    temperature_max_c: float | None = None
    temperature_mean_c: float | None = None
    temperature_median_c: float | None = None
    footprint_geojson: dict[str, Any] | None = None
    footprint_method: str | None = None
    footprint_accuracy_m: float | None = None
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.capture_id.strip() or not self.source_hash.strip():
            raise ValueError("capture_id and source_hash must not be empty")
        if self.image_kind not in {"thermal", "rgb"}:
            raise ValueError("image_kind must be thermal or rgb")
        if self.capture_time_utc is not None and (
            self.capture_time_utc.tzinfo is None
            or self.capture_time_utc.utcoffset() != dt.timedelta(0)
        ):
            raise ValueError("capture_time_utc must be timezone-aware UTC")
        if (self.camera_latitude is None) != (self.camera_longitude is None):
            raise ValueError("camera coordinates must be provided together")
        if (self.laser_target_latitude is None) != (self.laser_target_longitude is None):
            raise ValueError("laser target coordinates must be provided together")
        if self.camera_latitude is not None and not -90 <= self.camera_latitude <= 90:
            raise ValueError("camera latitude is outside [-90, 90]")
        if self.camera_longitude is not None and not -180 <= self.camera_longitude <= 180:
            raise ValueError("camera longitude is outside [-180, 180]")
        if self.laser_target_latitude is not None and not -90 <= self.laser_target_latitude <= 90:
            raise ValueError("laser target latitude is outside [-90, 90]")
        if self.laser_target_longitude is not None and not -180 <= self.laser_target_longitude <= 180:
            raise ValueError("laser target longitude is outside [-180, 180]")
        if self.footprint_geojson is not None:
            if not self.footprint_method or not self.footprint_method.strip():
                raise ValueError("footprint_method is required with footprint_geojson")
            if self.footprint_accuracy_m is None or self.footprint_accuracy_m <= 0:
                raise ValueError("positive footprint_accuracy_m is required with footprint_geojson")
        for name, value in asdict(self).items():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"{name} must be finite")


@dataclass(frozen=True)
class Stage1StoreSummary:
    analysis_run_id: str
    capture_id: str
    module_observations: int
    findings: int
    review_required_findings: int


def _duckdb():
    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError("DuckDB support requires: pip install -e '.[storage]'") from exc
    return duckdb


def connect_store(path: str | Path, *, read_only: bool = False):
    """Open a local store and create its schema unless it is read-only."""
    duckdb = _duckdb()
    database_path = Path(path)
    if not read_only:
        database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(str(database_path), read_only=read_only)
    if not read_only:
        initialize_schema(connection)
    return connection


def initialize_schema(connection: Any) -> None:
    """Create or migrate the local schema without downloading DuckDB extensions."""
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS store_metadata (
            key VARCHAR PRIMARY KEY,
            value VARCHAR NOT NULL
        );
        CREATE TABLE IF NOT EXISTS inspection_runs (
            run_id VARCHAR PRIMARY KEY,
            site_label VARCHAR,
            started_at_utc TIMESTAMPTZ,
            camera_model VARCHAR,
            source_root VARCHAR,
            created_at_utc TIMESTAMPTZ DEFAULT current_timestamp
        );

        CREATE TABLE IF NOT EXISTS captures (
            capture_id VARCHAR PRIMARY KEY,
            run_id VARCHAR,
            source_path VARCHAR NOT NULL,
            source_hash VARCHAR NOT NULL,
            image_kind VARCHAR NOT NULL CHECK (image_kind IN ('thermal', 'rgb')),
            capture_time_utc TIMESTAMPTZ,
            paired_capture_id VARCHAR,
            camera_model VARCHAR,
            width_px INTEGER,
            height_px INTEGER,
            gps_status VARCHAR,
            rtk_flag VARCHAR,
            camera_latitude DOUBLE,
            camera_longitude DOUBLE,
            absolute_altitude_m DOUBLE,
            relative_altitude_m DOUBLE,
            gimbal_pitch_deg DOUBLE,
            gimbal_yaw_deg DOUBLE,
            gimbal_roll_deg DOUBLE,
            laser_target_distance_m DOUBLE,
            laser_status VARCHAR,
            laser_target_latitude DOUBLE,
            laser_target_longitude DOUBLE,
            laser_target_absolute_altitude_m DOUBLE,
            focal_length_mm DOUBLE,
            gain_mode VARCHAR,
            emissivity DOUBLE,
            atmospheric_temperature_c DOUBLE,
            reflected_apparent_temperature_c DOUBLE,
            relative_humidity_pct DOUBLE,
            object_distance_m DOUBLE,
            temperature_array_uri VARCHAR,
            temperature_min_c DOUBLE,
            temperature_max_c DOUBLE,
            temperature_mean_c DOUBLE,
            temperature_median_c DOUBLE,
            footprint_geojson JSON,
            footprint_method VARCHAR,
            footprint_accuracy_m DOUBLE,
            warnings_json JSON NOT NULL DEFAULT '[]',
            created_at_utc TIMESTAMPTZ DEFAULT current_timestamp
        );

        CREATE TABLE IF NOT EXISTS module_observations (
            observation_id VARCHAR PRIMARY KEY,
            capture_id VARCHAR NOT NULL,
            frame_module_id VARCHAR NOT NULL,
            pixel_polygon_json JSON NOT NULL,
            visibility VARCHAR NOT NULL,
            temperature_min_c DOUBLE,
            temperature_max_c DOUBLE,
            temperature_mean_c DOUBLE,
            temperature_median_c DOUBLE,
            estimated_latitude DOUBLE,
            estimated_longitude DOUBLE,
            location_method VARCHAR,
            location_accuracy_m DOUBLE,
            quality_flags_json JSON NOT NULL DEFAULT '[]'
        );

        CREATE TABLE IF NOT EXISTS findings (
            finding_id VARCHAR PRIMARY KEY,
            observation_id VARCHAR NOT NULL,
            anomaly_type VARCHAR NOT NULL,
            assessment_status VARCHAR NOT NULL,
            delta_t_k DOUBLE NOT NULL,
            absolute_temperature_c DOUBLE NOT NULL,
            reference_temperature_c DOUBLE,
            confidence DOUBLE,
            pixel_polygon_json JSON NOT NULL,
            estimated_latitude DOUBLE,
            estimated_longitude DOUBLE,
            location_method VARCHAR,
            location_accuracy_m DOUBLE
        );

        CREATE TABLE IF NOT EXISTS finding_reviews (
            finding_id VARCHAR PRIMARY KEY,
            review_status VARCHAR NOT NULL CHECK (review_status IN ('ok', 'not_ok', 'wait')),
            reviewer VARCHAR,
            note VARCHAR,
            reviewed_at_utc TIMESTAMPTZ DEFAULT current_timestamp
        );

        CREATE TABLE IF NOT EXISTS measurement_profiles (
            analysis_run_id VARCHAR NOT NULL,
            capture_id VARCHAR NOT NULL,
            profile_id VARCHAR,
            profile_json JSON,
            result_fingerprint VARCHAR NOT NULL,
            PRIMARY KEY (analysis_run_id, capture_id)
        );
        CREATE TABLE IF NOT EXISTS invalidated_finding_reviews (
            finding_id VARCHAR NOT NULL,
            analysis_run_id VARCHAR NOT NULL,
            capture_id VARCHAR NOT NULL,
            review_status VARCHAR NOT NULL,
            reviewer VARCHAR,
            note VARCHAR,
            reviewed_at_utc TIMESTAMPTZ,
            invalidated_at_utc TIMESTAMPTZ DEFAULT current_timestamp,
            previous_result_fingerprint VARCHAR,
            replacement_result_fingerprint VARCHAR NOT NULL,
            reason VARCHAR NOT NULL
        );

        CREATE TABLE IF NOT EXISTS native_boundary_snapshots (
            result_id VARCHAR PRIMARY KEY,
            source_sha256 VARCHAR NOT NULL,
            source_name VARCHAR NOT NULL,
            engine_signature VARCHAR NOT NULL,
            report_json VARCHAR NOT NULL,
            created_at_utc TIMESTAMPTZ DEFAULT current_timestamp
        );
        CREATE SEQUENCE IF NOT EXISTS native_boundary_review_seq START 1;
        CREATE TABLE IF NOT EXISTS native_boundary_review_events (
            event_id BIGINT PRIMARY KEY DEFAULT nextval('native_boundary_review_seq'),
            result_id VARCHAR NOT NULL REFERENCES native_boundary_snapshots(result_id),
            review_status VARCHAR NOT NULL CHECK (review_status IN ('ok', 'not_ok', 'wait')),
            reviewer VARCHAR,
            note VARCHAR,
            reviewed_at_utc TIMESTAMPTZ DEFAULT current_timestamp
        );
        CREATE TABLE IF NOT EXISTS native_boundary_review_reasons (
            event_id BIGINT NOT NULL REFERENCES native_boundary_review_events(event_id),
            reason_code VARCHAR NOT NULL,
            PRIMARY KEY (event_id, reason_code)
        );

        CREATE TABLE IF NOT EXISTS finding_evidence (
            finding_id VARCHAR PRIMARY KEY,
            evidence_group_id VARCHAR,
            panel_support_status VARCHAR NOT NULL,
            panel_support_ratio DOUBLE,
            panel_texture_std DOUBLE,
            projected_latitude DOUBLE,
            projected_longitude DOUBLE,
            repeat_capture_count INTEGER NOT NULL DEFAULT 0,
            repeat_finding_count INTEGER NOT NULL DEFAULT 0,
            evidence_method VARCHAR NOT NULL,
            evidence_notes VARCHAR,
            created_at_utc TIMESTAMPTZ DEFAULT current_timestamp
        );

        CREATE TABLE IF NOT EXISTS asset_matches (
            observation_id VARCHAR PRIMARY KEY,
            match_status VARCHAR NOT NULL,
            asset_id VARCHAR,
            ranking_score DOUBLE,
            candidate_asset_ids_json JSON NOT NULL DEFAULT '[]',
            evidence VARCHAR
        );

        CREATE TABLE IF NOT EXISTS panel_assets (
            panel_id VARCHAR PRIMARY KEY,
            table_id VARCHAR NOT NULL,
            row_label VARCHAR NOT NULL,
            position INTEGER NOT NULL,
            latitude DOUBLE,
            longitude DOUBLE,
            crs VARCHAR,
            string_id VARCHAR,
            created_at_utc TIMESTAMPTZ DEFAULT current_timestamp
        );

        CREATE TABLE IF NOT EXISTS view_rankings (
            ranking_id VARCHAR PRIMARY KEY,
            run_id VARCHAR NOT NULL,
            panel_key VARCHAR NOT NULL,
            rgb_capture_id VARCHAR NOT NULL,
            rgb_path VARCHAR NOT NULL,
            thermal_capture_id VARCHAR,
            latitude DOUBLE,
            longitude DOUBLE,
            horizontal_accuracy_m DOUBLE,
            ranking_score DOUBLE NOT NULL,
            sharpness DOUBLE NOT NULL,
            clipped_fraction DOUBLE NOT NULL,
            contrast DOUBLE NOT NULL,
            view_rank INTEGER NOT NULL,
            selection_status VARCHAR NOT NULL,
            created_at_utc TIMESTAMPTZ DEFAULT current_timestamp
        );
        """
    )
    # DuckDB has no multi-statement CREATE OR ALTER construct. These additions migrate a v1
    # store in place and remain idempotent for every later open.
    for statement in (
        "ALTER TABLE finding_evidence ADD COLUMN IF NOT EXISTS panel_texture_std DOUBLE",
        "ALTER TABLE module_observations ADD COLUMN IF NOT EXISTS analysis_run_id VARCHAR",
        "ALTER TABLE module_observations ADD COLUMN IF NOT EXISTS assessment_status VARCHAR",
        "ALTER TABLE module_observations ADD COLUMN IF NOT EXISTS assessment_reasons_json JSON DEFAULT '[]'",
        "ALTER TABLE findings ADD COLUMN IF NOT EXISTS analysis_run_id VARCHAR",
        "ALTER TABLE findings ADD COLUMN IF NOT EXISTS reference_method VARCHAR",
        "ALTER TABLE findings ADD COLUMN IF NOT EXISTS n_region_px INTEGER",
        "ALTER TABLE findings ADD COLUMN IF NOT EXISTS n_reference_px INTEGER",
        "ALTER TABLE findings ADD COLUMN IF NOT EXISTS normalized_delta_t_k DOUBLE",
        "ALTER TABLE findings ADD COLUMN IF NOT EXISTS normalization_basis VARCHAR",
        "ALTER TABLE findings ADD COLUMN IF NOT EXISTS normalization_sources_json JSON DEFAULT '[]'",
        "ALTER TABLE findings ADD COLUMN IF NOT EXISTS normalization_flags_json JSON DEFAULT '[]'",
        "ALTER TABLE findings ADD COLUMN IF NOT EXISTS region_pixels_yx_json JSON",
        "ALTER TABLE findings ADD COLUMN IF NOT EXISTS region_max_temperature_c DOUBLE",
        "ALTER TABLE findings ADD COLUMN IF NOT EXISTS region_p95_temperature_c DOUBLE",
        "ALTER TABLE findings ADD COLUMN IF NOT EXISTS region_max_residual_k DOUBLE",
        "ALTER TABLE findings ADD COLUMN IF NOT EXISTS delta_method VARCHAR",
        "ALTER TABLE asset_matches ADD COLUMN IF NOT EXISTS evidence VARCHAR",
    ):
        connection.execute(statement)
    connection.execute(
        """
        INSERT INTO store_metadata VALUES ('schema_version', ?)
        ON CONFLICT (key) DO UPDATE SET value = excluded.value
        """,
        [SCHEMA_VERSION],
    )


_CAPTURE_COLUMNS = tuple(field.name for field in CaptureRecord.__dataclass_fields__.values())


def upsert_run(
    connection: Any,
    *,
    run_id: str,
    site_label: str | None,
    started_at_utc: dt.datetime | None,
    camera_model: str | None,
    source_root: str,
) -> None:
    """Insert or update one local import run."""
    if not run_id.strip():
        raise ValueError("run_id must not be empty")
    connection.execute(
        """
        INSERT INTO inspection_runs
            (run_id, site_label, started_at_utc, camera_model, source_root)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT (run_id) DO UPDATE SET
            site_label = excluded.site_label,
            started_at_utc = excluded.started_at_utc,
            camera_model = excluded.camera_model,
            source_root = excluded.source_root
        """,
        [run_id, site_label, started_at_utc, camera_model, source_root],
    )


def upsert_panel_assets(
    connection: Any,
    assets: Iterable[PanelAsset],
) -> int:
    """Upsert validated optional Stage 2 panel identities into the local store.

    This function stores declared coordinates and CRS verbatim. It never interprets
    camera/LRF coordinates as panel centres and never performs an implicit CRS conversion.
    """
    items = tuple(assets)
    validation = validate_panel_assets(items)
    if validation.errors:
        raise ValueError("cannot store invalid panel inventory: " + "; ".join(validation.errors))
    connection.execute("BEGIN TRANSACTION")
    try:
        connection.executemany(
            """
            INSERT INTO panel_assets (
                panel_id, table_id, row_label, position,
                latitude, longitude, crs, string_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (panel_id) DO UPDATE SET
                table_id = excluded.table_id,
                row_label = excluded.row_label,
                position = excluded.position,
                latitude = excluded.latitude,
                longitude = excluded.longitude,
                crs = excluded.crs,
                string_id = excluded.string_id
            """,
            [
                (
                    asset.panel_id,
                    asset.table_id,
                    asset.row,
                    asset.position,
                    asset.latitude,
                    asset.longitude,
                    asset.crs,
                    asset.string_id,
                )
                for asset in items
            ],
        )
        connection.execute("COMMIT")
    except Exception:
        connection.execute("ROLLBACK")
        raise
    return len(items)


def read_panel_assets(connection: Any) -> tuple[PanelAsset, ...]:
    """Read the declared Stage 2 inventory without applying coordinate assumptions."""
    rows = connection.execute(
        """
        SELECT panel_id, table_id, row_label, position,
               latitude, longitude, crs, string_id
        FROM panel_assets
        ORDER BY table_id, row_label, position, panel_id
        """
    ).fetchall()
    return tuple(PanelAsset(*row) for row in rows)


def update_capture_radiometry(connection: Any, result: Any) -> None:
    """Persist a native radiometry summary by source hash, never a resampled display array."""
    exists = connection.execute(
        "SELECT count(*) FROM captures WHERE source_hash = ? AND image_kind = 'thermal'",
        [result.source_hash],
    ).fetchone()[0]
    if exists != 1:
        raise KeyError(f"unknown thermal capture source hash: {result.source_hash}")
    connection.execute(
        """
        UPDATE captures
        SET temperature_array_uri = NULL,
            temperature_min_c = ?, temperature_max_c = ?,
            temperature_mean_c = ?, temperature_median_c = ?,
            emissivity = ?, object_distance_m = ?,
            atmospheric_temperature_c = ?, reflected_apparent_temperature_c = ?,
            width_px = ?, height_px = ?
        WHERE source_hash = ? AND image_kind = 'thermal'
        """,
        [
            result.temperature_min_c, result.temperature_max_c,
            result.temperature_mean_c, result.temperature_median_c,
            result.emissivity, result.distance_m,
            result.ambient_temp_c, result.reflected_temp_c,
            result.width_px, result.height_px, result.source_hash,
        ],
    )


def write_asset_matches(connection: Any, matches: Iterable[AssetMatch]) -> int:
    """Upsert Stage 2 match decisions without changing Stage 1 observations."""
    items = tuple(matches)
    connection.execute("BEGIN TRANSACTION")
    try:
        connection.executemany(
            """
            INSERT INTO asset_matches (
                observation_id, match_status, asset_id, ranking_score, candidate_asset_ids_json,
                evidence
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT (observation_id) DO UPDATE SET
                match_status = excluded.match_status,
                asset_id = excluded.asset_id,
                ranking_score = excluded.ranking_score,
                candidate_asset_ids_json = excluded.candidate_asset_ids_json,
                evidence = excluded.evidence
            """,
            [
                (
                    item.observation_id,
                    item.status.value,
                    item.asset_id,
                    item.ranking_score,
                    json.dumps(item.candidate_asset_ids, ensure_ascii=False),
                    item.evidence,
                )
                for item in items
            ],
        )
        connection.execute("COMMIT")
    except Exception:
        connection.execute("ROLLBACK")
        raise
    return len(items)


def upsert_capture(connection: Any, record: CaptureRecord) -> None:
    """Insert or replace a capture record using explicit, stable column names."""
    values = asdict(record)
    values["footprint_geojson"] = (
        None
        if record.footprint_geojson is None
        else json.dumps(record.footprint_geojson, allow_nan=False, separators=(",", ":"))
    )
    values["warnings"] = json.dumps(record.warnings, ensure_ascii=False)
    database_columns = [
        "warnings_json" if column == "warnings" else column for column in _CAPTURE_COLUMNS
    ]
    placeholders = ", ".join("?" for _ in database_columns)
    assignments = ", ".join(
        f"{column} = excluded.{column}" for column in database_columns if column != "capture_id"
    )
    connection.execute(
        f"""
        INSERT INTO captures ({', '.join(database_columns)}) VALUES ({placeholders})
        ON CONFLICT (capture_id) DO UPDATE SET {assignments}
        """,
        [values[column] for column in _CAPTURE_COLUMNS],
    )


def write_stage1_results(
    connection: Any,
    *,
    analysis_run_id: str,
    capture_id: str,
    module_results: Iterable[ModuleReport],
    measurement_profile: dict | None = None,
) -> Stage1StoreSummary:
    """Replace one capture's Stage 1 results for an analysis run atomically.

    Dense temperature matrices stay external. Exact candidate support is stored as compact JSON
    because it is sparse evidence, while the capture keeps the external matrix URI.
    """
    if not analysis_run_id.strip() or not capture_id.strip():
        raise ValueError("analysis_run_id and capture_id must not be empty")
    modules = tuple(module_results)
    module_ids = [item.module.module_id for item in modules]
    if len(module_ids) != len(set(module_ids)):
        raise ValueError("module IDs must be unique within an analysis run and capture")
    if not connection.execute(
        "SELECT count(*) FROM inspection_runs WHERE run_id = ?", [analysis_run_id]
    ).fetchone()[0]:
        raise KeyError(f"unknown analysis_run_id: {analysis_run_id}")
    if not connection.execute(
        "SELECT count(*) FROM captures WHERE capture_id = ?", [capture_id]
    ).fetchone()[0]:
        raise KeyError(f"unknown capture_id: {capture_id}")

    if measurement_profile is not None:
        verify_measurement_profile(measurement_profile)
        capture_hash = connection.execute('SELECT source_hash FROM captures WHERE capture_id=?',
                                          [capture_id]).fetchone()[0]
        if measurement_profile['source_sha256'] != capture_hash:
            raise ValueError('Measurement profile source differs from the capture')
    from pvdelta.contracts import dumps_strict

    fingerprint = hashlib.sha256(dumps_strict({'modules': modules,
        'measurement_profile': measurement_profile}).encode()).hexdigest()

    finding_count = sum(len(item.findings) for item in modules)
    review_required = sum(
        finding.assessment_status == "review_required"
        for item in modules
        for finding in item.findings
    )
    connection.execute("BEGIN TRANSACTION")
    try:
        previous = connection.execute('SELECT result_fingerprint FROM measurement_profiles '
            'WHERE analysis_run_id=? AND capture_id=?', [analysis_run_id, capture_id]).fetchone()
        if previous is None or previous[0] != fingerprint:
            old_findings = '''SELECT f.finding_id FROM findings f JOIN module_observations m
                ON f.observation_id=m.observation_id WHERE f.analysis_run_id=? AND m.capture_id=?'''
            connection.execute(f'''INSERT INTO invalidated_finding_reviews
                (finding_id,analysis_run_id,capture_id,review_status,reviewer,note,reviewed_at_utc,
                 previous_result_fingerprint,replacement_result_fingerprint,reason)
                SELECT r.finding_id,?,?,r.review_status,r.reviewer,r.note,r.reviewed_at_utc,?,?,?
                FROM finding_reviews r WHERE r.finding_id IN ({old_findings})''',
                [analysis_run_id, capture_id, previous[0] if previous else None, fingerprint,
                 'Analysis results or measurement profile changed; review no longer applies',
                 analysis_run_id, capture_id])
            for table in ('finding_reviews', 'finding_evidence'):
                connection.execute(f'DELETE FROM {table} WHERE finding_id IN ({old_findings})',
                                   [analysis_run_id, capture_id])
        connection.execute(
            """
            DELETE FROM findings
            WHERE observation_id IN (
                SELECT observation_id FROM module_observations
                WHERE analysis_run_id = ? AND capture_id = ?
            )
            """,
            [analysis_run_id, capture_id],
        )
        connection.execute(
            "DELETE FROM module_observations WHERE analysis_run_id = ? AND capture_id = ?",
            [analysis_run_id, capture_id],
        )
        for result in modules:
            module = result.module
            observation_id = f"{analysis_run_id}::{capture_id}::{module.module_id}"
            connection.execute(
                """
                INSERT INTO module_observations (
                    observation_id, capture_id, frame_module_id, pixel_polygon_json,
                    visibility, temperature_min_c, temperature_max_c, temperature_mean_c,
                    temperature_median_c, quality_flags_json, analysis_run_id,
                    assessment_status, assessment_reasons_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    observation_id,
                    capture_id,
                    module.module_id,
                    json.dumps(module.pixel_polygon, allow_nan=False, separators=(",", ":")),
                    module.visibility,
                    module.temp_min,
                    module.temp_max,
                    module.temp_mean,
                    module.temp_median,
                    json.dumps(module.quality_flags, ensure_ascii=False),
                    analysis_run_id,
                    result.assessment_status,
                    json.dumps(result.reasons, ensure_ascii=False),
                ],
            )
            for finding in result.findings:
                database_finding_id = f"{analysis_run_id}::{capture_id}::{finding.finding_id}"
                exact_support = (
                    None
                    if finding.region_pixels_yx is None
                    else json.dumps(finding.region_pixels_yx, separators=(",", ":"))
                )
                connection.execute(
                    """
                    INSERT INTO findings (
                        finding_id, observation_id, anomaly_type, assessment_status,
                        delta_t_k, absolute_temperature_c, reference_temperature_c,
                        confidence, pixel_polygon_json, analysis_run_id, reference_method,
                        n_region_px, n_reference_px, normalized_delta_t_k,
                        normalization_basis, normalization_sources_json,
                        normalization_flags_json, region_pixels_yx_json,
                        region_max_temperature_c, region_p95_temperature_c,
                        region_max_residual_k, delta_method
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        database_finding_id,
                        observation_id,
                        finding.anomaly_type,
                        finding.assessment_status,
                        finding.delta_t_c,
                        finding.absolute_t_c,
                        finding.reference_t_c,
                        finding.confidence,
                        json.dumps(finding.pixel_polygon, allow_nan=False, separators=(",", ":")),
                        analysis_run_id,
                        finding.reference_method,
                        finding.n_region_px,
                        finding.n_reference_px,
                        finding.normalized_delta_t_c,
                        finding.normalization_basis,
                        json.dumps(finding.normalization_sources, ensure_ascii=False),
                        json.dumps(finding.normalization_flags, ensure_ascii=False),
                        exact_support,
                        finding.region_max_t_c,
                        finding.region_p95_t_c,
                        finding.region_max_residual_k,
                        finding.delta_method,
                    ],
                )
        connection.execute('DELETE FROM measurement_profiles WHERE analysis_run_id=? AND capture_id=?',
                           [analysis_run_id, capture_id])
        connection.execute('INSERT INTO measurement_profiles VALUES (?,?,?,?,?)',
            [analysis_run_id, capture_id, measurement_profile['profile_id'] if measurement_profile else None,
             dumps_strict(measurement_profile) if measurement_profile else None, fingerprint])
        connection.execute("COMMIT")
    except Exception:
        connection.execute("ROLLBACK")
        raise
    return Stage1StoreSummary(
        analysis_run_id,
        capture_id,
        len(modules),
        finding_count,
        review_required,
    )


def upsert_finding_review(
    connection: Any,
    *,
    finding_id: str,
    review_status: str,
    reviewer: str | None = None,
    note: str | None = None,
) -> None:
    """Persist a human review without changing the detector's finding status."""
    if not finding_id.strip():
        raise ValueError("finding_id must not be empty")
    if review_status not in {"ok", "not_ok", "wait"}:
        raise ValueError("review_status must be ok, not_ok or wait")
    exists = connection.execute(
        "SELECT count(*) FROM findings WHERE finding_id = ?", [finding_id]
    ).fetchone()[0]
    if not exists:
        raise KeyError(f"unknown finding_id: {finding_id}")
    connection.execute(
        """
        INSERT INTO finding_reviews (finding_id, review_status, reviewer, note)
        VALUES (?, ?, ?, ?)
        ON CONFLICT (finding_id) DO UPDATE SET
            review_status = excluded.review_status,
            reviewer = excluded.reviewer,
            note = excluded.note,
            reviewed_at_utc = now()
        """,
        [finding_id, review_status, reviewer, note],
    )


def update_capture_footprint(
    connection: Any,
    *,
    capture_id: str,
    footprint_geojson: dict[str, Any],
    method: str,
    horizontal_accuracy_m: float,
) -> None:
    """Attach reviewed coverage geometry without replacing other capture metadata."""
    if not capture_id.strip():
        raise ValueError("capture_id must not be empty")
    if footprint_geojson.get("type") != "Polygon":
        raise ValueError("footprint_geojson must be a Polygon")
    if not method.strip():
        raise ValueError("footprint method must not be empty")
    if not math.isfinite(horizontal_accuracy_m) or horizontal_accuracy_m <= 0:
        raise ValueError("horizontal accuracy must be a positive finite number")
    exists = connection.execute(
        "SELECT count(*) FROM captures WHERE capture_id = ?", [capture_id]
    ).fetchone()[0]
    if not exists:
        raise KeyError(f"unknown capture_id: {capture_id}")
    payload = json.dumps(footprint_geojson, allow_nan=False, separators=(",", ":"))
    connection.execute(
        """
        UPDATE captures
        SET footprint_geojson = ?, footprint_method = ?, footprint_accuracy_m = ?
        WHERE capture_id = ?
        """,
        [payload, method, horizontal_accuracy_m, capture_id],
    )


def map_features(connection: Any) -> list[dict[str, Any]]:
    """Return local camera points and optional calibrated coverage polygons for mapping."""
    rows = connection.execute(
        """
        SELECT c.capture_id, c.source_path, c.image_kind,
               r.source_root,
               rgb.source_path AS rgb_source_path,
               CAST(c.capture_time_utc AS VARCHAR) AS capture_time_utc,
               c.camera_latitude, c.camera_longitude, c.laser_target_latitude,
               c.laser_target_longitude,
               c.relative_altitude_m, c.temperature_min_c, c.temperature_max_c,
               (SELECT count(*) FROM module_observations AS mo
                WHERE mo.capture_id = c.capture_id) AS module_observation_count,
               (SELECT count(*) FROM findings AS f
                JOIN module_observations AS fm USING (observation_id)
                WHERE fm.capture_id = c.capture_id) AS finding_count,
               CASE
                   WHEN c.footprint_method IS NOT NULL AND c.footprint_accuracy_m IS NOT NULL
                   THEN c.footprint_geojson
               END AS footprint_geojson,
               c.footprint_method, c.footprint_accuracy_m
        FROM captures AS c
        LEFT JOIN captures AS rgb ON rgb.capture_id = c.paired_capture_id
        LEFT JOIN inspection_runs AS r ON r.run_id = c.run_id
        WHERE c.camera_latitude IS NOT NULL AND c.camera_longitude IS NOT NULL
          AND c.image_kind = 'thermal'
        ORDER BY c.capture_time_utc NULLS LAST, c.capture_id
        """
    ).fetchall()
    columns = [item[0] for item in connection.description]
    features = []
    for row in rows:
        item = dict(zip(columns, row, strict=True))
        geometry = item.get("footprint_geojson")
        if isinstance(geometry, str):
            item["footprint_geojson"] = json.loads(geometry)
        features.append(item)
    return features


def write_view_rankings(
    connection: Any,
    *,
    run_id: str,
    rankings: Iterable[RankedView],
) -> int:
    """Atomically replace RGB view rankings for one import run.

    `panel_key` is supplied by an upstream reviewed matcher; this function never
    invents physical identity from filename similarity.
    """
    if not run_id.strip():
        raise ValueError("run_id must not be empty")
    items = tuple(rankings)
    if not connection.execute(
        "SELECT count(*) FROM inspection_runs WHERE run_id = ?", [run_id]
    ).fetchone()[0]:
        raise KeyError(f"unknown run_id: {run_id}")
    panel_keys = sorted({item.panel_key for item in items})
    connection.execute("BEGIN TRANSACTION")
    try:
        if panel_keys:
            connection.executemany(
                "DELETE FROM view_rankings WHERE run_id = ? AND panel_key = ?",
                [(run_id, key) for key in panel_keys],
            )
        for item in items:
            ranking_id = f"{run_id}::{item.panel_key}::{item.rgb_capture_id}"
            connection.execute(
                """
                INSERT INTO view_rankings (
                    ranking_id, run_id, panel_key, rgb_capture_id, rgb_path,
                    thermal_capture_id, latitude, longitude, horizontal_accuracy_m,
                    ranking_score, sharpness, clipped_fraction, contrast,
                    view_rank, selection_status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (ranking_id) DO UPDATE SET
                    rgb_path = excluded.rgb_path,
                    thermal_capture_id = excluded.thermal_capture_id,
                    latitude = excluded.latitude,
                    longitude = excluded.longitude,
                    horizontal_accuracy_m = excluded.horizontal_accuracy_m,
                    ranking_score = excluded.ranking_score,
                    sharpness = excluded.sharpness,
                    clipped_fraction = excluded.clipped_fraction,
                    contrast = excluded.contrast,
                    view_rank = excluded.view_rank,
                    selection_status = excluded.selection_status
                """,
                [
                    ranking_id, run_id, item.panel_key, item.rgb_capture_id, item.rgb_path,
                    item.thermal_capture_id, item.latitude, item.longitude,
                    item.horizontal_accuracy_m, item.ranking_score, item.sharpness,
                    item.clipped_fraction, item.contrast, item.rank, item.selection_status,
                ],
            )
        connection.execute("COMMIT")
    except Exception:
        connection.execute("ROLLBACK")
        raise
    return len(items)
