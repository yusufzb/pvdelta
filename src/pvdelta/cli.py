"""Command-line entry point for the explicit-region Stage 1 workflow."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from pvdelta.batch import run_batch
from pvdelta.config import DetectionConfig
from pvdelta.decode import RJpeg
from pvdelta.inventory import load_inventory_csv, load_inventory_geojson, validate_panel_assets
from pvdelta.manifest import prepare_manifest
from pvdelta.metadata import parse_capture_metadata
from pvdelta.models import ThermalFrame
from pvdelta.pipeline import execute_stage1
from pvdelta.regions import load_annotations, load_mask_archive
from pvdelta.store import connect_store, upsert_panel_assets, write_view_rankings
from pvdelta.view_selection import (
    VIEWS_SCHEMA_VERSION,
    ViewCandidate,
    load_rgb_file,
    rank_view_candidates,
    ranked_view_to_dict,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decode_frame(path: Path, sdk_lib: str | None = None, *, measurement_overrides=None,
                 surface_manifest=None) -> ThermalFrame:
    from pvdelta.measurement_profile import validate_overrides

    overrides = validate_overrides(measurement_overrides)
    original_hash = _sha256(path)
    metadata = parse_capture_metadata(path)
    with RJpeg(path, sdk_lib) as image:
        recorded = image.measurement_params
        if overrides:
            image.set_measurement_params(replace(recorded, **overrides))
        temperature = image.temperature()
        parameters = image.measurement_params
    if _sha256(path) != original_hash:
        raise ValueError('Source changed during SDK analysis')
    frame=ThermalFrame(
        frame_id=path.stem,
        source_path=str(path.resolve()),
        source_hash=original_hash,
        temperature_c=temperature,
        valid_mask=np.isfinite(temperature),
        width=temperature.shape[1],
        height=temperature.shape[0],
        capture_time=metadata.exposure_time_utc,
        camera_profile=metadata.camera_profile,
        raw_metadata={
            "measurement_params": asdict(parameters),
            "recorded_measurement_params": asdict(recorded),
            "measurement_overrides": overrides,
            "sdk_parameter_readback_verified": True,
            "metadata_warnings": list(metadata.warnings),
        },
    )
    from pvdelta.surface_constraints import apply_configured_surface

    return apply_configured_surface(frame,surface_manifest)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pvdelta")
    commands = parser.add_subparsers(dest="command", required=True)
    analyze = commands.add_parser("analyze", help="Analyze one DJI thermal R-JPEG")
    analyze.add_argument("input", type=Path)
    analyze.add_argument("--regions", type=Path, required=True)
    analyze.add_argument(
        "--mask-archive",
        type=Path,
        help="Optional NPZ with module_labels and measurement arrays on the native grid",
    )
    analyze.add_argument("--out", type=Path, required=True)
    analyze.add_argument("--run-id", required=True)
    analyze.add_argument("--database", type=Path)
    analyze.add_argument("--site-label")
    analyze.add_argument("--sdk-lib")
    analyze.add_argument("--measurement-params", type=Path,
                         help="Optional JSON of explicit SDK corrections for this analysis")
    analyze.add_argument("--margin-px", type=int, default=0)
    analyze.add_argument("--delta-k", type=float, default=2.5)
    analyze.add_argument("--min-region-px", type=int, default=4)
    analyze.add_argument("--min-reference-px", type=int, default=16)
    analyze.add_argument("--edge-erosion-px", type=int, default=1)
    analyze.add_argument(
        "--min-valid-fraction",
        type=float,
        default=0.95,
        help="minimum finite thermal-pixel fraction required inside a module",
    )
    analyze.add_argument("--direction", choices=("hot", "cold", "both"), default="both")
    analyze.add_argument("--overwrite", action="store_true")
    sdk_check = commands.add_parser(
        "sdk-check", help="Verify libdirp and decode one private DJI radiometric frame"
    )
    sdk_check.add_argument("input", type=Path)
    sdk_check.add_argument("--sdk-lib", required=True)
    scan = commands.add_parser("scan-flight", help="Resumable SDK decode of every original thermal frame")
    scan.add_argument("flight", type=Path)
    scan.add_argument("--out", type=Path, required=True)
    scan.add_argument("--sdk-lib")
    scan.add_argument("--resume", action="store_true")
    scan.add_argument("--measurement-params", type=Path,
                      help="Optional JSON of explicit SDK parameter overrides; recorded values remain in the report")
    replay = commands.add_parser('replay-screening', help='Remeasure stored panel geometry into a separate inspection database')
    replay.add_argument('--database', type=Path, required=True)
    replay.add_argument('--baseline-run', required=True)
    replay.add_argument('--run-id', required=True)
    replay.add_argument('--out', type=Path, required=True)
    replay.add_argument('--sdk-lib')
    replay.add_argument('--site-geometry', type=Path, help='Optional user site prior for later repeat-view checks')
    replay.add_argument('--measurement-params', type=Path)
    replay.add_argument('--per-frame-params', type=Path,
                        help='JSON object mapping every exact source hash to explicit SDK overrides')
    native = commands.add_parser('scan-native-flight', help='Provisional native thermal boundaries and anomaly analysis into a new private database')
    native.add_argument('--database', type=Path, required=True)
    native.add_argument('--acquisition', required=True)
    native.add_argument('--run-id', required=True)
    native.add_argument('--out', type=Path, required=True)
    native.add_argument('--sdk-lib')
    native.add_argument('--workers', type=int, default=1)
    native.add_argument('--measurement-params', type=Path)
    native.add_argument('--surface-manifest',type=Path,
                        help='Source-bound declared non-panel veto; unreferenced frames remain unverified')
    control_review = commands.add_parser("prepare-control-review", help="Prepare unmarked original RGB/thermal control review")
    control_review.add_argument("--controls", type=Path, required=True)
    control_review.add_argument("--sources", type=Path, required=True)
    control_review.add_argument("--source-root", type=Path, required=True)
    control_review.add_argument("--out", type=Path, required=True)
    control_review.add_argument("--sdk-lib")
    coverage = commands.add_parser("coverage-audit", help="Audit one stored analysis run read-only")
    coverage.add_argument("--database", type=Path, required=True)
    coverage.add_argument("--run-id", required=True)
    coverage.add_argument("--out", type=Path, required=True)
    coverage.add_argument("--overwrite", action="store_true")
    export = commands.add_parser('export-inspection', help='Read-only inspection handover with coverage, reviews and source provenance')
    export.add_argument('--database', type=Path, required=True)
    export.add_argument('--run-id', required=True)
    export.add_argument('--out', type=Path, required=True, help='New private output directory')
    verify_export = commands.add_parser('verify-inspection-export', help='Verify inspection export file hashes')
    verify_export.add_argument('directory', type=Path)
    validation = commands.add_parser(
        "validate-boundaries", help="Validate acquisition geometry against private full-panel references"
    )
    validation.add_argument("--database",type=Path,required=True)
    validation.add_argument("--acquisition",required=True)
    validation.add_argument("--references",type=Path,required=True)
    validation.add_argument("--out",type=Path,required=True,help="New private output directory")
    validation.add_argument("--sdk-lib")
    review_pack=commands.add_parser("prepare-boundary-review",help="Prepare unmarked native annotation images without detector predictions")
    review_pack.add_argument("--database",type=Path,required=True)
    review_pack.add_argument("--acquisition",required=True)
    review_pack.add_argument("--out",type=Path,required=True)
    review_pack.add_argument("--count",type=int,default=12)
    review_pack.add_argument("--seed",default="pvdelta-review-v1")
    review_pack.add_argument("--exclude",type=Path,action="append",default=[],help="Reference JSON or SHA256 list; may repeat")
    review_pack.add_argument("--sdk-lib")
    boundaries = commands.add_parser(
        "propose-boundaries", help="Export experimental RGB table/module proposals for review"
    )
    boundaries.add_argument("input", type=Path)
    boundaries.add_argument("--out", type=Path, required=True)
    boundaries.add_argument("--pitch-deg", type=float, help="Explicit pitch override; default reads image metadata")
    boundaries.add_argument("--rows", type=int, default=4)
    boundaries.add_argument("--module-length-m", type=float, default=2.278)
    boundaries.add_argument("--module-width-m", type=float, default=1.113)
    boundaries.add_argument("--tilt-deg", type=float, default=25)
    boundaries.add_argument("--cell-columns", type=int, default=24)
    boundaries.add_argument("--cell-rows", type=int, default=6)
    boundaries.add_argument("--overwrite", action="store_true")
    batch = commands.add_parser("batch", help="Analyze frames listed in a local manifest")
    batch.add_argument("manifest", type=Path)
    batch.add_argument("--out", type=Path, required=True)
    batch.add_argument("--database", type=Path)
    batch.add_argument("--sdk-lib")
    batch.add_argument("--overwrite", action="store_true")
    prepare = commands.add_parser("prepare-manifest", help="Match a flight folder to native regions")
    prepare.add_argument("flight", type=Path)
    prepare.add_argument("--regions-dir", type=Path, required=True)
    prepare.add_argument("--mask-dir", type=Path)
    prepare.add_argument("--out", type=Path, required=True)
    prepare.add_argument("--run-id", required=True)
    prepare.add_argument("--overwrite", action="store_true")
    rank = commands.add_parser("rank-views", help="Select the best RGB view per explicit panel group")
    rank.add_argument("manifest", type=Path)
    rank.add_argument("--out", type=Path, required=True)
    rank.add_argument("--database", type=Path)
    rank.add_argument("--overwrite", action="store_true")
    inventory = commands.add_parser("import-inventory", help="Load a local panel inventory into DuckDB")
    inventory.add_argument("input", type=Path)
    inventory.add_argument("--database", type=Path, required=True)
    inventory.add_argument("--format", choices=("auto", "csv", "geojson"), default="auto")
    inventory.add_argument("--source-crs", help="Explicit source CRS for GeoJSON coordinates")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == 'scan-native-flight':
        try:
            from pvdelta.native_flight import scan_native_flight

            result = scan_native_flight(args.database, args.acquisition, args.out, args.run_id,
                sdk_library=args.sdk_lib, workers=args.workers,
                surface_manifest=args.surface_manifest,
                measurement_overrides=json.loads(args.measurement_params.read_bytes()) if args.measurement_params else None)
        except (ImportError, OSError, RuntimeError, TypeError, ValueError, KeyError) as exc:
            print(f'pvdelta: {exc}', file=sys.stderr)
            return 1
        print(json.dumps({k: v for k, v in result.items() if k != 'frames'}, indent=2))
        return 0 if result['status'] == 'complete' else 3 if result['status'] == 'partial' else 1
    if args.command == 'replay-screening':
        try:
            from pvdelta.screening_replay import attach_site_geometry, replay_screening

            result = replay_screening(args.database, args.baseline_run, args.out, args.run_id,
                sdk_library=args.sdk_lib,
                measurement_overrides=json.loads(args.measurement_params.read_bytes()) if args.measurement_params else None,
                per_frame_overrides=json.loads(args.per_frame_params.read_bytes()) if args.per_frame_params else None)
            if args.site_geometry:
                result = attach_site_geometry(args.out, args.site_geometry)
        except (ImportError, OSError, RuntimeError, TypeError, ValueError, KeyError) as exc:
            print(f'pvdelta: {exc}', file=sys.stderr)
            return 1
        print(json.dumps({k: v for k, v in result.items() if k not in ('frames', 'per_frame_overrides')}, indent=2))
        return 0 if result['status'] == 'complete' else 3 if result['status'] == 'partial' else 1
    if args.command == "prepare-control-review":
        try:
            from pvdelta.control_review import prepare_control_review

            controls = json.loads(args.controls.read_bytes())
            sources = json.loads(args.sources.read_bytes())['frames']
            root = args.source_root.resolve()
            sources = [{k: str(root / item[k]) for k in ('thermal', 'rgb')} for item in sources]
            result = prepare_control_review(controls, sources, args.out, sdk_library=args.sdk_lib)
        except (ImportError, OSError, RuntimeError, ValueError, TypeError, KeyError) as exc:
            print(f"pvdelta: {exc}", file=sys.stderr)
            return 1
        print(json.dumps({'pack_id': result['pack_id'], 'output': str(args.out),
                          'frames': len(result['frames'])}, indent=2))
        return 0
    if args.command == "scan-flight":
        try:
            from pvdelta.flight_scan import scan_radiometric_flight

            report, _ = scan_radiometric_flight(args.flight, args.out,
                sdk_lib=args.sdk_lib, resume=args.resume,
                measurement_overrides=json.loads(args.measurement_params.read_bytes())
                    if args.measurement_params else None)
        except (ImportError, OSError, RuntimeError, ValueError, TypeError, KeyError) as exc:
            print(f"pvdelta: {exc}", file=sys.stderr)
            return 1
        print(json.dumps({k: v for k, v in report.items() if k != 'frames'}, indent=2))
        return 0 if report['status'] == 'complete' else 3 if report['status'] == 'partial' else 1
    if args.command == "prepare-boundary-review":
        try:
            from pvdelta.boundary_review_pack import prepare_boundary_review

            result=prepare_boundary_review(args.database,args.acquisition,args.out,count=args.count,
                seed=args.seed,exclusions=args.exclude,sdk_library=args.sdk_lib)
        except (ImportError,OSError,RuntimeError,TypeError,ValueError) as exc:
            print(f"pvdelta: {exc}",file=sys.stderr)
            return 1
        print(json.dumps({k:v for k,v in result.items() if k!='frames'},indent=2))
        return 0 if result['status']=='complete' else 3 if result['status']=='partial' else 1
    if args.command == "validate-boundaries":
        try:
            from pvdelta.boundary_validation_batch import run_boundary_validation

            result=run_boundary_validation(args.database,args.acquisition,args.references,args.out,
                                           sdk_library=args.sdk_lib)
        except (ImportError,OSError,RuntimeError,TypeError,ValueError) as exc:
            print(f"pvdelta: {exc}",file=sys.stderr)
            return 1
        print(json.dumps({k:result[k] for k in ('status','requested_frames','succeeded_frames',
            'failed_frames','unprocessed_frames','indexed_acquisition_frames',
            'indexed_records_without_reference','requested_reference_instances',
            'scored_reference_instances','aggregate_scores')},indent=2))
        return 0 if result['status']=='complete' else 3 if result['status']=='partial' else 1
    if args.command in {'export-inspection', 'verify-inspection-export'}:
        try:
            from pvdelta.inspection_export import export_inspection, verify_inspection_export

            result = (export_inspection(args.database, args.run_id, args.out)
                      if args.command == 'export-inspection' else verify_inspection_export(args.directory))
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            print(f'pvdelta: {exc}', file=sys.stderr)
            return 1
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    if args.command == "coverage-audit":
        try:
            from pvdelta.coverage_audit import read_coverage_audit

            output = args.out.resolve()
            if output == args.database.resolve():
                raise ValueError("audit output must differ from source database")
            if output.exists() and not args.overwrite:
                raise FileExistsError("audit output exists; use --overwrite")
            result = read_coverage_audit(args.database, args.run_id)
            content = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(content, encoding="utf-8")
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            print(f"pvdelta: {exc}", file=sys.stderr)
            return 1
        print(json.dumps({"run_id": args.run_id, "output": str(output),
                          "counts": result['counts'], "data_issues": result['data_issues']}, indent=2))
        return 0
    if args.command == "propose-boundaries":
        try:
            import cv2

            from pvdelta.table_boundaries import TableBoundaryConfig, refined_module_proposals

            output = args.out.resolve()
            if output == args.input.resolve():
                raise ValueError("output must differ from source image")
            if output.exists() and not args.overwrite:
                raise FileExistsError("boundary output exists; use --overwrite")
            image = cv2.imdecode(np.frombuffer(args.input.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("cannot decode RGB input")
            pitch = args.pitch_deg
            if pitch is None:
                pitch = parse_capture_metadata(args.input).camera_profile.gimbal_pitch
            config = TableBoundaryConfig(rows=args.rows, module_length_m=args.module_length_m,
                                         module_width_m=args.module_width_m, tilt_deg=args.tilt_deg,
                                         cell_columns=args.cell_columns, cell_rows=args.cell_rows)
            result = refined_module_proposals(image, config=config, gimbal_pitch_deg=pitch)
            result.update(source_name=args.input.name, source_sha256=_sha256(args.input),
                          coordinate_space="original_rgb_pixels", image_width_px=image.shape[1],
                          image_height_px=image.shape[0], config=asdict(config), gimbal_pitch_deg=pitch,
                          pitch_source="explicit_override" if args.pitch_deg is not None else "image_metadata")
            content = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(content, encoding="utf-8")
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
            print(f"pvdelta: {exc}", file=sys.stderr)
            return 1
        print(json.dumps({"status": result['status'], "modules": len(result['modules']),
                          "output": str(output)}, indent=2))
        return 0
    if args.command == "sdk-check":
        try:
            frame = decode_frame(args.input, args.sdk_lib)
            values = frame.temperature_c[np.isfinite(frame.temperature_c)]
            if not len(values):
                raise ValueError("SDK returned no finite temperature pixels")
            params = frame.raw_metadata.get("measurement_params", {})
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            print(f"pvdelta: {exc}", file=sys.stderr)
            return 1
        print(json.dumps({
            "status": "ok",
            "input": str(args.input.resolve()),
            "sdk_library": str(Path(args.sdk_lib).resolve()),
            "shape": list(frame.temperature_c.shape),
            "finite_pixels": int(values.size),
            "temperature_min_c": float(np.min(values)),
            "temperature_median_c": float(np.median(values)),
            "temperature_max_c": float(np.max(values)),
            "measurement_params": params,
        }, ensure_ascii=False, indent=2))
        return 0
    if args.command == "import-inventory":
        try:
            format_name = args.format
            if format_name == "auto":
                format_name = "geojson" if args.input.suffix.lower() in {".geojson", ".json"} else "csv"
            if format_name == "csv":
                assets = load_inventory_csv(args.input)
            else:
                assets = load_inventory_geojson(args.input, source_crs=args.source_crs)
            validation = validate_panel_assets(assets)
            connection = connect_store(args.database)
            try:
                imported = upsert_panel_assets(connection, assets)
            finally:
                connection.close()
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            print(f"pvdelta: {exc}", file=sys.stderr)
            return 1
        print(json.dumps({
            "format": format_name,
            "assets": imported,
            "warnings": list(validation.warnings),
            "database": str(args.database.resolve()),
        }, ensure_ascii=False, indent=2))
        return 0
    if args.command == "rank-views":
        manifest_file = args.manifest.resolve()
        try:
            payload = json.loads(manifest_file.read_text(encoding="utf-8"))
            if payload.get("schema_version") != VIEWS_SCHEMA_VERSION:
                raise ValueError(f"manifest schema_version must be {VIEWS_SCHEMA_VERSION}")
            run_id = payload.get("run_id")
            groups = payload.get("groups")
            if not isinstance(run_id, str) or not run_id.strip() or not isinstance(groups, list):
                raise ValueError("manifest needs non-empty run_id and groups list")
            candidates = []
            root = manifest_file.parent
            for group in groups:
                if not isinstance(group, dict) or not isinstance(group.get("panel_key"), str):
                    raise TypeError("each group needs a panel_key")
                views = group.get("views")
                if not isinstance(views, list) or not views:
                    raise ValueError(f"group {group['panel_key']} needs a non-empty views list")
                for view in views:
                    if not isinstance(view, dict) or not isinstance(view.get("rgb_path"), str):
                        raise TypeError("each view needs an rgb_path")
                    candidates.append(ViewCandidate(
                        panel_key=group["panel_key"],
                        rgb_capture_id=str(view["rgb_capture_id"]),
                        rgb_path=view["rgb_path"],
                        thermal_capture_id=view.get("thermal_capture_id"),
                        latitude=view.get("latitude"), longitude=view.get("longitude"),
                        horizontal_accuracy_m=view.get("horizontal_accuracy_m"),
                    ))
            rankings = rank_view_candidates(
                candidates, lambda relative: load_rgb_file(root / relative)
            )
            output = args.out.resolve()
            if output.exists() and not args.overwrite:
                raise FileExistsError("view ranking output exists; use --overwrite")
            output.parent.mkdir(parents=True, exist_ok=True)
            result = {
                "schema_version": VIEWS_SCHEMA_VERSION,
                "run_id": run_id,
                "groups": len({item.panel_key for item in rankings}),
                "views": [ranked_view_to_dict(item) for item in rankings],
            }
            output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            if args.database is not None:
                connection = connect_store(args.database)
                try:
                    write_view_rankings(connection, run_id=run_id, rankings=rankings)
                finally:
                    connection.close()
        except (FileExistsError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
            print(f"pvdelta: {exc}", file=sys.stderr)
            return 1
        selected = sum(item.selection_status == "selected" for item in rankings)
        print(json.dumps({"run_id": run_id, "groups": result["groups"], "views": len(rankings), "selected": selected, "output": str(output)}, indent=2))
        return 0
    if args.command == "prepare-manifest":
        try:
            payload = prepare_manifest(
                args.flight, args.regions_dir, args.out,
                run_id=args.run_id, mask_dir=args.mask_dir, overwrite=args.overwrite,
            )
        except (OSError, ValueError) as exc:
            print(f"pvdelta: {exc}", file=sys.stderr)
            return 1
        summary = {key: value for key, value in payload["preparation"].items() if key != "unresolved"}
        print(json.dumps(summary, indent=2))
        return {"ready": 0, "partial": 3, "blocked": 1}[summary["status"]]
    if args.command == "batch":
        try:
            result = run_batch(
                args.manifest,
                args.out,
                decoder=decode_frame,
                database_path=args.database,
                sdk_lib=args.sdk_lib,
                overwrite=args.overwrite,
            )
        except (FileExistsError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
            print(f"pvdelta: {exc}", file=sys.stderr)
            return 1
        print(
            json.dumps(
                {
                    "run_id": result.run_id,
                    "status": result.status,
                    "total_frames": result.total_frames,
                    "succeeded_frames": result.succeeded_frames,
                    "failed_frames": result.failed_frames,
                    "summary_json": str(result.summary_json),
                    "summary_csv": str(result.summary_csv),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0 if result.status == "complete" else (3 if result.status == "partial" else 1)
    try:
        frame = decode_frame(args.input, args.sdk_lib,
            measurement_overrides=json.loads(args.measurement_params.read_bytes())) \
            if args.measurement_params else decode_frame(args.input, args.sdk_lib)
        annotations = load_annotations(args.regions)
        module_masks = None
        if args.mask_archive is not None:
            if args.margin_px:
                raise ValueError("--margin-px must be zero with --mask-archive")
            module_masks = load_mask_archive(
                args.mask_archive,
                annotations,
                frame.temperature_c.shape,
            )
        config = DetectionConfig(
            delta_threshold_k=args.delta_k,
            min_region_px=args.min_region_px,
            edge_erosion_px=args.edge_erosion_px,
            min_reference_px=args.min_reference_px,
            min_valid_fraction=args.min_valid_fraction,
        )
        result = execute_stage1(
            frame,
            annotations,
            output_dir=args.out,
            run_id=args.run_id,
            region_margin_px=args.margin_px,
            config=config,
            direction=args.direction,
            database_path=args.database,
            site_label=args.site_label,
            overwrite=args.overwrite,
            module_masks=module_masks,
        )
    except (FileExistsError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"pvdelta: {exc}", file=sys.stderr)
        return 1
    payload = {
        "run_id": args.run_id,
        "modules": len(result.module_results),
        "findings": sum(len(item.findings) for item in result.module_results),
        "not_assessed": sum(
            item.assessment_status == "not_assessed" for item in result.module_results
        ),
        "jsonl": str(result.report_artifacts.jsonl_path),
        "csv": str(result.report_artifacts.csv_path),
        "overlay": str(result.report_artifacts.overlay_path),
        "database": None if args.database is None else str(args.database),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
