import duckdb

from pvdelta.ingest import discover_capture_files, index_flight_folder


def _xmp(kind: str) -> bytes:
    return f'''<x:xmpmeta xmlns:x="adobe:ns:meta/" xmlns:rdf="rdf"
      xmlns:drone-dji="dji"><rdf:RDF><rdf:Description
      drone-dji:ImageSource="{kind}" drone-dji:DroneModel="M4T"
      drone-dji:UTCAtExposure="2026-04-29T13:26:31"
      drone-dji:GpsLatitude="40.5" drone-dji:GpsLongitude="29.5"
      drone-dji:GimbalPitchDegree="-90" drone-dji:GimbalYawDegree="90"
      drone-dji:GimbalRollDegree="180" drone-dji:RelativeAltitude="15"
      drone-dji:LRFStatus="Normal" drone-dji:LRFTargetDistance="18.5"
      drone-dji:LRFTargetLat="40.5001" drone-dji:LRFTargetLon="29.5001"
      drone-dji:LRFTargetAbsAlt="100"/></rdf:RDF></x:xmpmeta>'''.encode()


def test_discovery_and_indexing_pair_thermal_and_rgb(tmp_path):
    flight = tmp_path / "flight"
    flight.mkdir()
    (flight / "DJI_20260101000001_0001_T.JPG").write_bytes(_xmp("InfraredCamera"))
    (flight / "DJI_20260101000000_0001_V.JPG").write_bytes(_xmp("WideCamera"))
    (flight / "ignore.jpg").write_bytes(b"not a capture")
    database = tmp_path / "inspection.duckdb"

    assert len(discover_capture_files(flight)) == 2
    summary = index_flight_folder(flight, database, run_id="run-1", site_label="synthetic")

    assert summary.indexed_captures == 2
    assert summary.thermal_captures == 1
    assert summary.rgb_captures == 1
    assert summary.paired_captures == 2
    assert summary.geolocated_captures == 2
    assert summary.laser_targets == 2
    connection = duckdb.connect(str(database), read_only=True)
    rows = connection.execute(
        """
        SELECT image_kind, paired_capture_id IS NOT NULL, camera_latitude,
               laser_target_latitude, source_path
        FROM captures ORDER BY image_kind
        """
    ).fetchall()
    connection.close()
    assert rows == [
        ("rgb", True, 40.5, 40.5001, "DJI_20260101000000_0001_V.JPG"),
        ("thermal", True, 40.5, 40.5001, "DJI_20260101000001_0001_T.JPG"),
    ]


def test_indexing_is_idempotent(tmp_path):
    flight = tmp_path / "flight"
    flight.mkdir()
    (flight / "DJI_20260101000001_0001_T.JPG").write_bytes(_xmp("InfraredCamera"))
    database = tmp_path / "inspection.duckdb"
    index_flight_folder(flight, database, run_id="run-1")
    index_flight_folder(flight, database, run_id="run-1")

    connection = duckdb.connect(str(database), read_only=True)
    assert connection.execute("SELECT count(*) FROM captures").fetchone()[0] == 1
    assert connection.execute("SELECT count(*) FROM inspection_runs").fetchone()[0] == 1
    connection.close()
