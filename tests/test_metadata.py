import datetime as dt
import struct

import pytest

from pvdelta.metadata import (
    assess_metadata_readiness,
    extract_dji_xmp,
    parse_camera_profile,
    parse_capture_metadata,
    visible_pair_candidate,
)


def _jpeg_with_exif_focal_length(numerator: int, denominator: int) -> bytes:
    # TIFF header + IFD0 Exif pointer + Exif IFD FocalLength rational.
    tiff = bytearray(b"II" + struct.pack("<H", 42) + struct.pack("<I", 8))
    tiff += struct.pack("<H", 1)
    tiff += struct.pack("<HHI", 0x8769, 4, 1) + struct.pack("<I", 26)
    tiff += struct.pack("<I", 0)
    tiff += struct.pack("<H", 1)
    tiff += struct.pack("<HHI", 0x920A, 5, 1) + struct.pack("<I", 44)
    tiff += struct.pack("<I", 0)
    tiff += struct.pack("<II", numerator, denominator)
    payload = b"Exif\x00\x00" + bytes(tiff)
    return b"\xff\xd8\xff\xe1" + struct.pack(">H", len(payload) + 2) + payload + b"\xff\xd9"


def test_extract_namespaced_attributes_and_elements(tmp_path):
    packet = b'''<x:xmpmeta xmlns:x="adobe:ns:meta/" xmlns:rdf="rdf" xmlns:drone-dji="dji"
      xmlns:tiff="tiff"><rdf:RDF><rdf:Description tiff:Make="DJI"
      drone-dji:DroneModel="M4T" drone-dji:GimbalPitchDegree="-89.5"
      drone-dji:RelativeAltitude="15.2"><drone-dji:GimbalYawDegree>91</drone-dji:GimbalYawDegree>
      </rdf:Description></rdf:RDF></x:xmpmeta>'''
    image = tmp_path / "synthetic.jpg"
    image.write_bytes(_jpeg_with_exif_focal_length(120, 10) + packet)
    values = extract_dji_xmp(image)
    assert values["drone-dji:DroneModel"] == "M4T"
    assert values["drone-dji:GimbalYawDegree"] == "91"
    profile = parse_camera_profile(image)
    assert profile.make == "DJI"
    assert profile.model == "M4T"
    assert profile.gimbal_pitch == -89.5
    assert profile.gimbal_yaw == 91
    assert profile.relative_altitude_m == 15.2
    assert profile.focal_length_mm == 12.0
    assert profile.focal_length_source == "image_exif"


def test_unverified_database_focal_length_is_not_invented_for_m4t(tmp_path):
    image = tmp_path / "synthetic.jpg"
    image.write_bytes(b'''<x:xmpmeta xmlns:x="adobe:ns:meta/" xmlns:rdf="rdf"
      xmlns:drone-dji="dji"><rdf:RDF><rdf:Description
      drone-dji:DroneModel="M4T"/></rdf:RDF></x:xmpmeta>''')
    profile = parse_camera_profile(image)
    assert profile.focal_length_mm is None
    assert profile.focal_length_source is None


def test_missing_and_malformed_xmp_are_distinct(tmp_path):
    image = tmp_path / "synthetic.jpg"
    image.write_bytes(b"no XMP")
    assert extract_dji_xmp(image) == {}
    profile = parse_camera_profile(image)
    assert profile.model == "Unknown"
    assert profile.gimbal_pitch is None
    assert profile.relative_altitude_m is None
    image.write_bytes(b"<x:xmpmeta><broken></x:xmpmeta>")
    with pytest.raises(ValueError, match="invalid XMP"):
        extract_dji_xmp(image)


def test_capture_metadata_keeps_time_sources_and_camera_position_separate(tmp_path):
    packet = b'''<x:xmpmeta xmlns:x="adobe:ns:meta/" xmlns:rdf="rdf"
      xmlns:drone-dji="dji" xmlns:xmp="xmp"><rdf:RDF><rdf:Description
      drone-dji:UTCAtExposure="2026-04-29T13:26:31"
      xmp:CreateDate="2026-04-29T16:26:12+03:00"
      drone-dji:GpsLatitude="40.5" drone-dji:GpsLongitude="29.5"
      drone-dji:GpsStatus="RTK"/></rdf:RDF></x:xmpmeta>'''
    image = tmp_path / "synthetic.jpg"
    image.write_bytes(packet)
    capture = parse_capture_metadata(image)
    assert capture.exposure_time_utc == dt.datetime(2026, 4, 29, 13, 26, 31, tzinfo=dt.timezone.utc)
    assert capture.created_time_utc == dt.datetime(2026, 4, 29, 13, 26, 12, tzinfo=dt.timezone.utc)
    assert (capture.camera_latitude, capture.camera_longitude) == (40.5, 29.5)
    assert capture.gps_status == "RTK"
    assert "XMP time fields differ" in capture.warnings[0]
    readiness = assess_metadata_readiness(capture)
    assert readiness.usable_for_camera_pose is False
    assert readiness.usable_for_panel_location is False
    assert any("intrinsics" in reason for reason in readiness.reasons)


def test_invalid_gps_and_ambiguous_timestamp_stay_unknown(tmp_path):
    image = tmp_path / "synthetic.jpg"
    image.write_bytes(b'''<x:xmpmeta xmlns:x="adobe:ns:meta/" xmlns:rdf="rdf"
      xmlns:drone-dji="dji" xmlns:xmp="xmp"><rdf:RDF><rdf:Description
      xmp:CreateDate="2026-04-29T16:26:12"
      drone-dji:GpsLatitude="999"/></rdf:RDF></x:xmpmeta>''')
    capture = parse_capture_metadata(image)
    assert capture.created_time_utc is None
    assert capture.camera_latitude is None
    assert capture.camera_longitude is None
    assert len(capture.warnings) == 2


@pytest.mark.parametrize("yaw", [90, 180])
def test_orientation_values_are_preserved_without_guessing_rotation(tmp_path, yaw):
    image = tmp_path / "frame_T.JPG"
    image.write_bytes(f'''<x:xmpmeta xmlns:x="adobe:ns:meta/" xmlns:rdf="rdf"
      xmlns:drone-dji="dji"><rdf:RDF><rdf:Description
      drone-dji:GimbalYawDegree="{yaw}"/></rdf:RDF></x:xmpmeta>'''.encode())
    assert parse_camera_profile(image).gimbal_yaw == yaw
    assert visible_pair_candidate(image) is None
    visible = tmp_path / "frame_V.JPG"
    visible.write_bytes(b"candidate")
    assert visible_pair_candidate(image) == visible


@pytest.mark.parametrize("distance, expected", [(18.5, 18.5), (-1, None), (float('nan'), None)])
def test_laser_target_distance_is_preserved_separately_from_camera_altitude(tmp_path, distance, expected):
    image = tmp_path / "synthetic.jpg"
    image.write_bytes(f'''<x:xmpmeta xmlns:x="adobe:ns:meta/" xmlns:rdf="rdf"
      xmlns:drone-dji="dji"><rdf:RDF><rdf:Description
      drone-dji:RelativeAltitude="15" drone-dji:LRFStatus="Normal"
      drone-dji:LRFTargetDistance="{distance}"/></rdf:RDF></x:xmpmeta>'''.encode())
    capture = parse_capture_metadata(image)
    assert capture.laser_target_distance_m == expected
    assert capture.laser_status == "Normal"
    assert capture.camera_profile.relative_altitude_m == 15


@pytest.mark.parametrize('thermal_name,visible_name,expected', [
    ('DJI_20310429162706_0420_T.JPG', 'DJI_20310429162705_0420_V.JPG', True),
    ('DJI_20310429162706_0420_T.JPG', 'DJI_20310429162707_0420_V.JPG', True),
    ('DJI_20310430000000_0420_T.JPG', 'DJI_20310429235959_0420_V.JPG', True),
    ('DJI_20310429162706_0420_T.JPG', 'dji_20310429162705_0420_v.JpG', True),
    ('DJI_20310429162706_0420_T.JPG', 'DJI_20310429162704_0420_V.JPG', False),
    ('DJI_20310429162706_0420_T.JPG', 'DJI_20310429162705_0421_V.JPG', False),
    ('DJI_20310429162706_0420_T.JPG', 'OTHER_20310429162705_0420_V.JPG', False),
    ('DJI_20310429162706_0420_T.JPG', 'DJI_20300429162706_0420_V.JPG', False),
])
def test_visible_suggestion_requires_same_capture_counter_and_one_second_bound(
        tmp_path, thermal_name, visible_name, expected):
    thermal = tmp_path/thermal_name
    thermal.write_bytes(b'original thermal placeholder')
    visible = tmp_path/visible_name
    visible.write_bytes(b'RGB placeholder')
    assert visible_pair_candidate(thermal) == (visible if expected else None)


@pytest.mark.parametrize('include_exact', [False, True])
def test_ambiguous_same_counter_suggestions_are_withheld_even_with_exact_name(tmp_path, include_exact):
    thermal = tmp_path/'DJI_20310429162706_0420_T.JPG'
    thermal.write_bytes(b'thermal')
    for second in ([5, 6, 7] if include_exact else [5, 7]):
        (tmp_path/f'DJI_2031042916270{second}_0420_V.JPG').write_bytes(b'RGB')
    assert visible_pair_candidate(thermal) is None


def test_matching_directory_and_invalid_date_cannot_supply_a_time_fallback(tmp_path):
    thermal = tmp_path/'DJI_20310429162706_0420_T.JPG'
    (tmp_path/'DJI_20310429162705_0420_V.JPG').mkdir()
    (tmp_path/'DJI_20311329162705_0420_V.JPG').write_bytes(b'invalid month')
    assert visible_pair_candidate(thermal) is None
