import math
import struct

import numpy as np
import pytest

from pvdelta.flir_fff import FlirDecodeError, PlanckCalibration, read_flir_rjpeg


def payload(*, endian='>', raw_order='<', camera_order='<', subtype=None):
    values = np.array([[3000, 3500, 4000], [4500, 5000, 5500]], dtype=np.uint16)
    raw = bytearray(32)
    struct.pack_into(raw_order+'HHH', raw, 0, 2, 3, 2)
    raw += values.astype(raw_order+'u2').tobytes()
    camera = bytearray(0x310)
    struct.pack_into(camera_order+'HHH', camera, 0, 2, 3, 2)
    for offset, value in {0x20: .95, 0x24: 10, 0x28: 293.15, 0x2c: 294.15,
                          0x30: 295.15, 0x34: 1., 0x3c: .3, 0x58: 348025.,
                          0x5c: 1428., 0x60: 1., 0x30c: 1.}.items():
        struct.pack_into(camera_order+'f', camera, offset, value)
    struct.pack_into(camera_order+'i', camera, 0x308, -425)
    camera[0xd4:0xd8] = b'XT2\0'
    data = bytearray(128)
    data[:4] = b'FFF\0'
    struct.pack_into(endian+'III', data, 20, 100, 64, 2)
    struct.pack_into(endian+'HHIIII', data, 64, 1, subtype or (1 if raw_order == '>' else 2), 100, 1, 128, len(raw))
    struct.pack_into(endian+'HHIIII', data, 96, 32, 1, 100, 1, 128+len(raw), len(camera))
    return data+raw+camera, values


def jpeg(data, *, chunk_size=200):
    chunks = [bytes(data[i:i+chunk_size]) for i in range(0, len(data), chunk_size)]
    segments = []
    for i, chunk in enumerate(chunks):
        content = b'FLIR\0\x01'+bytes([i, len(chunks)-1])+chunk
        segments.append(b'\xff\xe1'+struct.pack('>H', len(content)+2)+content)
    return b'\xff\xd8'+b''.join(segments)+b'\xff\xd9'


@pytest.mark.parametrize('endian,raw_order,camera_order', [('>', '<', '<'), ('<', '>', '<'), ('>', '>', '>')])
def test_independent_container_raw_camera_byte_orders_and_read_only_native_counts(endian, raw_order, camera_order):
    data, expected = payload(endian=endian, raw_order=raw_order, camera_order=camera_order)
    frame = read_flir_rjpeg(jpeg(data))
    np.testing.assert_array_equal(frame.raw_counts, expected)
    assert frame.raw_counts.shape == (2, 3) and frame.raw_counts.dtype == np.uint16
    assert not frame.raw_counts.flags.writeable and frame.camera_model == 'XT2'
    assert (frame.fff_byte_order, frame.raw_byte_order, frame.camera_byte_order) == (endian, raw_order, camera_order)
    assert frame.observation_settings.emissivity == pytest.approx(.95)
    assert frame.observation_settings.relative_humidity_fraction == pytest.approx(.3)
    assert frame.observation_settings.reflected_temp_c == pytest.approx(20., abs=1e-4)


def test_apparent_planck_inversion_round_trip_and_metadata_does_not_silently_correct_it():
    data, raw = payload()
    frame = read_flir_rjpeg(jpeg(data))
    measured = frame.apparent_temperature_c()
    c = frame.calibration
    reconstructed = c.r1/(c.r2*(np.exp(c.b/(measured+273.15))-c.f))-c.o
    np.testing.assert_allclose(reconstructed, raw, rtol=1e-12)
    expected = 1428/np.log(348025/(raw.astype(float)-425)+1)-273.15
    np.testing.assert_allclose(measured, expected)
    assert not measured.flags.writeable


def test_known_scalar_temperature_uses_embedded_coefficients_not_a_default_camera():
    # Analytic fixture: log argument is exactly 2, yielding exactly 300 K.
    c = PlanckCalibration(1000, 300*math.log(2), 1, 0, 1)
    assert c.apparent_temperature_c(np.array([[1000]], dtype=np.uint16))[0, 0] == pytest.approx(26.85, abs=1e-10)


@pytest.mark.parametrize('case', ['missing_chunk', 'duplicate_chunk', 'inconsistent_count', 'truncated_segment', 'not_jpeg', 'no_flir'])
def test_bad_jpeg_or_chunk_sequence_cannot_be_joined_by_guessing(case):
    data, _ = payload()
    encoded = jpeg(data)
    if case == 'not_jpeg':
        encoded = b'not JPEG'
    elif case == 'no_flir':
        encoded = b'\xff\xd8\xff\xd9'
    elif case == 'missing_chunk':
        size = struct.unpack_from('>H', encoded, 4)[0]
        encoded = encoded[:2]+encoded[4+size:]
    elif case == 'duplicate_chunk':
        size = struct.unpack_from('>H', encoded, 4)[0]
        encoded = encoded[:2]+encoded[2:4+size]+encoded[2:]
    elif case == 'inconsistent_count':
        encoded = bytearray(encoded)
        encoded[13] += 1
        encoded = bytes(encoded)
    else:
        encoded = encoded[:100]
    with pytest.raises(FlirDecodeError):
        read_flir_rjpeg(encoded)


@pytest.mark.parametrize('case', ['unsupported_version', 'bad_directory', 'outside_record', 'overlap',
                                  'png_subtype', 'bad_raw_order', 'bad_raw_length', 'bad_camera_order',
                                  'dimension_mismatch', 'nan_coefficient', 'zero_calibration'])
def test_invalid_or_unsupported_fff_fails_explicitly(case):
    data, _ = payload()
    raw_start = 128
    camera_start = 128+32+12
    if case == 'unsupported_version':
        struct.pack_into('>I', data, 20, 900)
    elif case == 'bad_directory':
        struct.pack_into('>I', data, 24, len(data))
    elif case == 'outside_record':
        struct.pack_into('>I', data, 64+12, len(data)+1)
    elif case == 'overlap':
        struct.pack_into('>I', data, 96+12, raw_start)
    elif case == 'png_subtype':
        struct.pack_into('>H', data, 64+2, 3)
    elif case == 'bad_raw_order':
        struct.pack_into('<H', data, raw_start, 7)
    elif case == 'bad_raw_length':
        struct.pack_into('<H', data, raw_start+2, 4)
    elif case == 'bad_camera_order':
        struct.pack_into('<H', data, camera_start, 7)
    elif case == 'dimension_mismatch':
        struct.pack_into('<H', data, camera_start+2, 4)
    elif case == 'nan_coefficient':
        struct.pack_into('<f', data, camera_start+0x58, float('nan'))
    else:
        struct.pack_into('<f', data, camera_start+0x30c, 0)
    with pytest.raises(FlirDecodeError):
        read_flir_rjpeg(jpeg(data))


@pytest.mark.parametrize('counts', [np.array([[4000.]]), np.array([4000]), np.array([[-5]]),
                                   np.array([[70000]]), np.array([[425]], dtype=np.uint16)])
def test_invalid_counts_are_not_clipped_or_removed(counts):
    with pytest.raises(FlirDecodeError):
        PlanckCalibration(348025, 1428, 1, -425, 1).apparent_temperature_c(counts)
