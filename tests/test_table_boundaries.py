import numpy as np
import pytest

cv2 = pytest.importorskip('cv2')

from pvdelta.table_boundaries import (
    TableBoundaryConfig,
    refined_module_proposals,
    table_module_proposals,
)


def synthetic_table(*, angle=0, hole=False):
    image = np.full((850, 1400, 3), (35, 135, 30), np.uint8)
    truth = []
    for col in range(6):
        for row in range(4):
            x, y = 180+col*150, 230+row*66
            cv2.rectangle(image, (x, y), (x+150, y+66), (108, 63, 55), -1)
            for cell in range(1, 24):
                cv2.line(image, (x+round(cell*150/24), y),
                         (x+round(cell*150/24), y+66), (138, 118, 110), 1)
            for cell in range(1, 6):
                cv2.line(image, (x, y+cell*11), (x+150, y+cell*11), (138, 118, 110), 1)
            cv2.line(image, (x+75, y), (x+75, y+66), (185, 185, 185), 1)
            cv2.rectangle(image, (x, y), (x+150, y+66), (245, 245, 245), 3)
            truth.append(np.float32([[x, y], [x+150, y], [x+150, y+66], [x, y+66]]))
    if hole:
        image[300:350, 490:620] = (35, 135, 30)
    if angle:
        matrix = cv2.getRotationMatrix2D((700, 425), angle, 1)
        image = cv2.warpAffine(image, matrix, (1400, 850), borderValue=(35, 135, 30))
        truth = [cv2.transform(p[None], matrix)[0] for p in truth]
    return image, truth


def iou(first, second):
    overlap, _ = cv2.intersectConvexConvex(np.float32(first), np.float32(second))
    return overlap / (abs(cv2.contourArea(first))+abs(cv2.contourArea(second))-overlap)


@pytest.mark.parametrize('angle', [0, 13, 83])
def test_observed_separators_recover_full_modules_without_half_cut_splits(angle):
    image, truth = synthetic_table(angle=angle)
    before = image.copy()
    result = table_module_proposals(image, gimbal_pitch_deg=-90)
    assert np.array_equal(image, before)
    assert len(result['modules']) >= 8
    for module in result['modules']:
        polygon = np.float32(module['polygon_rgb_xy'])
        assert max(iou(polygon, target) for target in truth) > .85
        assert module['source'] == 'observed_table_separators'
        assert module['assessment_status'] == 'unreviewed_rgb_geometry'


def test_blue_rectangles_without_separators_and_green_ground_are_not_filled():
    image = np.full((850, 1400, 3), (35, 135, 30), np.uint8)
    cv2.rectangle(image, (180, 230), (1080, 494), (108, 63, 55), -1)
    result = table_module_proposals(image, gimbal_pitch_deg=-90)
    assert not result['modules']
    assert all(t['status'] == 'unresolved' for t in result['tables'])


def test_missing_pose_and_oblique_views_are_explicitly_deferred():
    image, _ = synthetic_table()
    assert refined_module_proposals(image)['status'] == 'pose_unavailable'
    result = refined_module_proposals(image, gimbal_pitch_deg=-45)
    assert result['status'] == 'deferred_oblique_geometry'
    assert result['modules'] == []


def test_large_vegetation_gap_does_not_become_a_module():
    image, _ = synthetic_table(hole=True)
    result = table_module_proposals(image, gimbal_pitch_deg=-90)
    damaged = np.float32([[480, 296], [630, 296], [630, 362], [480, 362]])
    assert result['modules']
    assert all(iou(np.float32(m['polygon_rgb_xy']), damaged) < .2 for m in result['modules'])


def test_invalid_geometry_configuration_is_rejected():
    with pytest.raises(ValueError):
        TableBoundaryConfig(rows=1)
    with pytest.raises(ValueError):
        TableBoundaryConfig(tilt_deg=float('nan'))


def test_boundary_cli_keeps_rgb_geometry_explicit_and_protects_outputs(tmp_path):
    import json

    from pvdelta.cli import main

    image, _ = synthetic_table()
    source = tmp_path / 'synthetic_V.png'
    source.write_bytes(cv2.imencode('.png', image)[1].tobytes())
    before = source.read_bytes()
    output = tmp_path / 'boundaries.json'
    assert main(['propose-boundaries', str(source), '--out', str(output), '--pitch-deg', '-90']) == 0
    report = json.loads(output.read_text())
    assert report['coordinate_space'] == 'original_rgb_pixels'
    assert report['status'] == 'unreviewed_rgb_geometry'
    assert report['pitch_source'] == 'explicit_override'
    assert len(report['modules']) >= 8
    saved = output.read_bytes()
    assert main(['propose-boundaries', str(source), '--out', str(output)]) == 1
    assert output.read_bytes() == saved
    assert main(['propose-boundaries', str(source), '--out', str(source), '--overwrite']) == 1
    assert source.read_bytes() == before


def test_boundary_cli_missing_pose_does_not_assume_nadir(tmp_path):
    import json

    from pvdelta.cli import main

    image, _ = synthetic_table()
    source = tmp_path / 'synthetic_V.png'
    source.write_bytes(cv2.imencode('.png', image)[1].tobytes())
    output = tmp_path / 'unknown.json'
    assert main(['propose-boundaries', str(source), '--out', str(output)]) == 0
    report = json.loads(output.read_text())
    assert report['status'] == 'pose_unavailable'
    assert report['modules'] == []


def test_sheared_table_preserves_local_frame_geometry():
    image, truth = synthetic_table()
    matrix = np.float32([[1, 0, 0], [.2, 1, -100]])
    transformed = cv2.warpAffine(image, matrix, (1400, 850), borderValue=(35, 135, 30))
    truth = [cv2.transform(p[None], matrix)[0] for p in truth]
    result = table_module_proposals(transformed, gimbal_pitch_deg=-90)
    assert len(result['modules']) >= 8
    for module in result['modules']:
        polygon = np.float32(module['polygon_rgb_xy'])
        assert max(iou(polygon, target) for target in truth) > .85
