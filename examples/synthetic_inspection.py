"""Run a small inspection with generated temperature data and reviewed regions."""

import argparse
import hashlib
from pathlib import Path

import numpy as np

from pvdelta.models import ThermalFrame
from pvdelta.pipeline import analyze_annotations
from pvdelta.regions import ModuleAnnotation
from pvdelta.report import write_stage1_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path, help='Directory for the inspection reports')
    args = parser.parse_args()

    temperatures = np.full((64, 112), 30, dtype=np.float32)
    temperatures[26:33, 24:31] = 40
    source_hash = hashlib.sha256(temperatures.tobytes()).hexdigest()
    frame = ThermalFrame('synthetic', 'generated-thermal-grid', source_hash,
                         temperatures, None, 112, 64)
    modules = [
        ModuleAnnotation('demo-01', ((8, 10), (48, 10), (48, 54), (8, 54))),
        ModuleAnnotation('demo-02', ((64, 10), (104, 10), (104, 54), (64, 54))),
    ]
    reports = analyze_annotations(frame, modules, direction='hot')
    artifacts = write_stage1_report(args.output, 'synthetic', frame, reports,
                                    settings={'source': 'generated', 'regions': 'supplied'})
    for report in reports:
        print(f'{report.module.module_id} | {report.assessment_status}')
        for finding in report.findings:
            print(f'  {finding.anomaly_type} | local delta T {finding.delta_t_c:+.1f} K')
    print(f'Reports: {artifacts.jsonl_path}, {artifacts.csv_path}, {artifacts.overlay_path}')


if __name__ == '__main__':
    main()
