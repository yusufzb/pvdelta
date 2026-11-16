# Validation and verification

## Verified checks

| Verification | Evidence | Interpretation |
| --- | --- | --- |
| Native decoding | Eight public STR Drones M4T/M30T R-JPEGs, each 640 × 512. The wrapper and manufacturer CLI agreed on all 2,621,440 temperature values at the same effective settings. | Decoder-path agreement |
| Field processing | 1,239 thermal frames processed from a local M4T flight. | Evidence of processing at flight scale |
| Packaging and software regressions | The published 9eca5d0 revision completed the Windows/Ubuntu and Python 3.10/3.12 CI matrix. The real SDK regression path was also exercised locally. | Reproducible software checks |
| Reviewed example | Figure 3 retains measured ΔT and exact support alongside paired source views and an alignment caveat. | Traceability of a single candidate |

[Published CI run](https://github.com/yusufzb/pvdelta/actions/runs/43917278836) · [STR Drones source](https://huggingface.co/datasets/STRDrones/DJI) · [Test modules](../tests/) · [Figure 3](results.md#paired-field-observation)

## Replication and acceptance

Use original inputs, recorded SDK parameters and source hashes. Reuse identical effective settings when comparing two decoder paths. For independent detection evaluation, freeze settings before predictions and match predictions to reference regions using declared criteria. Report missed reference events and false detections, and give denominators by frame, region and physical module as applicable. The project includes [fixed evaluation](fixed-evaluation.md), [hotspot evaluation](hotspot-evaluation.md) and [coverage auditing](coverage-audit.md).

Verification of field accuracy and inspection conformity needs independent reference temperatures, measured plane-of-array irradiance and weather, operating state, viewing geometry and documented inspection procedure. A warm region may have multiple physical explanations. Repeat views can add evidence when same-panel identity, image geometry and conditions are supported. [IEC TS 62446-3:2017](https://webstore.iec.ch/en/publication/28628) · [Field evidence checklist](method.md#6-iec-ts-62446-3-and-field-evidence)
