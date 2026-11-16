# Image sources and reproduction

## Figure 1 · PV Delta project flight

Project-owned DJI Matrice 4T capture selected by the operator from the Streamlit
review. The publishing export preserves the complete operator-supplied thermal
(640 × 512) and RGB (1280 × 1280) views at their supplied pixel dimensions,
with camera metadata removed. The SDK replay matched the saved native
temperature hash. The white thermal ring is derived from exact saved region
pixels. The red locator is
enlarged for legibility. The RGB locator is approximate because the local edge
alignment check did not pass. Debris is the operator's site observation.

## Figure 3 · Public UAV thermal image of PV modules

Panagiotis Christakakis, Eleftheria-Maria Pechlivani and Periklis Dimou (2025),
*Thermal PV Panel Detection and Fault Detection Dataset for UAV-Based Inspection*.
[Zenodo record](https://doi.org/10.5281/zenodo.16420123) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

Source image in the archive's `test/images` directory:
`DJI_20230213154415_0130_T_JPG.rf.6719ca4c1ab5d3c4d383fa068f406aca.jpg`.
Source JPEG SHA-256:
`5152cec16a2f341c0198ff1dd7bed11e76c5062e34bde071cbf2468464b9664e`.

The published PNG preserves the decoded 640 × 512 RGB pixels. It was re-encoded
without the original metadata. The source is a processed research image, so
Figure 3 has no temperature scale.

## Figure 2 · Temperature map and row profile

Source: STR Drones, [DJI dataset](https://huggingface.co/datasets/STRDrones/DJI)
(MIT), [DJI_M4T_5.JPG](https://huggingface.co/datasets/STRDrones/DJI/blob/main/M4T/DJI_M4T_5.JPG).
Source R-JPEG SHA-256:
`ca01fd68c8285577865f17ab9c8c5aef32b6baf615f5bc6ab15c20cdc4470446`.
The plot is derived from the DJI Thermal SDK temperature grid with the recorded
measurement settings. The color scale is clipped to the 2nd and 98th percentiles
of finite grid values. The row profile plots the decoded temperatures directly.

| SDK parameter | Recorded value (rounded) |
| --- | --- |
| Emissivity | 0.94995 |
| Object distance | 5 m |
| Relative humidity | 50% |
| Reflected temperature | 15.30 °C |
| Ambient temperature | 15.37 °C |

Download the R-JPEG from the dataset and configure the DJI Thermal SDK.
From the repository root:

```bash
python -m pip install -e . matplotlib
python examples/plot_native_temperature.py DJI_M4T_5.JPG native-temperature-m4t.png --sdk-library /path/to/sdk/lib --row 256
```

The PNG contains the figure only. The original R-JPEG and its camera metadata
remain with the dataset provider.
