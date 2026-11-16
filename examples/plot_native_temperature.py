"""Export a temperature map and a row profile from a DJI radiometric JPEG."""

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pvdelta.decode import RJpeg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path, help='Original DJI radiometric JPEG')
    parser.add_argument('output', type=Path, help='Output PNG')
    parser.add_argument('--sdk-library', required=True, help='DJI Thermal SDK library or directory')
    parser.add_argument('--row', type=int, help='Native image row, zero-based (default: middle)')
    args = parser.parse_args()
    with RJpeg(args.source, args.sdk_library) as image:
        values = image.temperature()
    height, width = values.shape
    row = height // 2 if args.row is None else args.row
    if not 0 <= row < height or not np.isfinite(values).any():
        parser.error('Requires a valid row index and finite temperature data')

    low, high = np.percentile(values[np.isfinite(values)], [2, 98])
    if low == high:
        low, high = low - .5, high + .5
    figure, (thermal, profile) = plt.subplots(
        1, 2, figsize=(11, 4.3), width_ratios=(1.25, 1), layout='constrained',
    )
    figure.suptitle('Native radiometric temperature', fontsize=17, fontweight='bold')
    raster = thermal.imshow(values, cmap='inferno', vmin=low, vmax=high, interpolation='nearest')
    thermal.axhline(row, color='#65e7f0', linewidth=1, linestyle='--')
    thermal.set(title=f'Thermal grid · {width} × {height} pixels', xlabel='Column', ylabel='Row')
    figure.colorbar(raster, ax=thermal, label='Temperature (°C)', extend='both')
    profile.plot(values[row], color='#087f8c', linewidth=1.15)
    profile.set(title=f'Cross-section · row {row}', xlabel='Native column', ylabel='Temperature (°C)')
    profile.grid(alpha=.25)
    profile.spines[['right', 'top']].set_visible(False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, format='png', dpi=160, metadata={'Software': None})
    plt.close(figure)


if __name__ == '__main__':
    main()
