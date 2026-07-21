#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ucup_rsrp.data import discover_cells
from ucup_rsrp.raster import point_cloud_fingerprint, rasterize_point_cloud


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rasterize each unique challenge point cloud.")
    parser.add_argument("--data", type=Path, default=Path("TrainingData.26UCupSummer"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/rasters"))
    parser.add_argument("--resolution", type=float, default=4.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    mapping: dict[str, str] = {}
    cells = discover_cells(args.data)
    for index, cell in enumerate(cells, start=1):
        fingerprint = point_cloud_fingerprint(cell.point_cloud_path)
        destination = args.output / f"{fingerprint}.npz"
        mapping[cell.cell_id] = destination.name
        if destination.exists():
            print(f"[{index:02d}/{len(cells)}] {cell.cell_id}: cached {destination.name}")
            continue
        print(f"[{index:02d}/{len(cells)}] {cell.cell_id}: building {destination.name}")
        raster = rasterize_point_cloud(cell.point_cloud_path, resolution_m=args.resolution)
        raster.save(destination)
    (args.output / "cell_to_raster.json").write_text(
        json.dumps(mapping, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"Wrote mappings for {len(mapping)} cells; {len(set(mapping.values()))} unique rasters.")


if __name__ == "__main__":
    main()

