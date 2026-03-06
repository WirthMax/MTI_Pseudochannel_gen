#!/usr/bin/env python3
"""Replace the DAPI channel in backsub output with the registered Scan DAPI.

After MCMICRO runs with the Scan DAPI injected as an extra ASHLAR round
(cycle 999), this script:
  1. Reads markers.csv to find the Scan DAPI channel (cycle 999) and the
     Cycle1 DAPI channel (earliest cycle, not removed).
  2. Extracts the Scan DAPI plane from the registration mosaic.
  3. Overwrites the corresponding DAPI plane in the backsub OME-TIFF.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import tifffile


def parse_markers(markers_csv: Path) -> list[dict]:
    """Read markers.csv and return rows as list of dicts."""
    with open(markers_csv) as f:
        reader = csv.DictReader(f)
        return list(reader)


def find_scan_dapi(rows: list[dict]) -> dict:
    """Find the Scan DAPI row (cycle 999, marker DAPI)."""
    for row in rows:
        cycle = int(row["cycle_number"])
        marker = row["marker_name"].strip()
        if cycle == 999 and marker.upper() == "DAPI":
            return row
    raise ValueError("No Scan DAPI entry found (cycle 999, marker DAPI)")


def find_cycle1_dapi(rows: list[dict]) -> dict:
    """Find the earliest-cycle DAPI that is not removed."""
    candidates = []
    for row in rows:
        marker = row["marker_name"].strip()
        removed = row.get("remove", "").strip().upper() in ("TRUE", "1", "YES")
        if marker.upper() == "DAPI" and not removed:
            candidates.append(row)

    if not candidates:
        raise ValueError("No non-removed DAPI channel found in markers.csv")

    # Return the one with the smallest cycle number
    candidates.sort(key=lambda r: int(r["cycle_number"]))
    return candidates[0]


def compute_backsub_index(rows: list[dict], target_row: dict) -> int:
    """Compute the 0-based index of target_row in the backsub output.

    Backsub only contains non-removed channels, in channel_number order.
    """
    target_ch = int(target_row["channel_number"])
    index = 0
    for row in sorted(rows, key=lambda r: int(r["channel_number"])):
        removed = row.get("remove", "").strip().upper() in ("TRUE", "1", "YES")
        if removed:
            continue
        if int(row["channel_number"]) == target_ch:
            return index
        index += 1
    raise ValueError(
        f"Channel {target_ch} not found among non-removed channels"
    )


def find_single_tiff(directory: Path, label: str) -> Path:
    """Find a single TIFF file in a directory."""
    tiffs = list(directory.glob("*.tif")) + list(directory.glob("*.tiff"))
    if len(tiffs) == 0:
        raise FileNotFoundError(f"No TIFF files found in {label}: {directory}")
    if len(tiffs) > 1:
        print(f"Warning: multiple TIFFs in {label}, using {tiffs[0].name}")
    return tiffs[0]


def main():
    parser = argparse.ArgumentParser(
        description="Replace DAPI in backsub with registered Scan DAPI"
    )
    parser.add_argument(
        "--registration-dir", required=True, type=Path,
        help="Path to MCMICRO registration/ directory",
    )
    parser.add_argument(
        "--backsub-dir", required=True, type=Path,
        help="Path to MCMICRO backsub/ directory",
    )
    parser.add_argument(
        "--markers-csv", required=True, type=Path,
        help="Path to markers.csv",
    )
    args = parser.parse_args()

    # --- 1. Parse markers ---
    rows = parse_markers(args.markers_csv)
    scan_dapi = find_scan_dapi(rows)
    cycle1_dapi = find_cycle1_dapi(rows)

    # Registration mosaic channel index (0-based from 1-based channel_number)
    reg_channel = int(scan_dapi["channel_number"]) - 1
    # Backsub channel index (0-based, skipping removed channels)
    backsub_channel = compute_backsub_index(rows, cycle1_dapi)

    print(
        f"Scan DAPI: channel_number={scan_dapi['channel_number']} "
        f"(registration page {reg_channel})"
    )
    print(
        f"Cycle1 DAPI: channel_number={cycle1_dapi['channel_number']} "
        f"(backsub page {backsub_channel})"
    )

    # --- 2. Read Scan DAPI from registration mosaic ---
    reg_tiff = find_single_tiff(args.registration_dir, "registration")
    print(f"Reading Scan DAPI from {reg_tiff.name}, page {reg_channel}")

    with tifffile.TiffFile(reg_tiff) as tif:
        scan_data = tif.pages[reg_channel].asarray()

    # --- 3. Read backsub OME-TIFF ---
    backsub_tiff = find_single_tiff(args.backsub_dir, "backsub")
    print(f"Reading backsub from {backsub_tiff.name}")

    backsub_data = tifffile.imread(backsub_tiff)

    # Handle both (C, Y, X) and (Y, X) shapes
    if backsub_data.ndim == 2:
        raise ValueError("Backsub TIFF has only one page — cannot replace channel")

    # --- 4. Verify dimensions match ---
    if scan_data.shape != backsub_data[backsub_channel].shape:
        raise ValueError(
            f"Shape mismatch: Scan DAPI {scan_data.shape} vs "
            f"backsub channel {backsub_data[backsub_channel].shape}"
        )

    # --- 5. Replace ---
    backsub_data[backsub_channel] = scan_data.astype(backsub_data.dtype)
    print(f"Replaced backsub page {backsub_channel} with Scan DAPI")

    # --- 6. Write back ---
    # Read OME metadata if present, to preserve it
    with tifffile.TiffFile(backsub_tiff) as tif:
        ome_metadata = tif.ome_metadata

    if ome_metadata:
        tifffile.imwrite(
            backsub_tiff,
            backsub_data,
            ome=True,
            description=ome_metadata,
        )
    else:
        tifffile.imwrite(backsub_tiff, backsub_data)

    print(f"Wrote updated backsub to {backsub_tiff.name}")


if __name__ == "__main__":
    main()
