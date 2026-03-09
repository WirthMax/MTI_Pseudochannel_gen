#!/usr/bin/env python3
"""Replace the DAPI channel in backsub output with the registered Scan DAPI.

After MCMICRO runs with the Scan DAPI injected as an extra ASHLAR round
(cycle 999), this script:
  1. Reads markers.csv to find the Scan DAPI channel (cycle 999) and the
     Cycle1 DAPI channel (earliest cycle, not removed).
  2. Extracts the Scan DAPI plane from the registration mosaic.
  3. Overwrites the corresponding DAPI plane in the backsub OME-TIFF.

Two modes of operation:
  A) --registration-dir: Extract Scan DAPI from the registration mosaic
     using the cycle 999 entry in markers.csv (legacy).
  B) --scan-dapi-file: Read Scan DAPI directly from a standalone TIFF file.
     Use this when cycle 999 ended up in a separate exposure folder.
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


def read_scan_dapi_from_file(tiff_path: Path) -> np.ndarray:
    """Read Scan DAPI from a standalone TIFF file.

    Handles multi-page TIFFs by taking the first page.
    Validates that the result is a 2D image.
    """
    with tifffile.TiffFile(tiff_path) as tif:
        if len(tif.pages) > 1:
            print(f"Warning: {tiff_path.name} has {len(tif.pages)} pages, using first")
        data = tif.pages[0].asarray()

    if data.ndim != 2:
        raise ValueError(
            f"Expected 2D image from {tiff_path.name}, got shape {data.shape}"
        )
    return data


def main():
    parser = argparse.ArgumentParser(
        description="Replace DAPI in backsub with registered Scan DAPI"
    )
    parser.add_argument(
        "--registration-dir", type=Path,
        help="Path to MCMICRO registration/ directory "
             "(required unless --scan-dapi-file is given)",
    )
    parser.add_argument(
        "--backsub-dir", required=True, type=Path,
        help="Path to MCMICRO backsub/ directory",
    )
    parser.add_argument(
        "--markers-csv", required=True, type=Path,
        help="Path to markers.csv",
    )
    parser.add_argument(
        "--scan-dapi-file", type=Path,
        help="Path to a standalone Scan DAPI TIFF file. "
             "Use when cycle 999 is in a separate exposure folder.",
    )
    args = parser.parse_args()

    # Validate: need either --scan-dapi-file or --registration-dir
    if not args.scan_dapi_file and not args.registration_dir:
        parser.error("one of --scan-dapi-file or --registration-dir is required")

    # --- 1. Parse markers ---
    rows = parse_markers(args.markers_csv)
    cycle1_dapi = find_cycle1_dapi(rows)
    backsub_channel = compute_backsub_index(rows, cycle1_dapi)

    print(
        f"Cycle1 DAPI: channel_number={cycle1_dapi['channel_number']} "
        f"(backsub page {backsub_channel})"
    )

    # --- 2. Read Scan DAPI ---
    if args.scan_dapi_file:
        # Direct file path — skip markers.csv cycle-999 lookup
        print(f"Reading Scan DAPI from file: {args.scan_dapi_file}")
        scan_data = read_scan_dapi_from_file(args.scan_dapi_file)
    else:
        # Legacy path — extract from registration mosaic via cycle-999
        scan_dapi = find_scan_dapi(rows)
        reg_channel = int(scan_dapi["channel_number"]) - 1
        print(
            f"Scan DAPI: channel_number={scan_dapi['channel_number']} "
            f"(registration page {reg_channel})"
        )
        reg_tiff = find_single_tiff(args.registration_dir, "registration")
        print(f"Reading Scan DAPI from {reg_tiff.name}, page {reg_channel}")
        with tifffile.TiffFile(reg_tiff) as tif:
            scan_data = tif.pages[reg_channel].asarray()

    # --- 3. Read backsub and replace target channel ---
    backsub_tiff = find_single_tiff(args.backsub_dir, "backsub")
    print(f"Backsub TIFF: {backsub_tiff.name}")

    with tifffile.TiffFile(backsub_tiff) as tif:
        if len(tif.pages) < 2:
            raise ValueError("Backsub TIFF has only one page — cannot replace channel")

        page = tif.pages[backsub_channel]

        if scan_data.shape != (page.shape[0], page.shape[1]):
            raise ValueError(
                f"Shape mismatch: Scan DAPI {scan_data.shape} vs "
                f"backsub channel {page.shape}"
            )

        contiguous = page.is_contiguous
        if contiguous:
            # Fast path: overwrite page bytes in-place via memmap
            offset, size = contiguous
            print(f"In-place replacing backsub page {backsub_channel} (offset {offset})")
            mm = np.memmap(
                backsub_tiff, dtype=page.dtype, mode="r+",
                offset=offset, shape=page.shape,
            )
            mm[:] = scan_data.astype(page.dtype)
            del mm  # flush to disk
        else:
            # Fallback: read all channels, modify, write back
            print(f"Page not contiguous — falling back to full read/write")
            ome_metadata = tif.ome_metadata
            backsub_data = tifffile.imread(backsub_tiff)
            backsub_data[backsub_channel] = scan_data.astype(backsub_data.dtype)
            if ome_metadata:
                tifffile.imwrite(backsub_tiff, backsub_data, ome=True, description=ome_metadata)
            else:
                tifffile.imwrite(backsub_tiff, backsub_data)

    print(f"Replaced backsub page {backsub_channel} with Scan DAPI")


if __name__ == "__main__":
    main()
