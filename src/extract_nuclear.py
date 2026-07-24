#!/usr/bin/env python3
"""Extract a single named channel from MACSIMA/MCMICRO background-subtracted
OME-TIFFs and write it as a single-channel TIFF into a sibling ``Nuclear/`` folder.

Given a marker name (e.g. ``DAPI``) the script, for every ``background/`` folder it
finds under the supplied path:

  1. Reads ``markers_bs.csv`` and resolves the marker name to a plane index using
     the CSV *row order* (0-based), cross-checked against the channel names embedded
     in the image's OME-XML when those are present.
  2. Reads only that one plane -- peak memory is roughly one channel, not the whole
     stack, so a 56 GB / N-channel image only needs ~(56/N) GB of RAM.
  3. Writes it, tiled and (by default) losslessly compressed, to
     ``<roi>/Nuclear/<sample>_<marker>.ome.tif``, preserving dtype and pixel size.

Indexing note
-------------
MCMICRO's ``backsub`` re-indexes its output: the k-th row of ``markers_bs.csv`` is
the k-th plane of ``*_backsub.ome.tif``. The ``channel_number`` column refers to the
*original* raw acquisition and is **not** the plane index. This script therefore
uses CSV row order by default. When the OME-XML carries channel names, the script
matches on the name directly and warns if that disagrees with the CSV row position,
so either convention is handled correctly. Use ``--force-index`` to override.
"""

from __future__ import annotations

import argparse
import csv
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import tifffile


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MarkerRow:
    """One row of markers_bs.csv, with its 0-based position in the file."""

    row_index: int
    marker_name: str
    channel_number: int | None


@dataclass(frozen=True)
class Job:
    """A single unit of work: one background folder to process."""

    background_dir: Path
    tif_path: Path
    csv_path: Path
    roi_dir: Path  # parent of background/ -- where Nuclear/ is created


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def find_background_dirs(
    root: Path, csv_name: str, tif_glob: str
) -> list[Path]:
    """Return every directory at or under *root* that directly contains both a
    marker CSV and at least one matching TIFF. Handles pointing at a single
    background folder, a single ROI folder, or a whole batch tree uniformly."""
    hits: list[Path] = []
    candidates = [root] if root.is_dir() else []
    candidates += [p for p in root.rglob("*") if p.is_dir()]
    for d in candidates:
        if (d / csv_name).is_file() and any(d.glob(tif_glob)):
            hits.append(d)
    return sorted(set(hits))


def select_tif(background_dir: Path, tif_glob: str) -> Path:
    """Pick the OME-TIFF in *background_dir*. Prefer a file whose name contains
    'backsub' when the glob is ambiguous."""
    matches = sorted(background_dir.glob(tif_glob))
    if not matches:
        raise FileNotFoundError(
            f"No file matching {tif_glob!r} in {background_dir}"
        )
    if len(matches) == 1:
        return matches[0]
    backsub = [m for m in matches if "backsub" in m.name.lower()]
    if len(backsub) == 1:
        return backsub[0]
    listing = "\n  ".join(m.name for m in matches)
    raise ValueError(
        f"Multiple TIFFs match {tif_glob!r} in {background_dir}; "
        f"narrow --tif-glob. Candidates:\n  {listing}"
    )


def build_jobs(root: Path, csv_name: str, tif_glob: str) -> list[Job]:
    jobs: list[Job] = []
    for bg in find_background_dirs(root, csv_name, tif_glob):
        jobs.append(
            Job(
                background_dir=bg,
                tif_path=select_tif(bg, tif_glob),
                csv_path=bg / csv_name,
                roi_dir=bg.parent,
            )
        )
    return jobs


# --------------------------------------------------------------------------- #
# CSV parsing / marker resolution
# --------------------------------------------------------------------------- #
def load_markers(csv_path: Path) -> list[MarkerRow]:
    with csv_path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or "marker_name" not in reader.fieldnames:
            raise ValueError(
                f"{csv_path} has no 'marker_name' column "
                f"(found: {reader.fieldnames})"
            )
        rows: list[MarkerRow] = []
        for i, rec in enumerate(reader):
            name = (rec.get("marker_name") or "").strip()
            if not name:
                continue  # skip blank lines
            raw_cn = (rec.get("channel_number") or "").strip()
            try:
                cn: int | None = int(raw_cn) if raw_cn else None
            except ValueError:
                cn = None
            rows.append(MarkerRow(row_index=i, marker_name=name, channel_number=cn))
    if not rows:
        raise ValueError(f"No marker rows parsed from {csv_path}")
    return rows


def resolve_marker(
    rows: Sequence[MarkerRow], channel: str, ignore_case: bool
) -> MarkerRow:
    if ignore_case:
        matches = [r for r in rows if r.marker_name.lower() == channel.lower()]
    else:
        matches = [r for r in rows if r.marker_name == channel]
    if not matches:
        available = ", ".join(r.marker_name for r in rows)
        raise ValueError(
            f"Marker {channel!r} not found. Available markers:\n  {available}"
        )
    if len(matches) > 1:
        where = ", ".join(f"row {m.row_index}" for m in matches)
        raise ValueError(
            f"Marker {channel!r} is ambiguous ({where}). "
            f"Use an exact name or --force-index."
        )
    return matches[0]


# --------------------------------------------------------------------------- #
# OME-XML helpers (channel names + physical pixel size)
# --------------------------------------------------------------------------- #
def _localname(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def read_ome_channel_names(tif: tifffile.TiffFile) -> list[str] | None:
    """Return the ordered list of Channel/@Name from the OME-XML, or None."""
    xml = tif.ome_metadata
    if not xml:
        return None
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return None
    for el in root.iter():
        if _localname(el.tag) == "Pixels":
            names = [
                (c.get("Name") or "")
                for c in el
                if _localname(c.tag) == "Channel"
            ]
            if names and any(n for n in names):
                return names
            return None
    return None


def read_pixel_size(tif: tifffile.TiffFile) -> dict[str, object]:
    """Extract PhysicalSizeX/Y and units from the OME-XML if available."""
    out: dict[str, object] = {}
    xml = tif.ome_metadata
    if not xml:
        return out
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return out
    for el in root.iter():
        if _localname(el.tag) == "Pixels":
            for src, dst, unit_src, unit_dst in (
                ("PhysicalSizeX", "PhysicalSizeX", "PhysicalSizeXUnit", "PhysicalSizeXUnit"),
                ("PhysicalSizeY", "PhysicalSizeY", "PhysicalSizeYUnit", "PhysicalSizeYUnit"),
            ):
                val = el.get(src)
                if val:
                    try:
                        out[dst] = float(val)
                    except ValueError:
                        continue
                    out[unit_dst] = el.get(unit_src) or "µm"
            break
    return out


# --------------------------------------------------------------------------- #
# Plane index resolution + reading
# --------------------------------------------------------------------------- #
def base_level(series: tifffile.TiffPageSeries) -> tifffile.TiffPageSeries:
    """Full-resolution level of a (possibly pyramidal) series."""
    levels = getattr(series, "levels", None)
    return levels[0] if levels else series


def channel_axis_len(base: tifffile.TiffPageSeries) -> int:
    axes = base.axes
    stripped = axes.replace("Y", "").replace("X", "")
    if stripped in ("", "S"):  # single channel (S = samples, e.g. RGB kept as YXS)
        return 1
    if stripped == "C":
        return base.shape[axes.index("C")]
    raise ValueError(
        f"Unexpected axis layout {axes!r}; this script handles planar C,Y,X "
        f"images. Inspect the file with tifffile."
    )


def resolve_plane_index(
    row: MarkerRow,
    ome_names: list[str] | None,
    n_planes: int,
    n_csv_rows: int,
    channel: str,
    ignore_case: bool,
) -> int:
    """Decide which plane to pull, preferring the image's own channel names."""
    # 1) Trust embedded OME channel names when they exist and contain the marker.
    if ome_names:
        def _eq(a: str, b: str) -> bool:
            return a.lower() == b.lower() if ignore_case else a == b

        name_hits = [i for i, n in enumerate(ome_names) if _eq(n, channel)]
        if len(name_hits) == 1:
            idx = name_hits[0]
            if idx != row.row_index:
                print(
                    f"    note: OME-XML places {channel!r} at plane {idx}, "
                    f"CSV row order says {row.row_index}. Using OME-XML "
                    f"(the image's own metadata).",
                    file=sys.stderr,
                )
            return idx
        if len(name_hits) > 1:
            print(
                f"    warning: {channel!r} appears {len(name_hits)}x in OME-XML; "
                f"falling back to CSV row order.",
                file=sys.stderr,
            )

    # 2) Fall back to CSV row order, but sanity-check plane count.
    if n_planes != n_csv_rows:
        raise ValueError(
            f"Plane count mismatch: image has {n_planes} planes but "
            f"markers_bs.csv has {n_csv_rows} rows, and the OME-XML has no "
            f"usable channel names to disambiguate. Refusing to guess -- "
            f"verify the pairing, then use --force-index if the mapping is known."
        )
    if not 0 <= row.row_index < n_planes:
        raise ValueError(
            f"Row index {row.row_index} out of range for {n_planes}-plane image."
        )
    return row.row_index


def read_plane(base: tifffile.TiffPageSeries, plane_index: int) -> np.ndarray:
    """Read one 2D plane, decoding only that page's tiles."""
    pages = base.pages
    if not 0 <= plane_index < len(pages):
        raise IndexError(
            f"Plane {plane_index} out of range (image has {len(pages)} pages)."
        )
    page = pages[plane_index]
    if page is None:
        raise ValueError(f"Page {plane_index} is missing in the series.")
    arr = page.asarray()
    return np.squeeze(arr)


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #
def write_single_channel(
    out_path: Path,
    plane: np.ndarray,
    channel_name: str,
    pixel_size: dict[str, object],
    compression: str | None,
    tile: int,
) -> None:
    if plane.ndim != 2:
        raise ValueError(f"Expected a 2D plane, got shape {plane.shape}.")
    metadata: dict[str, object] = {"axes": "YX", "Channel": {"Name": [channel_name]}}
    metadata.update(pixel_size)
    # BigTIFF is decided from the *uncompressed* size (offsets are reserved before
    # compression); >~3.8 GB needs 64-bit offsets.
    bigtiff = plane.nbytes > 3_800_000_000
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".part")
    tifffile.imwrite(
        tmp,
        plane,
        ome=True,
        metadata=metadata,
        tile=(tile, tile),
        compression=compression,
        bigtiff=bigtiff,
    )
    tmp.replace(out_path)  # atomic: partial files never look complete


# --------------------------------------------------------------------------- #
# Per-job processing
# --------------------------------------------------------------------------- #
def sample_stem(tif_path: Path) -> str:
    name = tif_path.name
    for ext in (".ome.tif", ".ome.tiff", ".tif", ".tiff"):
        if name.lower().endswith(ext):
            name = name[: -len(ext)]
            break
    if name.lower().endswith("_backsub"):
        name = name[: -len("_backsub")]
    return name


def process_job(
    job: Job,
    channel: str,
    ignore_case: bool,
    force_index: int | None,
    out_dir_name: str,
    compression: str | None,
    tile: int,
    overwrite: bool,
    list_only: bool,
    dry_run: bool,
) -> bool:
    """Process one background folder. Returns True on a successful extraction."""
    rel = job.background_dir
    print(f"[{rel}]")
    rows = load_markers(job.csv_path)

    with tifffile.TiffFile(job.tif_path) as tif:
        base = base_level(tif.series[0])
        n_planes = channel_axis_len(base)
        ome_names = read_ome_channel_names(tif)
        pixel_size = read_pixel_size(tif)

        if list_only:
            print(f"    image: {job.tif_path.name}")
            print(f"    planes: {n_planes}   csv rows: {len(rows)}   "
                  f"dtype: {base.dtype}   shape: {tuple(base.shape)}")
            print(f"    OME channel names: "
                  f"{'yes' if ome_names else 'no'}")
            for r in rows:
                embedded = (
                    ome_names[r.row_index]
                    if ome_names and r.row_index < len(ome_names)
                    else "-"
                )
                print(f"      plane {r.row_index:>3}  csv={r.marker_name:<24} "
                      f"ome={embedded}")
            return False

        if force_index is not None:
            plane_index = force_index
            print(f"    marker {channel!r} -> plane {plane_index} (forced)")
        else:
            row = resolve_marker(rows, channel, ignore_case)
            plane_index = resolve_plane_index(
                row, ome_names, n_planes, len(rows), channel, ignore_case
            )
            print(f"    marker {channel!r} -> plane {plane_index} "
                  f"(csv row {row.row_index}, channel_number "
                  f"{row.channel_number})")

        out_dir = job.roi_dir / out_dir_name
        out_path = out_dir / f"{sample_stem(job.tif_path)}_{channel}.ome.tif"

        if out_path.exists() and not overwrite:
            print(f"    skip: {out_path} exists (use --overwrite)")
            return False

        page = base.pages[plane_index]
        est_gb = (page.size * base.dtype.itemsize) / 1e9 if page else 0.0
        print(f"    reading plane -> ~{est_gb:.2f} GB in RAM, writing {out_path}")

        if dry_run:
            print("    dry-run: not reading/writing")
            return False

        plane = read_plane(base, plane_index)

    write_single_channel(
        out_path, plane, channel, pixel_size, compression, tile
    )
    size_gb = out_path.stat().st_size / 1e9
    print(f"    done: {out_path} ({size_gb:.2f} GB, {plane.shape} {plane.dtype})")
    return True


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  # one ROI, extract DAPI\n"
            "  %(prog)s DAPI .../rack-01-well-A01-roi-001-exp-2\n\n"
            "  # whole batch (recurses to find every background/ folder)\n"
            "  %(prog)s DAPI .\n\n"
            "  # just inspect the marker->plane mapping, read nothing large\n"
            "  %(prog)s DAPI . --list\n"
        ),
    )
    p.add_argument("channel", help="Marker name to extract (e.g. DAPI).")
    p.add_argument(
        "path",
        type=Path,
        help="A background/ folder, an ROI folder, or a tree to search.",
    )
    p.add_argument(
        "--out-dir-name",
        default="Nuclear",
        help="Name of the sibling output folder (default: Nuclear).",
    )
    p.add_argument(
        "--csv-name",
        default="markers_bs.csv",
        help="Marker CSV filename (default: markers_bs.csv).",
    )
    p.add_argument(
        "--tif-glob",
        default="*.ome.tif",
        help="Glob for the source image (default: *.ome.tif).",
    )
    p.add_argument(
        "--compression",
        default="zlib",
        choices=["none", "zlib", "lzw"],
        help="Output compression (default: zlib/deflate, lossless).",
    )
    p.add_argument(
        "--tile", type=int, default=512, help="Output tile size (default: 512)."
    )
    p.add_argument(
        "--ignore-case",
        action="store_true",
        help="Case-insensitive marker matching.",
    )
    p.add_argument(
        "--force-index",
        type=int,
        default=None,
        metavar="N",
        help="Bypass name lookup and extract plane N (0-based).",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs (default: skip, for resumability).",
    )
    p.add_argument(
        "--list",
        dest="list_only",
        action="store_true",
        help="List markers and resolved plane indices, then exit. Reads no pixels.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve everything and report, but read/write nothing large.",
    )
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.path.exists():
        print(f"error: path does not exist: {args.path}", file=sys.stderr)
        return 2

    compression = None if args.compression == "none" else args.compression
    jobs = build_jobs(args.path, args.csv_name, args.tif_glob)
    if not jobs:
        print(
            f"error: no folder under {args.path} contains both {args.csv_name!r} "
            f"and {args.tif_glob!r}",
            file=sys.stderr,
        )
        return 2

    print(f"Found {len(jobs)} background folder(s).\n")
    written = 0
    failed = 0
    for job in jobs:
        try:
            if process_job(
                job,
                channel=args.channel,
                ignore_case=args.ignore_case,
                force_index=args.force_index,
                out_dir_name=args.out_dir_name,
                compression=compression,
                tile=args.tile,
                overwrite=args.overwrite,
                list_only=args.list_only,
                dry_run=args.dry_run,
            ):
                written += 1
        except (ValueError, FileNotFoundError, IndexError, OSError) as exc:
            failed += 1
            print(f"    ERROR: {exc}", file=sys.stderr)
        print()

    if not args.list_only:
        print(f"Summary: {written} written, {failed} failed, "
              f"{len(jobs) - written - failed} skipped.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
