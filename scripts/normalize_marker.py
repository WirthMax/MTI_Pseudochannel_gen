#!/usr/bin/env python3
"""Export a single marker from an MCMICRO / OME-TIFF and normalize it for
manual coregistration.

Some markers "don't work" as stains but their weak/uneven signal still traces
the tissue background (autofluorescence, morphology). This script pulls one
channel out of a multi-channel stack and stretches its contrast so that faint
tissue structure becomes clearly visible by eye, then writes a viewer-friendly
image (PNG or TIFF) you can drag into Fiji / BigWarp / QuPath and align by hand.

Normalization pipeline:
    robust percentile stretch  ->  method (clahe | percentile | gamma)
                               ->  optional gamma  ->  optional invert
                               ->  scale to uint8 (PNG) or uint16 (TIFF)

CLAHE (Contrast-Limited Adaptive Histogram Equalization, the default) locally
equalizes contrast so structure pops even where illumination/signal is uneven -
ideal for reading tissue outline off a marker that barely worked.

The marker can be selected by name (needs channel names, from a markers.csv
sidecar or the OME-XML metadata) or by raw plane index (needs nothing). A plain
single-channel input is used as-is.
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import tifffile

# Make the src/ packages importable when run from a checkout (scripts/ is a
# sibling of src/, not inside it).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pseudochannel.core import normalize_channel  # noqa: E402
from pseudochannel.io import (  # noqa: E402
    load_marker_names,
    load_mcmicro_markers,
)


# --------------------------------------------------------------------------- #
# Channel discovery / plane reading
# --------------------------------------------------------------------------- #
def _series_layout(series) -> tuple[int, int]:
    """Return (n_channels, n_z) for a tifffile series, tolerating 2/3/4D."""
    shape = series.shape
    ndim = len(shape)
    if ndim == 2:
        return 1, 1
    if ndim == 3:  # (C, Y, X)
        return shape[0], 1
    if ndim == 4:  # (C, Z, Y, X)
        return shape[0], shape[1]
    raise ValueError(f"Unexpected image dimensions: {shape}")


def _read_plane(series, idx: int, n_z: int, z: int = 0) -> np.ndarray:
    """Decode only the page for channel `idx` (mirrors OMETiffChannels)."""
    ndim = len(series.shape)
    if ndim == 2:
        return series.pages[0].asarray()
    if ndim == 3:
        return series.pages[idx].asarray()
    # 4D (C, Z, Y, X): pages laid out as C*Z
    return series.pages[idx * n_z + z].asarray()


def _read_ome_channel_names(tif: tifffile.TiffFile) -> list[str]:
    """Ordered Channel/@Name entries from OME-XML, or [] if unavailable."""
    xml = tif.ome_metadata
    if not xml:
        return []
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return []
    names: list[str] = []
    for chan in root.iter():
        if chan.tag.rsplit("}", 1)[-1] == "Channel":
            names.append(chan.attrib.get("Name", ""))
    return names


def _is_mcmicro_markers(csv_path: Path) -> bool:
    """True if the CSV header contains a `marker_name` column (MCMICRO format)."""
    try:
        with open(csv_path, newline="") as f:
            header = f.readline()
    except OSError:
        return False
    return "marker_name" in [c.strip().lower() for c in header.split(",")]


def _resolve_marker_names(
    input_path: Path,
    markers_csv: Optional[Path],
    mcmicro_markers: bool,
    tif: tifffile.TiffFile,
) -> tuple[list[str], str]:
    """Return (channel_names, source_label).

    Preference: explicit --markers-csv > sibling markers.csv > OME-XML metadata.
    """
    csv_path = markers_csv
    if csv_path is None:
        sibling = input_path.parent / "markers.csv"
        if sibling.exists():
            csv_path = sibling

    if csv_path is not None:
        if not csv_path.exists():
            raise FileNotFoundError(f"markers file not found: {csv_path}")
        # MCMICRO markers.csv has a `marker_name` column; auto-detect it so the
        # header row and `remove` filtering are handled correctly without
        # requiring --mcmicro-markers.
        if mcmicro_markers or _is_mcmicro_markers(csv_path):
            names = load_mcmicro_markers(csv_path)
        else:
            names = load_marker_names(csv_path)
        return names, f"markers file {csv_path.name}"

    return _read_ome_channel_names(tif), "OME-XML metadata"


def _match_index(names: Sequence[str], marker: str) -> int:
    """Index of `marker` in `names` (exact, then case-insensitive)."""
    if marker in names:
        return list(names).index(marker)
    lowered = [n.lower() for n in names]
    if marker.lower() in lowered:
        return lowered.index(marker.lower())
    raise ValueError(
        f"marker {marker!r} not found. Available channels: "
        + (", ".join(n for n in names if n) or "(none)")
    )


def load_marker_plane(
    input_path: Path,
    marker: Optional[str],
    channel_index: Optional[int],
    markers_csv: Optional[Path],
    mcmicro_markers: bool,
) -> tuple[np.ndarray, str]:
    """Load a single 2D plane and a short label describing it."""
    with tifffile.TiffFile(str(input_path)) as tif:
        if not tif.series:
            raise ValueError(f"No image series found in {input_path}")
        series = tif.series[0]
        n_channels, n_z = _series_layout(series)

        # Single-channel input: nothing to select.
        if n_channels == 1:
            if marker or channel_index not in (None, 0):
                print(
                    "  note: input is single-channel; ignoring marker/index "
                    "selection.",
                    file=sys.stderr,
                )
            return _read_plane(series, 0, n_z), "ch0"

        if channel_index is not None:
            if channel_index < 0 or channel_index >= n_channels:
                raise IndexError(
                    f"--channel-index {channel_index} out of range "
                    f"(0-{n_channels - 1})"
                )
            return _read_plane(series, channel_index, n_z), f"ch{channel_index}"

        if marker is not None:
            names, source = _resolve_marker_names(
                input_path, markers_csv, mcmicro_markers, tif
            )
            if not names:
                raise ValueError(
                    "cannot resolve marker by name: no markers.csv found and no "
                    "channel names in OME-XML. Use --channel-index instead."
                )
            if len(names) != n_channels:
                print(
                    f"  warning: {len(names)} names in {source} but "
                    f"{n_channels} image channels; matching by position.",
                    file=sys.stderr,
                )
            idx = _match_index(names, marker)
            return _read_plane(series, idx, n_z), marker

        raise ValueError(
            "multi-channel input: specify --marker NAME or --channel-index N "
            "(use --list to see available channels)."
        )


def list_channels(
    input_path: Path,
    markers_csv: Optional[Path],
    mcmicro_markers: bool,
) -> None:
    """Print channel index -> name mapping and exit."""
    with tifffile.TiffFile(str(input_path)) as tif:
        series = tif.series[0]
        n_channels, _ = _series_layout(series)
        names, source = _resolve_marker_names(
            input_path, markers_csv, mcmicro_markers, tif
        )
    print(f"{input_path}: {n_channels} channel(s)")
    if names:
        print(f"channel names (from {source}):")
        for i in range(n_channels):
            name = names[i] if i < len(names) else "?"
            print(f"  [{i:>3}] {name}")
    else:
        print("  no channel names available; select with --channel-index.")


# --------------------------------------------------------------------------- #
# Normalization
# --------------------------------------------------------------------------- #
def normalize_for_display(
    plane: np.ndarray,
    method: str = "clahe",
    low: float = 1.0,
    high: float = 99.5,
    gamma: float = 1.0,
    clahe_clip: float = 0.01,
    clahe_kernel: Optional[int] = None,
    invert: bool = False,
    dtype: str = "uint8",
) -> np.ndarray:
    """Stretch/equalize a plane to reveal faint tissue background.

    Returns a uint8 or uint16 array ready to write.
    """
    # Robust global percentile stretch to [0, 1] (reused repo helper).
    img = normalize_channel(
        plane, method="percentile", percentile_low=low, percentile_high=high
    )

    if method == "clahe":
        from skimage import exposure  # lazy: only needed for this branch

        kernel = int(clahe_kernel) if clahe_kernel else None
        img = exposure.equalize_adapthist(
            img, kernel_size=kernel, clip_limit=clahe_clip
        ).astype(np.float32)
    elif method in ("percentile", "gamma"):
        pass  # percentile stretch already applied above
    else:
        raise ValueError(f"unknown --method: {method}")

    if gamma != 1.0:
        img = np.power(np.clip(img, 0, 1), gamma, dtype=np.float32)

    if invert:
        img = 1.0 - img

    img = np.clip(img, 0, 1)
    if dtype == "uint16":
        return (img * 65535.0 + 0.5).astype(np.uint16)
    return (img * 255.0 + 0.5).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def resolve_dtype(output_path: Path, dtype_arg: str) -> str:
    if dtype_arg != "auto":
        return dtype_arg
    return "uint8" if output_path.suffix.lower() == ".png" else "uint16"


def write_image(img: np.ndarray, output_path: Path) -> None:
    suffix = output_path.suffix.lower()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if suffix == ".png":
        from PIL import Image

        Image.MAX_IMAGE_PIXELS = None  # allow very large tissue images
        if img.dtype != np.uint8:
            raise ValueError("PNG output requires uint8 (use --dtype uint8).")
        Image.fromarray(img).save(output_path)
    elif suffix in (".tif", ".tiff"):
        big = img.nbytes > 3_800_000_000
        tifffile.imwrite(
            str(output_path),
            img,
            compression="zlib",
            tile=(512, 512),
            bigtiff=big,
        )
    else:
        raise ValueError(f"unsupported output extension: {output_path.suffix}")


IMAGE_EXTS = (".tif", ".tiff")
_IMAGE_STEM_EXTS = (".ome.tif", ".ome.tiff", ".tiff", ".tif")


def _safe(text: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in text)


def _image_stem(path: Path) -> str:
    """Filename without image extension (handles multi-part `.ome.tif`)."""
    name = path.name
    for ext in _IMAGE_STEM_EXTS:
        if name.lower().endswith(ext):
            return name[: -len(ext)]
    return path.stem


def output_ext(args, explicit: Optional[Path]) -> str:
    if explicit is not None:
        return explicit.suffix
    return ".png" if args.format == "png" else ".tif"


def build_output_path(
    job: "Job", label: str, args, taken: set[Path]
) -> Path:
    """Compute the output path for a job (batch-aware, collision-safe)."""
    ext = output_ext(args, None)
    suffix = args.suffix if args.suffix is not None else f"_{_safe(label)}_{args.method}"
    fname = f"{job.base_stem}{suffix}{ext}"

    if args.output_dir is None:
        # Write next to the source image.
        return job.input_path.parent / fname

    # Consolidated output dir: disambiguate colliding names with the job prefix.
    out = args.output_dir / fname
    if out in taken:
        out = args.output_dir / f"{_safe(job.prefix)}_{fname}"
    n = 1
    while out in taken:
        out = args.output_dir / f"{_safe(job.prefix)}_{n}_{fname}"
        n += 1
    taken.add(out)
    return out


# --------------------------------------------------------------------------- #
# Batch discovery
# --------------------------------------------------------------------------- #
@dataclass
class Job:
    """One image to process, with its marker file and naming info.

    base_stem is the descriptive filename stem for outputs. For MCMICRO trees
    it embeds the experiment folder name (the image basenames repeat between
    experiments, so the basename alone is not unique). prefix is an extra
    disambiguator used only if two jobs still collide in a consolidated
    --output-dir.
    """

    input_path: Path
    markers_csv: Optional[Path]
    prefix: str
    base_stem: str


def discover_jobs(
    input_path: Path,
    *,
    mcmicro: bool,
    glob: str,
    recursive: bool,
    markers_csv: Optional[Path],
) -> list[Job]:
    """Expand an input path into a list of jobs.

    - A file           -> one job.
    - A dir + --mcmicro -> reuse find_mcmicro_experiments (per-experiment
                           image + markers.csv).
    - A dir (plain)     -> every matching image in the folder (recursive with
                           --recursive); markers.csv defaults to a sibling.
    """
    if input_path.is_file():
        stem = _image_stem(input_path)
        return [Job(input_path, markers_csv, stem, stem)]

    if mcmicro:
        from pseudochannel.batch import find_mcmicro_experiments

        jobs = []
        for exp in find_mcmicro_experiments(input_path):
            image_path = exp["image_path"]
            exp_name = exp["experiment_path"].name
            img_stem = _image_stem(image_path)
            # Embed the experiment folder name so outputs stay distinguishable
            # even when per-ROI image basenames repeat across experiments.
            base_stem = img_stem if img_stem.startswith(exp_name) else f"{exp_name}_{img_stem}"
            jobs.append(
                Job(
                    input_path=image_path,
                    markers_csv=markers_csv or exp["marker_path"],
                    prefix=exp_name,
                    base_stem=base_stem,
                )
            )
        return jobs

    matches = input_path.rglob(glob) if recursive else input_path.glob(glob)
    files = sorted(
        p for p in matches if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    return [Job(p, markers_csv, p.parent.name, _image_stem(p)) for p in files]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  # CLAHE-normalized PNG of a marker, by name (uses sibling markers.csv)\n"
            "  %(prog)s image.ome.tif --marker CD31 -o cd31_bg.png\n\n"
            "  # by raw plane index, no CSV needed\n"
            "  %(prog)s image.ome.tif --channel-index 7 -o plane7.png\n\n"
            "  # brighten dark tissue with gamma, write 16-bit TIFF\n"
            "  %(prog)s image.ome.tif --marker Vimentin --gamma 0.5 -o v.tif\n\n"
            "  # batch: every TIFF in a folder -> PNGs beside each source\n"
            "  %(prog)s ./images --marker CD31\n\n"
            "  # batch: whole MCMICRO tree -> one output dir, per-experiment markers.csv\n"
            "  %(prog)s ./mcmicro_root --mcmicro --marker CD31 --output-dir ./bg_pngs\n\n"
            "  # batch with a custom suffix written next to each source\n"
            "  %(prog)s ./images --channel-index 0 --suffix _bg --format tif\n\n"
            "  # just inspect the available channels\n"
            "  %(prog)s image.ome.tif --list\n"
        ),
    )
    p.add_argument(
        "input", type=Path,
        help="Input TIFF/OME-TIFF, OR a directory to batch-process.",
    )
    p.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Single-file output path; extension picks the format (.png -> "
        "8-bit, .tif/.tiff -> 16-bit). Not valid with a directory input (use "
        "--output-dir / --suffix). Default: <input>_<marker>_<method>.png next "
        "to the input.",
    )

    sel = p.add_argument_group("marker selection")
    sel.add_argument("--marker", help="Marker name to extract (e.g. CD31).")
    sel.add_argument(
        "--channel-index", type=int, default=None, metavar="N",
        help="Extract raw plane N (0-based); no markers.csv needed.",
    )
    sel.add_argument(
        "--markers-csv", type=Path, default=None,
        help="Marker names file for name lookup (default: markers.csv next to "
        "the input, else OME-XML channel names).",
    )
    sel.add_argument(
        "--mcmicro-markers", action="store_true",
        help="Parse --markers-csv as MCMICRO format (marker_name + remove cols).",
    )

    norm = p.add_argument_group("normalization")
    norm.add_argument(
        "--method", choices=["clahe", "percentile", "gamma"], default="clahe",
        help="Contrast method (default: clahe - best for faint tissue).",
    )
    norm.add_argument(
        "--low", type=float, default=1.0,
        help="Low percentile for the initial stretch (default: 1.0).",
    )
    norm.add_argument(
        "--high", type=float, default=99.5,
        help="High percentile for the initial stretch (default: 99.5).",
    )
    norm.add_argument(
        "--gamma", type=float, default=1.0,
        help="Gamma applied after the method; <1 brightens dark tissue "
        "(default: 1.0 = off).",
    )
    norm.add_argument(
        "--clahe-clip", type=float, default=0.01,
        help="CLAHE clip limit, 0-1; higher = more contrast/noise "
        "(default: 0.01).",
    )
    norm.add_argument(
        "--clahe-kernel", type=int, default=None, metavar="PX",
        help="CLAHE tile size in pixels (default: skimage auto ~1/8 of image).",
    )
    norm.add_argument(
        "--invert", action="store_true",
        help="Invert intensities (dark tissue on light background).",
    )

    batch = p.add_argument_group("batch (directory input)")
    batch.add_argument(
        "--mcmicro", action="store_true",
        help="Treat the input directory as an MCMICRO tree: process each "
        "background/ OME-TIFF using its sibling markers.csv.",
    )
    batch.add_argument(
        "--glob", default="*.tif*",
        help="Glob for images in a plain folder (default: *.tif*). "
        "Ignored with --mcmicro.",
    )
    batch.add_argument(
        "-r", "--recursive", action="store_true",
        help="Recurse into subfolders when scanning a plain folder.",
    )
    batch.add_argument(
        "--output-dir", type=Path, default=None,
        help="Write all outputs into this one directory (names disambiguated "
        "on collision). Default: beside each source image.",
    )
    batch.add_argument(
        "--suffix", default=None,
        help="Output filename suffix appended to the source stem "
        "(default: _<marker>_<method>).",
    )
    batch.add_argument(
        "--format", choices=["png", "tif"], default="png",
        help="Output format when no explicit -o is given (default: png).",
    )

    p.add_argument(
        "--dtype", choices=["auto", "uint8", "uint16"], default="auto",
        help="Output bit depth (default: auto from extension).",
    )
    p.add_argument(
        "--list", dest="list_only", action="store_true",
        help="List channels and exit; reads no pixels.",
    )
    p.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing outputs (default: skip/refuse).",
    )
    return p


def process_job(job: Job, args, taken: set[Path], explicit: Optional[Path]) -> str:
    """Process one image. Returns 'written', 'skipped', or 'failed'."""
    plane, label = load_marker_plane(
        job.input_path,
        marker=args.marker,
        channel_index=args.channel_index,
        markers_csv=job.markers_csv,
        mcmicro_markers=args.mcmicro_markers,
    )

    output_path = explicit or build_output_path(job, label, args, taken)
    if output_path.exists() and not args.overwrite:
        print(f"  skip (exists): {output_path}")
        return "skipped"

    out_dtype = resolve_dtype(output_path, args.dtype)
    print(
        f"  {job.input_path.name} [{label}] {plane.shape} {plane.dtype} "
        f"-> {args.method} -> {output_path.name} ({out_dtype})"
    )

    img = normalize_for_display(
        plane,
        method=args.method,
        low=args.low,
        high=args.high,
        gamma=args.gamma,
        clahe_clip=args.clahe_clip,
        clahe_kernel=args.clahe_kernel,
        invert=args.invert,
        dtype=out_dtype,
    )
    write_image(img, output_path)
    print(f"  wrote {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")
    return "written"


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.input.exists():
        print(f"error: input does not exist: {args.input}", file=sys.stderr)
        return 2

    if args.marker is not None and args.channel_index is not None:
        print(
            "error: use either --marker or --channel-index, not both.",
            file=sys.stderr,
        )
        return 2

    is_batch = args.input.is_dir()
    if is_batch and args.output is not None:
        print(
            "error: -o/--output is for a single file; use --output-dir / "
            "--suffix for a directory input.",
            file=sys.stderr,
        )
        return 2

    jobs = discover_jobs(
        args.input,
        mcmicro=args.mcmicro,
        glob=args.glob,
        recursive=args.recursive,
        markers_csv=args.markers_csv,
    )

    if not jobs:
        print(f"error: no images found under {args.input}", file=sys.stderr)
        if is_batch and not args.mcmicro:
            from pseudochannel.batch import find_mcmicro_experiments

            if find_mcmicro_experiments(args.input):
                print(
                    "  hint: this looks like an MCMICRO tree; try --mcmicro.",
                    file=sys.stderr,
                )
        return 2

    if args.list_only:
        for job in jobs:
            list_channels(job.input_path, job.markers_csv, args.mcmicro_markers)
        return 0

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    if is_batch:
        print(f"Processing {len(jobs)} image(s) from {args.input}\n")

    taken: set[Path] = set()
    written = skipped = failed = 0
    for job in jobs:
        explicit = args.output if not is_batch else None
        try:
            status = process_job(job, args, taken, explicit)
        except (ValueError, IndexError, FileNotFoundError, KeyError, OSError) as exc:
            print(f"  ERROR ({job.input_path.name}): {exc}", file=sys.stderr)
            failed += 1
            continue
        written += status == "written"
        skipped += status == "skipped"

    if is_batch or failed:
        print(
            f"\nSummary: {written} written, {skipped} skipped, {failed} failed."
        )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
