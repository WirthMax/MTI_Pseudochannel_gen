"""Superpixel feature extraction from multi-channel images.

Segmentation-free alternative to cell-based feature extraction: the image is
divided into a regular grid of square regions ("superpixels") and per-region
marker statistics (mean / sum / std) are aggregated into a feature table that
plugs directly into the downstream ``analysis`` pipeline.

The aggregation kernel generalizes
``analysis.features._extract_marker_intensities_vectorized`` -- an integer
label image plus ``np.bincount`` -- so any grid label image is aggregated in a
single vectorized pass over each channel.
"""

import sys
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
import tifffile

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from pseudochannel.io import OMETiffChannels
from pseudochannel.batch import find_mcmicro_experiments

# Statistics available per marker. Column names are "{marker}_{stat}".
VALID_STATS = ("mean", "sum", "std")

# Metadata columns emitted alongside the per-marker statistics. Kept in sync
# with the additions to analysis.preprocessing.METADATA_COLUMNS so downstream
# marker auto-detection ignores them.
SUPERPIXEL_METADATA_COLUMNS = [
    "label",
    "tile_row",
    "tile_col",
    "centroid_y",
    "centroid_x",
    "n_pixels",
    "total_signal",
]


def build_superpixel_labels(
    shape: tuple[int, int],
    size: int,
) -> tuple[np.ndarray, int, int]:
    """Build a row-major square-grid label image.

    Each ``size`` x ``size`` square receives a unique integer label. Squares at
    the right/bottom edges are simply smaller than ``size`` when the image
    dimensions are not exact multiples. Labels start at 1 (there is no
    background label -- the grid tiles the whole image).

    Args:
        shape: (height, width) of the image.
        size: Side length of each square superpixel, in pixels.

    Returns:
        Tuple of (label_img, n_rows, n_cols) where ``label_img`` is an int32
        array of ``shape`` with values ``1..n_rows*n_cols``.
    """
    if size < 1:
        raise ValueError(f"Superpixel size must be >= 1, got {size}")

    height, width = shape
    row_idx = np.arange(height) // size
    col_idx = np.arange(width) // size
    n_rows = int(row_idx[-1]) + 1
    n_cols = int(col_idx[-1]) + 1

    # label = row * n_cols + col + 1  (row-major, 1-based)
    label_img = (row_idx[:, None] * n_cols + col_idx[None, :] + 1).astype(np.int32)
    return label_img, n_rows, n_cols


def _aggregate_regions(
    label_img: np.ndarray,
    channels: dict[str, np.ndarray],
    markers: Sequence[str],
    stats: Sequence[str] = VALID_STATS,
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    """Aggregate per-region marker statistics over an integer label image.

    Generalizes ``features._extract_marker_intensities_vectorized`` to also
    produce sum and std using the sum-of-squares identity, all via
    ``np.bincount`` (one pass per channel).

    Args:
        label_img: 2D int array with labels ``1..N`` (0 is treated as
            background and ignored, though the superpixel grid uses no 0).
        channels: Dict mapping marker_name -> 2D intensity array.
        markers: Marker names to aggregate.
        stats: Which statistics to compute (subset of VALID_STATS).

    Returns:
        Tuple of (labels, stat_arrays, counts):
        - labels: 1D array of the unique non-zero labels (sorted).
        - stat_arrays: dict "{marker}_{stat}" -> 1D array aligned with labels.
        - counts: 1D array of pixel counts per label (aligned with labels).
    """
    invalid = [s for s in stats if s not in VALID_STATS]
    if invalid:
        raise ValueError(f"Unknown stat(s) {invalid}; valid: {VALID_STATS}")

    labels = np.unique(label_img)
    labels = labels[labels != 0]

    if len(labels) == 0:
        return labels, {}, np.array([], dtype=np.int64)

    mask_flat = label_img.ravel()
    minlength = int(labels.max()) + 1
    counts_full = np.bincount(mask_flat, minlength=minlength)
    counts = counts_full[labels]

    stat_arrays: dict[str, np.ndarray] = {}
    need_mean = "mean" in stats or "std" in stats

    for marker in markers:
        if marker not in channels:
            continue

        channel_data = np.asarray(channels[marker], dtype=np.float64).ravel()
        sums = np.bincount(mask_flat, weights=channel_data, minlength=minlength)

        if "sum" in stats:
            stat_arrays[f"{marker}_sum"] = sums[labels]

        if need_mean:
            with np.errstate(divide="ignore", invalid="ignore"):
                means_full = sums / counts_full
                means_full = np.nan_to_num(means_full, nan=0.0)
            means = means_full[labels]
            if "mean" in stats:
                stat_arrays[f"{marker}_mean"] = means

        if "std" in stats:
            sumsq = np.bincount(
                mask_flat, weights=channel_data * channel_data, minlength=minlength
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                mean_sq_full = sumsq / counts_full
                mean_sq_full = np.nan_to_num(mean_sq_full, nan=0.0)
            variance = mean_sq_full[labels] - means * means
            variance = np.clip(variance, 0.0, None)  # guard tiny negatives
            stat_arrays[f"{marker}_std"] = np.sqrt(variance)

    return labels, stat_arrays, counts


def extract_superpixel_features(
    channels: Union[dict[str, np.ndarray], OMETiffChannels],
    size: int,
    markers: Optional[list[str]] = None,
    stats: Sequence[str] = VALID_STATS,
    remove_empty: bool = False,
    empty_threshold: float = 0.0,
    roi_name: Optional[str] = None,
) -> pd.DataFrame:
    """Compute a superpixel feature table from multi-channel image data.

    Args:
        channels: Dict-like of marker_name -> 2D array (e.g. OMETiffChannels).
            All channels must share the same (H, W) shape.
        size: Superpixel square side length in pixels.
        markers: Markers to aggregate. If None, uses all channel keys.
        stats: Statistics per marker (subset of "mean", "sum", "std").
        remove_empty: If True, drop superpixels whose ``total_signal`` is
            <= ``empty_threshold`` (background / low-signal regions).
        empty_threshold: Threshold on ``total_signal`` (sum of per-marker mean
            intensities) for the empty filter. Default 0 drops only regions with
            no signal at all.
        roi_name: Optional ROI identifier inserted as the first column.

    Returns:
        DataFrame with metadata columns (label, tile_row, tile_col, centroid_y,
        centroid_x, n_pixels, total_signal) followed by "{marker}_{stat}"
        columns. One row per (kept) superpixel.
    """
    if markers is None:
        markers = list(channels.keys())

    # Determine image shape from the channel container.
    if isinstance(channels, OMETiffChannels):
        shape = channels.shape
    else:
        shape = np.asarray(channels[markers[0]]).shape[:2]

    label_img, n_rows, n_cols = build_superpixel_labels(tuple(shape), size)

    labels, stat_arrays, counts = _aggregate_regions(
        label_img, channels, markers, stats=stats
    )

    if len(labels) == 0:
        return pd.DataFrame({"label": []})

    # Grid geometry derived analytically from the row-major labels.
    tile_row = (labels - 1) // n_cols
    tile_col = (labels - 1) % n_cols
    centroid_y = tile_row * size + np.minimum(
        size, np.asarray(shape[0]) - tile_row * size
    ) / 2.0
    centroid_x = tile_col * size + np.minimum(
        size, np.asarray(shape[1]) - tile_col * size
    ) / 2.0

    # total_signal: sum of per-marker means when available, else derive from
    # sums / pixel count. Used as the emptiness proxy.
    mean_cols = [c for c in stat_arrays if c.endswith("_mean")]
    if mean_cols:
        total_signal = np.sum([stat_arrays[c] for c in mean_cols], axis=0)
    else:
        sum_cols = [c for c in stat_arrays if c.endswith("_sum")]
        if sum_cols:
            total = np.sum([stat_arrays[c] for c in sum_cols], axis=0)
            total_signal = total / np.maximum(counts, 1)
        else:
            total_signal = np.zeros(len(labels), dtype=np.float64)

    data = {
        "label": labels,
        "tile_row": tile_row,
        "tile_col": tile_col,
        "centroid_y": centroid_y,
        "centroid_x": centroid_x,
        "n_pixels": counts,
        "total_signal": total_signal,
    }
    # Preserve marker-then-stat ordering for readability.
    for marker in markers:
        for stat in stats:
            col = f"{marker}_{stat}"
            if col in stat_arrays:
                data[col] = stat_arrays[col]

    df = pd.DataFrame(data)

    if remove_empty:
        df = df[df["total_signal"] > empty_threshold].reset_index(drop=True)

    if roi_name is not None:
        df.insert(0, "ROI", roi_name)

    return df


def build_superpixel_mask(
    channels: Union[dict[str, np.ndarray], OMETiffChannels],
    size: int,
    remove_empty: bool = False,
    empty_threshold: float = 0.0,
    markers: Optional[list[str]] = None,
) -> np.ndarray:
    """Build a superpixel label image, optionally dropping empty regions.

    Each square gets a unique label (row-major, 1..N). When ``remove_empty`` is
    set, superpixels whose ``total_signal`` (sum of per-marker mean intensities)
    is ``<= empty_threshold`` are set to 0 (background). Retained superpixels
    keep their original label IDs, so the mask lines up with the ``label``
    column of :func:`extract_superpixel_features`.

    Args:
        channels: Dict-like of marker_name -> 2D array (e.g. OMETiffChannels).
        size: Superpixel square side length in pixels.
        remove_empty: If True, zero out empty superpixels.
        empty_threshold: Threshold on total_signal for the empty filter.
        markers: Markers used to compute total_signal. If None, uses all.

    Returns:
        int32 label image of the image shape (0 = background/removed).
    """
    if markers is None:
        markers = list(channels.keys())

    if isinstance(channels, OMETiffChannels):
        shape = channels.shape
    else:
        shape = np.asarray(channels[markers[0]]).shape[:2]

    label_img, _, _ = build_superpixel_labels(tuple(shape), size)

    if not remove_empty:
        return label_img

    labels, stat_arrays, _ = _aggregate_regions(
        label_img, channels, markers, stats=("mean",)
    )
    if len(labels) == 0:
        return label_img

    mean_cols = [c for c in stat_arrays if c.endswith("_mean")]
    if mean_cols:
        total_signal = np.sum([stat_arrays[c] for c in mean_cols], axis=0)
    else:
        total_signal = np.zeros(len(labels), dtype=np.float64)

    keep = labels[total_signal > empty_threshold]
    return _apply_keep_lut(label_img, keep)


def _apply_keep_lut(label_img: np.ndarray, keep_labels: np.ndarray) -> np.ndarray:
    """Zero out every label not in ``keep_labels`` (preserving kept IDs)."""
    lut = np.zeros(int(label_img.max()) + 1, dtype=label_img.dtype)
    keep_labels = np.asarray(keep_labels, dtype=label_img.dtype)
    lut[keep_labels] = keep_labels
    return lut[label_img]


def save_superpixel_masks(
    label_img: np.ndarray,
    output_dir: Union[str, Path],
    basename: str = "superpixel",
    formats: Sequence[str] = ("cellpose", "macsiqview"),
    compress: bool = True,
) -> dict[str, Path]:
    """Save a superpixel label image in Cellpose and/or MacsIQView formats.

    - ``cellpose``: a plain label-mask TIFF (uint16, or uint32 when labels
      exceed 65535). Unique label per superpixel, adjacent squares touching.
      The label IDs match the feature table's ``label`` column.
    - ``macsiqview``: the binary separated mask from ``Separate_masks.py`` --
      1px background gaps carved between differently-labeled neighbors, then
      binarized to uint8 0/1 (mirrors ``Separate_masks.save_image``).

    Args:
        label_img: 2D int label image (0 = background).
        output_dir: Directory to write into (created if missing).
        basename: Filename stem; outputs are ``{basename}_cellpose.tif`` and
            ``{basename}_MacsIQView.tif``.
        formats: Which formats to write (subset of "cellpose", "macsiqview").
        compress: Write compressed TIFFs.

    Returns:
        Dict mapping each requested format to the written Path.
    """
    valid = {"cellpose", "macsiqview"}
    invalid = [f for f in formats if f not in valid]
    if invalid:
        raise ValueError(f"Unknown mask format(s) {invalid}; valid: {sorted(valid)}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    compression = "zlib" if compress else None
    written: dict[str, Path] = {}

    if "cellpose" in formats:
        max_label = int(label_img.max())
        dtype = np.uint16 if max_label <= np.iinfo(np.uint16).max else np.uint32
        cp_path = output_dir / f"{basename}_cellpose.tif"
        tifffile.imwrite(str(cp_path), label_img.astype(dtype), compression=compression)
        written["cellpose"] = cp_path

    if "macsiqview" in formats:
        sep = _separate_borders(label_img.copy())
        sep[sep != 0] = 1  # binarize, matching Separate_masks CLI behavior
        mq_path = output_dir / f"{basename}_MacsIQView.tif"
        tifffile.imwrite(str(mq_path), sep.astype(np.uint8), compression=compression)
        written["macsiqview"] = mq_path

    return written


def _separate_borders(label_img: np.ndarray) -> np.ndarray:
    """Apply Separate_masks.separation_border_inplace, importing it lazily.

    ``Separate_masks`` is a top-level script under ``src/`` (not part of the
    ``analysis`` package) and pulls in Numba, so it is imported on demand.
    """
    try:
        from Separate_masks import separation_border_inplace
    except ImportError:
        # Ensure src/ (parent of this package) is importable, then retry.
        src_dir = str(Path(__file__).resolve().parent.parent)
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)
        try:
            from Separate_masks import separation_border_inplace
        except ImportError as e:
            raise ImportError(
                "The MacsIQView mask format needs Separate_masks.py (and Numba). "
                f"Could not import it from {src_dir}: {e}"
            ) from e
    return separation_border_inplace(label_img)


def compute_superpixel_features(
    image_path: Union[str, Path],
    marker_file: Union[str, Path],
    size: int,
    mcmicro_markers: bool = True,
    markers: Optional[list[str]] = None,
    stats: Sequence[str] = VALID_STATS,
    remove_empty: bool = False,
    empty_threshold: float = 0.0,
    roi_name: Optional[str] = None,
    output_path: Optional[Union[str, Path]] = None,
    save_masks: bool = False,
    mask_dir: Optional[Union[str, Path]] = None,
    mask_basename: str = "superpixel",
    mask_formats: Sequence[str] = ("cellpose", "macsiqview"),
) -> pd.DataFrame:
    """Single-image superpixel feature extraction entry point.

    Opens the OME-TIFF lazily (all channels retained -- no default exclusion),
    aggregates superpixel statistics, and optionally writes the feature CSV and
    the superpixel masks (Cellpose label TIFF + MacsIQView binary).

    Args:
        image_path: Path to the OME-TIFF with marker channels.
        marker_file: Path to markers.csv (or plain marker list).
        size: Superpixel square side length in pixels.
        mcmicro_markers: Parse ``marker_file`` as MCMICRO format if True.
        markers: Markers to aggregate. If None, uses all channels.
        stats: Statistics per marker (subset of "mean", "sum", "std").
        remove_empty: Drop empty superpixels (see extract_superpixel_features).
        empty_threshold: Threshold on total_signal for the empty filter.
        roi_name: Optional ROI identifier column.
        output_path: If given, write the feature table as CSV here.
        save_masks: If True, write the superpixel masks (see mask_* args). When
            ``remove_empty`` is set, the masks are thresholded to match the table.
        mask_dir: Directory for the mask TIFFs. Defaults to the CSV's parent
            when ``output_path`` is given, else the current directory.
        mask_basename: Filename stem for the mask outputs.
        mask_formats: Which mask formats to write (subset of "cellpose",
            "macsiqview").

    Returns:
        The superpixel feature DataFrame.
    """
    image_path = Path(image_path)
    marker_file = Path(marker_file)

    channels = OMETiffChannels(
        image_path,
        marker_file,
        mcmicro_markers=mcmicro_markers,
        exclude_channels=set(),  # keep ALL channels as features
    )
    try:
        df = extract_superpixel_features(
            channels,
            size=size,
            markers=markers,
            stats=stats,
            remove_empty=remove_empty,
            empty_threshold=empty_threshold,
            roi_name=roi_name,
        )
        shape = channels.shape
    finally:
        channels.close()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

    if save_masks:
        # Rebuild the (thresholded) label image from the computed table -- the
        # kept labels are exactly df["label"], so no second aggregation pass.
        label_img, _, _ = build_superpixel_labels(tuple(shape), size)
        if remove_empty and "label" in df:
            label_img = _apply_keep_lut(label_img, df["label"].to_numpy())
        target_dir = (
            Path(mask_dir)
            if mask_dir is not None
            else (output_path.parent if output_path is not None else Path("."))
        )
        save_superpixel_masks(
            label_img, target_dir, basename=mask_basename, formats=mask_formats
        )

    return df


def compute_superpixel_features_batch(
    root_path: Union[str, Path],
    size: int,
    background_folder: str = "background",
    marker_filename: str = "markers.csv",
    output_folder: str = "analysis",
    output_filename: str = "superpixel_features.csv",
    mcmicro_markers: bool = True,
    markers: Optional[list[str]] = None,
    stats: Sequence[str] = VALID_STATS,
    remove_empty: bool = False,
    empty_threshold: float = 0.0,
    save_masks: bool = False,
    mask_basename: str = "superpixel",
    mask_formats: Sequence[str] = ("cellpose", "macsiqview"),
    progress: bool = True,
    save_individual: bool = True,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Batch superpixel feature extraction over MCMICRO experiment folders.

    Mirrors ``features.extract_features_batch``: discovers experiments via
    ``find_mcmicro_experiments`` (each ``background/`` folder with a sibling
    markers.csv), extracts a superpixel table per ROI, writes per-ROI CSVs to
    ``<rack>/<output_folder>/``, and returns the concatenated table with an
    ``ROI`` column.

    Args:
        root_path: Root directory containing experiment folders.
        size: Superpixel square side length in pixels.
        background_folder: Name of the image subfolder.
        marker_filename: Name of the marker file.
        output_folder: Subfolder for feature output files.
        output_filename: Filename for the per-ROI feature CSV.
        mcmicro_markers: Parse marker files as MCMICRO format.
        markers: Markers to aggregate. If None, uses all channels.
        stats: Statistics per marker.
        remove_empty: Drop empty superpixels.
        empty_threshold: Threshold on total_signal for the empty filter.
        save_masks: If True, also write the superpixel masks per ROI (Cellpose
            label TIFF + MacsIQView binary), into ``<rack>/<output_folder>/``.
            Thresholded to match the table when ``remove_empty`` is set.
        mask_basename: Filename stem for the per-ROI mask outputs.
        mask_formats: Which mask formats to write (subset of "cellpose",
            "macsiqview").
        progress: Show a tqdm progress bar if available.
        save_individual: Write a CSV per experiment.
        overwrite: If False, skip experiments with an existing output CSV.

    Returns:
        Combined DataFrame with features from all experiments (``ROI`` column).
    """
    root_path = Path(root_path)

    experiments = find_mcmicro_experiments(
        root_path,
        background_folder=background_folder,
        marker_filename=marker_filename,
    )

    if not experiments:
        print(f"No MCMICRO experiments found in {root_path}")
        return pd.DataFrame()

    # Skip already-processed experiments unless overwrite.
    if not overwrite:
        to_process = []
        skipped = 0
        for exp_info in experiments:
            out_path = exp_info["background_path"].parent / output_folder / output_filename
            if out_path.exists():
                skipped += 1
            else:
                to_process.append(exp_info)
        if skipped > 0:
            print(f"Skipping {skipped} already processed (use overwrite=True)")
        pending = to_process
    else:
        pending = list(experiments)

    if not pending:
        print("All experiments already have superpixel features. Loading existing.")
        all_dfs = []
        for exp_info in experiments:
            out_path = exp_info["background_path"].parent / output_folder / output_filename
            if out_path.exists():
                all_dfs.append(pd.read_csv(out_path))
        return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

    print(f"Extracting superpixel features from {len(pending)} experiments")

    if progress and tqdm is not None:
        iterator = tqdm(pending, desc="Superpixel features")
    else:
        iterator = pending

    all_dfs = []
    for exp_info in iterator:
        try:
            relative_path = exp_info["background_path"].parent.relative_to(root_path)
            roi_name = str(relative_path).replace("/", "_").replace("\\", "_")

            output_dir = exp_info["background_path"].parent / output_folder
            csv_path = output_dir / output_filename if save_individual else None

            df = compute_superpixel_features(
                image_path=exp_info["image_path"],
                marker_file=exp_info["marker_path"],
                size=size,
                mcmicro_markers=mcmicro_markers,
                markers=markers,
                stats=stats,
                remove_empty=remove_empty,
                empty_threshold=empty_threshold,
                roi_name=roi_name,
                output_path=csv_path,
                save_masks=save_masks,
                mask_dir=output_dir,
                mask_basename=mask_basename,
                mask_formats=mask_formats,
            )
            all_dfs.append(df)

        except Exception as e:
            exp_name = exp_info["experiment_path"].name
            print(f"Error extracting superpixel features from {exp_name}: {e}")
            continue

    if not all_dfs:
        return pd.DataFrame()

    return pd.concat(all_dfs, ignore_index=True)
