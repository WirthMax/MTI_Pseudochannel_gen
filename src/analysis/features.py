"""Feature extraction from segmentation masks and multi-channel images."""

from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import tifffile
from skimage.measure import regionprops_table

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from pseudochannel.io import load_mcmicro_markers, OMETiffChannels
from pseudochannel.batch import find_mcmicro_experiments


def extract_morphology(
    mask: np.ndarray,
    properties: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Extract morphological features from a segmentation mask.

    Args:
        mask: 2D integer array where each cell has a unique label (0=background).
        properties: List of properties to extract. If None, uses default set:
            ['label', 'centroid', 'area', 'eccentricity', 'major_axis_length',
             'minor_axis_length', 'perimeter'].

    Returns:
        DataFrame with one row per cell and columns for each property.
        Centroid columns are named 'centroid_y' and 'centroid_x'.
    """
    if properties is None:
        properties = [
            "label",
            "centroid",
            "area",
            "eccentricity",
            "major_axis_length",
            "minor_axis_length",
            "perimeter",
        ]

    props_table = regionprops_table(mask, properties=properties)
    df = pd.DataFrame(props_table)

    # Rename centroid columns for clarity
    if "centroid-0" in df.columns:
        df = df.rename(columns={"centroid-0": "centroid_y", "centroid-1": "centroid_x"})

    return df


def extract_marker_intensities(
    mask: np.ndarray,
    channels: dict[str, np.ndarray],
    markers: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Extract mean marker intensities per cell.

    Args:
        mask: 2D integer array where each cell has a unique label (0=background).
        channels: Dict mapping marker_name -> 2D intensity array.
        markers: List of marker names to extract. If None, uses all channels.

    Returns:
        DataFrame with columns ['label', marker1, marker2, ...] containing
        mean intensity of each marker within each cell.
    """
    if markers is None:
        markers = list(channels.keys())

    # Get unique cell labels (excluding background)
    labels = np.unique(mask)
    labels = labels[labels != 0]

    results = {"label": labels}

    for marker in markers:
        if marker not in channels:
            continue

        channel_data = channels[marker]
        intensities = []

        for label in labels:
            cell_mask = mask == label
            cell_values = channel_data[cell_mask]
            mean_intensity = np.mean(cell_values) if cell_values.size > 0 else 0.0
            intensities.append(mean_intensity)

        results[marker] = intensities

    return pd.DataFrame(results)


def _extract_marker_intensities_vectorized(
    mask: np.ndarray,
    channels: dict[str, np.ndarray],
    markers: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Vectorized version of marker intensity extraction (faster for large images).

    Args:
        mask: 2D integer array where each cell has a unique label (0=background).
        channels: Dict mapping marker_name -> 2D intensity array.
        markers: List of marker names to extract. If None, uses all channels.

    Returns:
        DataFrame with columns ['label', marker1, marker2, ...].
    """
    if markers is None:
        markers = list(channels.keys())

    # Get unique labels
    labels = np.unique(mask)
    labels = labels[labels != 0]

    if len(labels) == 0:
        # No cells found
        return pd.DataFrame({"label": []})

    results = {"label": labels}

    for marker in markers:
        if marker not in channels:
            continue

        channel_data = channels[marker].ravel()
        mask_flat = mask.ravel()

        # Use bincount for fast sum computation
        sums = np.bincount(mask_flat, weights=channel_data, minlength=labels.max() + 1)
        counts = np.bincount(mask_flat, minlength=labels.max() + 1)

        # Compute means (avoid division by zero)
        with np.errstate(divide="ignore", invalid="ignore"):
            means = sums / counts
            means = np.nan_to_num(means, nan=0.0)

        # Extract only the labels we have
        results[marker] = means[labels]

    return pd.DataFrame(results)


def extract_features(
    mask_path: Union[str, Path],
    image_path: Union[str, Path],
    marker_file: Union[str, Path],
    morphology_properties: Optional[list[str]] = None,
    markers: Optional[list[str]] = None,
    mcmicro_markers: bool = True,
    exclude_markers: Optional[list[str]] = None,
    roi_name: Optional[str] = None,
) -> pd.DataFrame:
    """Extract morphology and marker intensity features from a segmented image.

    Main entry point for single-experiment feature extraction.

    Args:
        mask_path: Path to segmentation mask TIFF (integer labels, 0=background).
        image_path: Path to OME-TIFF with marker channels.
        marker_file: Path to markers.csv file.
        morphology_properties: Properties to extract (see extract_morphology).
        markers: Specific markers to extract intensities for. If None, uses all.
        mcmicro_markers: If True, parse marker_file as MCMICRO format.
        exclude_markers: List of marker names to exclude from intensity extraction.
        roi_name: Optional ROI identifier to add as a column.

    Returns:
        DataFrame with morphology and intensity features per cell.
    """
    mask_path = Path(mask_path)
    image_path = Path(image_path)
    marker_file = Path(marker_file)

    # Load mask
    mask = tifffile.imread(str(mask_path))
    if mask.ndim > 2:
        mask = mask[0] if mask.ndim == 3 else mask[0, 0]
    mask = mask.astype(np.int32)

    # Load channels
    channels = OMETiffChannels(
        image_path,
        marker_file,
        mcmicro_markers=mcmicro_markers,
        exclude_channels=set() if exclude_markers is None else set(),
    )

    # Determine which markers to extract
    available_markers = list(channels.keys())
    if markers is not None:
        markers_to_use = [m for m in markers if m in available_markers]
    else:
        markers_to_use = available_markers

    if exclude_markers:
        exclude_set = {m.lower() for m in exclude_markers}
        markers_to_use = [m for m in markers_to_use if m.lower() not in exclude_set]

    # Extract morphology
    morphology_df = extract_morphology(mask, properties=morphology_properties)

    # Extract intensities using vectorized method for speed
    intensity_df = _extract_marker_intensities_vectorized(mask, channels, markers_to_use)

    # Merge on label
    features_df = morphology_df.merge(intensity_df, on="label", how="left")

    # Add ROI name if provided
    if roi_name is not None:
        features_df.insert(0, "ROI", roi_name)

    channels.close()

    return features_df


def extract_features_batch(
    root_path: Union[str, Path],
    background_folder: str = "background",
    marker_filename: str = "markers.csv",
    segmentation_folder: str = "segmentation",
    mask_filename: str = "seg_mask.tif",
    output_folder: str = "analysis",
    output_filename: str = "features.csv",
    morphology_properties: Optional[list[str]] = None,
    markers: Optional[list[str]] = None,
    mcmicro_markers: bool = True,
    exclude_markers: Optional[list[str]] = None,
    progress: bool = True,
    save_individual: bool = True,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Batch extract features from all MCMICRO experiments.

    Finds experiments with segmentation masks, extracts features from each,
    and optionally saves individual CSV files to analysis/ subfolders.

    Args:
        root_path: Root directory containing experiment folders.
        background_folder: Name of subfolder containing images.
        marker_filename: Name of marker file.
        segmentation_folder: Subfolder containing segmentation masks.
        mask_filename: Filename of segmentation mask.
        output_folder: Subfolder for feature output files.
        output_filename: Filename for feature CSV.
        morphology_properties: Properties to extract (see extract_morphology).
        markers: Specific markers to extract. If None, uses all.
        mcmicro_markers: Use MCMICRO marker format.
        exclude_markers: Marker names to exclude.
        progress: Show progress bar.
        save_individual: If True, save CSV for each experiment.
        overwrite: If False, skip experiments with existing feature files.

    Returns:
        Combined DataFrame with features from all experiments.
        Includes 'ROI' column identifying the source experiment.
    """
    root_path = Path(root_path)

    # Find experiments
    experiments = find_mcmicro_experiments(
        root_path,
        background_folder=background_folder,
        marker_filename=marker_filename,
    )

    if not experiments:
        print(f"No MCMICRO experiments found in {root_path}")
        return pd.DataFrame()

    # Filter to those with segmentation masks
    with_seg = []
    for exp_info in experiments:
        mask_path = (
            exp_info["background_path"].parent / segmentation_folder / mask_filename
        )
        if mask_path.exists():
            exp_info["mask_path"] = mask_path
            with_seg.append(exp_info)

    if not with_seg:
        print(f"No experiments with segmentation masks found")
        return pd.DataFrame()

    # Filter out already processed unless overwrite=True
    if not overwrite:
        to_process = []
        skipped = 0
        for exp_info in with_seg:
            output_path = (
                exp_info["background_path"].parent / output_folder / output_filename
            )
            if output_path.exists():
                skipped += 1
            else:
                to_process.append(exp_info)

        if skipped > 0:
            print(f"Skipping {skipped} already processed (use overwrite=True)")
        with_seg = to_process

    if not with_seg:
        print("All experiments already have features extracted.")
        # Load and combine existing features
        all_dfs = []
        for exp_info in experiments:
            output_path = (
                exp_info["background_path"].parent / output_folder / output_filename
            )
            if output_path.exists():
                df = pd.read_csv(output_path)
                all_dfs.append(df)
        return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

    print(f"Extracting features from {len(with_seg)} experiments")

    all_dfs = []

    if progress and tqdm is not None:
        iterator = tqdm(with_seg, desc="Extracting features")
    else:
        iterator = with_seg

    for exp_info in iterator:
        try:
            # Generate ROI name from path
            relative_path = exp_info["background_path"].parent.relative_to(root_path)
            roi_name = str(relative_path).replace("/", "_").replace("\\", "_")

            features_df = extract_features(
                mask_path=exp_info["mask_path"],
                image_path=exp_info["image_path"],
                marker_file=exp_info["marker_path"],
                morphology_properties=morphology_properties,
                markers=markers,
                mcmicro_markers=mcmicro_markers,
                exclude_markers=exclude_markers,
                roi_name=roi_name,
            )

            # Save individual CSV if requested
            if save_individual:
                output_dir = exp_info["background_path"].parent / output_folder
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / output_filename
                features_df.to_csv(output_path, index=False)

            all_dfs.append(features_df)

        except Exception as e:
            exp_name = exp_info["experiment_path"].name
            print(f"Error extracting features from {exp_name}: {e}")
            continue

    if not all_dfs:
        return pd.DataFrame()

    combined_df = pd.concat(all_dfs, ignore_index=True)
    return combined_df
