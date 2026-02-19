"""Export utilities for QuPath GeoJSON annotations."""

import json
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import tifffile

try:
    from skimage import measure
except ImportError:
    measure = None

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from pseudochannel.batch import find_mcmicro_experiments


# 45 distinct colors for cluster visualization (hex format)
# Designed for maximum distinguishability on both light and dark backgrounds
CLUSTER_COLORS = [
    "#e6194b",  # Red
    "#3cb44b",  # Green
    "#ffe119",  # Yellow
    "#4363d8",  # Blue
    "#f58231",  # Orange
    "#911eb4",  # Purple
    "#46f0f0",  # Cyan
    "#f032e6",  # Magenta
    "#bcf60c",  # Lime
    "#fabebe",  # Pink
    "#008080",  # Teal
    "#e6beff",  # Lavender
    "#9a6324",  # Brown
    "#fffac8",  # Beige
    "#800000",  # Maroon
    "#aaffc3",  # Mint
    "#808000",  # Olive
    "#ffd8b1",  # Apricot
    "#000075",  # Navy
    "#808080",  # Gray
    "#000000",  # Black
    "#ffffff",  # White
    "#aa6e28",  # Sienna
    "#1e90ff",  # Dodger Blue
    "#ff69b4",  # Hot Pink
    "#8b0000",  # Dark Red
    "#006400",  # Dark Green
    "#ffa500",  # Orange2
    "#00ced1",  # Dark Turquoise
    "#9400d3",  # Dark Violet
    "#ff1493",  # Deep Pink
    "#00ff7f",  # Spring Green
    "#dc143c",  # Crimson
    "#00bfff",  # Deep Sky Blue
    "#228b22",  # Forest Green
    "#daa520",  # Goldenrod
    "#ff4500",  # Orange Red
    "#2f4f4f",  # Dark Slate Gray
    "#7fff00",  # Chartreuse
    "#d2691e",  # Chocolate
    "#ff00ff",  # Fuchsia
    "#1e90ff",  # Royal Blue
    "#adff2f",  # Green Yellow
    "#ff6347",  # Tomato
    "#40e0d0",  # Turquoise
]


def hex_to_rgba(hex_color: str, alpha: int = 255) -> list[int]:
    """Convert hex color to RGBA list.

    Args:
        hex_color: Color in '#RRGGBB' format.
        alpha: Alpha value (0-255).

    Returns:
        List of [R, G, B, A] values (0-255).
    """
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return [r, g, b, alpha]


def get_cluster_color(cluster_id: Union[int, str], alpha: int = 128) -> list[int]:
    """Get RGBA color for a cluster ID.

    Args:
        cluster_id: Cluster identifier (int or str convertible to int).
        alpha: Alpha value for transparency.

    Returns:
        RGBA color list.
    """
    try:
        idx = int(cluster_id) % len(CLUSTER_COLORS)
    except (ValueError, TypeError):
        # Hash string cluster IDs
        idx = hash(str(cluster_id)) % len(CLUSTER_COLORS)

    return hex_to_rgba(CLUSTER_COLORS[idx], alpha)


def mask_to_contours(
    mask: np.ndarray,
    simplify_tolerance: float = 1.0,
) -> dict[int, list[np.ndarray]]:
    """Extract polygon contours from a segmentation mask.

    Args:
        mask: 2D integer array with cell labels (0=background).
        simplify_tolerance: Tolerance for contour simplification.
            Higher values produce simpler polygons with fewer points.

    Returns:
        Dict mapping label -> list of contour arrays.
        Each contour is shape (N, 2) with (x, y) coordinates.
    """
    if measure is None:
        raise ImportError("skimage is required for contour extraction")

    labels = np.unique(mask)
    labels = labels[labels != 0]

    contours = {}

    for label in labels:
        cell_mask = (mask == label).astype(np.uint8)

        # Find contours
        cell_contours = measure.find_contours(cell_mask, level=0.5)

        if not cell_contours:
            continue

        # Simplify and convert to (x, y) format
        simplified = []
        for contour in cell_contours:
            # find_contours returns (row, col) = (y, x), swap to (x, y)
            contour_xy = contour[:, ::-1]

            # Simplify using Douglas-Peucker if tolerance > 0
            if simplify_tolerance > 0:
                contour_xy = _simplify_contour(contour_xy, simplify_tolerance)

            if len(contour_xy) >= 3:
                simplified.append(contour_xy)

        if simplified:
            contours[int(label)] = simplified

    return contours


def _simplify_contour(
    contour: np.ndarray,
    tolerance: float,
) -> np.ndarray:
    """Simplify a contour using Douglas-Peucker algorithm.

    Args:
        contour: Array of (x, y) points, shape (N, 2).
        tolerance: Maximum distance from line to point.

    Returns:
        Simplified contour array.
    """
    if len(contour) <= 3:
        return contour

    try:
        from skimage.measure import approximate_polygon
        return approximate_polygon(contour, tolerance=tolerance)
    except ImportError:
        # Fallback: simple decimation
        step = max(1, int(tolerance))
        return contour[::step]


def clean_and_close_contours(
    contours: dict[int, list[np.ndarray]],
) -> dict[int, list[np.ndarray]]:
    """Ensure contours are closed for GeoJSON compatibility.

    Args:
        contours: Dict from mask_to_contours().

    Returns:
        Dict with closed contours (first point == last point).
    """
    cleaned = {}

    for label, label_contours in contours.items():
        closed_contours = []
        for contour in label_contours:
            if len(contour) < 3:
                continue

            # Close contour if not already closed
            if not np.allclose(contour[0], contour[-1]):
                contour = np.vstack([contour, contour[0:1]])

            closed_contours.append(contour)

        if closed_contours:
            cleaned[label] = closed_contours

    return cleaned


def _contour_to_geojson_coords(contour: np.ndarray) -> list[list[float]]:
    """Convert contour array to GeoJSON coordinate list.

    Args:
        contour: Array of shape (N, 2) with (x, y) coordinates.

    Returns:
        List of [x, y] pairs for GeoJSON.
    """
    return [[float(x), float(y)] for x, y in contour]


def export_to_geojson(
    mask_path: Union[str, Path],
    output_path: Union[str, Path],
    cluster_assignments: Optional[dict[int, Union[int, str]]] = None,
    simplify_tolerance: float = 1.0,
    alpha: int = 128,
    include_properties: Optional[dict[int, dict]] = None,
) -> Path:
    """Export segmentation mask to QuPath-compatible GeoJSON.

    Creates a GeoJSON FeatureCollection where each cell is a Feature
    with polygon geometry and optional cluster classification.

    Args:
        mask_path: Path to segmentation mask TIFF.
        output_path: Path for output GeoJSON file.
        cluster_assignments: Dict mapping cell label -> cluster ID.
            If provided, cells are colored by cluster.
        simplify_tolerance: Tolerance for polygon simplification.
        alpha: Alpha value for cell fill color (0-255).
        include_properties: Dict mapping label -> additional properties to include.

    Returns:
        Path to saved GeoJSON file.
    """
    mask_path = Path(mask_path)
    output_path = Path(output_path)

    # Load mask
    mask = tifffile.imread(str(mask_path))
    if mask.ndim > 2:
        mask = mask[0] if mask.ndim == 3 else mask[0, 0]
    mask = mask.astype(np.int32)

    # Extract and clean contours
    contours = mask_to_contours(mask, simplify_tolerance=simplify_tolerance)
    contours = clean_and_close_contours(contours)

    # Build features
    features = []

    for label, label_contours in contours.items():
        # Get cluster assignment
        cluster = None
        if cluster_assignments is not None and label in cluster_assignments:
            cluster = cluster_assignments[label]

        # Determine color
        if cluster is not None:
            color = get_cluster_color(cluster, alpha=alpha)
        else:
            color = [128, 128, 128, alpha]  # Gray for unassigned

        # Build properties
        properties = {
            "objectType": "cell",
            "classification": {
                "name": f"Cluster {cluster}" if cluster is not None else "Unclassified",
                "color": color,
            },
            "measurements": {
                "label": int(label),
            },
        }

        # Add additional properties if provided
        if include_properties is not None and label in include_properties:
            properties["measurements"].update(include_properties[label])

        # Use largest contour as exterior, others as holes
        if len(label_contours) == 1:
            # Single polygon
            coords = [_contour_to_geojson_coords(label_contours[0])]
            geometry = {"type": "Polygon", "coordinates": coords}
        else:
            # Multiple contours: largest is exterior, rest are holes
            areas = [len(c) for c in label_contours]
            main_idx = np.argmax(areas)

            exterior = _contour_to_geojson_coords(label_contours[main_idx])
            holes = [
                _contour_to_geojson_coords(label_contours[i])
                for i in range(len(label_contours))
                if i != main_idx
            ]
            coords = [exterior] + holes
            geometry = {"type": "Polygon", "coordinates": coords}

        feature = {
            "type": "Feature",
            "id": f"cell_{label}",
            "geometry": geometry,
            "properties": properties,
        }
        features.append(feature)

    # Build FeatureCollection
    geojson = {
        "type": "FeatureCollection",
        "features": features,
    }

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(geojson, f)

    return output_path


def export_mcmicro_batch(
    root_path: Union[str, Path],
    features_df: pd.DataFrame,
    cluster_col: str = "cluster",
    label_col: str = "label",
    roi_col: str = "ROI",
    background_folder: str = "background",
    marker_filename: str = "markers.csv",
    segmentation_folder: str = "segmentation",
    mask_filename: str = "seg_mask.tif",
    output_folder: str = "qupath",
    output_filename: str = "annotations.geojson",
    simplify_tolerance: float = 1.0,
    alpha: int = 128,
    progress: bool = True,
    overwrite: bool = False,
) -> list[Path]:
    """Batch export GeoJSON annotations for MCMICRO experiments.

    Args:
        root_path: Root directory containing experiment folders.
        features_df: DataFrame with cluster assignments. Must have columns
            for ROI, label, and cluster.
        cluster_col: Column name for cluster assignments.
        label_col: Column name for cell labels.
        roi_col: Column name for ROI identifier.
        background_folder: Name of background subfolder.
        marker_filename: Name of marker file.
        segmentation_folder: Subfolder containing masks.
        mask_filename: Filename of segmentation mask.
        output_folder: Subfolder for GeoJSON outputs.
        output_filename: Filename for GeoJSON.
        simplify_tolerance: Contour simplification tolerance.
        alpha: Fill alpha for cells.
        progress: Show progress bar.
        overwrite: If False, skip existing outputs.

    Returns:
        List of paths to created GeoJSON files.
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
        return []

    # Filter to those with masks
    with_seg = []
    for exp_info in experiments:
        mask_path = (
            exp_info["background_path"].parent / segmentation_folder / mask_filename
        )
        if mask_path.exists():
            exp_info["mask_path"] = mask_path
            with_seg.append(exp_info)

    if not with_seg:
        print("No experiments with segmentation masks found")
        return []

    # Filter out existing unless overwrite
    if not overwrite:
        to_export = []
        skipped = 0
        for exp_info in with_seg:
            output_path = (
                exp_info["background_path"].parent / output_folder / output_filename
            )
            if output_path.exists():
                skipped += 1
            else:
                to_export.append(exp_info)

        if skipped > 0:
            print(f"Skipping {skipped} existing exports (use overwrite=True)")
        with_seg = to_export

    if not with_seg:
        print("All experiments already exported")
        return []

    print(f"Exporting {len(with_seg)} experiments to GeoJSON")

    outputs = []

    if progress and tqdm is not None:
        iterator = tqdm(with_seg, desc="Exporting GeoJSON")
    else:
        iterator = with_seg

    for exp_info in iterator:
        try:
            # Get ROI name
            relative_path = exp_info["background_path"].parent.relative_to(root_path)
            roi_name = str(relative_path).replace("/", "_").replace("\\", "_")

            # Get cluster assignments for this ROI
            roi_df = features_df[features_df[roi_col] == roi_name]

            if roi_df.empty:
                print(f"  No features for ROI: {roi_name}")
                continue

            # Build cluster assignments dict
            cluster_assignments = dict(
                zip(roi_df[label_col].astype(int), roi_df[cluster_col])
            )

            # Export
            output_path = (
                exp_info["background_path"].parent / output_folder / output_filename
            )

            result = export_to_geojson(
                mask_path=exp_info["mask_path"],
                output_path=output_path,
                cluster_assignments=cluster_assignments,
                simplify_tolerance=simplify_tolerance,
                alpha=alpha,
            )
            outputs.append(result)

        except Exception as e:
            exp_name = exp_info["experiment_path"].name
            print(f"Error exporting {exp_name}: {e}")
            continue

    return outputs
