"""Tile mask merging utilities for reassembling segmented tiles."""

from dataclasses import dataclass
from pathlib import Path
import math
from typing import Optional, Union

import numpy as np
from scipy import ndimage
import tifffile

from .split import TileInfo


@dataclass
class CellProperties:
    """Cached properties for a cell to avoid recomputation."""
    label: int
    bbox: tuple[int, int, int, int]  # (y_min, y_max, x_min, x_max)
    centroid: tuple[float, float]  # (y, x)
    area: int


def load_tile_masks(
    tile_dir: Union[str, Path],
    tile_infos: list[TileInfo],
    mask_pattern: str = "tile_r{row}_c{col}_cp_masks.tif",
) -> dict[tuple[int, int], np.ndarray]:
    """Load segmented mask tiles from directory.

    Args:
        tile_dir: Directory containing tile mask files
        tile_infos: List of TileInfo objects describing tile positions
        mask_pattern: Filename pattern for mask files. Must include {row} and {col}.

    Returns:
        Dict mapping (row, col) to mask array

    Raises:
        FileNotFoundError: If any expected mask file is missing
    """
    tile_dir = Path(tile_dir)
    masks = {}

    missing = []
    for info in tile_infos:
        filename = mask_pattern.format(row=info.row, col=info.col)
        mask_path = tile_dir / filename

        if not mask_path.exists():
            missing.append(str(mask_path))
            continue

        mask = tifffile.imread(str(mask_path))
        # Handle multi-dimensional masks (take first slice if needed)
        if mask.ndim > 2:
            mask = mask[0]
        masks[(info.row, info.col)] = mask

    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} tile mask files:\n" + "\n".join(missing[:10])
            + ("\n..." if len(missing) > 10 else "")
        )

    return masks


def find_cells_in_region(
    mask: np.ndarray,
    y_start: int,
    y_end: int,
    x_start: int,
    x_end: int,
) -> set[int]:
    """Find cell labels that have pixels in the specified region.

    Args:
        mask: Segmentation mask with integer cell labels
        y_start: Start Y coordinate of region
        y_end: End Y coordinate of region (exclusive)
        x_start: Start X coordinate of region
        x_end: End X coordinate of region (exclusive)

    Returns:
        Set of cell labels present in the region (excluding 0/background)
    """
    region = mask[y_start:y_end, x_start:x_end]
    labels = set(np.unique(region))
    labels.discard(0)  # Remove background
    return labels


def compute_cell_properties_batch(
    mask: np.ndarray,
    labels: Optional[list[int]] = None,
) -> dict[int, CellProperties]:
    """Compute properties for all cells in mask using vectorized operations.

    Uses scipy.ndimage for efficient batch computation of centroids and
    find_objects for bounding boxes.

    Args:
        mask: Segmentation mask with integer cell labels
        labels: Optional list of labels to compute properties for.
                If None, computes for all non-zero labels in mask.

    Returns:
        Dict mapping label to CellProperties
    """
    if labels is None:
        labels = np.unique(mask)
        labels = labels[labels != 0].tolist()

    if not labels:
        return {}

    # Use find_objects for bounding boxes - returns list of slice tuples
    # Index i corresponds to label i+1
    max_label = max(labels)
    slices = ndimage.find_objects(mask, max_label=max_label)

    # Compute centroids in batch using center_of_mass
    # Need to pass labels as array for batch computation
    labels_arr = np.array(labels)
    centroids = ndimage.center_of_mass(np.ones_like(mask), mask, labels_arr)

    # Build properties dict
    properties = {}
    for i, label in enumerate(labels):
        # Get bounding box from slices (index is label - 1)
        slice_tuple = slices[label - 1] if label <= len(slices) else None

        if slice_tuple is not None:
            y_slice, x_slice = slice_tuple
            bbox = (y_slice.start, y_slice.stop, x_slice.start, x_slice.stop)
            # Compute area from the slice region
            region = mask[y_slice, x_slice]
            area = int(np.sum(region == label))
        else:
            # Fallback for missing slice
            ys, xs = np.where(mask == label)
            if len(ys) == 0:
                continue
            bbox = (int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1)
            area = len(ys)

        centroid = centroids[i] if not np.isnan(centroids[i]).any() else None
        if centroid is None:
            # Fallback
            ys, xs = np.where(mask == label)
            if len(ys) > 0:
                centroid = (float(np.mean(ys)), float(np.mean(xs)))
            else:
                continue

        properties[label] = CellProperties(
            label=label,
            bbox=bbox,
            centroid=(float(centroid[0]), float(centroid[1])),
            area=area,
        )

    return properties


def bbox_intersects_region(
    bbox: tuple[int, int, int, int],
    region_y_start: int,
    region_y_end: int,
    region_x_start: int,
    region_x_end: int,
) -> bool:
    """Check if a cell's bounding box intersects with a region.

    Args:
        bbox: Cell bounding box (y_min, y_max, x_min, x_max)
        region_y_start, region_y_end: Region Y bounds
        region_x_start, region_x_end: Region X bounds

    Returns:
        True if bounding box intersects region
    """
    y_min, y_max, x_min, x_max = bbox

    # Check for non-intersection (any of these means no overlap)
    if y_max <= region_y_start or y_min >= region_y_end:
        return False
    if x_max <= region_x_start or x_min >= region_x_end:
        return False

    return True


def compute_cell_iou(
    mask1: np.ndarray,
    mask2: np.ndarray,
    label1: int,
    label2: int,
) -> float:
    """Compute Intersection over Union between two cells.

    Args:
        mask1: First segmentation mask
        mask2: Second segmentation mask
        label1: Cell label in mask1
        label2: Cell label in mask2

    Returns:
        IoU value between 0 and 1
    """
    cell1 = mask1 == label1
    cell2 = mask2 == label2

    intersection = np.sum(cell1 & cell2)
    union = np.sum(cell1 | cell2)

    if union == 0:
        return 0.0

    return intersection / union


def _compute_cell_centroid(mask: np.ndarray, label: int) -> Optional[tuple[float, float]]:
    """Compute centroid of a cell.

    Args:
        mask: Segmentation mask
        label: Cell label

    Returns:
        (y, x) centroid coordinates, or None if cell not found
    """
    ys, xs = np.where(mask == label)
    if len(ys) == 0:
        return None
    return (float(np.mean(ys)), float(np.mean(xs)))


def _get_cell_centroid_in_image_coords(
    mask: np.ndarray,
    label: int,
    y_offset: int = 0,
    x_offset: int = 0,
) -> Optional[tuple[float, float]]:
    """Get cell centroid in original image coordinates.

    Args:
        mask: Segmentation mask (can be full tile or merged mask)
        label: Cell label to find
        y_offset: Y offset to add (for tile-local to image coords)
        x_offset: X offset to add (for tile-local to image coords)

    Returns:
        (y, x) centroid in image coordinates, or None if cell not found
    """
    centroid = _compute_cell_centroid(mask, label)
    if centroid is None:
        return None
    return (centroid[0] + y_offset, centroid[1] + x_offset)


def _euclidean_distance(
    point1: tuple[float, float],
    point2: tuple[float, float],
) -> float:
    """Compute Euclidean distance between two points.

    Args:
        point1: (y, x) coordinates of first point
        point2: (y, x) coordinates of second point

    Returns:
        Euclidean distance
    """
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def _cell_area_in_region(
    mask: np.ndarray,
    label: int,
    y_start: int,
    y_end: int,
    x_start: int,
    x_end: int,
) -> int:
    """Count pixels of a cell within a region.

    Args:
        mask: Segmentation mask
        label: Cell label
        y_start, y_end, x_start, x_end: Region bounds

    Returns:
        Number of pixels in region
    """
    region = mask[y_start:y_end, x_start:x_end]
    return int(np.sum(region == label))


def compute_iou_matrix(
    overlap1: np.ndarray,
    overlap2: np.ndarray,
    labels1: list[int],
    labels2: list[int],
) -> np.ndarray:
    """Compute IoU matrix between all pairs of labels in two overlap regions.

    This is more efficient than computing IoU one pair at a time when there
    are many labels to compare.

    Args:
        overlap1: First overlap region mask
        overlap2: Second overlap region mask
        labels1: Labels from first mask present in overlap
        labels2: Labels from second mask present in overlap

    Returns:
        IoU matrix of shape (len(labels1), len(labels2))
    """
    n1, n2 = len(labels1), len(labels2)

    if n1 == 0 or n2 == 0:
        return np.zeros((n1, n2), dtype=np.float32)

    # Pre-compute binary masks for all labels (avoids repeated comparisons)
    # Store as list of boolean arrays for each label
    masks1 = [(overlap1 == l) for l in labels1]
    masks2 = [(overlap2 == l) for l in labels2]

    # Pre-compute areas for union calculation
    areas1 = np.array([np.sum(m) for m in masks1], dtype=np.float32)
    areas2 = np.array([np.sum(m) for m in masks2], dtype=np.float32)

    iou_matrix = np.zeros((n1, n2), dtype=np.float32)

    for i, m1 in enumerate(masks1):
        for j, m2 in enumerate(masks2):
            intersection = np.sum(m1 & m2)
            union = areas1[i] + areas2[j] - intersection
            if union > 0:
                iou_matrix[i, j] = intersection / union

    return iou_matrix


def apply_label_map_vectorized(
    region: np.ndarray,
    label_map: dict[int, int],
) -> np.ndarray:
    """Apply label mapping to a region using vectorized lookup table.

    Args:
        region: Mask region with old labels
        label_map: Mapping from old labels to new labels

    Returns:
        Region with mapped labels
    """
    if not label_map:
        return np.zeros_like(region, dtype=np.int32)

    max_old = max(label_map.keys())
    # Build lookup table
    lut = np.zeros(max_old + 1, dtype=np.int32)
    for old, new in label_map.items():
        lut[old] = new

    # Ensure region values don't exceed lut size
    region_clipped = np.clip(region, 0, max_old)
    return lut[region_clipped]


def match_cells_in_overlap(
    mask_existing: np.ndarray,
    mask_new_full: np.ndarray,
    tile_info: TileInfo,
    overlap_y_start: int,
    overlap_y_end: int,
    overlap_x_start: int,
    overlap_x_end: int,
    iou_threshold: float = 0.5,
    max_centroid_distance: float = 50.0,
    min_iou_for_centroid_match: float = 0.01,
) -> tuple[list[tuple[int, int]], list[int]]:
    """Match cells between existing merged mask and new tile in overlap region.

    Uses a hybrid matching strategy:
    1. Primary: IoU >= iou_threshold in overlap region
    2. Fallback: Centroid distance < max_centroid_distance AND IoU > min_iou

    The fallback handles edge cases where a cell has minimal pixels in the
    overlap region (causing unreliable IoU) but centroids clearly indicate
    it's the same cell.

    Args:
        mask_existing: Already-merged mask (in original image coordinates)
        mask_new_full: Full new tile mask (in tile-local coordinates)
        tile_info: TileInfo for the new tile (for coordinate transforms)
        overlap_y_start: Overlap region Y start in original image coords
        overlap_y_end: Overlap region Y end in original image coords
        overlap_x_start: Overlap region X start in original image coords
        overlap_x_end: Overlap region X end in original image coords
        iou_threshold: Minimum IoU for primary matching (default 0.5)
        max_centroid_distance: Maximum centroid distance for fallback matching (default 50px)
        min_iou_for_centroid_match: Minimum IoU required for centroid-based match (default 0.01)

    Returns:
        Tuple of:
        - List of (existing_label, new_label) matched pairs
        - List of unmatched labels from new tile (new cells)
    """
    # Get existing cells in overlap region
    existing_labels = find_cells_in_region(
        mask_existing,
        overlap_y_start,
        overlap_y_end,
        overlap_x_start,
        overlap_x_end,
    )

    # Determine overlap region in tile-local coordinates
    tile_overlap_y_start = overlap_y_start - tile_info.y_start
    tile_overlap_y_end = overlap_y_end - tile_info.y_start
    tile_overlap_x_start = overlap_x_start - tile_info.x_start
    tile_overlap_x_end = overlap_x_end - tile_info.x_start

    # Clamp to valid tile bounds
    tile_overlap_y_start = max(0, tile_overlap_y_start)
    tile_overlap_y_end = min(mask_new_full.shape[0], tile_overlap_y_end)
    tile_overlap_x_start = max(0, tile_overlap_x_start)
    tile_overlap_x_end = min(mask_new_full.shape[1], tile_overlap_x_end)

    # Get new cells in the overlap region of the new tile
    new_labels = find_cells_in_region(
        mask_new_full,
        tile_overlap_y_start,
        tile_overlap_y_end,
        tile_overlap_x_start,
        tile_overlap_x_end,
    )

    # Extract overlap regions for IoU comparison
    existing_overlap = mask_existing[
        overlap_y_start:overlap_y_end,
        overlap_x_start:overlap_x_end,
    ]
    new_overlap = mask_new_full[
        tile_overlap_y_start:tile_overlap_y_end,
        tile_overlap_x_start:tile_overlap_x_end,
    ]

    # Early return if no cells to match
    if not existing_labels or not new_labels:
        return [], list(new_labels)

    # Convert to sorted lists for consistent indexing
    existing_labels_list = sorted(existing_labels)
    new_labels_list = sorted(new_labels)

    # Compute IoU matrix in batch (more efficient than per-pair)
    iou_matrix = compute_iou_matrix(
        existing_overlap, new_overlap,
        existing_labels_list, new_labels_list
    )

    # Pre-compute centroids for fallback matching (batch computation)
    # Only compute for cells that might need centroid fallback
    existing_props = compute_cell_properties_batch(mask_existing, existing_labels_list)
    new_props = compute_cell_properties_batch(mask_new_full, new_labels_list)

    matches = []
    matched_new = set()

    # Match cells using IoU matrix
    for i, existing_label in enumerate(existing_labels_list):
        best_match = None
        best_score = -1.0

        for j, new_label in enumerate(new_labels_list):
            if new_label in matched_new:
                continue

            iou = iou_matrix[i, j]

            # Primary match: IoU threshold met
            if iou >= iou_threshold:
                score = iou + 1.0  # Offset to prioritize IoU matches
                if score > best_score:
                    best_score = score
                    best_match = new_label
                continue

            # Fallback: Centroid distance match (requires minimal overlap)
            if iou >= min_iou_for_centroid_match:
                # Use cached centroids
                existing_centroid = existing_props.get(existing_label)
                new_centroid = new_props.get(new_label)

                if existing_centroid is not None and new_centroid is not None:
                    # Convert new centroid to image coordinates
                    centroid_existing = existing_centroid.centroid
                    centroid_new = (
                        new_centroid.centroid[0] + tile_info.y_start,
                        new_centroid.centroid[1] + tile_info.x_start,
                    )

                    distance = _euclidean_distance(centroid_existing, centroid_new)

                    if distance < max_centroid_distance:
                        score = 1.0 - (distance / max_centroid_distance)
                        if score > best_score:
                            best_score = score
                            best_match = new_label

        if best_match is not None:
            matches.append((existing_label, best_match))
            matched_new.add(best_match)

    unmatched_new = [label for label in new_labels_list if label not in matched_new]

    return matches, unmatched_new


def merge_tile_masks(
    tile_masks: dict[tuple[int, int], np.ndarray],
    tile_infos: list[TileInfo],
    image_shape: tuple[int, int],
    iou_threshold: float = 0.5,
    max_centroid_distance: float = 50.0,
    min_iou_for_centroid_match: float = 0.01,
) -> np.ndarray:
    """Merge segmented tile masks back into full image.

    Strategy:
    1. Create empty output mask of original shape
    2. Process tiles in order (top-left to bottom-right)
    3. For each tile:
       a. Place non-overlap (core) region directly with new labels
       b. For overlap regions with already-placed tiles:
          - Find matching cells using hybrid IoU + centroid distance
          - Assign consistent label to matched cells
    4. Relabel to consecutive 1..N

    The hybrid matching strategy prevents cell duplication at tile boundaries
    by using centroid distance as a fallback when IoU is unreliable (e.g.,
    when a cell has only a few pixels in the overlap region).

    Args:
        tile_masks: Dict mapping (row, col) to mask array
        tile_infos: List of TileInfo objects (same order as used for splitting)
        image_shape: Original image shape (height, width)
        iou_threshold: Minimum IoU for primary matching (default 0.5)
        max_centroid_distance: Maximum centroid distance for fallback matching
            in pixels (default 50.0, ~1.5x typical cell diameter)
        min_iou_for_centroid_match: Minimum IoU required for centroid-based
            match to prevent matching completely disjoint cells (default 0.01)

    Returns:
        Merged segmentation mask with consecutive labels 1..N
    """
    height, width = image_shape

    # Use int32 to handle large cell counts
    merged = np.zeros((height, width), dtype=np.int32)

    # Track next available label
    next_label = 1

    # Build lookup for tile infos
    info_by_pos = {(t.row, t.col): t for t in tile_infos}

    # Process tiles in row-major order
    sorted_positions = sorted(tile_masks.keys())

    for row, col in sorted_positions:
        info = info_by_pos[(row, col)]
        tile_mask = tile_masks[(row, col)]

        # Get unique labels in this tile (excluding background)
        tile_labels = set(np.unique(tile_mask))
        tile_labels.discard(0)

        if not tile_labels:
            continue

        # Create mapping from old labels to new labels
        label_map = {}
        matched_merged_labels = set()  # Track which merged labels came from matches

        # Handle overlaps with already-processed tiles
        # Previous tiles placed their FULL area (including overlaps) into
        # merged, so the overlap region has data to match against.

        # Check left overlap (with tile at col-1)
        if col > 0 and (row, col - 1) in tile_masks:
            overlap_width = info.overlap_left

            if overlap_width > 0:
                overlap_x_start = info.x_start
                overlap_x_end = info.x_start + overlap_width
                overlap_y_start = info.y_start
                overlap_y_end = info.y_end

                matches, unmatched = match_cells_in_overlap(
                    merged,
                    tile_mask,
                    info,
                    overlap_y_start,
                    overlap_y_end,
                    overlap_x_start,
                    overlap_x_end,
                    iou_threshold,
                    max_centroid_distance,
                    min_iou_for_centroid_match,
                )

                for existing_label, new_label in matches:
                    label_map[new_label] = existing_label
                    matched_merged_labels.add(existing_label)

        # Check top overlap (with tile at row-1)
        if row > 0 and (row - 1, col) in tile_masks:
            overlap_height = info.overlap_top

            if overlap_height > 0:
                overlap_y_start = info.y_start
                overlap_y_end = info.y_start + overlap_height
                overlap_x_start = info.x_start
                overlap_x_end = info.x_end

                matches, unmatched = match_cells_in_overlap(
                    merged,
                    tile_mask,
                    info,
                    overlap_y_start,
                    overlap_y_end,
                    overlap_x_start,
                    overlap_x_end,
                    iou_threshold,
                    max_centroid_distance,
                    min_iou_for_centroid_match,
                )

                for existing_label, new_label in matches:
                    if new_label not in label_map:
                        label_map[new_label] = existing_label
                        matched_merged_labels.add(existing_label)

        # Assign new labels to unmatched cells
        for label in tile_labels:
            if label not in label_map:
                label_map[label] = next_label
                next_label += 1

        # --- Place the FULL tile into merged ---
        # Map the entire tile mask using label_map
        mapped_full = apply_label_map_vectorized(tile_mask, label_map)

        # Get the destination view (tile's full footprint in merged)
        dest = merged[info.y_start:info.y_end, info.x_start:info.x_end]

        # Core bounds in tile-local coordinates
        tile_core_y_start = info.overlap_top
        tile_core_y_end = tile_mask.shape[0] - info.overlap_bottom if info.overlap_bottom > 0 else tile_mask.shape[0]
        tile_core_x_start = info.overlap_left
        tile_core_x_end = tile_mask.shape[1] - info.overlap_right if info.overlap_right > 0 else tile_mask.shape[1]

        # 1) Core region: always overwrite (this tile owns it)
        dest[tile_core_y_start:tile_core_y_end,
             tile_core_x_start:tile_core_x_end] = \
            mapped_full[tile_core_y_start:tile_core_y_end,
                        tile_core_x_start:tile_core_x_end]

        # 2) Overlap regions: place the full tile area outside core
        #    - Matched cells → overwrite (unifies the cell across tiles)
        #    - Unmatched new cells → fill only where merged is empty
        #    - Existing cells from previous tiles → preserve
        is_core = np.zeros(tile_mask.shape, dtype=bool)
        is_core[tile_core_y_start:tile_core_y_end,
                tile_core_x_start:tile_core_x_end] = True

        has_overlap_cell = ~is_core & (mapped_full > 0)

        if matched_merged_labels and np.any(has_overlap_cell):
            # Build fast lookup for matched labels
            max_lut = max(max(matched_merged_labels), int(mapped_full.max())) + 1
            is_matched_lut = np.zeros(max_lut, dtype=bool)
            for label in matched_merged_labels:
                is_matched_lut[label] = True

            is_matched_pixel = is_matched_lut[np.clip(mapped_full, 0, max_lut - 1)] & has_overlap_cell

            # Matched cells: overwrite to unify across tile boundary
            dest[is_matched_pixel] = mapped_full[is_matched_pixel]

            # New cells: fill only where empty
            is_new_pixel = has_overlap_cell & ~is_matched_pixel
            fill_mask = is_new_pixel & (dest == 0)
            dest[fill_mask] = mapped_full[fill_mask]
        elif np.any(has_overlap_cell):
            # No matches — fill only where empty (e.g. first tile)
            fill_mask = has_overlap_cell & (dest == 0)
            dest[fill_mask] = mapped_full[fill_mask]

    # Relabel to consecutive integers
    merged = relabel_mask(merged)

    return merged


def relabel_mask(mask: np.ndarray) -> np.ndarray:
    """Relabel mask to have consecutive labels 1..N.

    Uses a lookup table for O(H×W) single-pass relabeling instead of
    O(N×H×W) per-label iteration.

    Args:
        mask: Segmentation mask with arbitrary integer labels

    Returns:
        Relabeled mask with consecutive labels starting from 1
    """
    unique_labels = np.unique(mask)
    unique_labels = unique_labels[unique_labels != 0]  # Exclude background

    if len(unique_labels) == 0:
        return mask.copy()

    # Check if already consecutive
    if np.array_equal(unique_labels, np.arange(1, len(unique_labels) + 1)):
        return mask.copy()

    # Build lookup table: lut[old_label] = new_label
    # Single pass O(H×W) instead of O(N×H×W)
    max_label = int(mask.max())
    lut = np.zeros(max_label + 1, dtype=np.int32)
    lut[unique_labels] = np.arange(1, len(unique_labels) + 1, dtype=np.int32)

    # Apply lookup table in single pass
    return lut[mask]


def save_merged_mask(
    mask: np.ndarray,
    output_path: Union[str, Path],
    compress: bool = True,
) -> Path:
    """Save merged segmentation mask to TIFF file.

    Args:
        mask: Merged segmentation mask
        output_path: Output file path
        compress: Whether to use compression

    Returns:
        Path to saved file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Use uint32 for label masks (can have many cells)
    if mask.max() > 65535:
        dtype = np.uint32
    elif mask.max() > 255:
        dtype = np.uint16
    else:
        dtype = np.uint8

    mask_out = mask.astype(dtype)

    if compress:
        tifffile.imwrite(str(output_path), mask_out, compression="zlib")
    else:
        tifffile.imwrite(str(output_path), mask_out)

    return output_path


def compute_cellwise_iou(
    mask_reference: np.ndarray,
    mask_test: np.ndarray,
    iou_threshold_for_match: float = 0.1,
) -> dict:
    """Compute cell-wise IoU between reference and test masks.

    Matches cells between masks and computes IoU for each matched pair.
    Useful for evaluating how well a split/merged mask matches the original.

    Uses optimized approach:
    - Pre-computes bounding boxes to limit region searches
    - Uses scipy.ndimage.find_objects for efficient bbox extraction
    - Only creates boolean masks within cell bounding boxes

    Args:
        mask_reference: Reference segmentation mask (e.g., from single-run)
        mask_test: Test segmentation mask (e.g., from tiled/merged processing)
        iou_threshold_for_match: Minimum IoU to consider a cell matched

    Returns:
        Dictionary containing:
        - 'mean_iou': Average IoU across all matched cells
        - 'median_iou': Median IoU across all matched cells
        - 'matched_ious': List of IoU values for matched cells
        - 'n_reference_cells': Number of cells in reference mask
        - 'n_test_cells': Number of cells in test mask
        - 'n_matched': Number of matched cells
        - 'n_unmatched_reference': Reference cells with no match
        - 'n_unmatched_test': Test cells with no match
        - 'matched_pairs': List of (ref_label, test_label, iou) tuples
    """
    # Get unique labels (excluding background)
    ref_labels_arr = np.unique(mask_reference)
    ref_labels_arr = ref_labels_arr[ref_labels_arr != 0]
    test_labels_arr = np.unique(mask_test)
    test_labels_arr = test_labels_arr[test_labels_arr != 0]

    ref_labels = set(ref_labels_arr.tolist())
    test_labels = set(test_labels_arr.tolist())

    if not ref_labels or not test_labels:
        return {
            'mean_iou': 0.0,
            'median_iou': 0.0,
            'matched_ious': [],
            'n_reference_cells': len(ref_labels),
            'n_test_cells': len(test_labels),
            'n_matched': 0,
            'n_unmatched_reference': len(ref_labels),
            'n_unmatched_test': len(test_labels),
            'matched_pairs': [],
        }

    # Pre-compute bounding boxes for all cells using find_objects
    # This is much faster than repeated np.where calls
    max_ref_label = int(ref_labels_arr.max())
    max_test_label = int(test_labels_arr.max())

    ref_slices = ndimage.find_objects(mask_reference, max_label=max_ref_label)
    test_slices = ndimage.find_objects(mask_test, max_label=max_test_label)

    # Pre-compute areas for test labels (avoids recomputation)
    test_areas = {}
    for test_label in test_labels:
        sl = test_slices[test_label - 1] if test_label <= len(test_slices) else None
        if sl is not None:
            region = mask_test[sl]
            test_areas[test_label] = int(np.sum(region == test_label))

    # For each reference cell, find best matching test cell
    matched_pairs = []
    matched_test = set()

    for ref_label in ref_labels:
        ref_sl = ref_slices[ref_label - 1] if ref_label <= len(ref_slices) else None
        if ref_sl is None:
            continue

        # Extract reference cell region and mask
        y_slice, x_slice = ref_sl
        ref_region = mask_reference[y_slice, x_slice]
        ref_mask_local = ref_region == ref_label
        ref_area = int(np.sum(ref_mask_local))

        # Find test cells that overlap with this reference cell's bbox
        test_region = mask_test[y_slice, x_slice]
        overlapping_test_labels = set(np.unique(test_region))
        overlapping_test_labels.discard(0)

        best_iou = 0.0
        best_test_label = None

        for test_label in overlapping_test_labels:
            if test_label in matched_test:
                continue

            # Compute IoU using local regions where possible
            test_mask_local = test_region == test_label
            intersection = int(np.sum(ref_mask_local & test_mask_local))

            if intersection == 0:
                continue

            # Union = ref_area + test_area - intersection
            test_area = test_areas.get(test_label, 0)
            union = ref_area + test_area - intersection

            if union > 0:
                iou = intersection / union
                if iou > best_iou:
                    best_iou = iou
                    best_test_label = test_label

        if best_iou >= iou_threshold_for_match and best_test_label is not None:
            matched_pairs.append((ref_label, best_test_label, best_iou))
            matched_test.add(best_test_label)

    # Compute statistics
    ious = [iou for _, _, iou in matched_pairs]

    return {
        'mean_iou': float(np.mean(ious)) if ious else 0.0,
        'median_iou': float(np.median(ious)) if ious else 0.0,
        'matched_ious': ious,
        'n_reference_cells': len(ref_labels),
        'n_test_cells': len(test_labels),
        'n_matched': len(matched_pairs),
        'n_unmatched_reference': len(ref_labels) - len(matched_pairs),
        'n_unmatched_test': len(test_labels) - len(matched_test),
        'matched_pairs': matched_pairs,
    }


def evaluate_merge_quality(
    mask_original: np.ndarray,
    mask_merged: np.ndarray,
    iou_threshold: float = 0.5,
) -> dict:
    """Evaluate quality of merged mask against original single-run mask.

    Provides comprehensive metrics for comparing tiled/merged segmentation
    against a reference segmentation computed on the full image.

    Args:
        mask_original: Reference mask from single-run segmentation
        mask_merged: Mask from split/segment/merge pipeline
        iou_threshold: IoU threshold for considering cells well-matched

    Returns:
        Dictionary containing:
        - 'cellwise_iou': Results from compute_cellwise_iou
        - 'fraction_well_matched': Fraction of cells with IoU >= threshold
        - 'cell_count_ratio': n_merged / n_original
        - 'pixel_agreement': Fraction of pixels with same label (foreground only)
        - 'dice_coefficient': Dice coefficient treating masks as binary
    """
    cellwise = compute_cellwise_iou(mask_original, mask_merged)

    # Fraction of reference cells that are well-matched
    well_matched = sum(1 for iou in cellwise['matched_ious'] if iou >= iou_threshold)
    fraction_well_matched = (
        well_matched / cellwise['n_reference_cells']
        if cellwise['n_reference_cells'] > 0 else 0.0
    )

    # Cell count ratio
    cell_count_ratio = (
        cellwise['n_test_cells'] / cellwise['n_reference_cells']
        if cellwise['n_reference_cells'] > 0 else 0.0
    )

    # Binary mask agreement (foreground vs background)
    fg_original = mask_original > 0
    fg_merged = mask_merged > 0

    intersection = np.sum(fg_original & fg_merged)
    union = np.sum(fg_original | fg_merged)
    sum_areas = np.sum(fg_original) + np.sum(fg_merged)

    dice = 2 * intersection / sum_areas if sum_areas > 0 else 0.0
    pixel_agreement = intersection / union if union > 0 else 0.0

    return {
        'cellwise_iou': cellwise,
        'fraction_well_matched': fraction_well_matched,
        'cell_count_ratio': cell_count_ratio,
        'pixel_agreement': pixel_agreement,
        'dice_coefficient': dice,
    }
