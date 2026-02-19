"""Image tiling utilities for large-scale segmentation.

This module provides tools for:
- Splitting large images into overlapping tiles for memory-efficient processing
- Merging segmented tile masks back together with proper cell deduplication
- Handling cells that span tile boundaries via IoU-based matching
"""

from .split import (
    TileInfo,
    compute_tile_grid,
    extract_tile,
    split_image,
    save_tile_info,
    load_tile_info,
)
from .merge import (
    load_tile_masks,
    find_cells_in_region,
    compute_cell_iou,
    match_cells_in_overlap,
    merge_tile_masks,
    relabel_mask,
    save_merged_mask,
    compute_cellwise_iou,
    evaluate_merge_quality,
)

__all__ = [
    # Split functions
    "TileInfo",
    "compute_tile_grid",
    "extract_tile",
    "split_image",
    "save_tile_info",
    "load_tile_info",
    # Merge functions
    "load_tile_masks",
    "find_cells_in_region",
    "compute_cell_iou",
    "match_cells_in_overlap",
    "merge_tile_masks",
    "relabel_mask",
    "save_merged_mask",
    # Evaluation functions
    "compute_cellwise_iou",
    "evaluate_merge_quality",
]
