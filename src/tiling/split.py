"""Image splitting utilities for tiled processing."""

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Union

import numpy as np
import tifffile


@dataclass
class TileInfo:
    """Metadata for a single tile.

    Attributes:
        row: Tile row index in the grid
        col: Tile column index in the grid
        y_start: Start Y coordinate in original image
        y_end: End Y coordinate in original image (exclusive)
        x_start: Start X coordinate in original image
        x_end: End X coordinate in original image (exclusive)
        overlap_top: Number of overlap pixels at top edge
        overlap_bottom: Number of overlap pixels at bottom edge
        overlap_left: Number of overlap pixels at left edge
        overlap_right: Number of overlap pixels at right edge
    """

    row: int
    col: int
    y_start: int
    y_end: int
    x_start: int
    x_end: int
    overlap_top: int
    overlap_bottom: int
    overlap_left: int
    overlap_right: int

    @property
    def height(self) -> int:
        """Tile height in pixels."""
        return self.y_end - self.y_start

    @property
    def width(self) -> int:
        """Tile width in pixels."""
        return self.x_end - self.x_start

    @property
    def shape(self) -> tuple[int, int]:
        """Tile shape (height, width)."""
        return (self.height, self.width)

    @property
    def core_y_start(self) -> int:
        """Y start of non-overlap region within tile."""
        return self.overlap_top

    @property
    def core_y_end(self) -> int:
        """Y end of non-overlap region within tile."""
        return self.height - self.overlap_bottom

    @property
    def core_x_start(self) -> int:
        """X start of non-overlap region within tile."""
        return self.overlap_left

    @property
    def core_x_end(self) -> int:
        """X end of non-overlap region within tile."""
        return self.width - self.overlap_right

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TileInfo":
        """Create TileInfo from dictionary."""
        return cls(**d)


def compute_tile_grid(
    shape: tuple[int, int],
    tile_size: int = 2048,
    overlap: int = 200,
) -> list[TileInfo]:
    """Compute tile grid coordinates for an image.

    Tiles are arranged to cover the entire image with the specified overlap.
    Edge tiles may be smaller than tile_size if the image doesn't divide evenly.

    Args:
        shape: Image shape as (height, width)
        tile_size: Target size for each tile (before overlap)
        overlap: Number of pixels to overlap between adjacent tiles.
            Should be at least 2x the expected cell diameter.

    Returns:
        List of TileInfo objects describing each tile's position and overlaps
    """
    height, width = shape

    if tile_size <= overlap:
        raise ValueError(
            f"tile_size ({tile_size}) must be greater than overlap ({overlap})"
        )

    # Calculate step size (tile size minus overlap)
    step = tile_size - overlap

    # Calculate number of tiles needed
    n_rows = max(1, (height - overlap + step - 1) // step)
    n_cols = max(1, (width - overlap + step - 1) // step)

    tiles = []

    for row in range(n_rows):
        for col in range(n_cols):
            # Calculate tile boundaries
            y_start = row * step
            x_start = col * step

            # Ensure we don't exceed image bounds
            y_end = min(y_start + tile_size, height)
            x_end = min(x_start + tile_size, width)

            # Adjust start if end tile is too small
            # (pull it back to maintain minimum size)
            min_tile_size = overlap + 1
            if y_end - y_start < min_tile_size and y_start > 0:
                y_start = max(0, y_end - tile_size)
            if x_end - x_start < min_tile_size and x_start > 0:
                x_start = max(0, x_end - tile_size)

            # Calculate overlap regions
            overlap_top = overlap if row > 0 else 0
            overlap_bottom = overlap if row < n_rows - 1 and y_end < height else 0
            overlap_left = overlap if col > 0 else 0
            overlap_right = overlap if col < n_cols - 1 and x_end < width else 0

            tiles.append(
                TileInfo(
                    row=row,
                    col=col,
                    y_start=y_start,
                    y_end=y_end,
                    x_start=x_start,
                    x_end=x_end,
                    overlap_top=overlap_top,
                    overlap_bottom=overlap_bottom,
                    overlap_left=overlap_left,
                    overlap_right=overlap_right,
                )
            )

    return tiles


def extract_tile(
    image: np.ndarray,
    tile_info: TileInfo,
) -> np.ndarray:
    """Extract a single tile from an image.

    Args:
        image: Source image array (2D or multi-channel 3D with channels first)
        tile_info: TileInfo describing the tile location

    Returns:
        Tile array with the same number of dimensions as input
    """
    if image.ndim == 2:
        return image[
            tile_info.y_start : tile_info.y_end,
            tile_info.x_start : tile_info.x_end,
        ].copy()
    elif image.ndim == 3:
        # Assume channels-first format (C, H, W)
        return image[
            :,
            tile_info.y_start : tile_info.y_end,
            tile_info.x_start : tile_info.x_end,
        ].copy()
    else:
        raise ValueError(f"Expected 2D or 3D image, got {image.ndim}D")


def split_image(
    image: np.ndarray,
    tile_size: int = 2048,
    overlap: int = 200,
    output_dir: Optional[Union[str, Path]] = None,
    filename_pattern: str = "tile_r{row}_c{col}.tif",
) -> tuple[list[tuple[np.ndarray, TileInfo]], list[TileInfo]]:
    """Split an image into overlapping tiles.

    Args:
        image: Source image array (2D or 3D with channels first)
        tile_size: Target size for each tile
        overlap: Overlap between adjacent tiles in pixels
        output_dir: If provided, save tiles to this directory
        filename_pattern: Pattern for tile filenames. Must include {row} and {col}.

    Returns:
        Tuple of:
        - List of (tile_array, tile_info) tuples
        - List of all TileInfo objects
    """
    if image.ndim == 2:
        shape = image.shape
    elif image.ndim == 3:
        shape = image.shape[1:]  # (H, W) from (C, H, W)
    else:
        raise ValueError(f"Expected 2D or 3D image, got {image.ndim}D")

    tile_infos = compute_tile_grid(shape, tile_size, overlap)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    tiles = []
    for info in tile_infos:
        tile = extract_tile(image, info)
        tiles.append((tile, info))

        if output_dir is not None:
            filename = filename_pattern.format(row=info.row, col=info.col)
            output_path = output_dir / filename
            tifffile.imwrite(str(output_path), tile)

    return tiles, tile_infos


def save_tile_info(
    tile_infos: list[TileInfo],
    output_path: Union[str, Path],
    image_shape: Optional[tuple[int, int]] = None,
    tile_size: Optional[int] = None,
    overlap: Optional[int] = None,
) -> Path:
    """Save tile metadata to JSON file.

    Args:
        tile_infos: List of TileInfo objects
        output_path: Path to save JSON file
        image_shape: Original image shape (height, width)
        tile_size: Tile size used for splitting
        overlap: Overlap used for splitting

    Returns:
        Path to saved file
    """
    output_path = Path(output_path)

    # Calculate grid dimensions
    max_row = max(t.row for t in tile_infos)
    max_col = max(t.col for t in tile_infos)

    data = {
        "n_rows": max_row + 1,
        "n_cols": max_col + 1,
        "n_tiles": len(tile_infos),
        "tiles": [t.to_dict() for t in tile_infos],
    }

    if image_shape is not None:
        data["image_shape"] = list(image_shape)
    if tile_size is not None:
        data["tile_size"] = tile_size
    if overlap is not None:
        data["overlap"] = overlap

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    return output_path


def load_tile_info(input_path: Union[str, Path]) -> tuple[list[TileInfo], dict]:
    """Load tile metadata from JSON file.

    Args:
        input_path: Path to tile_info.json file

    Returns:
        Tuple of:
        - List of TileInfo objects
        - Dict with metadata (n_rows, n_cols, image_shape, etc.)
    """
    input_path = Path(input_path)

    with open(input_path) as f:
        data = json.load(f)

    tile_infos = [TileInfo.from_dict(t) for t in data["tiles"]]

    metadata = {k: v for k, v in data.items() if k != "tiles"}

    return tile_infos, metadata
