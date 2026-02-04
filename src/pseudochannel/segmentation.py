"""Cellpose segmentation wrapper with lazy imports.

Cellpose is an optional dependency — all imports happen inside functions
so this module loads without it installed.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np


def _gpu_available() -> bool:
    """Check if a CUDA GPU is available via PyTorch."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


@dataclass
class CellposeConfig:
    """Configuration for Cellpose segmentation."""

    model_type: str = "cyto3"
    diameter: Optional[float] = None  # None = auto-estimate
    flow_threshold: float = 0.4
    cellprob_threshold: float = 0.0
    gpu: Optional[bool] = None  # None = auto-detect
    min_size: int = 15

    def __post_init__(self):
        if self.gpu is None:
            self.gpu = _gpu_available()

    def to_dict(self) -> dict:
        """Serialize to dict for YAML round-tripping."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CellposeConfig":
        """Create from dict (e.g. loaded from YAML)."""
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in known_fields}
        return cls(**filtered)


def _check_cellpose() -> bool:
    """Check whether cellpose is importable.

    Caches only successful imports so that installing cellpose mid-session
    is picked up on the next call.
    """
    if getattr(_check_cellpose, "_available", False):
        return True
    try:
        import cellpose  # noqa: F401
        _check_cellpose._available = True
        return True
    except ImportError:
        return False


def create_cellpose_model(config: CellposeConfig):
    """Create a Cellpose model instance.

    Args:
        config: CellposeConfig with model parameters.

    Returns:
        cellpose.models.Cellpose model instance.

    Raises:
        ImportError: If cellpose is not installed.
    """
    if not _check_cellpose():
        raise ImportError(
            "Cellpose is not installed. Install it with:\n"
            "  pip install cellpose\n"
            "For GPU support:\n"
            "  pip install cellpose torch --extra-index-url "
            "https://download.pytorch.org/whl/cu118"
        )
    from cellpose import models

    return models.Cellpose(model_type=config.model_type, gpu=config.gpu)


def run_segmentation(
    model,
    pseudochannel: np.ndarray,
    nuclear: Optional[np.ndarray],
    config: CellposeConfig,
) -> np.ndarray:
    """Run Cellpose segmentation on a pseudochannel image.

    Args:
        model: Cellpose model from create_cellpose_model().
        pseudochannel: (H, W) float32 array in [0, 1].
        nuclear: Optional (H, W) float32 nuclear marker in [0, 1].
        config: CellposeConfig with eval parameters.

    Returns:
        Integer mask array (H, W) where 0=background, 1..N=cell IDs.
    """
    # Scale from [0, 1] to [0, 255] for Cellpose
    pseudo_u8 = (np.clip(pseudochannel, 0, 1) * 255).astype(np.float32)

    if nuclear is not None:
        nuclear_u8 = (np.clip(nuclear, 0, 1) * 255).astype(np.float32)
        img = np.stack([pseudo_u8, nuclear_u8], axis=-1)  # (H, W, 2)
        channels = [1, 2]  # cyto=ch1, nuc=ch2
    else:
        img = pseudo_u8  # (H, W)
        channels = [0, 0]  # grayscale

    masks, flows, styles, diams = model.eval(
        img,
        diameter=config.diameter,
        channels=channels,
        flow_threshold=config.flow_threshold,
        cellprob_threshold=config.cellprob_threshold,
        min_size=config.min_size,
    )

    return masks


def extract_mask_contours(masks: np.ndarray) -> list[np.ndarray]:
    """Extract boundary contours from a label mask.

    Args:
        masks: Integer mask array (H, W), 0=background.

    Returns:
        List of (N, 2) arrays with (x, y) coordinates for each cell boundary.
    """
    try:
        from skimage.measure import find_contours
    except ImportError:
        # Fallback: use numpy gradient-based edge detection
        return _extract_contours_numpy(masks)

    contours = []
    for cell_id in range(1, masks.max() + 1):
        cell_mask = (masks == cell_id).astype(np.float64)
        cell_contours = find_contours(cell_mask, 0.5)
        for c in cell_contours:
            # find_contours returns (row, col) = (y, x); convert to (x, y)
            xy = c[:, ::-1]
            contours.append(xy)

    return contours


def _extract_contours_numpy(masks: np.ndarray) -> list[np.ndarray]:
    """Fallback contour extraction using numpy when skimage is unavailable.

    Finds boundary pixels via gradient and returns them as coordinate arrays.
    """
    contours = []
    for cell_id in range(1, masks.max() + 1):
        cell_mask = masks == cell_id
        # Find boundary: pixels in mask adjacent to pixels not in mask
        gy = np.diff(cell_mask.astype(np.int8), axis=0)
        gx = np.diff(cell_mask.astype(np.int8), axis=1)

        edge_y = np.zeros_like(cell_mask)
        edge_x = np.zeros_like(cell_mask)
        edge_y[:-1, :] |= gy != 0
        edge_y[1:, :] |= gy != 0
        edge_x[:, :-1] |= gx != 0
        edge_x[:, 1:] |= gx != 0

        boundary = (edge_y | edge_x) & cell_mask
        ys, xs = np.where(boundary)
        if len(xs) > 0:
            # Sort boundary pixels by angle from centroid for rough ordering
            cx, cy = xs.mean(), ys.mean()
            angles = np.arctan2(ys - cy, xs - cx)
            order = np.argsort(angles)
            coords = np.column_stack([xs[order], ys[order]])
            contours.append(coords)

    return contours


def overlay_contours_on_axes(
    ax,
    contours: list[np.ndarray],
    color: str = "lime",
    linewidth: float = 1.0,
    alpha: float = 0.8,
) -> list:
    """Draw contour lines on a matplotlib Axes.

    Args:
        ax: Matplotlib Axes to draw on.
        contours: List of (N, 2) xy-coordinate arrays.
        color: Line color.
        linewidth: Line width.
        alpha: Line opacity.

    Returns:
        List of Line2D artists (for later removal).
    """
    lines = []
    for c in contours:
        (line,) = ax.plot(
            c[:, 0], c[:, 1],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
        )
        lines.append(line)
    return lines
