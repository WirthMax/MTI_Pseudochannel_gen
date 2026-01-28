"""Tissue Pseudochannel Generator for Multiplex Tissue Imaging data.

This package provides tools for:
- Loading multi-channel TIFF images
- Interactive weight tuning with real-time preview
- Generating weighted pseudochannel composites
- Batch processing of large datasets
"""

from .io import (
    DEFAULT_EXCLUDED_CHANNELS,
    MACSIMA_PATTERN,
    load_channel_folder,
    load_marker_names,
    load_ome_tiff,
    OMETiffChannels,
    parse_channel_name,
)
from .core import compute_pseudochannel
from .preview import create_preview_stack, downsample_image
from .config import save_config, load_config
from .batch import process_dataset

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_EXCLUDED_CHANNELS",
    "MACSIMA_PATTERN",
    "load_channel_folder",
    "load_marker_names",
    "load_ome_tiff",
    "OMETiffChannels",
    "parse_channel_name",
    "compute_pseudochannel",
    "create_preview_stack",
    "downsample_image",
    "save_config",
    "load_config",
    "process_dataset",
]
