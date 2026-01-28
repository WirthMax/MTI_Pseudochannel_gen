"""Efficient preview generation for large images."""

import numpy as np


def downsample_image(
    img: np.ndarray,
    target_size: int = 512,
    method: str = "stride",
) -> np.ndarray:
    """Fast downsampling of large images.

    Args:
        img: Input image array (2D or 3D)
        target_size: Target size for the longest dimension
        method: Downsampling method
            - "stride": Fast strided access (default)
            - "block": Block averaging (better quality)

    Returns:
        Downsampled image as float32 array
    """
    height, width = img.shape[:2]
    max_dim = max(height, width)

    if max_dim <= target_size:
        return img.astype(np.float32)

    scale = target_size / max_dim
    new_height = int(height * scale)
    new_width = int(width * scale)

    if method == "stride":
        stride_h = height // new_height
        stride_w = width // new_width
        downsampled = img[::stride_h, ::stride_w]

    elif method == "block":
        stride_h = height // new_height
        stride_w = width // new_width

        trim_h = new_height * stride_h
        trim_w = new_width * stride_w
        trimmed = img[:trim_h, :trim_w]

        reshaped = trimmed.reshape(
            new_height, stride_h,
            new_width, stride_w
        )
        downsampled = reshaped.mean(axis=(1, 3))

    else:
        raise ValueError(f"Unknown downsampling method: {method}")

    return downsampled.astype(np.float32)


def create_preview_stack(
    channels: dict[str, np.ndarray],
    target_size: int = 512,
    method: str = "stride",
) -> dict[str, np.ndarray]:
    """Create downsampled previews of all channels for interactive use.

    Loads and downsamples once at startup; previews are reused for all
    slider updates during interactive weight tuning.

    Args:
        channels: Dict of channel_name -> array (can be memory-mapped)
        target_size: Target size for longest dimension
        method: Downsampling method ("stride" or "block")

    Returns:
        Dict of channel_name -> downsampled float32 array
    """
    previews = {}

    for name, channel in channels.items():
        previews[name] = downsample_image(channel, target_size, method)

    return previews


def get_preview_scale(
    original_shape: tuple[int, ...],
    preview_shape: tuple[int, ...],
) -> tuple[float, float]:
    """Get the scale factors between original and preview.

    Args:
        original_shape: Shape of original image
        preview_shape: Shape of preview image

    Returns:
        Tuple of (scale_y, scale_x)
    """
    scale_y = preview_shape[0] / original_shape[0]
    scale_x = preview_shape[1] / original_shape[1]
    return scale_y, scale_x


def extract_region_preview(
    channels: dict[str, np.ndarray],
    y: int,
    x: int,
    size: int = 512,
) -> dict[str, np.ndarray]:
    """Extract a region from full-resolution channels for detailed preview.

    Useful for inspecting specific areas at full resolution.

    Args:
        channels: Dict of channel_name -> full-res array
        y: Top-left y coordinate
        x: Top-left x coordinate
        size: Size of region to extract

    Returns:
        Dict of channel_name -> region array
    """
    regions = {}

    for name, channel in channels.items():
        height, width = channel.shape[:2]

        y_end = min(y + size, height)
        x_end = min(x + size, width)
        y_start = max(0, y)
        x_start = max(0, x)

        region = channel[y_start:y_end, x_start:x_end]
        regions[name] = region.astype(np.float32)

    return regions
