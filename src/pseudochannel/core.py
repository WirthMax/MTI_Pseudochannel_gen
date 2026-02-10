"""Core pseudochannel computation functions."""

from typing import Literal

import numpy as np


def normalize_channel(
    img: np.ndarray,
    method: Literal["minmax", "percentile"] = "minmax",
    percentile_low: float = 1.0,
    percentile_high: float = 99.0,
) -> np.ndarray:
    """Normalize a channel image to 0-1 range.

    Args:
        img: Input image array
        method: Normalization method
            - "minmax": Scale by min/max values
            - "percentile": Scale by percentile values (more robust to outliers)
        percentile_low: Low percentile for "percentile" method
        percentile_high: High percentile for "percentile" method

    Returns:
        Normalized float32 array in range [0, 1]
    """
    img = img.astype(np.float32)

    if method == "minmax":
        vmin = img.min()
        vmax = img.max()
    elif method == "percentile":
        vmin = np.percentile(img, percentile_low)
        vmax = np.percentile(img, percentile_high)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    if vmax - vmin < 1e-10:
        return np.zeros_like(img, dtype=np.float32)

    normalized = (img - vmin) / (vmax - vmin)
    return np.clip(normalized, 0, 1)


def compute_pseudochannel(
    channels: dict[str, np.ndarray],
    weights: dict[str, float],
    normalize: Literal["minmax", "percentile", "none"] = "minmax",
    normalize_output: bool = True,
) -> np.ndarray:
    """Compute weighted sum of channels to create a pseudochannel.

    Args:
        channels: Dict of channel_name -> array (preview or full-res)
        weights: Dict of channel_name -> weight (typically 0-1)
        normalize: Normalization method for input channels
            - "minmax": Scale each channel by min/max
            - "percentile": Scale by 1st/99th percentile
            - "none": Use raw values
        normalize_output: If True, normalize the final output to 0-1

    Returns:
        Pseudochannel image as float32 array

    Raises:
        ValueError: If weights reference unknown channels
    """
    unknown_channels = set(weights.keys()) - set(channels.keys())
    if unknown_channels:
        raise ValueError(f"Unknown channels in weights: {unknown_channels}")

    active_weights = {k: v for k, v in weights.items() if v != 0}

    if not active_weights:
        first_channel = next(iter(channels.values()))
        return np.zeros(first_channel.shape, dtype=np.float32)

    result = None

    for channel_name, weight in active_weights.items():
        channel_data = channels[channel_name]

        if normalize != "none":
            channel_data = normalize_channel(channel_data, method=normalize)
        else:
            channel_data = channel_data.astype(np.float32)

        weighted = channel_data * weight

        if result is None:
            result = weighted
        else:
            result = result + weighted

    if normalize_output and result is not None:
        result = normalize_channel(result, method="minmax")

    return result


def _compute_global_channel_stats(
    channels: dict[str, np.ndarray],
    channel_names: list[str],
    method: Literal["minmax", "percentile"],
) -> dict[str, tuple[float, float]]:
    """Compute global min/max or percentiles for each channel.

    Args:
        channels: Dict of channel_name -> array
        channel_names: List of channel names to compute stats for
        method: Normalization method ("minmax" or "percentile")

    Returns:
        Dict mapping channel_name -> (vmin, vmax)
    """
    stats = {}
    for name in channel_names:
        arr = channels[name]
        if method == "percentile":
            vmin = float(np.percentile(arr, 1))
            vmax = float(np.percentile(arr, 99))
        else:  # minmax
            vmin = float(arr.min())
            vmax = float(arr.max())
        stats[name] = (vmin, vmax)
    return stats


def compute_pseudochannel_chunked(
    channels: dict[str, np.ndarray],
    weights: dict[str, float],
    normalize: Literal["minmax", "percentile", "none"] = "minmax",
    chunk_size: int = 1024,
) -> np.ndarray:
    """Compute pseudochannel in chunks with GLOBAL normalization.

    Uses a two-pass approach for memory-efficient processing:
    1. Compute global min/max for each weighted channel
    2. Apply uniform normalization during chunked processing

    This ensures consistent normalization across the entire image,
    avoiding patch artifacts that occur with per-chunk normalization.

    Args:
        channels: Dict of channel_name -> memory-mapped array
        weights: Dict of channel_name -> weight
        normalize: Normalization method
        chunk_size: Size of chunks to process

    Returns:
        Pseudochannel image as float32 array
    """
    active_weights = {k: v for k, v in weights.items() if v != 0}

    if not active_weights:
        first_channel = next(iter(channels.values()))
        return np.zeros(first_channel.shape[:2], dtype=np.float32)

    # Pass 1: Compute global normalization parameters
    if normalize != "none":
        global_stats = _compute_global_channel_stats(
            channels, list(active_weights.keys()), normalize
        )
    else:
        global_stats = None

    # Pass 2: Chunked computation with uniform normalization
    first_channel = next(iter(channels.values()))
    height, width = first_channel.shape[:2]
    result = np.zeros((height, width), dtype=np.float32)

    for y in range(0, height, chunk_size):
        y_end = min(y + chunk_size, height)
        for x in range(0, width, chunk_size):
            x_end = min(x + chunk_size, width)

            chunk_result = np.zeros((y_end - y, x_end - x), dtype=np.float32)

            for channel_name, weight in active_weights.items():
                chunk_data = channels[channel_name][y:y_end, x:x_end].astype(np.float32)

                # Apply GLOBAL normalization
                if global_stats is not None:
                    vmin, vmax = global_stats[channel_name]
                    if vmax - vmin > 1e-10:
                        chunk_data = np.clip((chunk_data - vmin) / (vmax - vmin), 0, 1)
                    else:
                        chunk_data = np.zeros_like(chunk_data)

                chunk_result += chunk_data * weight

            result[y:y_end, x:x_end] = chunk_result

    # Final output normalization
    vmin, vmax = result.min(), result.max()
    if vmax - vmin > 1e-10:
        result = (result - vmin) / (vmax - vmin)

    return result
