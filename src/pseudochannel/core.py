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


def compute_pseudochannel_chunked(
    channels: dict[str, np.ndarray],
    weights: dict[str, float],
    normalize: Literal["minmax", "percentile", "none"] = "minmax",
    chunk_size: int = 1024,
) -> np.ndarray:
    """Compute pseudochannel in chunks for memory efficiency.

    Useful for very large images that don't fit in RAM.

    Args:
        channels: Dict of channel_name -> memory-mapped array
        weights: Dict of channel_name -> weight
        normalize: Normalization method
        chunk_size: Size of chunks to process

    Returns:
        Pseudochannel image as float32 array
    """
    first_channel = next(iter(channels.values()))
    height, width = first_channel.shape[:2]

    result = np.zeros((height, width), dtype=np.float32)

    active_weights = {k: v for k, v in weights.items() if v != 0}

    if not active_weights:
        return result

    for y in range(0, height, chunk_size):
        y_end = min(y + chunk_size, height)

        for x in range(0, width, chunk_size):
            x_end = min(x + chunk_size, width)

            chunk_channels = {
                name: arr[y:y_end, x:x_end]
                for name, arr in channels.items()
            }

            chunk_result = compute_pseudochannel(
                chunk_channels,
                active_weights,
                normalize=normalize,
                normalize_output=False,
            )

            result[y:y_end, x:x_end] = chunk_result

    vmin, vmax = result.min(), result.max()
    if vmax - vmin > 1e-10:
        result = (result - vmin) / (vmax - vmin)

    return result
