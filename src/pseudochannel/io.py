"""Image loading utilities for multi-channel TIFF data."""

import csv
import re
from pathlib import Path
from typing import Optional, Union

import numpy as np
import tifffile

# Default channels to exclude (case-insensitive matching)
DEFAULT_EXCLUDED_CHANNELS = {
    "DAPI",              # Nuclear stain
    "None",              # Empty channels
    "APC",               # Autofluorescence/filter channel
    "FITC",              # Autofluorescence/filter channel
    "PE",                # Autofluorescence/filter channel
    "FE",                # Iron/artifact
    "Autofluorescence",
    "Empty",
    "Background",
    "AF",                # Autofluorescence abbreviation
}


def _should_exclude_channel(
    channel_name: str,
    exclude_channels: set[str] | list[str] | None,
) -> bool:
    """Check if a channel should be excluded based on its name.

    Args:
        channel_name: Name of the channel to check
        exclude_channels: Set/list of channel names to exclude (case-insensitive).
            If None, uses DEFAULT_EXCLUDED_CHANNELS.
            If empty set/list, excludes nothing.

    Returns:
        True if channel should be excluded, False otherwise
    """
    if exclude_channels is None:
        exclude_set = DEFAULT_EXCLUDED_CHANNELS
    else:
        exclude_set = set(exclude_channels)

    # Case-insensitive comparison
    channel_lower = channel_name.lower()
    return any(excl.lower() == channel_lower for excl in exclude_set)


# Regex pattern to extract marker name from filenames like:
# C-000_S-000_AFB_APC_R-01_W-A-1_ROI-08_A-None.tif -> "None"
# C-001_S-000_S_APC_R-01_W-A-1_ROI-08_A-VG_C-234TCR.tif -> "VG_C-234TCR"
_MARKER_NAME_PATTERN = re.compile(r"_A-([^.]+)$")


def parse_channel_name(filename: str) -> str:
    """Extract marker/channel name from filename.

    Supports two naming conventions:
    1. Complex filenames with '_A-<marker>' suffix:
       'C-000_S-000_AFB_APC_R-01_W-A-1_ROI-08_A-None.tif' -> 'None'
       'C-001_S-000_S_APC_R-01_W-A-1_ROI-08_A-CD45_C-REA123.tif' -> 'CD45_C-REA123'
    2. Simple filenames where stem is the marker name:
       'CD45.tif' -> 'CD45'
       'DAPI_channel.tiff' -> 'DAPI_channel'

    Args:
        filename: The filename (with or without path)

    Returns:
        Extracted marker/channel name
    """
    stem = Path(filename).stem

    # Try to match the _A-<marker> pattern
    match = _MARKER_NAME_PATTERN.search(stem)
    if match:
        return match.group(1)

    # Fall back to using the full stem as the channel name
    return stem


def load_channel_folder(
    folder_path: Union[str, Path],
    extensions: tuple[str, ...] = (".tif", ".tiff"),
    use_memmap: bool = True,
    exclude_channels: set[str] | list[str] | None = None,
) -> dict[str, np.ndarray]:
    """Load all channel TIFFs from folder.

    Args:
        folder_path: Path to folder containing channel TIFF files
        extensions: File extensions to include
        use_memmap: If True, memory-map files for efficiency with large images
        exclude_channels: Channel names to exclude (case-insensitive).
            If None, uses DEFAULT_EXCLUDED_CHANNELS.
            Pass empty set/list to include all channels.

    Returns:
        Dict mapping channel_name -> array (memory-mapped if use_memmap=True)

    Raises:
        FileNotFoundError: If folder doesn't exist
        ValueError: If no TIFF files found
    """
    folder_path = Path(folder_path)

    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    if not folder_path.is_dir():
        raise ValueError(f"Path is not a directory: {folder_path}")

    channels = {}

    tiff_files = sorted(
        f for f in folder_path.iterdir()
        if f.is_file() and f.suffix.lower() in extensions
    )

    if not tiff_files:
        raise ValueError(f"No TIFF files found in {folder_path}")

    for tiff_file in tiff_files:
        channel_name = parse_channel_name(tiff_file.name)

        # Skip excluded channels
        if _should_exclude_channel(channel_name, exclude_channels):
            continue

        if use_memmap:
            img = tifffile.memmap(str(tiff_file), mode="r")
        else:
            img = tifffile.imread(str(tiff_file))

        channels[channel_name] = img

    if not channels:
        raise ValueError(
            f"No channels loaded from {folder_path} after applying exclusions"
        )

    return channels


def get_channel_info(channels: dict[str, np.ndarray]) -> dict:
    """Get information about loaded channels.

    Args:
        channels: Dict from load_channel_folder

    Returns:
        Dict with channel metadata (shape, dtype, etc.)
    """
    info = {}
    for name, arr in channels.items():
        info[name] = {
            "shape": arr.shape,
            "dtype": str(arr.dtype),
            "min": float(np.min(arr)) if not hasattr(arr, "_mmap") else None,
            "max": float(np.max(arr)) if not hasattr(arr, "_mmap") else None,
        }
    return info


def validate_channels(channels: dict[str, np.ndarray]) -> bool:
    """Validate that all channels have the same shape.

    Args:
        channels: Dict from load_channel_folder

    Returns:
        True if all channels have matching shapes

    Raises:
        ValueError: If shapes don't match
    """
    if not channels:
        raise ValueError("No channels provided")

    shapes = [(name, arr.shape) for name, arr in channels.items()]
    first_name, first_shape = shapes[0]

    for name, shape in shapes[1:]:
        if shape != first_shape:
            raise ValueError(
                f"Shape mismatch: {first_name} has shape {first_shape}, "
                f"but {name} has shape {shape}"
            )

    return True


def load_marker_names(
    marker_file: Union[str, Path],
    column: Optional[Union[int, str]] = None,
) -> list[str]:
    """Load marker/channel names from a file.

    Supports multiple formats:
    - Plain text: one marker name per line
    - CSV/TSV: specify column by index or header name

    Args:
        marker_file: Path to file containing marker names
        column: For CSV/TSV files, the column to read.
            - None: treat as plain text (one name per line)
            - int: column index (0-based)
            - str: column header name

    Returns:
        List of marker names in order

    Raises:
        FileNotFoundError: If marker file doesn't exist
        ValueError: If file is empty or column not found
    """
    marker_file = Path(marker_file)

    if not marker_file.exists():
        raise FileNotFoundError(f"Marker file not found: {marker_file}")

    suffix = marker_file.suffix.lower()

    if column is None and suffix not in (".csv", ".tsv"):
        with open(marker_file) as f:
            names = [line.strip() for line in f if line.strip()]
    else:
        delimiter = "\t" if suffix == ".tsv" else ","

        with open(marker_file, newline="") as f:
            reader = csv.reader(f, delimiter=delimiter)
            rows = list(reader)

        if not rows:
            raise ValueError(f"Empty marker file: {marker_file}")

        if isinstance(column, str):
            header = rows[0]
            if column not in header:
                raise ValueError(f"Column '{column}' not found in {marker_file}")
            col_idx = header.index(column)
            rows = rows[1:]
        elif isinstance(column, int):
            col_idx = column
        else:
            col_idx = 0

        names = [row[col_idx].strip() for row in rows if row and row[col_idx].strip()]

    if not names:
        raise ValueError(f"No marker names found in {marker_file}")

    return names


def load_ome_tiff(
    tiff_path: Union[str, Path],
    marker_file: Union[str, Path],
    marker_column: Optional[Union[int, str]] = None,
    use_memmap: bool = True,
    exclude_channels: set[str] | list[str] | None = None,
) -> dict[str, np.ndarray]:
    """Load a multi-channel OME-TIFF with marker names from a separate file.

    Args:
        tiff_path: Path to OME-TIFF file containing all channels
        marker_file: Path to file with marker names (one per channel)
        marker_column: Column to read from CSV/TSV marker file (see load_marker_names)
        use_memmap: If True, memory-map the file for efficiency
        exclude_channels: Channel names to exclude (case-insensitive).
            If None, uses DEFAULT_EXCLUDED_CHANNELS.
            Pass empty set/list to include all channels.

    Returns:
        Dict mapping channel_name -> 2D array (view into the OME-TIFF)

    Raises:
        FileNotFoundError: If files don't exist
        ValueError: If number of markers doesn't match channels
    """
    tiff_path = Path(tiff_path)

    if not tiff_path.exists():
        raise FileNotFoundError(f"TIFF file not found: {tiff_path}")

    marker_names = load_marker_names(marker_file, column=marker_column)

    if use_memmap:
        data = tifffile.memmap(str(tiff_path), mode="r")
    else:
        data = tifffile.imread(str(tiff_path))

    if data.ndim == 2:
        if len(marker_names) != 1:
            raise ValueError(
                f"Single 2D image but {len(marker_names)} marker names provided"
            )
        name = marker_names[0]
        if _should_exclude_channel(name, exclude_channels):
            raise ValueError(
                f"Single channel '{name}' is excluded - no channels to load"
            )
        return {name: data}

    if data.ndim == 3:
        n_channels = data.shape[0]
    elif data.ndim == 4:
        n_channels = data.shape[0]
        data = data[:, 0, :, :]
    else:
        raise ValueError(f"Unexpected array dimensions: {data.ndim}")

    if len(marker_names) != n_channels:
        raise ValueError(
            f"Mismatch: {n_channels} channels in TIFF but "
            f"{len(marker_names)} marker names provided"
        )

    channels = {}
    for i, name in enumerate(marker_names):
        # Skip excluded channels
        if _should_exclude_channel(name, exclude_channels):
            continue
        channels[name] = data[i]

    if not channels:
        raise ValueError(
            f"No channels loaded from {tiff_path} after applying exclusions"
        )

    return channels


class OMETiffChannels:
    """Wrapper for OME-TIFF that provides dict-like access to channels.

    This class keeps the underlying memory-mapped array open and provides
    views into individual channels, which is more memory-efficient than
    creating separate arrays for each channel.
    """

    def __init__(
        self,
        tiff_path: Union[str, Path],
        marker_file: Union[str, Path],
        marker_column: Optional[Union[int, str]] = None,
        exclude_channels: set[str] | list[str] | None = None,
    ):
        """Initialize OME-TIFF channel accessor.

        Args:
            tiff_path: Path to OME-TIFF file
            marker_file: Path to marker names file
            marker_column: Column for CSV/TSV files
            exclude_channels: Channel names to exclude (case-insensitive).
                If None, uses DEFAULT_EXCLUDED_CHANNELS.
                Pass empty set/list to include all channels.
        """
        self.tiff_path = Path(tiff_path)
        all_marker_names = load_marker_names(marker_file, column=marker_column)

        self._data = tifffile.memmap(str(self.tiff_path), mode="r")

        if self._data.ndim == 3:
            self._channel_axis = 0
        elif self._data.ndim == 4:
            self._channel_axis = 0
            self._z_slice = 0
        else:
            raise ValueError(f"Unexpected dimensions: {self._data.ndim}")

        n_channels = self._data.shape[self._channel_axis]
        if len(all_marker_names) != n_channels:
            raise ValueError(
                f"Mismatch: {n_channels} channels but "
                f"{len(all_marker_names)} marker names"
            )

        # Filter out excluded channels
        self._all_marker_names = all_marker_names
        self.marker_names = [
            name for name in all_marker_names
            if not _should_exclude_channel(name, exclude_channels)
        ]

        if not self.marker_names:
            raise ValueError(
                f"No channels remaining after applying exclusions to {tiff_path}"
            )

        # Map from name to original index in the TIFF
        self._name_to_idx = {
            name: i for i, name in enumerate(all_marker_names)
            if name in self.marker_names
        }

    def __getitem__(self, name: str) -> np.ndarray:
        """Get channel by name."""
        if name not in self._name_to_idx:
            raise KeyError(f"Unknown channel: {name}")

        idx = self._name_to_idx[name]

        if self._data.ndim == 3:
            return self._data[idx]
        else:
            return self._data[idx, self._z_slice]

    def __contains__(self, name: str) -> bool:
        """Check if channel exists."""
        return name in self._name_to_idx

    def __iter__(self):
        """Iterate over channel names."""
        return iter(self.marker_names)

    def __len__(self) -> int:
        """Number of channels."""
        return len(self.marker_names)

    def keys(self):
        """Channel names."""
        return self._name_to_idx.keys()

    def values(self):
        """Channel arrays."""
        for name in self.marker_names:
            yield self[name]

    def items(self):
        """(name, array) pairs."""
        for name in self.marker_names:
            yield name, self[name]

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of individual channel images (H, W)."""
        if self._data.ndim == 3:
            return self._data.shape[1:]
        else:
            return self._data.shape[2:]

    @property
    def dtype(self):
        """Data type of the array."""
        return self._data.dtype
