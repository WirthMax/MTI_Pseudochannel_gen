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


# Default regex pattern for MACSima filenames:
# C-000_S-000_AFB_APC_R-01_W-A-1_ROI-08_A-None.tif -> "None"
# C-001_S-000_S_APC_R-01_W-A-1_ROI-08_A-VG_C-234TCR.tif -> "VG"
# The marker name follows "_A-" up to the next underscore or dot (strips _C_<Clone> suffix)
MACSIMA_PATTERN = r"_A-([^_.]+)"

# Pattern to detect MACSima DAPI files and extract their C-number
# Matches: C-001_S-000_S_DAPI_R-01_W-A-1_ROI-08_A-DAPI.tif
MACSIMA_DAPI_PATTERN = re.compile(r"^C-(\d+)_.*_S_DAPI_.*_A-DAPI\.", re.IGNORECASE)

# Compiled default pattern
_DEFAULT_MARKER_PATTERN = re.compile(MACSIMA_PATTERN)


class FolderChannels(dict):
    """Dict of channels with optional nuclear marker path.

    Extends dict so it can be used anywhere a regular channel dict is expected,
    while also carrying metadata about the auto-detected nuclear marker.
    """

    def __init__(self, *args, nuclear_path: Optional[Path] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.nuclear_path = nuclear_path


def parse_channel_name(
    filename: str,
    pattern: Optional[Union[str, re.Pattern]] = None,
) -> str:
    """Extract marker/channel name from filename.

    By default uses the MACSima naming convention where marker names
    follow '_A-' in the filename. Custom patterns can be provided for
    other instruments.

    Args:
        filename: The filename (with or without path)
        pattern: Optional regex pattern with one capture group for the marker name.
            Can be a string or compiled re.Pattern. If None, uses MACSima pattern.
            Examples:
                r"_A-([^.]+)$"           - MACSima (default)
                r"^([^_]+)_"             - Marker at start before first underscore
                r"_channel_([^_]+)_"     - Marker after '_channel_'

    Returns:
        Extracted marker/channel name
    """
    stem = Path(filename).stem

    # Compile pattern if string
    if pattern is None:
        compiled = _DEFAULT_MARKER_PATTERN
    elif isinstance(pattern, str):
        compiled = re.compile(pattern)
    else:
        compiled = pattern

    # Try to match the pattern
    match = compiled.search(stem)
    if match:
        return match.group(1)

    # Fall back to using the full stem as the channel name
    return stem


def load_channel_folder(
    folder_path: Union[str, Path],
    extensions: tuple[str, ...] = (".tif", ".tiff"),
    use_memmap: bool = True,
    exclude_channels: set[str] | list[str] | None = None,
    marker_pattern: Optional[Union[str, re.Pattern]] = None,
    macsima_mode: bool = False,
) -> Union[dict[str, np.ndarray], FolderChannels]:
    """Load all channel TIFFs from folder.

    Args:
        folder_path: Path to folder containing channel TIFF files
        extensions: File extensions to include
        use_memmap: If True, memory-map files for efficiency with large images
        exclude_channels: Channel names to exclude (case-insensitive).
            If None, uses DEFAULT_EXCLUDED_CHANNELS.
            Pass empty set/list to include all channels.
        marker_pattern: Regex pattern to extract marker name from filename.
            Must contain one capture group. If None, uses MACSima pattern.
            See parse_channel_name() for examples.
        macsima_mode: If True, enables MACSima-specific handling:
            - Uses MACSIMA_PATTERN for marker name extraction
            - Auto-detects DAPI files and keeps only the one with lowest C-number
            - Returns FolderChannels with nuclear_path set

    Returns:
        Dict mapping channel_name -> array (memory-mapped if use_memmap=True).
        If macsima_mode=True, returns FolderChannels with nuclear_path attribute.

    Raises:
        FileNotFoundError: If folder doesn't exist
        ValueError: If no TIFF files found
    """
    folder_path = Path(folder_path)

    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    if not folder_path.is_dir():
        raise ValueError(f"Path is not a directory: {folder_path}")

    # Use MACSima pattern if macsima_mode and no custom pattern
    if macsima_mode and marker_pattern is None:
        marker_pattern = MACSIMA_PATTERN

    tiff_files = sorted(
        f for f in folder_path.iterdir()
        if f.is_file() and f.suffix.lower() in extensions
    )

    if not tiff_files:
        raise ValueError(f"No TIFF files found in {folder_path}")

    # In MACSima mode, find DAPI files and select the one with lowest C-number
    nuclear_path = None
    dapi_files_to_skip = set()

    if macsima_mode:
        dapi_candidates = []
        for tiff_file in tiff_files:
            match = MACSIMA_DAPI_PATTERN.match(tiff_file.name)
            if match:
                c_number = int(match.group(1))
                dapi_candidates.append((c_number, tiff_file))

        if dapi_candidates:
            # Sort by C-number, keep only the lowest
            dapi_candidates.sort(key=lambda x: x[0])
            nuclear_path = dapi_candidates[0][1]
            # Skip all other DAPI files
            dapi_files_to_skip = {f for _, f in dapi_candidates[1:]}

    channels = {}

    for tiff_file in tiff_files:
        # Skip duplicate DAPI files in MACSima mode
        if tiff_file in dapi_files_to_skip:
            continue

        channel_name = parse_channel_name(tiff_file.name, pattern=marker_pattern)

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

    if macsima_mode:
        return FolderChannels(channels, nuclear_path=nuclear_path)

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


def load_mcmicro_markers(
    marker_file: Union[str, Path],
    keep_removed: bool = False,
) -> list[str]:
    """Load marker names from MCMICRO markers.csv format.

    MCMICRO marker files have columns like:
        channel_number,cycle_number,marker_name,Filter,background,exposure,remove

    This function extracts marker_name and optionally filters out rows
    where remove=TRUE.

    Args:
        marker_file: Path to MCMICRO markers.csv file
        keep_removed: If False (default), skip rows where remove=TRUE.
            Set to True to include all markers.

    Returns:
        List of marker names in channel order

    Raises:
        FileNotFoundError: If marker file doesn't exist
        ValueError: If file format is invalid
    """
    marker_file = Path(marker_file)

    if not marker_file.exists():
        raise FileNotFoundError(f"Marker file not found: {marker_file}")

    with open(marker_file, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"Empty marker file: {marker_file}")

    # Check for required column
    if "marker_name" not in rows[0]:
        raise ValueError(
            f"Column 'marker_name' not found in {marker_file}. "
            "Expected MCMICRO format with columns: "
            "channel_number,cycle_number,marker_name,Filter,background,exposure,remove"
        )

    names = []
    for row in rows:
        # Check remove column if it exists
        if not keep_removed and "remove" in row:
            remove_val = row["remove"].strip().upper()
            if remove_val in ("TRUE", "1", "YES"):
                continue

        name = row["marker_name"].strip()
        if name:
            names.append(name)

    if not names:
        raise ValueError(f"No marker names found in {marker_file} after filtering")

    return names


def load_ome_tiff(
    tiff_path: Union[str, Path],
    marker_file: Union[str, Path],
    marker_column: Optional[Union[int, str]] = None,
    use_memmap: bool = True,
    exclude_channels: set[str] | list[str] | None = None,
    mcmicro_markers: bool = False,
) -> dict[str, np.ndarray]:
    """Load a multi-channel OME-TIFF with marker names from a separate file.

    Note: This function loads ALL channels into memory at once. For large files,
    consider using OMETiffChannels instead, which loads channels on-demand.

    Args:
        tiff_path: Path to OME-TIFF file containing all channels
        marker_file: Path to file with marker names (one per channel)
        marker_column: Column to read from CSV/TSV marker file (see load_marker_names)
        use_memmap: If True, memory-map the file for efficiency
        exclude_channels: Channel names to exclude (case-insensitive).
            If None, uses DEFAULT_EXCLUDED_CHANNELS.
            Pass empty set/list to include all channels.
        mcmicro_markers: If True, parse marker_file as MCMICRO format
            (with marker_name column and remove column for filtering).
            This filters out channels marked remove=TRUE.

    Returns:
        Dict mapping channel_name -> 2D array (view into the OME-TIFF)

    Raises:
        FileNotFoundError: If files don't exist
        ValueError: If number of markers doesn't match channels
    """
    tiff_path = Path(tiff_path)

    if not tiff_path.exists():
        raise FileNotFoundError(f"TIFF file not found: {tiff_path}")

    if mcmicro_markers:
        marker_names = load_mcmicro_markers(marker_file)
    else:
        marker_names = load_marker_names(marker_file, column=marker_column)

    if use_memmap:
        try:
            data = tifffile.memmap(str(tiff_path), mode="r")
        except ValueError:
            # Compressed TIFFs can't be memory-mapped, fall back to imread
            data = tifffile.imread(str(tiff_path))
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


def detect_input_mode(
    channel_folder: Optional[Union[str, Path]],
    ome_tiff_path: Optional[Union[str, Path]],
    marker_file: Optional[Union[str, Path]],
    prefer_ome_tiff: bool = False,
) -> str:
    """Detect which input mode to use based on provided paths.

    Args:
        channel_folder: Path to folder with channel TIFFs (or None)
        ome_tiff_path: Path to OME-TIFF file (or None)
        marker_file: Path to marker names file (required for OME-TIFF)
        prefer_ome_tiff: If both inputs available, prefer OME-TIFF

    Returns:
        'folder' or 'ome_tiff'

    Raises:
        ValueError: If no valid input configuration detected
    """
    has_folder = channel_folder is not None and Path(channel_folder).is_dir()
    has_ome_tiff = (
        ome_tiff_path is not None
        and Path(ome_tiff_path).is_file()
        and marker_file is not None
        and Path(marker_file).is_file()
    )

    if has_folder and has_ome_tiff:
        return 'ome_tiff' if prefer_ome_tiff else 'folder'
    elif has_folder:
        return 'folder'
    elif has_ome_tiff:
        return 'ome_tiff'
    elif ome_tiff_path is not None and marker_file is None:
        raise ValueError("OME-TIFF path provided but marker file is missing")
    else:
        raise ValueError(
            "No valid input detected. Provide either:\n"
            "  - channel_folder (path to folder with TIFF files)\n"
            "  - ome_tiff_path and marker_file (for OME-TIFF input)"
        )


class OMETiffChannels:
    """Wrapper for OME-TIFF that provides dict-like access to channels.

    This class uses lazy loading - it opens the file instantly (reading only
    metadata) and loads individual channel data on-demand when accessed.
    This is much more efficient for large files, especially compressed ones.

    Use as a context manager to ensure proper cleanup:
        with OMETiffChannels(path, markers) as channels:
            img = channels["CD45"]
    """

    def __init__(
        self,
        tiff_path: Union[str, Path],
        marker_file: Union[str, Path],
        marker_column: Optional[Union[int, str]] = None,
        exclude_channels: set[str] | list[str] | None = None,
        mcmicro_markers: bool = False,
    ):
        """Initialize OME-TIFF channel accessor.

        Args:
            tiff_path: Path to OME-TIFF file
            marker_file: Path to marker names file
            marker_column: Column for CSV/TSV files
            exclude_channels: Channel names to exclude (case-insensitive).
                If None, uses DEFAULT_EXCLUDED_CHANNELS.
                Pass empty set/list to include all channels.
            mcmicro_markers: If True, parse marker_file as MCMICRO format
                (with marker_name column and remove column for filtering).
        """
        self.tiff_path = Path(tiff_path)
        if mcmicro_markers:
            all_marker_names = load_mcmicro_markers(marker_file)
        else:
            all_marker_names = load_marker_names(marker_file, column=marker_column)

        # Open file lazily - only reads metadata, not pixel data
        self._tiff = tifffile.TiffFile(str(self.tiff_path))

        # Get the first series (standard for OME-TIFF)
        if not self._tiff.series:
            raise ValueError(f"No image series found in {tiff_path}")
        self._series = self._tiff.series[0]

        # Get shape from series metadata
        series_shape = self._series.shape
        ndim = len(series_shape)

        if ndim == 2:
            # Single 2D image
            n_channels = 1
            self._z_slice = None
        elif ndim == 3:
            # (C, Y, X)
            n_channels = series_shape[0]
            self._z_slice = None
        elif ndim == 4:
            # (C, Z, Y, X) - take first Z slice
            n_channels = series_shape[0]
            self._z_slice = 0
        else:
            raise ValueError(f"Unexpected dimensions: {ndim}")

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

        # Cache for loaded channels (optional, can reduce repeated reads)
        self._cache: dict[str, np.ndarray] = {}

    def __getitem__(self, name: str) -> np.ndarray:
        """Get channel by name (loads on demand)."""
        if name not in self._name_to_idx:
            raise KeyError(f"Unknown channel: {name}")

        # Return cached if available
        if name in self._cache:
            return self._cache[name]

        idx = self._name_to_idx[name]

        # Read only this channel from the file using page-level access
        # This avoids loading the entire multi-channel array
        series_shape = self._series.shape
        ndim = len(series_shape)

        if ndim == 2:
            # Single 2D image - just one page
            data = self._series.pages[0].asarray()
        elif ndim == 3:
            # (C, Y, X) - each channel is a separate page
            data = self._series.pages[idx].asarray()
        elif ndim == 4:
            # (C, Z, Y, X) - pages are arranged as C*Z
            # Page index = channel_idx * n_z_slices + z_slice
            n_z = series_shape[1]
            page_idx = idx * n_z + self._z_slice
            data = self._series.pages[page_idx].asarray()
        else:
            # Fallback: read specific page by index
            data = self._series.pages[idx].asarray()

        self._cache[name] = data
        return data

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

    def get_channel_by_index(self, idx: int) -> np.ndarray:
        """Get channel by original index (bypasses exclusion filtering).

        This is useful for loading channels like DAPI that may be excluded
        from the main marker list but still needed for visualization.

        Args:
            idx: Original channel index in the TIFF file

        Returns:
            Channel data as numpy array
        """
        if idx < 0 or idx >= len(self._all_marker_names):
            raise IndexError(f"Channel index {idx} out of range (0-{len(self._all_marker_names)-1})")

        series_shape = self._series.shape
        ndim = len(series_shape)

        if ndim == 2:
            return self._series.pages[0].asarray()
        elif ndim == 3:
            return self._series.pages[idx].asarray()
        elif ndim == 4:
            n_z = series_shape[1]
            page_idx = idx * n_z + self._z_slice
            return self._series.pages[page_idx].asarray()
        else:
            return self._series.pages[idx].asarray()

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of individual channel images (H, W)."""
        series_shape = self._series.shape
        ndim = len(series_shape)
        if ndim == 2:
            return series_shape
        elif ndim == 3:
            return series_shape[1:]
        else:
            return series_shape[2:]

    @property
    def dtype(self):
        """Data type of the array."""
        return self._series.dtype

    def close(self):
        """Close the underlying file."""
        if hasattr(self, "_tiff") and self._tiff is not None:
            self._tiff.close()
            self._tiff = None
        self._cache.clear()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False

    def __del__(self):
        """Cleanup on deletion."""
        self.close()
