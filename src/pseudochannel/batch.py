"""Batch processing for pseudochannel generation."""

from pathlib import Path
from typing import Optional, Union

import numpy as np
import tifffile

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from .config import get_normalization_from_config, get_weights_from_config, load_config
from .core import compute_pseudochannel, compute_pseudochannel_chunked
from .io import load_channel_folder, load_ome_tiff, OMETiffChannels, validate_channels


def process_single_folder(
    input_folder: Union[str, Path],
    weights: dict[str, float],
    output_path: Union[str, Path],
    normalization: str = "minmax",
    use_chunked: bool = True,
    chunk_size: int = 1024,
    output_dtype: str = "uint16",
) -> Path:
    """Process a single folder and save pseudochannel output.

    Args:
        input_folder: Path to folder containing channel TIFFs
        weights: Dict of channel_name -> weight
        output_path: Path for output TIFF file
        normalization: Normalization method
        use_chunked: Use chunked processing for memory efficiency
        chunk_size: Chunk size for chunked processing
        output_dtype: Output data type ("uint16" or "float32")

    Returns:
        Path to saved output file
    """
    channels = load_channel_folder(input_folder, use_memmap=True)
    validate_channels(channels)

    if use_chunked:
        result = compute_pseudochannel_chunked(
            channels,
            weights,
            normalize=normalization,
            chunk_size=chunk_size,
        )
    else:
        result = compute_pseudochannel(
            channels,
            weights,
            normalize=normalization,
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_dtype == "uint16":
        output_data = (result * 65535).astype(np.uint16)
    else:
        output_data = result

    tifffile.imwrite(str(output_path), output_data)

    return output_path


def process_ome_tiff(
    tiff_path: Union[str, Path],
    marker_file: Union[str, Path],
    weights: dict[str, float],
    output_path: Union[str, Path],
    marker_column: Optional[Union[int, str]] = None,
    normalization: str = "minmax",
    use_chunked: bool = True,
    chunk_size: int = 1024,
    output_dtype: str = "uint16",
) -> Path:
    """Process an OME-TIFF file and save pseudochannel output.

    Args:
        tiff_path: Path to OME-TIFF file
        marker_file: Path to file with marker names
        weights: Dict of channel_name -> weight
        output_path: Path for output TIFF file
        marker_column: Column for CSV/TSV marker files
        normalization: Normalization method
        use_chunked: Use chunked processing for memory efficiency
        chunk_size: Chunk size for chunked processing
        output_dtype: Output data type ("uint16" or "float32")

    Returns:
        Path to saved output file
    """
    channels = load_ome_tiff(tiff_path, marker_file, marker_column, use_memmap=True)
    validate_channels(channels)

    if use_chunked:
        result = compute_pseudochannel_chunked(
            channels,
            weights,
            normalize=normalization,
            chunk_size=chunk_size,
        )
    else:
        result = compute_pseudochannel(
            channels,
            weights,
            normalize=normalization,
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_dtype == "uint16":
        output_data = (result * 65535).astype(np.uint16)
    else:
        output_data = result

    tifffile.imwrite(str(output_path), output_data)

    return output_path


def process_ome_tiff_batch(
    tiff_files: list[Union[str, Path]],
    marker_files: Union[str, Path, list[Union[str, Path]]],
    config_path: Union[str, Path],
    output_folder: Union[str, Path],
    marker_column: Optional[Union[int, str]] = None,
    output_suffix: str = "_pseudochannel",
    use_chunked: bool = True,
    chunk_size: int = 1024,
    output_dtype: str = "uint16",
    progress: bool = True,
) -> list[Path]:
    """Apply config to multiple OME-TIFF files.

    Args:
        tiff_files: List of paths to OME-TIFF files
        marker_files: Either a single marker file (used for all TIFFs)
            or a list of marker files (one per TIFF)
        config_path: Path to YAML config file
        output_folder: Directory for output files
        marker_column: Column for CSV/TSV marker files
        output_suffix: Suffix to add to output filenames
        use_chunked: Use chunked processing for memory efficiency
        chunk_size: Chunk size for chunked processing
        output_dtype: Output data type ("uint16" or "float32")
        progress: Show progress bar

    Returns:
        List of paths to saved output files
    """
    config = load_config(config_path)
    weights = get_weights_from_config(config)
    normalization = get_normalization_from_config(config)

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    tiff_list = list(tiff_files)

    if isinstance(marker_files, (str, Path)):
        marker_list = [Path(marker_files)] * len(tiff_list)
    else:
        marker_list = [Path(m) for m in marker_files]
        if len(marker_list) != len(tiff_list):
            raise ValueError(
                f"Number of marker files ({len(marker_list)}) must match "
                f"number of TIFF files ({len(tiff_list)})"
            )

    outputs = []

    if progress and tqdm is not None:
        iterator = tqdm(zip(tiff_list, marker_list), total=len(tiff_list), desc="Processing")
    else:
        iterator = zip(tiff_list, marker_list)

    for tiff_path, marker_file in iterator:
        tiff_path = Path(tiff_path)
        tiff_name = tiff_path.stem

        output_name = f"{tiff_name}{output_suffix}.tif"
        output_path = output_folder / output_name

        try:
            result_path = process_ome_tiff(
                tiff_path,
                marker_file,
                weights,
                output_path,
                marker_column=marker_column,
                normalization=normalization,
                use_chunked=use_chunked,
                chunk_size=chunk_size,
                output_dtype=output_dtype,
            )
            outputs.append(result_path)

        except Exception as e:
            print(f"Error processing {tiff_path}: {e}")
            continue

    return outputs


def process_dataset(
    input_folders: list[Union[str, Path]],
    config_path: Union[str, Path],
    output_folder: Union[str, Path],
    output_suffix: str = "_pseudochannel",
    use_chunked: bool = True,
    chunk_size: int = 1024,
    output_dtype: str = "uint16",
    progress: bool = True,
) -> list[Path]:
    """Apply config to multiple image folders.

    Args:
        input_folders: List of paths to folders containing channel TIFFs
        config_path: Path to YAML config file
        output_folder: Directory for output files
        output_suffix: Suffix to add to output filenames
        use_chunked: Use chunked processing for memory efficiency
        chunk_size: Chunk size for chunked processing
        output_dtype: Output data type ("uint16" or "float32")
        progress: Show progress bar

    Returns:
        List of paths to saved output files
    """
    config = load_config(config_path)
    weights = get_weights_from_config(config)
    normalization = get_normalization_from_config(config)

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    outputs = []
    folders = list(input_folders)

    if progress and tqdm is not None:
        iterator = tqdm(folders, desc="Processing")
    else:
        iterator = folders

    for input_folder in iterator:
        input_folder = Path(input_folder)
        folder_name = input_folder.name

        output_name = f"{folder_name}{output_suffix}.tif"
        output_path = output_folder / output_name

        try:
            result_path = process_single_folder(
                input_folder,
                weights,
                output_path,
                normalization=normalization,
                use_chunked=use_chunked,
                chunk_size=chunk_size,
                output_dtype=output_dtype,
            )
            outputs.append(result_path)

        except Exception as e:
            print(f"Error processing {input_folder}: {e}")
            continue

    return outputs


def find_channel_folders(
    root_path: Union[str, Path],
    pattern: str = "*.tif",
    min_files: int = 2,
) -> list[Path]:
    """Find folders containing channel TIFFs.

    Args:
        root_path: Root directory to search
        pattern: Glob pattern for TIFF files
        min_files: Minimum number of TIFF files to qualify

    Returns:
        List of folder paths containing channel TIFFs
    """
    root_path = Path(root_path)
    folders = []

    for folder in root_path.iterdir():
        if not folder.is_dir():
            continue

        tiff_files = list(folder.glob(pattern))

        if len(tiff_files) >= min_files:
            folders.append(folder)

    return sorted(folders)


def batch_process_directory(
    root_path: Union[str, Path],
    config_path: Union[str, Path],
    output_folder: Optional[Union[str, Path]] = None,
    **kwargs,
) -> list[Path]:
    """Convenience function to process all channel folders in a directory.

    Args:
        root_path: Root directory containing channel folders
        config_path: Path to YAML config file
        output_folder: Directory for outputs (default: root_path/pseudochannels)
        **kwargs: Additional arguments passed to process_dataset

    Returns:
        List of paths to saved output files
    """
    root_path = Path(root_path)

    if output_folder is None:
        output_folder = root_path / "pseudochannels"

    folders = find_channel_folders(root_path)

    if not folders:
        print(f"No channel folders found in {root_path}")
        return []

    print(f"Found {len(folders)} channel folders to process")

    return process_dataset(
        folders,
        config_path,
        output_folder,
        **kwargs,
    )


def find_mcmicro_experiments(
    root_path: Union[str, Path],
    background_folder: str = "background",
    marker_filename: str = "markers.csv",
) -> list[dict]:
    """Find MCMICRO experiment folders with background images.

    Looks for experiment folders containing a background subfolder
    with an OME-TIFF image and markers.csv file.

    Args:
        root_path: Root directory containing experiment folders
        background_folder: Name of subfolder containing the image (default: "background")
        marker_filename: Name of marker file (default: "markers.csv")

    Returns:
        List of dicts with keys:
        - experiment_path: Path to experiment folder
        - image_path: Path to OME-TIFF image
        - marker_path: Path to markers.csv
    """
    root_path = Path(root_path)
    experiments = []

    for experiment_dir in sorted(root_path.iterdir()):
        if not experiment_dir.is_dir():
            continue

        bg_dir = experiment_dir / background_folder
        if not bg_dir.is_dir():
            continue

        # Find marker file
        marker_path = bg_dir / marker_filename
        if not marker_path.is_file():
            continue

        # Find OME-TIFF image (look for common patterns)
        image_path = None
        for pattern in ["*.ome.tiff", "*.ome.tif", "*.tiff", "*.tif"]:
            matches = list(bg_dir.glob(pattern))
            # Exclude marker files and other CSVs
            matches = [m for m in matches if m.suffix.lower() in (".tif", ".tiff")]
            if matches:
                image_path = matches[0]
                break

        if image_path is None:
            continue

        experiments.append({
            "experiment_path": experiment_dir,
            "image_path": image_path,
            "marker_path": marker_path,
        })

    return experiments


def process_mcmicro_experiment(
    experiment_info: dict,
    weights: dict[str, float],
    normalization: str = "minmax",
    output_folder: str = "pseudochannel",
    output_filename: str = "pseudochannel.tif",
    mcmicro_markers: bool = True,
    use_chunked: bool = True,
    chunk_size: int = 1024,
    output_dtype: str = "uint16",
) -> Path:
    """Process a single MCMICRO experiment.

    Args:
        experiment_info: Dict from find_mcmicro_experiments()
        weights: Dict of channel_name -> weight
        normalization: Normalization method
        output_folder: Subfolder name for output (default: "pseudochannel")
        output_filename: Output filename (default: "pseudochannel.tif")
        mcmicro_markers: Use MCMICRO marker format (default: True)
        use_chunked: Use chunked processing for memory efficiency
        chunk_size: Chunk size for chunked processing
        output_dtype: Output data type ("uint16" or "float32")

    Returns:
        Path to saved output file
    """
    experiment_path = experiment_info["experiment_path"]
    image_path = experiment_info["image_path"]
    marker_path = experiment_info["marker_path"]

    # Load channels
    channels = OMETiffChannels(
        image_path,
        marker_path,
        mcmicro_markers=mcmicro_markers,
    )
    validate_channels(channels)

    # Compute pseudochannel
    if use_chunked:
        result = compute_pseudochannel_chunked(
            channels,
            weights,
            normalize=normalization,
            chunk_size=chunk_size,
        )
    else:
        result = compute_pseudochannel(
            channels,
            weights,
            normalize=normalization,
        )

    # Save output
    output_dir = experiment_path / output_folder
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_filename

    if output_dtype == "uint16":
        output_data = (result * 65535).astype(np.uint16)
    else:
        output_data = result

    tifffile.imwrite(str(output_path), output_data)

    # Close the OMETiffChannels to release file handle
    channels.close()

    return output_path


def process_mcmicro_batch(
    root_path: Union[str, Path],
    config_path: Union[str, Path],
    background_folder: str = "background",
    marker_filename: str = "markers.csv",
    output_folder: str = "pseudochannel",
    output_filename: str = "pseudochannel.tif",
    mcmicro_markers: bool = True,
    use_chunked: bool = True,
    chunk_size: int = 1024,
    output_dtype: str = "uint16",
    progress: bool = True,
) -> list[Path]:
    """Batch process MCMICRO experiment folders.

    Finds all experiment folders with a background subfolder containing
    an OME-TIFF and markers.csv, processes each, and saves output to
    a pseudochannel subfolder within each experiment.

    Expected folder structure:
        root_path/
        ├── experiment1/
        │   ├── background/
        │   │   ├── image.ome.tiff
        │   │   └── markers.csv
        │   └── pseudochannel/  <- output created here
        ├── experiment2/
        │   └── ...

    Args:
        root_path: Root directory containing experiment folders
        config_path: Path to YAML config file with weights
        background_folder: Name of subfolder containing images (default: "background")
        marker_filename: Name of marker file (default: "markers.csv")
        output_folder: Subfolder name for outputs (default: "pseudochannel")
        output_filename: Output filename (default: "pseudochannel.tif")
        mcmicro_markers: Use MCMICRO marker format (default: True)
        use_chunked: Use chunked processing for memory efficiency
        chunk_size: Chunk size for chunked processing
        output_dtype: Output data type ("uint16" or "float32")
        progress: Show progress bar

    Returns:
        List of paths to saved output files
    """
    config = load_config(config_path)
    weights = get_weights_from_config(config)
    normalization = get_normalization_from_config(config)

    experiments = find_mcmicro_experiments(
        root_path,
        background_folder=background_folder,
        marker_filename=marker_filename,
    )

    if not experiments:
        print(f"No MCMICRO experiments found in {root_path}")
        print(f"  (looking for '{background_folder}/' with '{marker_filename}' and an image)")
        return []

    print(f"Found {len(experiments)} MCMICRO experiments to process")

    outputs = []

    if progress and tqdm is not None:
        iterator = tqdm(experiments, desc="Processing experiments")
    else:
        iterator = experiments

    for exp_info in iterator:
        try:
            result_path = process_mcmicro_experiment(
                exp_info,
                weights,
                normalization=normalization,
                output_folder=output_folder,
                output_filename=output_filename,
                mcmicro_markers=mcmicro_markers,
                use_chunked=use_chunked,
                chunk_size=chunk_size,
                output_dtype=output_dtype,
            )
            outputs.append(result_path)

        except Exception as e:
            exp_name = exp_info["experiment_path"].name
            print(f"Error processing {exp_name}: {e}")
            continue

    return outputs
