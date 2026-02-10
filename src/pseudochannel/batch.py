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

    Recursively searches for 'background' folders containing an OME-TIFF
    image, with markers.csv in the parent folder. Supports nested structures:

        root/EXP_.../1/rack-.../
        ├── markers.csv          <- marker file here
        ├── background/          <- image here
        │   └── image.ome.tiff
        └── pseudochannel/       <- output goes here

    Args:
        root_path: Root directory to search
        background_folder: Name of folder containing the image (default: "background")
        marker_filename: Name of marker file (default: "markers.csv")

    Returns:
        List of dicts with keys:
        - experiment_path: Path to top-level experiment folder (for naming)
        - background_path: Path to background folder (for output placement)
        - image_path: Path to OME-TIFF image
        - marker_path: Path to markers.csv
    """
    root_path = Path(root_path)
    experiments = []

    # Recursively find all 'background' folders
    for bg_dir in root_path.rglob(background_folder):
        if not bg_dir.is_dir():
            continue

        # Find marker file in parent folder (sibling to background)
        marker_path = bg_dir.parent / marker_filename
        if not marker_path.is_file():
            continue

        # Find OME-TIFF image inside background folder
        image_path = None
        for pattern in ["*.ome.tiff", "*.ome.tif", "*.tiff", "*.tif"]:
            matches = list(bg_dir.glob(pattern))
            matches = [m for m in matches if m.suffix.lower() in (".tif", ".tiff")]
            if matches:
                image_path = matches[0]
                break

        if image_path is None:
            continue

        # Find top-level experiment folder (first child of root_path in the path)
        # e.g., root/EXP_.../1/rack-.../background -> EXP_...
        try:
            relative = bg_dir.relative_to(root_path)
            experiment_name = relative.parts[0]
            experiment_path = root_path / experiment_name
        except (ValueError, IndexError):
            experiment_path = bg_dir.parent

        experiments.append({
            "experiment_path": experiment_path,
            "background_path": bg_dir,
            "image_path": image_path,
            "marker_path": marker_path,
        })

    return sorted(experiments, key=lambda x: str(x["image_path"]))


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
    image_path = experiment_info["image_path"]
    marker_path = experiment_info["marker_path"]
    background_path = experiment_info["background_path"]

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

    # Save output as sibling to background folder
    # e.g., .../rack-.../background/ -> .../rack-.../pseudochannel/
    output_dir = background_path.parent / output_folder
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
    overwrite: bool = False,
) -> list[Path]:
    """Batch process MCMICRO experiment folders.

    Recursively finds all 'background' folders containing an OME-TIFF
    and markers.csv, processes each, and saves output to a sibling
    'pseudochannel' folder.

    Supports nested folder structures like:
        root_path/
        ├── EXP_.../
        │   └── 1/
        │       └── rack-.../
        │           ├── markers.csv
        │           ├── background/
        │           │   └── image.ome.tiff
        │           └── pseudochannel/  <- output created here
        ├── EXP_.../
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
        overwrite: If False (default), skip experiments that already have output.
            If True, recompute all experiments.

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
        print(f"  (looking for '{background_folder}/' folders with '{marker_filename}' and a TIFF image)")
        return []

    # Filter out already processed experiments unless overwrite=True
    if not overwrite:
        to_process = []
        skipped = 0
        for exp_info in experiments:
            output_path = exp_info["background_path"].parent / output_folder / output_filename
            if output_path.exists():
                skipped += 1
            else:
                to_process.append(exp_info)
        experiments = to_process
        if skipped > 0:
            print(f"Skipping {skipped} already processed experiments (use overwrite=True to recompute)")

    if not experiments:
        print("All experiments already processed.")
        return []

    print(f"Processing {len(experiments)} experiments:")
    for exp in experiments[:5]:
        print(f"  {exp['experiment_path'].name} -> {exp['background_path'].parent.name}/{background_folder}/")
    if len(experiments) > 5:
        print(f"  ... and {len(experiments) - 5} more")

    outputs = []

    if progress and tqdm is not None:
        iterator = tqdm(experiments, desc="Processing")
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
            img_name = exp_info["image_path"].name
            print(f"Error processing {exp_name}/{img_name}: {e}")
            continue

    return outputs
