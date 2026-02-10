# Pseudochannel Generator for Multiplex Tissue Imaging

Originally written for **MACSima** data, but works with any multiplex imaging platform.

Most cell segmentation tools expect two channels: one for nuclei (usually DAPI) and one for cell boundaries (membrane or cytoplasm). But multiplex imaging gives you 30, 40, sometimes 50+ markers. Which ones do you pick for segmentation? And what if no single marker cleanly outlines your cells?

This tool lets you combine multiple channels into a single "pseudochannel" that works better for segmentation than any individual marker alone.

## The Problem

Say you're segmenting immune cells in tumor tissue. You've got CD45, CD3, CD8, CD11b, CD68, and a dozen other markers. Some stain membranes, some stain cytoplasm, some are sparse. If you just pick one for your segmentation algorithm, you'll miss cells that don't express that marker.

The solution: blend multiple channels together with different weights until you get a composite that captures all cell boundaries. That's what this does.

## Features

- **Interactive weight tuning** with real-time preview
- **Cellpose segmentation preview** - test your pseudochannel directly in the zoom view
- **Batch segmentation** - run Cellpose on entire MCMICRO experiment folders
- **Auto DAPI detection** for both OME-TIFF and MACSima folder formats
- **GPU acceleration** - auto-detects CUDA when available
- **Batch processing** - apply weights to entire datasets

## Quick Start

```bash
conda env create -f environment.yaml
conda activate Pseudochannel_gen
jupyter lab
```

Open `notebooks/pseudochannel_explorer.ipynb` and point it at your data.

## Workflow

### 1. Load your data

The tool handles two common formats:

**Folder of TIFFs** - one file per channel
```python
CHANNEL_FOLDER = "/path/to/your/roi/"
```

By default, marker names are extracted using the MACSima naming convention (`_A-<marker>` at the end):
```
C-001_S-000_S_APC_R-01_W-A-1_ROI-08_A-CD45_C-2B11.tif  →  "CD45"
```

**MACSima mode** - For MACSima data, use `macsima_mode=True` to enable automatic DAPI detection. This finds all DAPI images and keeps only the one with the lowest cycle number (C-number) as the nuclear marker:
```python
explorer = create_interactive_explorer(
    "/path/to/roi/",
    macsima_mode=True  # Auto-detects DAPI, uses MACSima naming pattern
)
```

For other instruments, pass a custom regex with one capture group:
```python
channels = load_channel_folder("/path/to/data/", marker_pattern=r"^([^_]+)_")
```

**OME-TIFF** - single file with a separate marker list
```python
OME_TIFF_PATH = "/path/to/image.ome.tiff"
MARKER_FILE = "/path/to/markers.txt"
```

**MCMICRO format** - If your marker file is in MCMICRO format (with `marker_name` and `remove` columns), use the `mcmicro_markers=True` flag. This automatically filters out channels marked `remove=TRUE`:

```python
from pseudochannel import load_ome_tiff, OMETiffChannels, create_interactive_explorer

# With load_ome_tiff
channels = load_ome_tiff(
    "/path/to/image.ome.tiff",
    "/path/to/markers.csv",
    mcmicro_markers=True,  # Filters out remove=TRUE rows
)
```

**Large or compressed OME-TIFF files** - Use `OMETiffChannels` instead of `load_ome_tiff()`. It opens instantly (only reads metadata) and loads individual channels on-demand, which is much faster and uses less memory:

```python
# This loads ALL channels into memory at once (slow for large files)
channels = load_ome_tiff(path, markers)

# This opens instantly and loads channels only when accessed (fast)
with OMETiffChannels(path, markers) as channels:
    cd45 = channels["CD45"]  # Only loads this channel
```

### 2. Tune weights interactively

The notebook launches a widget with sliders for each channel. Drag them around and watch the preview update. The preview is downsampled so it's fast even with large images.

You can also overlay DAPI in blue to see how your membrane pseudochannel aligns with nuclei - useful for checking that cell boundaries make sense.

Draw a rectangle on the preview to zoom in at full resolution and check the details.

### 2b. Preview segmentation (optional)

Once you have a zoom region selected, you can preview Cellpose segmentation directly in the widget:

1. Click **Segment** to run Cellpose on the zoomed region
2. Mask contours appear overlaid on the image in green
3. Toggle **Show Masks** to hide/show the contours
4. Adjust parameters with the sliders:
   - **Diameter**: Cell size in pixels (0 = auto-estimate)
   - **Flow thr**: Flow error threshold (lower = stricter)
   - **Prob thr**: Cell probability threshold (higher = fewer cells)

Cellpose uses your current pseudochannel weights and the nuclear marker (if available) for two-channel segmentation.

**Export config for batch processing**: Once you've tuned the segmentation parameters, export them for use with `segment_mcmicro_batch()`:

```python
cellpose_config = explorer.get_cellpose_config()
```

**Note**: Cellpose is an optional dependency. Install it with:
```bash
pip install cellpose

# For GPU support (much faster):
pip install cellpose torch --extra-index-url https://download.pytorch.org/whl/cu118
```

GPU is auto-detected when available.

### 3. Save your config

Once you're happy with the weights, save them to a YAML file:

```python
save_config(
    weights=explorer.get_weights(),
    output_path="configs/membrane_weights.yaml",
    name="membrane",
    description="CD45 + CD3 + pan-CK blend for immune/epithelial boundaries"
)
```

### 4. Batch process

Apply the same weights to all your ROIs:

```python
batch_process_directory(
    root_path="/data/experiment/",
    config_path="configs/membrane_weights.yaml",
    output_folder="/data/experiment/pseudochannels/"
)
```

#### MCMICRO Batch Processing

For MCMICRO-style folder structures, use `process_mcmicro_batch()`. It recursively finds all experiments with a `background/` folder containing an OME-TIFF and a sibling `markers.csv`:

```python
from pseudochannel import find_mcmicro_experiments, process_mcmicro_batch

# Preview what will be processed
experiments = find_mcmicro_experiments("/data/CRC/")
print(f"Found {len(experiments)} experiments")

# Process all experiments
output_paths = process_mcmicro_batch(
    root_path="/data/CRC/",
    config_path="configs/membrane_weights.yaml",
    mcmicro_markers=True,  # Uses marker_name column, filters remove=TRUE
)
```

Output structure:
```
experiment/
├── markers.csv
├── background/
│   └── image.ome.tiff
└── pseudochannel/           <- Created
    └── pseudochannel.tif
```

### 5. Batch segmentation

After generating pseudochannels, run Cellpose segmentation on all experiments:

```python
from pseudochannel import segment_mcmicro_batch

seg_outputs = segment_mcmicro_batch(
    root_path="/data/CRC/",
    config=explorer.get_cellpose_config(),  # Use tuned parameters from widget
    mcmicro_markers=True,
)
print(f"Segmented {len(seg_outputs)} experiments")
```

Or do both pseudochannel generation and segmentation in one call:

```python
from pseudochannel import process_and_segment_mcmicro_batch

pseudo_paths, seg_paths = process_and_segment_mcmicro_batch(
    root_path="/data/CRC/",
    config_path="configs/membrane_weights.yaml",
    mcmicro_markers=True,
)
```

**Config options for segmentation:**

| Source | Usage |
|--------|-------|
| Widget explorer | `config=explorer.get_cellpose_config()` |
| YAML file | `config="path/to/config.yaml"` (extracts `cellpose` section) |
| Direct | `config=CellposeConfig(diameter=30, flow_threshold=0.4)` |
| Defaults | `config=None` (auto GPU, cyto3 model) |

#### CellposeConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_type` | str | `"cyto3"` | Cellpose model to use. Options: `"cyto3"` (latest cytoplasm), `"cyto2"`, `"cyto"`, `"nuclei"` |
| `diameter` | float \| None | `None` | Expected cell diameter in pixels. `None` = auto-estimate from image |
| `flow_threshold` | float | `0.4` | Flow error threshold. Lower = stricter matching, fewer fragmented cells (range: 0-1) |
| `cellprob_threshold` | float | `0.0` | Cell probability threshold. Higher = fewer cells, only high-confidence detections (range: -6 to 6) |
| `gpu` | bool \| None | `None` | Use GPU acceleration. `None` = auto-detect CUDA availability |
| `min_size` | int | `15` | Minimum cell size in pixels. Cells smaller than this are removed |

**Example: Create config programmatically**
```python
from pseudochannel import CellposeConfig

config = CellposeConfig(
    model_type="cyto3",
    diameter=35,              # Set if you know your cell size
    flow_threshold=0.4,       # Lower for cleaner boundaries
    cellprob_threshold=0.0,   # Raise to reduce false positives
    min_size=15,
)
```

**Example: YAML config with cellpose section**

You can add a `cellpose` section to your weights YAML file. When you pass this file to `segment_mcmicro_batch()`, the cellpose parameters are extracted automatically:

```yaml
# configs/membrane_weights.yaml
name: membrane
description: CD45 + CD3 blend for immune cell boundaries

weights:
  CD45: 0.3
  CD3: 0.2
  CD8: 0.15
  pan-CK: 0.25

cellpose:
  model_type: cyto3
  diameter: 35
  flow_threshold: 0.4
  cellprob_threshold: 0.0
  min_size: 15
```

```python
# Both pseudochannel weights AND cellpose config come from the same file
seg_outputs = segment_mcmicro_batch(
    root_path="/data/CRC/",
    config="configs/membrane_weights.yaml",
)
```

Output structure after segmentation:
```
experiment/
├── markers.csv
├── background/
│   └── image.ome.tiff
├── pseudochannel/
│   └── pseudochannel.tif
└── segmentation/            <- Created
    ├── seg_mask.tif        <- uint32 label mask
    └── seg_flows.pkl       <- Cellpose flows (pickle format)
```

**Skip-existing behavior**: Both `process_mcmicro_batch()` and `segment_mcmicro_batch()` skip already processed experiments by default. Use `overwrite=True` to recompute.

## What gets excluded by default

DAPI, autofluorescence channels, empty channels, and a few other common non-markers are excluded from the weight sliders by default. You probably don't want these in your membrane composite anyway. Override with `exclude_channels=[]` if you need them.

## Project structure

```
├── src/pseudochannel/      # Main package
│   ├── core.py             # Pseudochannel computation
│   ├── io.py               # TIFF/OME-TIFF loading
│   ├── widgets.py          # Interactive Jupyter widget
│   ├── segmentation.py     # Cellpose wrapper (optional)
│   ├── batch.py            # Batch processing & segmentation
│   ├── config.py           # Config save/load
│   └── preview.py          # Image downsampling for previews
├── notebooks/
│   └── pseudochannel_explorer.ipynb
├── configs/                # Your saved weight configs
└── outputs/                # Generated pseudochannels
```

## API Reference

### Core functions

```python
from pseudochannel import (
    # I/O
    load_channel_folder,      # Load folder of individual TIFFs
    load_ome_tiff,            # Load OME-TIFF with marker file
    OMETiffChannels,          # Lazy-loading OME-TIFF wrapper

    # Processing
    compute_pseudochannel,    # Compute weighted pseudochannel

    # Config
    save_config,              # Save weights to YAML
    load_config,              # Load weights from YAML

    # Batch processing
    process_mcmicro_batch,    # Generate pseudochannels for MCMICRO experiments
    segment_mcmicro_batch,    # Segment MCMICRO experiments with Cellpose
    process_and_segment_mcmicro_batch,  # Both in one call

    # Segmentation
    CellposeConfig,           # Cellpose parameters dataclass
    SegmentationResult,       # Full segmentation output (masks, flows, etc.)
    run_segmentation,         # Run Cellpose, return masks only
    run_segmentation_full,    # Run Cellpose, return full results
)
```

## Tips

- Start with channels you know stain cell membranes or cytoplasm
- Keep weights low (0.1-0.3) and add more channels rather than cranking one up
- The "percentile" normalization handles hot pixels better than "minmax"
- For batch processing, test on one ROI first to make sure the weights transfer well

## Requirements

- Python 3.10+
- numpy, tifffile, matplotlib, ipywidgets, ipympl
- JupyterLab (for the interactive widget)

**Optional** (for segmentation preview):
- cellpose (+ torch for GPU support)
- scikit-image (for contour extraction, falls back to numpy if missing)

See `environment.yaml` for the full list.
