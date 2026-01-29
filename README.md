# Pseudochannel Generator for Multiplex Tissue Imaging

Originally written for **MACSima** data, but works with any multiplex imaging platform.

Most cell segmentation tools expect two channels: one for nuclei (usually DAPI) and one for cell boundaries (membrane or cytoplasm). But multiplex imaging gives you 30, 40, sometimes 50+ markers. Which ones do you pick for segmentation? And what if no single marker cleanly outlines your cells?

This tool lets you combine multiple channels into a single "pseudochannel" that works better for segmentation than any individual marker alone.

## The Problem

Say you're segmenting immune cells in tumor tissue. You've got CD45, CD3, CD8, CD11b, CD68, and a dozen other markers. Some stain membranes, some stain cytoplasm, some are sparse. If you just pick one for your segmentation algorithm, you'll miss cells that don't express that marker.

The solution: blend multiple channels together with different weights until you get a composite that captures all cell boundaries. That's what this does.

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
C-001_S-000_S_APC_R-01_W-A-1_ROI-08_A-CD45.tif  →  "CD45"
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

## What gets excluded by default

DAPI, autofluorescence channels, empty channels, and a few other common non-markers are excluded from the weight sliders by default. You probably don't want these in your membrane composite anyway. Override with `exclude_channels=[]` if you need them.

## Project structure

```
├── src/pseudochannel/      # Main package
│   ├── core.py             # Pseudochannel computation
│   ├── io.py               # TIFF/OME-TIFF loading
│   ├── widgets.py          # Interactive Jupyter widget
│   ├── batch.py            # Batch processing
│   └── config.py           # Config save/load
├── notebooks/
│   └── pseudochannel_explorer.ipynb
├── configs/                # Your saved weight configs
└── outputs/                # Generated pseudochannels
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

See `environment.yaml` for the full list.
