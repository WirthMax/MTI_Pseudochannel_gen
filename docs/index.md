# Tissue Pseudochannel Generator

Combine multiple multiplex tissue imaging channels into weighted **pseudochannels** that improve cell segmentation quality.

Originally written for **MACSima** data, but works with any multiplex imaging platform.

## The Problem

Most cell segmentation tools expect two channels: one for nuclei (DAPI) and one for cell boundaries. But multiplex imaging gives you 30-50+ markers. No single marker cleanly outlines all cells.

This tool lets you **blend multiple channels** with different weights into a single composite that captures all cell boundaries better than any individual marker alone.

## Features

- **Interactive weight tuning** with real-time preview in Jupyter
- **Cellpose segmentation preview** — test your pseudochannel directly in the zoom view
- **Batch processing** — apply weights to entire MCMICRO experiment folders
- **Batch segmentation** — run Cellpose on all experiments with tuned parameters
- **Auto DAPI detection** for both OME-TIFF and MACSima folder formats
- **GPU acceleration** — auto-detects CUDA when available
- **Tiled segmentation** — split/merge workflow for images too large to segment in one pass
- **Downstream analysis** — feature extraction, clustering, visualization, QuPath export

## Quick Start

```bash
conda env create -f environment.yaml
conda activate Pseudochannel_gen
jupyter lab
```

Open `notebooks/pseudochannel_explorer.ipynb` and point it at your data. See the [Installation](getting-started/installation.md) and [Quick Start](getting-started/quickstart.md) guides for details.

## Workflow Overview

1. **Load** your multi-channel image (folder of TIFFs, OME-TIFF, or MCMICRO format)
2. **Tune** channel weights interactively with the Jupyter widget
3. **Preview** Cellpose segmentation on a zoomed region (optional)
4. **Save** your weights to a YAML config
5. **Batch process** all ROIs/experiments with the saved config
6. **Analyze** — extract features, cluster cells, export to QuPath

## Packages

| Package | Purpose |
|---------|---------|
| [`pseudochannel`](api/pseudochannel/index.md) | Core: loading, weight tuning, pseudochannel computation, batch processing, segmentation |
| [`tiling`](api/tiling/index.md) | Split large images into overlapping tiles, merge segmented masks with cell deduplication |
| [`analysis`](api/analysis/index.md) | Feature extraction, clustering (PCA/Harmony/Louvain/UMAP), visualization, QuPath export |

## Project Structure

```
src/
├── pseudochannel/        # Core package
│   ├── core.py           # Pseudochannel computation
│   ├── io.py             # TIFF/OME-TIFF loading
│   ├── widgets.py        # Interactive Jupyter widget
│   ├── segmentation.py   # Cellpose wrapper (optional)
│   ├── batch.py          # Batch processing & segmentation
│   ├── config.py         # Config save/load
│   └── preview.py        # Image downsampling for previews
├── tiling/               # Large image tiling
│   ├── split.py          # Split images into tiles
│   └── merge.py          # Merge segmented tile masks
└── analysis/             # Downstream analysis
    ├── features.py       # Per-cell feature extraction
    ├── preprocessing.py  # Scaling, filtering, normalization
    ├── clustering.py     # PCA, Harmony, Louvain, UMAP
    ├── visualization.py  # Heatmaps, UMAP plots
    └── export.py         # QuPath GeoJSON export
```
