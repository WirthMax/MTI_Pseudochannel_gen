# Installation

## Requirements

- Python 3.10+
- numpy, tifffile, matplotlib, ipywidgets, ipympl
- JupyterLab (for the interactive widget)

## Conda Setup (Recommended)

```bash
git clone https://github.com/WirthMax/MTI_Pseudochannel_gen.git
cd MTI_Pseudochannel_gen

conda env create -f environment.yaml
conda activate Pseudochannel_gen
```

This installs all core dependencies. See `environment.yaml` for the full list.

## Optional Dependencies

### Cellpose (segmentation)

Required for segmentation preview in the widget and batch segmentation:

```bash
pip install cellpose
```

### GPU Support

For GPU-accelerated segmentation (much faster):

```bash
pip install cellpose torch --extra-index-url https://download.pytorch.org/whl/cu118
```

GPU is auto-detected when available. No code changes needed.

### Downstream Analysis

For the analysis pipeline (clustering, visualization, QuPath export):

```bash
pip install scanpy harmonypy geopandas
```

### scikit-image

Used for contour extraction in the segmentation preview. Falls back to numpy if missing:

```bash
pip install scikit-image
```

## Verify Installation

```bash
conda activate Pseudochannel_gen
jupyter lab
```

Open `notebooks/pseudochannel_explorer.ipynb` — if the widget renders with sliders, you're set.
