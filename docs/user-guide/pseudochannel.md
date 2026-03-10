# Pseudochannels

## Concept

Most cell segmentation tools expect two channels: nuclei (DAPI) and cell boundaries (membrane/cytoplasm). With 30-50+ markers from multiplex imaging, no single marker captures all cell boundaries.

A **pseudochannel** is a weighted combination of multiple marker channels into a single composite image. By blending membrane markers like CD45, E-cadherin, and pan-CK with different weights, you get a composite that outlines more cells than any individual marker.

## Input Formats

### Folder of TIFFs

One TIFF per channel. Marker names are extracted from filenames using a regex pattern.

**MACSima convention** (default): `_A-<marker>` at the end of the filename:
```
C-001_S-000_S_APC_R-01_W-A-1_ROI-08_A-CD45_C-2B11.tif  →  "CD45"
```

**Custom pattern**: pass a regex with one capture group:
```python
channels = load_channel_folder("/path/to/data/", marker_pattern=r"^([^_]+)_")
```

### OME-TIFF

Single multi-channel file with a separate marker list (`.txt` or `.csv`):

```python
channels = load_ome_tiff("/path/to/image.ome.tiff", "/path/to/markers.txt")
```

For large or compressed files, use `OMETiffChannels` for lazy loading:

```python
with OMETiffChannels(path, markers) as channels:
    cd45 = channels["CD45"]  # Only this channel is loaded
```

### MCMICRO Format

Use `mcmicro_markers=True` to read MCMICRO-style `markers.csv` files. Channels with `remove=TRUE` are filtered out automatically:

```python
channels = load_ome_tiff(path, markers_csv, mcmicro_markers=True)
```

## Normalization

Before combining channels, each is normalized to [0, 1]. Two modes:

- **minmax** — linear scaling from min to max. Simple but sensitive to hot pixels.
- **percentile** — clips to the 1st–99th percentile range, then scales. Handles outliers better.

!!! tip
    Use `"percentile"` normalization (the default) for most data. It handles hot pixels and background noise better than `"minmax"`.

## Excluded Channels

By default, common non-membrane channels are excluded from weight sliders:

- DAPI, autofluorescence (AF), PE, FE, and other non-markers

Override with `exclude_channels=[]` if you need them. The full default list is in `DEFAULT_EXCLUDED_CHANNELS`.

## Computing a Pseudochannel

```python
from pseudochannel import compute_pseudochannel

result = compute_pseudochannel(
    channels,           # dict[str, ndarray] — marker name → 2D image
    weights,            # dict[str, float] — marker name → weight
    normalization="percentile",
)
```

The result is a 2D float32 array normalized to [0, 1].

For very large images that don't fit in memory, use `compute_pseudochannel_chunked` which processes the image in a two-pass streaming fashion.
