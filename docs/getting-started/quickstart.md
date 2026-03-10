# Quick Start

A condensed walkthrough: load data, tune weights, save config, batch process.

## 1. Load Your Data

Open `notebooks/pseudochannel_explorer.ipynb` in JupyterLab and set the path to your data:

=== "Folder of TIFFs"

    ```python
    CHANNEL_FOLDER = "/path/to/your/roi/"
    ```

=== "MACSima Format"

    ```python
    explorer = create_interactive_explorer(
        "/path/to/roi/",
        macsima_mode=True  # Auto-detects DAPI, uses MACSima naming
    )
    ```

=== "OME-TIFF"

    ```python
    OME_TIFF_PATH = "/path/to/image.ome.tiff"
    MARKER_FILE = "/path/to/markers.txt"
    ```

=== "MCMICRO Format"

    ```python
    channels = load_ome_tiff(
        "/path/to/image.ome.tiff",
        "/path/to/markers.csv",
        mcmicro_markers=True,  # Filters out remove=TRUE rows
    )
    ```

## 2. Tune Weights

The widget shows sliders for each channel. Drag them to adjust weights and watch the preview update in real time. The preview is downsampled for speed.

- Toggle DAPI overlay (blue) to check alignment with nuclei
- Draw a rectangle to zoom in at full resolution

## 3. Preview Segmentation (Optional)

In the zoom view:

1. Click **Segment** to run Cellpose on the zoomed region
2. Mask contours appear in green
3. Adjust **Diameter**, **Flow thr**, and **Prob thr** sliders
4. Export config: `cellpose_config = explorer.get_cellpose_config()`

## 4. Save Config

```python
from pseudochannel import save_config

save_config(
    weights=explorer.get_weights(),
    output_path="configs/membrane_weights.yaml",
    name="membrane",
    description="CD45 + CD3 + pan-CK blend for immune/epithelial boundaries"
)
```

## 5. Batch Process

Apply the same weights to all ROIs:

```python
from pseudochannel import process_mcmicro_batch

output_paths = process_mcmicro_batch(
    root_path="/data/CRC/",
    config_path="configs/membrane_weights.yaml",
    mcmicro_markers=True,
)
```

Then run batch segmentation:

```python
from pseudochannel import segment_mcmicro_batch

seg_outputs = segment_mcmicro_batch(
    root_path="/data/CRC/",
    config=explorer.get_cellpose_config(),
    mcmicro_markers=True,
)
```

Or do both in one call:

```python
from pseudochannel import process_and_segment_mcmicro_batch

pseudo_paths, seg_paths = process_and_segment_mcmicro_batch(
    root_path="/data/CRC/",
    config_path="configs/membrane_weights.yaml",
    mcmicro_markers=True,
)
```

## Next Steps

- [Interactive Explorer](../user-guide/explorer.md) — detailed widget guide
- [Batch Processing](../user-guide/batch.md) — MCMICRO batch workflows
- [Configuration Reference](../user-guide/config.md) — YAML format and CellposeConfig options
- [Tiled Segmentation](../user-guide/tiling.md) — for images too large to segment in one pass
