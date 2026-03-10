# Interactive Explorer

The pseudochannel explorer is a Jupyter widget for real-time weight tuning with visual feedback.

## Launching the Widget

Open `notebooks/pseudochannel_explorer.ipynb` in JupyterLab:

```python
from pseudochannel import create_interactive_explorer

explorer = create_interactive_explorer("/path/to/data/")

# MACSima mode (auto DAPI detection)
explorer = create_interactive_explorer("/path/to/data/", macsima_mode=True)
```

## Weight Sliders

Each non-excluded channel gets a slider (0.0–1.0). Dragging a slider updates the preview immediately. The preview is downsampled for speed, even with large images.

**Tips:**

- Start with channels you know stain cell membranes or cytoplasm
- Keep weights low (0.1–0.3) and add more channels rather than cranking one up
- Use percentile normalization for data with hot pixels

## DAPI Overlay

Toggle the DAPI overlay to show nuclei in blue over your pseudochannel. Useful for checking that cell boundaries align with nuclei.

In MACSima mode, the tool auto-detects the best DAPI image (lowest cycle number).

## Zoom View

Draw a rectangle on the preview to zoom in at full resolution. This is where you check fine details before committing to weights.

## Segmentation Preview

Once you have a zoom region:

1. Click **Segment** to run Cellpose on the zoomed area
2. Green contours overlay the image showing detected cell boundaries
3. Toggle **Show Masks** to hide/show contours
4. Adjust parameters with sliders:

| Slider | Description | Range |
|--------|-------------|-------|
| **Diameter** | Expected cell size in pixels (0 = auto) | 0–200 |
| **Flow thr** | Flow error threshold (lower = stricter) | 0–1 |
| **Prob thr** | Cell probability threshold (higher = fewer cells) | -6–6 |

!!! note "Cellpose Required"
    Segmentation preview requires `cellpose` to be installed. See [Installation](../getting-started/installation.md#cellpose-segmentation).

## Exporting

### Weights

```python
weights = explorer.get_weights()  # dict[str, float]
```

### Cellpose Config

```python
cellpose_config = explorer.get_cellpose_config()  # CellposeConfig dataclass
```

Use these with [`save_config()`](../api/pseudochannel/config.md) and [`segment_mcmicro_batch()`](../api/pseudochannel/batch.md).
