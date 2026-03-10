# Tiled Segmentation

For images too large to segment in one pass (GPU/system memory limits), use the tiling workflow: split into overlapping tiles, segment each tile, merge results.

## When to Use Tiling

- Images larger than ~10,000 x 10,000 pixels
- GPU or system memory limits
- Segmentation tools crash on large images

## Tile Size Recommendations

| Image Size | Tile Size | Overlap |
|------------|-----------|---------|
| < 10k px | No tiling needed | - |
| 10k–20k px | 2048 px | 200 px |
| 20k–50k px | 4096 px | 300 px |
| > 50k px | 4096–8192 px | 400 px |

The overlap should be at least **2x the expected cell diameter** to ensure cells at tile boundaries are detected and deduplicated.

## Workflow

Open `notebooks/tiled_segmentation.ipynb` or use the Python API:

### 1. Split

```python
from tiling import split_image, save_tile_info

tiles, tile_infos = split_image(
    image, tile_size=2048, overlap=200, output_dir="tiles/"
)
save_tile_info(tile_infos, "tiles/tile_info.json", image_shape=image.shape)
```

### 2. Segment

Segment each tile using Cellpose (GUI or Python) or any other tool. Each tile produces a label mask saved to the same directory.

### 3. Merge

```python
from tiling import load_tile_info, load_tile_masks, merge_tile_masks

tile_infos, metadata = load_tile_info("tiles/tile_info.json")
tile_masks = load_tile_masks("tiles/", tile_infos)
merged = merge_tile_masks(tile_masks, tile_infos, metadata['image_shape'])
```

## How Merge Works

1. **Full-tile placement**: each tile places its entire footprint onto the output canvas
2. **Core overwrite**: the non-overlapping center of each tile overwrites directly
3. **Overlap matching**: cells in overlap regions are matched between adjacent tiles using sparse IoU (only comparing cells that share pixels)
4. **Centroid fallback**: unmatched cells in overlaps are kept if their centroid falls within the overlap region
5. **Vectorized relabeling**: a lookup table maps all per-tile labels to globally unique labels in one pass

This ensures no gaps, no duplicate cells, and efficient processing even for very large images.
