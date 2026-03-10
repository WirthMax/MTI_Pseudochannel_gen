# Downstream Analysis

The `analysis` package provides a complete pipeline from segmentation masks to cell phenotyping and spatial export.

## Pipeline Overview

```
Segmentation masks + Multi-channel images
  → Feature extraction (morphology + marker intensities)
  → Preprocessing (scaling, filtering, outlier removal)
  → Clustering (PCA → Harmony → Louvain/Leiden → UMAP)
  → Visualization (heatmaps, UMAP plots, composition)
  → Export (QuPath GeoJSON)
```

Open `notebooks/downstream_analysis.ipynb` for an interactive walkthrough.

## Feature Extraction

Extract per-cell morphology and marker intensities from masks:

```python
from analysis import extract_features

features = extract_features(
    mask,        # 2D label mask (uint32)
    channels,    # dict[str, ndarray] — marker name → 2D image
)
```

Returns a DataFrame with one row per cell and columns for area, centroid, eccentricity, and mean/median intensity per marker.

For batch processing across multiple ROIs, use `extract_features_batch()`.

## Preprocessing

```python
from analysis import scale_features, filter_markers, filter_cells, remove_outliers

# Scale marker columns
df = scale_features(df, method="zscore")

# Filter low-quality markers or cells
df = filter_markers(df, min_expression=0.1)
df = filter_cells(df, min_area=50, max_area=5000)
df = remove_outliers(df, method="iqr")
```

## Clustering

Uses scanpy under the hood (PCA → optional Harmony batch correction → Louvain/Leiden → UMAP):

```python
from analysis import ClusteringConfig, to_anndata, run_clustering

config = ClusteringConfig(
    n_pcs=15,
    resolution=0.5,
    method="leiden",
    use_harmony=True,
    batch_key="roi",
)

adata = to_anndata(df)
adata = run_clustering(adata, config)
```

## Visualization

```python
from analysis import plot_umap, plot_heatmap, plot_composition

plot_umap(adata, color="leiden")
plot_heatmap(adata, groupby="leiden")
plot_composition(adata, groupby="leiden", splitby="roi")
```

## QuPath Export

Export cell boundaries and cluster labels to GeoJSON for visualization in QuPath:

```python
from analysis import export_geojson

export_geojson(
    mask,
    adata,
    output_path="cells.geojson",
    label_column="leiden",
)
```

!!! note "Optional Dependencies"
    The analysis pipeline requires `scanpy`, `harmonypy`, and `geopandas`. See [Installation](../getting-started/installation.md#downstream-analysis).
