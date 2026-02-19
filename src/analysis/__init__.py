"""Downstream analysis pipeline for segmented multiplex tissue imaging data.

This package provides tools for:
- Feature extraction from segmentation masks and multi-channel images
- Data preprocessing (scaling, filtering, normalization)
- Clustering and dimensionality reduction (PCA, Harmony, Louvain)
- Visualization (heatmaps, UMAP, composition plots)
- Export to QuPath GeoJSON format
"""

from .features import (
    extract_morphology,
    extract_marker_intensities,
    extract_features,
    extract_features_batch,
)

from .preprocessing import (
    METADATA_COLUMNS,
    identify_marker_columns,
    scale_features,
    filter_markers,
    filter_cells,
    remove_outliers,
    impute_missing,
)

from .clustering import (
    ClusteringConfig,
    to_anndata,
    run_pca,
    run_harmony,
    run_louvain,
    run_leiden,
    run_umap,
    run_clustering,
    clustering_pipeline,
    get_cluster_assignments,
    add_clusters_to_dataframe,
)

from .visualization import (
    plot_marker_heatmap,
    plot_dotplot,
    plot_cluster_composition,
    plot_marker_distributions,
    plot_umap,
    plot_cluster_markers,
    plot_pca_variance,
)

from .export import (
    CLUSTER_COLORS,
    hex_to_rgba,
    get_cluster_color,
    mask_to_contours,
    clean_and_close_contours,
    export_to_geojson,
    export_mcmicro_batch,
)

__version__ = "0.1.0"

__all__ = [
    # features
    "extract_morphology",
    "extract_marker_intensities",
    "extract_features",
    "extract_features_batch",
    # preprocessing
    "METADATA_COLUMNS",
    "identify_marker_columns",
    "scale_features",
    "filter_markers",
    "filter_cells",
    "remove_outliers",
    "impute_missing",
    # clustering
    "ClusteringConfig",
    "to_anndata",
    "run_pca",
    "run_harmony",
    "run_louvain",
    "run_leiden",
    "run_umap",
    "run_clustering",
    "clustering_pipeline",
    "get_cluster_assignments",
    "add_clusters_to_dataframe",
    # visualization
    "plot_marker_heatmap",
    "plot_dotplot",
    "plot_cluster_composition",
    "plot_marker_distributions",
    "plot_umap",
    "plot_cluster_markers",
    "plot_pca_variance",
    # export
    "CLUSTER_COLORS",
    "hex_to_rgba",
    "get_cluster_color",
    "mask_to_contours",
    "clean_and_close_contours",
    "export_to_geojson",
    "export_mcmicro_batch",
]
