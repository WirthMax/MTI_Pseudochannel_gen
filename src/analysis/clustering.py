"""Clustering and dimensionality reduction using AnnData and scanpy."""

from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
import pandas as pd

try:
    import anndata as ad
    import scanpy as sc
except ImportError:
    ad = None
    sc = None

try:
    import harmonypy
    HARMONY_AVAILABLE = True
except ImportError:
    harmonypy = None
    HARMONY_AVAILABLE = False

from .preprocessing import identify_marker_columns


@dataclass
class ClusteringConfig:
    """Configuration for the clustering pipeline.

    Attributes:
        n_pcs: Number of principal components for PCA.
        harmony_vars: Variables for Harmony batch correction (e.g., ['ROI']).
            Set to None or empty list to skip Harmony.
        n_neighbors: Number of neighbors for neighborhood graph.
        louvain_resolution: Resolution parameter for Louvain clustering.
            Higher values produce more clusters.
        use_harmony: Whether to use Harmony batch correction.
            Auto-detected from harmony_vars if not explicitly set.
        random_state: Random seed for reproducibility.
    """
    n_pcs: int = 30
    harmony_vars: list[str] = field(default_factory=lambda: ["ROI"])
    n_neighbors: int = 15
    louvain_resolution: float = 1.0
    use_harmony: Optional[bool] = None
    random_state: int = 42

    def __post_init__(self):
        if self.use_harmony is None:
            self.use_harmony = bool(self.harmony_vars)


def _check_scanpy():
    """Check if scanpy is available."""
    if sc is None:
        raise ImportError(
            "scanpy is required for clustering. "
            "Install with: pip install scanpy"
        )


def _check_anndata():
    """Check if anndata is available."""
    if ad is None:
        raise ImportError(
            "anndata is required for clustering. "
            "Install with: pip install anndata"
        )


def to_anndata(
    df: pd.DataFrame,
    marker_columns: Optional[list[str]] = None,
    obs_columns: Optional[list[str]] = None,
) -> "ad.AnnData":
    """Convert a features DataFrame to AnnData object.

    Args:
        df: DataFrame with cell features (rows=cells, columns=features).
        marker_columns: Columns to use as the main data matrix (X).
            If None, auto-detects marker columns.
        obs_columns: Columns to store as cell metadata (obs).
            If None, uses all non-marker columns.

    Returns:
        AnnData object with:
        - X: Marker intensity matrix
        - obs: Cell metadata (ROI, morphology, etc.)
        - var: Marker names as index
    """
    _check_anndata()

    if marker_columns is None:
        marker_columns = identify_marker_columns(df)

    if obs_columns is None:
        obs_columns = [c for c in df.columns if c not in marker_columns]

    # Extract data matrix
    X = df[marker_columns].values.astype(np.float32)

    # Extract obs metadata
    obs = df[obs_columns].copy() if obs_columns else pd.DataFrame(index=df.index)
    obs.index = obs.index.astype(str)

    # Create var (marker metadata)
    var = pd.DataFrame(index=marker_columns)

    adata = ad.AnnData(X=X, obs=obs, var=var)

    return adata


def run_pca(
    adata: "ad.AnnData",
    n_comps: int = 30,
    random_state: int = 42,
) -> "ad.AnnData":
    """Run PCA dimensionality reduction.

    Args:
        adata: AnnData object with expression matrix in X.
        n_comps: Number of principal components to compute.
        random_state: Random seed for reproducibility.

    Returns:
        AnnData with PCA results in adata.obsm['X_pca'].
        Modifies adata in place and returns it.
    """
    _check_scanpy()

    sc.tl.pca(adata, n_comps=n_comps, random_state=random_state)

    return adata


def run_harmony(
    adata: "ad.AnnData",
    batch_key: Union[str, list[str]] = "ROI",
    use_rep: str = "X_pca",
) -> "ad.AnnData":
    """Run Harmony batch correction.

    Args:
        adata: AnnData object with PCA results.
        batch_key: Column(s) in adata.obs to use as batch variable(s).
        use_rep: Representation to correct (default: 'X_pca').

    Returns:
        AnnData with corrected embedding in adata.obsm['X_pca_harmony'].
        Modifies adata in place and returns it.
    """
    _check_scanpy()

    if not HARMONY_AVAILABLE:
        raise ImportError(
            "harmonypy is required for batch correction. "
            "Install with: pip install harmonypy"
        )

    # Ensure batch_key is a list
    if isinstance(batch_key, str):
        batch_key = [batch_key]

    # Check that batch keys exist
    for key in batch_key:
        if key not in adata.obs.columns:
            raise ValueError(f"Batch key '{key}' not found in adata.obs")

    # Run Harmony via scanpy external
    sc.external.pp.harmony_integrate(
        adata,
        key=batch_key,
        basis=use_rep,
        adjusted_basis="X_pca_harmony",
    )

    return adata


def run_louvain(
    adata: "ad.AnnData",
    resolution: float = 1.0,
    n_neighbors: int = 15,
    use_rep: Optional[str] = None,
    random_state: int = 42,
) -> "ad.AnnData":
    """Run Louvain clustering.

    Args:
        adata: AnnData object with dimensionality reduction.
        resolution: Resolution parameter for Louvain (higher = more clusters).
        n_neighbors: Number of neighbors for neighborhood graph.
        use_rep: Representation to use. If None, auto-selects:
            X_pca_harmony if available, else X_pca.
        random_state: Random seed for reproducibility.

    Returns:
        AnnData with cluster assignments in adata.obs['louvain'].
        Modifies adata in place and returns it.
    """
    _check_scanpy()

    # Auto-select representation
    if use_rep is None:
        if "X_pca_harmony" in adata.obsm:
            use_rep = "X_pca_harmony"
        elif "X_pca" in adata.obsm:
            use_rep = "X_pca"
        else:
            raise ValueError("No PCA representation found. Run run_pca() first.")

    # Build neighbor graph
    sc.pp.neighbors(
        adata,
        n_neighbors=n_neighbors,
        use_rep=use_rep,
        random_state=random_state,
    )

    # Run Louvain clustering
    sc.tl.louvain(adata, resolution=resolution, random_state=random_state)

    return adata


def run_leiden(
    adata: "ad.AnnData",
    resolution: float = 1.0,
    n_neighbors: int = 15,
    use_rep: Optional[str] = None,
    random_state: int = 42,
) -> "ad.AnnData":
    """Run Leiden clustering (alternative to Louvain).

    Args:
        adata: AnnData object with dimensionality reduction.
        resolution: Resolution parameter for Leiden (higher = more clusters).
        n_neighbors: Number of neighbors for neighborhood graph.
        use_rep: Representation to use. If None, auto-selects.
        random_state: Random seed for reproducibility.

    Returns:
        AnnData with cluster assignments in adata.obs['leiden'].
        Modifies adata in place and returns it.
    """
    _check_scanpy()

    # Auto-select representation
    if use_rep is None:
        if "X_pca_harmony" in adata.obsm:
            use_rep = "X_pca_harmony"
        elif "X_pca" in adata.obsm:
            use_rep = "X_pca"
        else:
            raise ValueError("No PCA representation found. Run run_pca() first.")

    # Build neighbor graph
    sc.pp.neighbors(
        adata,
        n_neighbors=n_neighbors,
        use_rep=use_rep,
        random_state=random_state,
    )

    # Run Leiden clustering
    sc.tl.leiden(adata, resolution=resolution, random_state=random_state)

    return adata


def run_umap(
    adata: "ad.AnnData",
    n_components: int = 2,
    min_dist: float = 0.5,
    random_state: int = 42,
) -> "ad.AnnData":
    """Run UMAP dimensionality reduction for visualization.

    Args:
        adata: AnnData object with neighbor graph (from clustering).
        n_components: Number of UMAP dimensions.
        min_dist: Minimum distance parameter for UMAP.
        random_state: Random seed for reproducibility.

    Returns:
        AnnData with UMAP embedding in adata.obsm['X_umap'].
        Modifies adata in place and returns it.
    """
    _check_scanpy()

    if "neighbors" not in adata.uns:
        raise ValueError("No neighbor graph found. Run clustering first.")

    sc.tl.umap(
        adata,
        n_components=n_components,
        min_dist=min_dist,
        random_state=random_state,
    )

    return adata


def run_clustering(
    adata: "ad.AnnData",
    config: Optional[ClusteringConfig] = None,
) -> "ad.AnnData":
    """Run full clustering pipeline: PCA -> Harmony -> Louvain.

    Args:
        adata: AnnData object with expression matrix in X.
        config: Clustering configuration. If None, uses defaults.

    Returns:
        AnnData with:
        - adata.obsm['X_pca']: PCA embedding
        - adata.obsm['X_pca_harmony']: Harmony-corrected embedding (if used)
        - adata.obs['louvain']: Cluster assignments
        - adata.obsm['X_umap']: UMAP embedding

        Modifies adata in place and returns it.
    """
    _check_scanpy()

    if config is None:
        config = ClusteringConfig()

    # Step 1: PCA
    n_pcs = min(config.n_pcs, adata.n_vars - 1, adata.n_obs - 1)
    run_pca(adata, n_comps=n_pcs, random_state=config.random_state)

    # Step 2: Harmony batch correction (optional)
    if config.use_harmony and config.harmony_vars:
        # Check if all harmony vars exist
        missing = [v for v in config.harmony_vars if v not in adata.obs.columns]
        if missing:
            print(f"Warning: Harmony vars not found in obs: {missing}. Skipping Harmony.")
        else:
            try:
                run_harmony(adata, batch_key=config.harmony_vars)
            except Exception as e:
                print(f"Warning: Harmony failed: {e}. Continuing without batch correction.")

    # Step 3: Louvain clustering
    run_louvain(
        adata,
        resolution=config.louvain_resolution,
        n_neighbors=config.n_neighbors,
        random_state=config.random_state,
    )

    # Step 4: UMAP for visualization
    run_umap(adata, random_state=config.random_state)

    return adata


def clustering_pipeline(
    df: pd.DataFrame,
    config: Optional[ClusteringConfig] = None,
    marker_columns: Optional[list[str]] = None,
    obs_columns: Optional[list[str]] = None,
) -> "ad.AnnData":
    """Convenience wrapper: DataFrame -> AnnData -> clustering.

    Args:
        df: DataFrame with cell features.
        config: Clustering configuration.
        marker_columns: Columns to use as expression data.
        obs_columns: Columns to use as metadata.

    Returns:
        AnnData object with clustering results.
    """
    adata = to_anndata(df, marker_columns=marker_columns, obs_columns=obs_columns)
    run_clustering(adata, config=config)
    return adata


def get_cluster_assignments(
    adata: "ad.AnnData",
    cluster_key: str = "louvain",
) -> pd.Series:
    """Extract cluster assignments as a pandas Series.

    Args:
        adata: AnnData object with clustering results.
        cluster_key: Key in adata.obs containing cluster assignments.

    Returns:
        Series with cluster assignments, indexed by cell.
    """
    if cluster_key not in adata.obs.columns:
        raise ValueError(f"Cluster key '{cluster_key}' not found in adata.obs")

    return adata.obs[cluster_key]


def add_clusters_to_dataframe(
    df: pd.DataFrame,
    adata: "ad.AnnData",
    cluster_key: str = "louvain",
    column_name: str = "cluster",
) -> pd.DataFrame:
    """Add cluster assignments back to original DataFrame.

    Args:
        df: Original features DataFrame.
        adata: AnnData with clustering results.
        cluster_key: Key in adata.obs with cluster assignments.
        column_name: Name for new cluster column in df.

    Returns:
        DataFrame with added cluster column.
    """
    if len(df) != adata.n_obs:
        raise ValueError(
            f"DataFrame length ({len(df)}) doesn't match AnnData ({adata.n_obs})"
        )

    df_out = df.copy()
    df_out[column_name] = adata.obs[cluster_key].values

    return df_out
