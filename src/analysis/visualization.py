"""Visualization functions for clustered cell data."""

from typing import Optional, Union

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    import seaborn as sns
except ImportError:
    sns = None

try:
    import scanpy as sc
except ImportError:
    sc = None

from .preprocessing import identify_marker_columns


def _check_matplotlib():
    """Check if matplotlib is available."""
    if plt is None:
        raise ImportError("matplotlib is required for plotting")


def _check_seaborn():
    """Check if seaborn is available."""
    if sns is None:
        raise ImportError("seaborn is required for this plot")


def _check_scanpy():
    """Check if scanpy is available."""
    if sc is None:
        raise ImportError("scanpy is required for this plot")


def plot_marker_heatmap(
    adata: "sc.AnnData",
    groupby: str = "louvain",
    markers: Optional[list[str]] = None,
    standard_scale: str = "var",
    cmap: str = "viridis",
    figsize: Optional[tuple[float, float]] = None,
    save: Optional[str] = None,
    show: bool = True,
    **kwargs,
) -> Optional[plt.Figure]:
    """Plot marker expression heatmap grouped by clusters.

    Uses scanpy's matrixplot to show mean expression of markers
    per cluster as a heatmap.

    Args:
        adata: AnnData object with clustering results.
        groupby: Column in adata.obs to group by (default: 'louvain').
        markers: List of markers to plot. If None, uses all.
        standard_scale: Standardization: 'var' (per marker), 'obs' (per cluster), None.
        cmap: Colormap name.
        figsize: Figure size (width, height).
        save: Path to save figure. If None, don't save.
        show: Whether to display the plot.
        **kwargs: Additional arguments passed to sc.pl.matrixplot.

    Returns:
        Figure object if show=False, else None.
    """
    _check_scanpy()
    _check_matplotlib()

    if markers is None:
        markers = list(adata.var_names)

    ax = sc.pl.matrixplot(
        adata,
        var_names=markers,
        groupby=groupby,
        standard_scale=standard_scale,
        cmap=cmap,
        figsize=figsize,
        save=save,
        show=show,
        **kwargs,
    )

    return ax


def plot_dotplot(
    adata: "sc.AnnData",
    groupby: str = "louvain",
    markers: Optional[list[str]] = None,
    standard_scale: str = "var",
    cmap: str = "Reds",
    figsize: Optional[tuple[float, float]] = None,
    save: Optional[str] = None,
    show: bool = True,
    **kwargs,
) -> Optional[plt.Figure]:
    """Plot marker expression as a dot plot.

    Shows both expression level (color) and fraction of expressing cells (size).

    Args:
        adata: AnnData object with clustering results.
        groupby: Column in adata.obs to group by.
        markers: List of markers to plot. If None, uses all.
        standard_scale: Standardization: 'var', 'obs', or None.
        cmap: Colormap name.
        figsize: Figure size.
        save: Path to save figure.
        show: Whether to display.
        **kwargs: Additional arguments to sc.pl.dotplot.

    Returns:
        Figure if show=False.
    """
    _check_scanpy()
    _check_matplotlib()

    if markers is None:
        markers = list(adata.var_names)

    ax = sc.pl.dotplot(
        adata,
        var_names=markers,
        groupby=groupby,
        standard_scale=standard_scale,
        cmap=cmap,
        figsize=figsize,
        save=save,
        show=show,
        **kwargs,
    )

    return ax


def plot_cluster_composition(
    df: pd.DataFrame,
    cluster_col: str = "cluster",
    group_col: str = "ROI",
    normalize: bool = True,
    figsize: tuple[float, float] = (12, 6),
    cmap: str = "tab20",
    title: str = "Cluster Composition by Group",
    save: Optional[str] = None,
    show: bool = True,
) -> Optional[plt.Figure]:
    """Plot cluster composition as stacked bar chart.

    Shows the proportion (or count) of each cluster within each group.

    Args:
        df: DataFrame with cluster assignments.
        cluster_col: Column containing cluster labels.
        group_col: Column containing group labels (e.g., ROI, sample).
        normalize: If True, show proportions. If False, show counts.
        figsize: Figure size.
        cmap: Colormap for clusters.
        title: Plot title.
        save: Path to save figure.
        show: Whether to display.

    Returns:
        Figure if show=False.
    """
    _check_matplotlib()

    # Compute crosstab
    ct = pd.crosstab(df[group_col], df[cluster_col])

    if normalize:
        ct = ct.div(ct.sum(axis=1), axis=0)

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    ct.plot(kind="bar", stacked=True, ax=ax, colormap=cmap, edgecolor="white", linewidth=0.5)

    ax.set_xlabel(group_col)
    ax.set_ylabel("Proportion" if normalize else "Count")
    ax.set_title(title)
    ax.legend(title=cluster_col, bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
        return None

    return fig


def plot_marker_distributions(
    df: pd.DataFrame,
    markers: Optional[list[str]] = None,
    hue: Optional[str] = None,
    kind: str = "kde",
    ncols: int = 4,
    figsize_per_plot: tuple[float, float] = (3, 2.5),
    palette: Optional[str] = None,
    save: Optional[str] = None,
    show: bool = True,
) -> Optional[plt.Figure]:
    """Plot distributions of marker intensities.

    Creates a grid of histograms or KDE plots for each marker.

    Args:
        df: DataFrame with marker intensities.
        markers: List of markers to plot. If None, auto-detects.
        hue: Column to use for color grouping (e.g., 'cluster').
        kind: Plot type: 'kde', 'hist', or 'box'.
        ncols: Number of columns in subplot grid.
        figsize_per_plot: Size of each subplot.
        palette: Seaborn color palette name.
        save: Path to save figure.
        show: Whether to display.

    Returns:
        Figure if show=False.
    """
    _check_matplotlib()
    _check_seaborn()

    if markers is None:
        markers = identify_marker_columns(df)

    n_markers = len(markers)
    nrows = (n_markers + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_plot[0] * ncols, figsize_per_plot[1] * nrows),
    )
    axes = np.array(axes).flatten()

    for i, marker in enumerate(markers):
        ax = axes[i]

        if kind == "kde":
            if hue:
                for group in df[hue].unique():
                    subset = df[df[hue] == group][marker]
                    sns.kdeplot(subset, ax=ax, label=str(group), fill=True, alpha=0.3)
                ax.legend(fontsize="small")
            else:
                sns.kdeplot(df[marker], ax=ax, fill=True)

        elif kind == "hist":
            if hue:
                for group in df[hue].unique():
                    subset = df[df[hue] == group][marker]
                    ax.hist(subset, alpha=0.5, label=str(group), bins=30)
                ax.legend(fontsize="small")
            else:
                ax.hist(df[marker], bins=30, edgecolor="white")

        elif kind == "box":
            if hue:
                sns.boxplot(data=df, x=hue, y=marker, ax=ax, palette=palette)
                ax.tick_params(axis="x", rotation=45)
            else:
                sns.boxplot(data=df, y=marker, ax=ax)

        ax.set_title(marker, fontsize=10)
        ax.set_xlabel("")

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()

    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
        return None

    return fig


def plot_umap(
    adata: "sc.AnnData",
    color: Union[str, list[str]] = "louvain",
    size: float = 10,
    palette: Optional[str] = None,
    figsize: tuple[float, float] = (8, 6),
    title: Optional[str] = None,
    save: Optional[str] = None,
    show: bool = True,
    **kwargs,
) -> Optional[plt.Figure]:
    """Plot UMAP embedding colored by cluster or feature.

    Args:
        adata: AnnData object with UMAP in obsm['X_umap'].
        color: Variable(s) to color by. Can be obs column or var name.
        size: Point size.
        palette: Color palette name.
        figsize: Figure size.
        title: Plot title.
        save: Path to save figure.
        show: Whether to display.
        **kwargs: Additional arguments to sc.pl.umap.

    Returns:
        Figure if show=False.
    """
    _check_scanpy()
    _check_matplotlib()

    if "X_umap" not in adata.obsm:
        raise ValueError("UMAP not found. Run run_umap() first.")

    ax = sc.pl.umap(
        adata,
        color=color,
        size=size,
        palette=palette,
        title=title,
        save=save,
        show=show,
        **kwargs,
    )

    return ax


def plot_cluster_markers(
    adata: "sc.AnnData",
    cluster_col: str = "louvain",
    n_markers: int = 5,
    method: str = "wilcoxon",
    save: Optional[str] = None,
    show: bool = True,
) -> Optional[plt.Figure]:
    """Plot top marker genes per cluster.

    Runs differential expression to find markers and displays them.

    Args:
        adata: AnnData object with clustering.
        cluster_col: Column with cluster assignments.
        n_markers: Number of top markers to show per cluster.
        method: Statistical test: 'wilcoxon', 't-test', etc.
        save: Path to save figure.
        show: Whether to display.

    Returns:
        Figure if show=False.
    """
    _check_scanpy()
    _check_matplotlib()

    # Run differential expression if not already done
    if "rank_genes_groups" not in adata.uns:
        sc.tl.rank_genes_groups(adata, groupby=cluster_col, method=method)

    ax = sc.pl.rank_genes_groups_dotplot(
        adata,
        n_genes=n_markers,
        save=save,
        show=show,
    )

    return ax


def plot_pca_variance(
    adata: "sc.AnnData",
    n_pcs: int = 30,
    figsize: tuple[float, float] = (8, 4),
    save: Optional[str] = None,
    show: bool = True,
) -> Optional[plt.Figure]:
    """Plot PCA variance explained (scree plot).

    Args:
        adata: AnnData with PCA results.
        n_pcs: Number of PCs to show.
        figsize: Figure size.
        save: Path to save.
        show: Whether to display.

    Returns:
        Figure if show=False.
    """
    _check_matplotlib()

    if "pca" not in adata.uns:
        raise ValueError("PCA not found. Run run_pca() first.")

    variance_ratio = adata.uns["pca"]["variance_ratio"][:n_pcs]
    cumulative = np.cumsum(variance_ratio)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Individual variance
    ax1.bar(range(1, len(variance_ratio) + 1), variance_ratio)
    ax1.set_xlabel("PC")
    ax1.set_ylabel("Variance Explained")
    ax1.set_title("Individual PC Variance")

    # Cumulative variance
    ax2.plot(range(1, len(cumulative) + 1), cumulative, marker="o", markersize=4)
    ax2.axhline(0.9, color="red", linestyle="--", alpha=0.5, label="90%")
    ax2.set_xlabel("Number of PCs")
    ax2.set_ylabel("Cumulative Variance")
    ax2.set_title("Cumulative Variance Explained")
    ax2.legend()

    plt.tight_layout()

    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
        return None

    return fig
