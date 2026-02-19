"""Preprocessing utilities for feature data normalization and filtering."""

from typing import Optional, Union

import numpy as np
import pandas as pd

try:
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
except ImportError:
    StandardScaler = None
    MinMaxScaler = None


# Common metadata/morphology column names to exclude from marker detection
METADATA_COLUMNS = {
    "label",
    "roi",
    "centroid_x",
    "centroid_y",
    "centroid-0",
    "centroid-1",
    "area",
    "eccentricity",
    "major_axis_length",
    "minor_axis_length",
    "perimeter",
    "solidity",
    "extent",
    "orientation",
    "cluster",
    "cell_id",
    "sample",
    "sample_id",
    "experiment",
    "batch",
    "patient",
    "condition",
}


def identify_marker_columns(
    df: pd.DataFrame,
    exclude_columns: Optional[list[str]] = None,
) -> list[str]:
    """Auto-detect marker intensity columns vs metadata columns.

    Identifies columns that are likely marker intensities by excluding
    known metadata columns and non-numeric columns.

    Args:
        df: DataFrame with cell features.
        exclude_columns: Additional column names to exclude.

    Returns:
        List of column names identified as marker intensities.
    """
    exclude_set = METADATA_COLUMNS.copy()
    if exclude_columns:
        exclude_set.update(c.lower() for c in exclude_columns)

    marker_cols = []
    for col in df.columns:
        # Skip non-numeric columns
        if not np.issubdtype(df[col].dtype, np.number):
            continue

        # Skip known metadata columns (case-insensitive)
        if col.lower() in exclude_set:
            continue

        marker_cols.append(col)

    return marker_cols


def scale_features(
    df: pd.DataFrame,
    method: str = "standard",
    columns: Optional[list[str]] = None,
    exclude_columns: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Scale/normalize feature columns.

    Args:
        df: DataFrame with cell features.
        method: Scaling method:
            - 'standard': StandardScaler (z-score, mean=0, std=1)
            - 'minmax': MinMaxScaler (scale to [0, 1])
            - 'log': log1p transform (log(1 + x))
        columns: Specific columns to scale. If None, auto-detects markers.
        exclude_columns: Columns to exclude from auto-detection.

    Returns:
        DataFrame with scaled values (copy, original unchanged).

    Raises:
        ImportError: If sklearn not installed for standard/minmax methods.
        ValueError: If unknown scaling method.
    """
    df_scaled = df.copy()

    if columns is None:
        columns = identify_marker_columns(df, exclude_columns)

    if not columns:
        return df_scaled

    if method == "standard":
        if StandardScaler is None:
            raise ImportError("sklearn required for StandardScaler")
        scaler = StandardScaler()
        df_scaled[columns] = scaler.fit_transform(df[columns])

    elif method == "minmax":
        if MinMaxScaler is None:
            raise ImportError("sklearn required for MinMaxScaler")
        scaler = MinMaxScaler()
        df_scaled[columns] = scaler.fit_transform(df[columns])

    elif method == "log":
        # log1p transform: log(1 + x)
        for col in columns:
            df_scaled[col] = np.log1p(df[col].clip(lower=0))

    else:
        raise ValueError(f"Unknown scaling method: {method}")

    return df_scaled


def filter_markers(
    df: pd.DataFrame,
    exclude_markers: Optional[list[str]] = None,
    include_markers: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Filter DataFrame to include/exclude specific marker columns.

    Args:
        df: DataFrame with cell features.
        exclude_markers: Marker column names to remove.
        include_markers: If provided, keep only these markers (plus metadata).

    Returns:
        DataFrame with filtered columns.
    """
    if exclude_markers is None and include_markers is None:
        return df

    marker_cols = identify_marker_columns(df)
    metadata_cols = [c for c in df.columns if c not in marker_cols]

    if include_markers is not None:
        # Keep only specified markers
        include_set = {m.lower() for m in include_markers}
        keep_markers = [c for c in marker_cols if c.lower() in include_set]
    else:
        keep_markers = marker_cols

    if exclude_markers is not None:
        # Remove excluded markers
        exclude_set = {m.lower() for m in exclude_markers}
        keep_markers = [c for c in keep_markers if c.lower() not in exclude_set]

    return df[metadata_cols + keep_markers]


def filter_cells(
    df: pd.DataFrame,
    min_area: Optional[float] = None,
    max_area: Optional[float] = None,
    area_column: str = "area",
    min_values: Optional[dict[str, float]] = None,
    max_values: Optional[dict[str, float]] = None,
) -> pd.DataFrame:
    """Filter cells based on morphology or intensity criteria.

    Args:
        df: DataFrame with cell features.
        min_area: Minimum cell area (inclusive).
        max_area: Maximum cell area (inclusive).
        area_column: Name of area column.
        min_values: Dict of {column: min_value} for additional filters.
        max_values: Dict of {column: max_value} for additional filters.

    Returns:
        Filtered DataFrame (copy).
    """
    mask = pd.Series(True, index=df.index)

    if min_area is not None and area_column in df.columns:
        mask &= df[area_column] >= min_area

    if max_area is not None and area_column in df.columns:
        mask &= df[area_column] <= max_area

    if min_values:
        for col, min_val in min_values.items():
            if col in df.columns:
                mask &= df[col] >= min_val

    if max_values:
        for col, max_val in max_values.items():
            if col in df.columns:
                mask &= df[col] <= max_val

    return df[mask].copy()


def remove_outliers(
    df: pd.DataFrame,
    columns: Optional[list[str]] = None,
    method: str = "percentile",
    lower: float = 0.01,
    upper: float = 0.99,
) -> pd.DataFrame:
    """Remove outlier cells based on feature values.

    Args:
        df: DataFrame with cell features.
        columns: Columns to check for outliers. If None, uses all marker columns.
        method: Outlier detection method:
            - 'percentile': Remove cells outside [lower, upper] percentiles
            - 'zscore': Remove cells with |z-score| > upper (lower ignored)
        lower: Lower percentile threshold (for 'percentile' method).
        upper: Upper percentile threshold (for 'percentile') or z-score
            threshold (for 'zscore' method).

    Returns:
        Filtered DataFrame with outliers removed.
    """
    if columns is None:
        columns = identify_marker_columns(df)

    if not columns:
        return df.copy()

    mask = pd.Series(True, index=df.index)

    if method == "percentile":
        for col in columns:
            q_low = df[col].quantile(lower)
            q_high = df[col].quantile(upper)
            mask &= (df[col] >= q_low) & (df[col] <= q_high)

    elif method == "zscore":
        for col in columns:
            mean = df[col].mean()
            std = df[col].std()
            if std > 0:
                z = np.abs((df[col] - mean) / std)
                mask &= z <= upper

    else:
        raise ValueError(f"Unknown outlier method: {method}")

    return df[mask].copy()


def impute_missing(
    df: pd.DataFrame,
    columns: Optional[list[str]] = None,
    method: str = "median",
) -> pd.DataFrame:
    """Impute missing values in feature columns.

    Args:
        df: DataFrame with cell features.
        columns: Columns to impute. If None, uses all marker columns.
        method: Imputation method:
            - 'median': Fill with column median
            - 'mean': Fill with column mean
            - 'zero': Fill with 0

    Returns:
        DataFrame with imputed values (copy).
    """
    df_imputed = df.copy()

    if columns is None:
        columns = identify_marker_columns(df)

    for col in columns:
        if col not in df_imputed.columns:
            continue

        if method == "median":
            fill_value = df_imputed[col].median()
        elif method == "mean":
            fill_value = df_imputed[col].mean()
        elif method == "zero":
            fill_value = 0
        else:
            raise ValueError(f"Unknown imputation method: {method}")

        df_imputed[col] = df_imputed[col].fillna(fill_value)

    return df_imputed
