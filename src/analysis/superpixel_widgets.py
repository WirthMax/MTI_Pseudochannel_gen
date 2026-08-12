"""Interactive ipywidgets explorer for superpixel grid tuning.

Provides a slider-driven preview so the superpixel square size and the
empty-region removal threshold can be tuned visually before exporting a
feature table. Heavy UI dependencies (ipywidgets, matplotlib) are imported
lazily so ``analysis.superpixels`` stays importable in headless runs.
"""

from typing import Optional, Union

import numpy as np
import pandas as pd

try:
    import ipywidgets as widgets
except ImportError:
    widgets = None

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from pseudochannel.io import OMETiffChannels
from pseudochannel.preview import create_preview_stack, get_preview_scale
from .superpixels import build_superpixel_labels, extract_superpixel_features


def _check_ipywidgets():
    """Ensure the interactive UI dependencies are available."""
    if widgets is None:
        raise ImportError("ipywidgets is required for the superpixel explorer")
    if plt is None:
        raise ImportError("matplotlib is required for the superpixel explorer")


class SuperpixelExplorer:
    """Interactive superpixel-grid preview with slider-tunable square size.

    Displays a downsampled background image with the superpixel grid overlaid.
    Dragging the size slider re-tiles the grid live; toggling "remove empty"
    shades the superpixels that would be dropped and reports the kept count.
    ``export_features`` runs the full-resolution extraction with the current
    settings.
    """

    def __init__(
        self,
        channels: Union[dict, OMETiffChannels],
        display_target: int = 800,
        initial_size: int = 64,
        min_size: int = 8,
        max_size: int = 512,
        step: int = 8,
    ):
        _check_ipywidgets()

        self.channels = channels
        self.full_shape = (
            channels.shape
            if isinstance(channels, OMETiffChannels)
            else np.asarray(next(iter(channels.values()))).shape[:2]
        )
        self.marker_names = list(channels.keys())
        self.features_df: Optional[pd.DataFrame] = None

        # Downsample every channel once for a responsive preview.
        self._previews = create_preview_stack(channels, target_size=display_target)
        self._preview_shape = next(iter(self._previews.values())).shape
        self._scale_y, self._scale_x = get_preview_scale(
            self.full_shape, self._preview_shape
        )
        # Summed-signal thumbnail used as the emptiness proxy (mirrors
        # total_signal = sum of per-marker means at full resolution).
        self._signal_thumb = np.sum(
            list(self._previews.values()), axis=0
        ).astype(np.float32)

        self._build_widgets(initial_size, min_size, max_size, step)

    # ------------------------------------------------------------------ UI ---
    def _build_widgets(self, initial_size, min_size, max_size, step):
        self.size_slider = widgets.IntSlider(
            value=initial_size, min=min_size, max=max_size, step=step,
            description="Size (px)", continuous_update=False,
            readout=True, style={"description_width": "90px"},
            layout=widgets.Layout(width="320px"),
        )
        self.remove_empty_toggle = widgets.Checkbox(
            value=False, description="Remove empty superpixels",
            indent=False, layout=widgets.Layout(width="230px"),
        )
        self.threshold_slider = widgets.FloatSlider(
            value=0.0, min=0.0, max=float(np.percentile(self._signal_thumb, 99)) or 1.0,
            step=0.01, description="Empty ≤", continuous_update=False,
            readout=True, readout_format=".3f", disabled=True,
            style={"description_width": "90px"},
            layout=widgets.Layout(width="320px"),
        )
        self.display_dropdown = widgets.Dropdown(
            options=["max projection"] + self.marker_names,
            value="max projection", description="Show",
            style={"description_width": "90px"},
            layout=widgets.Layout(width="260px"),
        )
        self.export_button = widgets.Button(
            description="Compute table", button_style="success", icon="table",
        )
        self.status = widgets.HTML(value="")
        self.output = widgets.Output()

        controls = widgets.VBox([
            widgets.HBox([self.size_slider, self.display_dropdown]),
            widgets.HBox([self.remove_empty_toggle, self.threshold_slider]),
            widgets.HBox([self.export_button, self.status]),
        ])
        self.main_widget = widgets.VBox([
            widgets.HTML("<h3>Superpixel explorer</h3>"),
            controls,
            self.output,
        ])

        # Wire callbacks.
        self.size_slider.observe(self._on_change, names="value")
        self.remove_empty_toggle.observe(self._on_toggle_empty, names="value")
        self.threshold_slider.observe(self._on_change, names="value")
        self.display_dropdown.observe(self._on_change, names="value")
        self.export_button.on_click(self._on_export_click)

    def _on_toggle_empty(self, change):
        self.threshold_slider.disabled = not change["new"]
        self._render()

    def _on_change(self, change):
        self._render()

    # -------------------------------------------------------------- render ---
    def _background_thumb(self) -> np.ndarray:
        choice = self.display_dropdown.value
        if choice == "max projection":
            return np.max(list(self._previews.values()), axis=0)
        return self._previews[choice]

    def _empty_cell_flags(self, size: int):
        """Per-superpixel emptiness on the thumbnail (approximate preview).

        Returns (labels, empty_mask_flags, n_total) where flags aligns with the
        thumbnail grid labels. n_total is the true full-resolution superpixel
        count.
        """
        _, n_rows, n_cols = build_superpixel_labels(self.full_shape, size)
        n_total = n_rows * n_cols

        # Thumbnail-space square size, at least 1px.
        size_thumb = max(1, int(round(size * (self._scale_y + self._scale_x) / 2)))
        thumb_labels, _, _ = build_superpixel_labels(self._preview_shape, size_thumb)

        flat = thumb_labels.ravel()
        minlength = int(flat.max()) + 1
        counts = np.bincount(flat, minlength=minlength)
        sums = np.bincount(flat, weights=self._signal_thumb.ravel(), minlength=minlength)
        with np.errstate(divide="ignore", invalid="ignore"):
            means = np.nan_to_num(sums / counts, nan=0.0)

        threshold = self.threshold_slider.value if self.remove_empty_toggle.value else -1.0
        empty_per_label = means <= threshold
        empty_img = empty_per_label[thumb_labels]
        return empty_img, int((means > threshold).sum()), n_total

    def _render(self):
        size = self.size_slider.value
        h, w = self.full_shape

        bg = self._background_thumb()
        vmax = np.percentile(bg, 99) or 1.0

        empty_img, kept_thumb, n_total = self._empty_cell_flags(size)

        with self.output:
            self.output.clear_output(wait=True)
            fig, ax = plt.subplots(figsize=(8, 8 * h / w))
            fig.canvas.header_visible = False
            ax.imshow(bg, cmap="gray", vmin=0, vmax=vmax, extent=[0, w, h, 0])

            # Shade superpixels that would be removed (semi-transparent red).
            if self.remove_empty_toggle.value:
                overlay = np.zeros((*empty_img.shape, 4), dtype=np.float32)
                overlay[empty_img] = [1.0, 0.0, 0.0, 0.35]
                ax.imshow(overlay, extent=[0, w, h, 0])

            # Grid lines at multiples of the square size (cheap vlines/hlines).
            xs = np.arange(0, w + 1, size)
            ys = np.arange(0, h + 1, size)
            ax.vlines(xs, 0, h, colors="cyan", linewidth=0.5, alpha=0.7)
            ax.hlines(ys, 0, w, colors="cyan", linewidth=0.5, alpha=0.7)
            ax.set_xlim(0, w)
            ax.set_ylim(h, 0)
            ax.axis("off")
            plt.show()

        kept = kept_thumb if self.remove_empty_toggle.value else n_total
        self.status.value = (
            f"<b>{n_total}</b> superpixels ({size}px)"
            + (f" &middot; keeping <b>{kept}</b> (≈preview)" if self.remove_empty_toggle.value else "")
        )

    # -------------------------------------------------------------- export ---
    def export_features(
        self,
        output_path=None,
        save_labels=None,
    ) -> pd.DataFrame:
        """Run full-resolution extraction with the current slider settings.

        Args:
            output_path: If given, write the feature table CSV here.
            save_labels: If given, write the grid label image TIFF here.

        Returns:
            The superpixel feature DataFrame (also stored on ``.features_df``).
        """
        import tifffile

        size = self.size_slider.value
        if save_labels is not None:
            label_img, _, _ = build_superpixel_labels(self.full_shape, size)
            tifffile.imwrite(str(save_labels), label_img)

        df = extract_superpixel_features(
            self.channels,
            size=size,
            remove_empty=self.remove_empty_toggle.value,
            empty_threshold=self.threshold_slider.value,
        )
        self.features_df = df

        if output_path is not None:
            from pathlib import Path
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)

        return df

    def _on_export_click(self, _button):
        df = self.export_features()
        n_markers = len([c for c in df.columns if c.endswith(("_mean", "_sum", "_std"))])
        self.status.value = (
            f"Computed table: <b>{len(df)}</b> rows &times; "
            f"<b>{n_markers}</b> marker columns"
        )

    def display(self):
        """Show the widget and render the initial preview."""
        from IPython.display import display as _display
        _display(self.main_widget)
        self._render()


def create_superpixel_explorer(
    channels: Union[dict, OMETiffChannels],
    display_target: int = 800,
    initial_size: int = 64,
    min_size: int = 8,
    max_size: int = 512,
    step: int = 8,
) -> SuperpixelExplorer:
    """Build, display, and return a :class:`SuperpixelExplorer`.

    Args:
        channels: Dict-like of marker_name -> 2D array (e.g. OMETiffChannels).
        display_target: Longest-edge size of the preview thumbnails.
        initial_size: Initial superpixel square size (px).
        min_size / max_size / step: Bounds/step of the size slider.

    Returns:
        The live explorer object; call ``.export_features(...)`` after tuning.
    """
    explorer = SuperpixelExplorer(
        channels,
        display_target=display_target,
        initial_size=initial_size,
        min_size=min_size,
        max_size=max_size,
        step=step,
    )
    explorer.display()
    return explorer
