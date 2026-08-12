"""Interactive ipywidgets explorer for superpixel grid tuning.

Provides a slider-driven preview so the superpixel square size and the
empty-region removal threshold can be tuned visually before exporting a
feature table. Heavy UI dependencies (ipywidgets, matplotlib) are imported
lazily so ``analysis.superpixels`` stays importable in headless runs.
"""

import gc
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
from pseudochannel.preview import downsample_image, get_preview_scale
from .superpixels import (
    build_superpixel_labels,
    extract_superpixel_features,
    save_superpixel_masks,
    evict_channel_cache,
    _apply_keep_lut,
)


# Dropdown label for thresholding on the summed signal across all channels.
_TOTAL_SIGNAL_OPTION = "total signal (all)"


def _check_ipywidgets():
    """Ensure the interactive UI dependencies are available."""
    if widgets is None:
        raise ImportError("ipywidgets is required for the superpixel explorer")
    if plt is None:
        raise ImportError("matplotlib is required for the superpixel explorer")


class SuperpixelExplorer:
    """Interactive superpixel-grid preview with slider-tunable square size.

    Displays a downsampled background image with the superpixel grid overlaid.
    Dragging the size slider re-tiles the grid live; toggling "remove empty" and
    the threshold slider drops superpixels whose total signal is at or below the
    cutoff, which simply lose their grid outline. ``export_features`` runs the
    full-resolution extraction with the current settings.

    Efficient like the pseudochannel explorer: the figure and its artists are
    created once and every interaction only calls ``set_data``/``draw_idle`` (no
    figure churn), and only two small thumbnails are retained (max-projection +
    summed signal) rather than a downsampled copy of every channel.
    """

    def __init__(
        self,
        channels: Union[dict, OMETiffChannels],
        display_target: int = 512,
        initial_size: int = 64,
        min_size: int = 8,
        max_size: int = 512,
        step: int = 8,
    ):
        _check_ipywidgets()

        self.channels = channels
        self.display_target = display_target
        self.full_shape = (
            channels.shape
            if isinstance(channels, OMETiffChannels)
            else np.asarray(next(iter(channels.values()))).shape[:2]
        )
        self.marker_names = list(channels.keys())
        self.features_df: Optional[pd.DataFrame] = None

        # Stream over channels once, keeping only two small thumbnails: the
        # max-projection (default background) and the summed-signal proxy for
        # emptiness. Per-channel thumbnails are NOT retained -- individual
        # channels are downsampled on demand and single-slot cached, so peak
        # memory stays at one full-res channel + these accumulators.
        self._max_proj: Optional[np.ndarray] = None
        self._signal_thumb: Optional[np.ndarray] = None
        for name in self.marker_names:
            thumb = self._load_thumb(name, evict=True)
            if self._max_proj is None:
                self._max_proj = thumb.copy()
                self._signal_thumb = thumb.astype(np.float32, copy=True)
            else:
                np.maximum(self._max_proj, thumb, out=self._max_proj)
                self._signal_thumb += thumb
            del thumb
        gc.collect()
        self._preview_shape = self._max_proj.shape
        self._scale_y, self._scale_x = get_preview_scale(
            self.full_shape, self._preview_shape
        )
        # Small bounded cache of per-channel thumbnails (display + threshold
        # channels), so we never retain more than a couple at once.
        self._chan_cache: dict[str, np.ndarray] = {}
        self._chan_cache_max = 3

        # Persistent figure/artist handles (created once in display()).
        self._fig = None
        self._ax = None
        self._bg_img = None
        self._overlay_img = None

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
        # Which channel drives the empty filter (default: summed signal).
        self.threshold_channel_dropdown = widgets.Dropdown(
            options=[_TOTAL_SIGNAL_OPTION] + self.marker_names,
            value=_TOTAL_SIGNAL_OPTION, description="Threshold on",
            style={"description_width": "90px"},
            layout=widgets.Layout(width="260px"),
        )
        # Absolute cutoff on the chosen signal. Range spans up to the 99th
        # percentile of that signal's thumbnail (a usable upper bound).
        thresh_max = self._thresh_slider_max()
        self.threshold_slider = widgets.FloatSlider(
            value=0.0, min=0.0, max=thresh_max, step=thresh_max / 200 or 0.01,
            description="Empty ≤", continuous_update=False,
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
            widgets.HBox([self.remove_empty_toggle, self.threshold_channel_dropdown]),
            widgets.HBox([self.threshold_slider]),
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
        self.threshold_channel_dropdown.observe(self._on_threshold_channel_change, names="value")
        self.display_dropdown.observe(self._on_change, names="value")
        self.export_button.on_click(self._on_export_click)

    def _thresh_slider_max(self) -> float:
        """99th-percentile upper bound of the currently selected threshold
        signal thumbnail."""
        return float(np.percentile(self._threshold_signal_thumb(), 99)) or 1.0

    def _on_toggle_empty(self, change):
        self.threshold_slider.disabled = not change["new"]
        self._render()

    def _on_threshold_channel_change(self, change):
        # Re-scale the slider range to the newly selected signal, then redraw.
        new_max = self._thresh_slider_max()
        if new_max >= self.threshold_slider.max:
            self.threshold_slider.max = new_max
        else:
            self.threshold_slider.value = min(self.threshold_slider.value, new_max)
            self.threshold_slider.max = new_max
        self.threshold_slider.step = new_max / 200 or 0.01
        self._render()

    def _on_change(self, change):
        self._render()

    # -------------------------------------------------------------- render ---
    def _load_thumb(self, name: str, evict: bool = True) -> np.ndarray:
        """Downsample one channel to a thumbnail, then release the full-res
        source (drop local ref + evict the OMETiffChannels cache entry) so
        full-resolution channels never accumulate in memory."""
        thumb = downsample_image(self.channels[name], self.display_target)
        if evict:
            evict_channel_cache(self.channels, name)
        return thumb

    def _get_channel_thumb(self, name: str) -> np.ndarray:
        """Thumbnail for a single channel, bounded-cached (small thumbnails)."""
        if name not in self._chan_cache:
            if len(self._chan_cache) >= self._chan_cache_max:
                self._chan_cache.pop(next(iter(self._chan_cache)))  # evict oldest
            self._chan_cache[name] = self._load_thumb(name, evict=True)
        return self._chan_cache[name]

    def _background_thumb(self) -> np.ndarray:
        choice = self.display_dropdown.value
        if choice == "max projection":
            return self._max_proj
        return self._get_channel_thumb(choice)

    def _threshold_signal_thumb(self) -> np.ndarray:
        """Thumbnail whose per-superpixel mean drives the empty filter: the
        summed signal, or a single channel when one is selected."""
        choice = self.threshold_channel_dropdown.value
        if choice == _TOTAL_SIGNAL_OPTION:
            return self._signal_thumb
        return self._get_channel_thumb(choice)

    def _kept_cells(self, size: int):
        """Compute the kept-superpixel mask on the thumbnail (preview approx).

        Returns (thumb_labels, kept_img, n_total, kept_fraction):
        - thumb_labels: thumbnail-resolution grid label image.
        - kept_img: bool array (thumbnail res), True where the pixel's superpixel
          is retained under the current absolute total-signal cutoff.
        - n_total: true full-resolution superpixel count.
        - kept_fraction: fraction of superpixels retained.
        """
        _, n_rows, n_cols = build_superpixel_labels(self.full_shape, size)
        n_total = n_rows * n_cols

        # Thumbnail-space square size, at least 1px.
        size_thumb = max(1, int(round(size * (self._scale_y + self._scale_x) / 2)))
        thumb_labels, _, _ = build_superpixel_labels(self._preview_shape, size_thumb)

        signal_thumb = self._threshold_signal_thumb()
        flat = thumb_labels.ravel()
        minlength = int(flat.max()) + 1
        counts = np.bincount(flat, minlength=minlength)
        sums = np.bincount(flat, weights=signal_thumb.ravel(), minlength=minlength)
        with np.errstate(divide="ignore", invalid="ignore"):
            means = np.nan_to_num(sums / counts, nan=0.0)

        kept_per_label = np.ones(minlength, dtype=bool)
        kept_per_label[0] = False  # label 0 is unused (grid tiles from 1)
        if self.remove_empty_toggle.value:
            cutoff = self.threshold_slider.value
            kept_per_label[1:] = means[1:] > cutoff

        kept_img = kept_per_label[thumb_labels]
        kept_fraction = float(kept_per_label[1:].mean()) if minlength > 1 else 1.0
        return thumb_labels, kept_img, n_total, kept_fraction

    @staticmethod
    def _cell_border(label_img: np.ndarray) -> np.ndarray:
        """Boolean mask of superpixel boundary pixels (label differs from a
        4-neighbor, plus the image edge)."""
        b = np.zeros(label_img.shape, dtype=bool)
        b[:, :-1] |= label_img[:, :-1] != label_img[:, 1:]
        b[:, 1:] |= label_img[:, 1:] != label_img[:, :-1]
        b[:-1, :] |= label_img[:-1, :] != label_img[1:, :]
        b[1:, :] |= label_img[1:, :] != label_img[:-1, :]
        b[0, :] = b[-1, :] = b[:, 0] = b[:, -1] = True
        return b

    def _create_figure(self):
        """Create the figure and its two AxesImages once; updates reuse them.

        Mirrors the pseudochannel explorer: the figure and artists live for the
        whole session and each interaction only calls ``set_data``/``draw_idle``,
        so figures never accumulate in matplotlib's registry.
        """
        # Close any prior figure this explorer created before making a new one,
        # so re-running the cell never leaves orphaned figures in matplotlib's
        # registry (a classic ipympl memory leak).
        if self._fig is not None:
            plt.close(self._fig)
            self._fig = self._ax = self._bg_img = self._overlay_img = None
        h, w = self.full_shape
        with self.output:
            self.output.clear_output(wait=True)
            self._fig, self._ax = plt.subplots(figsize=(8, 8 * h / w))
            self._fig.canvas.header_visible = False
            self._fig.canvas.toolbar_visible = False
            self._fig.canvas.resizable = False
            blank_bg = np.zeros(self._preview_shape, dtype=np.float32)
            blank_ov = np.zeros((*self._preview_shape, 4), dtype=np.float32)
            self._bg_img = self._ax.imshow(
                blank_bg, cmap="gray", vmin=0, vmax=1, extent=[0, w, h, 0]
            )
            self._overlay_img = self._ax.imshow(
                blank_ov, extent=[0, w, h, 0], interpolation="nearest"
            )
            self._ax.set_xlim(0, w)
            self._ax.set_ylim(h, 0)
            self._ax.axis("off")
            self._fig.tight_layout(pad=0)
            plt.show()

    def _render(self):
        if self._fig is None:  # figure not created yet
            return
        size = self.size_slider.value

        bg = self._background_thumb()
        vmax = float(np.percentile(bg, 99)) or 1.0

        thumb_labels, kept_img, n_total, kept_fraction = self._kept_cells(size)
        # Outline only the retained superpixels; removed blocks lose their grid so
        # the grid visibly dissolves where superpixels are dropped.
        outline = self._cell_border(thumb_labels) & kept_img
        overlay = np.zeros((*outline.shape, 4), dtype=np.float32)
        overlay[outline] = [0.0, 1.0, 1.0, 0.8]  # cyan grid on kept cells

        # Update artists in place -- no new figure, no clear_output.
        self._bg_img.set_data(bg)
        self._bg_img.set_clim(0, vmax)
        self._overlay_img.set_data(overlay)
        self._fig.canvas.draw_idle()

        if self.remove_empty_toggle.value:
            n_kept = int(round(kept_fraction * n_total))
            src = self.threshold_channel_dropdown.value
            self.status.value = (
                f"<b>{n_total}</b> superpixels ({size}px) &middot; keeping "
                f"<b>{n_kept}</b> (~{kept_fraction * 100:.0f}%, {src} &gt; {self.threshold_slider.value:.3f})"
            )
        else:
            self.status.value = f"<b>{n_total}</b> superpixels ({size}px)"

    # -------------------------------------------------------------- export ---
    def export_features(
        self,
        output_path=None,
        save_masks: bool = False,
        mask_dir=None,
        mask_basename: str = "superpixel",
        mask_formats=("cellpose", "macsiqview"),
    ) -> pd.DataFrame:
        """Run full-resolution extraction with the current slider settings.

        Args:
            output_path: If given, write the feature table CSV here.
            save_masks: If True, also save the superpixel masks (Cellpose label
                TIFF + MacsIQView binary). When "Remove empty superpixels" is on,
                the masks are thresholded to match the table.
            mask_dir: Directory for the mask TIFFs. Defaults to the CSV's parent
                when ``output_path`` is given, else the current directory.
            mask_basename: Filename stem for the mask outputs.
            mask_formats: Which mask formats to write.

        Returns:
            The superpixel feature DataFrame (also stored on ``.features_df``).
            Saved mask paths, if any, are stored on ``.mask_paths``.
        """
        from pathlib import Path

        size = self.size_slider.value
        remove_empty = self.remove_empty_toggle.value
        empty_threshold = self.threshold_slider.value
        choice = self.threshold_channel_dropdown.value
        threshold_marker = None if choice == _TOTAL_SIGNAL_OPTION else choice

        try:
            df = extract_superpixel_features(
                self.channels,
                size=size,
                remove_empty=remove_empty,
                empty_threshold=empty_threshold,
                threshold_marker=threshold_marker,
            )
            self.features_df = df

            if output_path is not None:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(output_path, index=False)

            self.mask_paths = {}
            if save_masks:
                # Rebuild the (thresholded) label image from the computed table;
                # the kept labels are exactly df["label"].
                label_img, _, _ = build_superpixel_labels(self.full_shape, size)
                if remove_empty and "label" in df:
                    label_img = _apply_keep_lut(label_img, df["label"].to_numpy())
                target_dir = (
                    Path(mask_dir)
                    if mask_dir is not None
                    else (output_path.parent if output_path is not None else Path("."))
                )
                self.mask_paths = save_superpixel_masks(
                    label_img, target_dir, basename=mask_basename, formats=mask_formats
                )
                del label_img
        finally:
            # Extraction loads every channel full-res; release them all (the
            # per-channel eviction already runs during aggregation, this is a
            # belt-and-suspenders clear) and reclaim memory before returning.
            evict_channel_cache(self.channels)
            gc.collect()

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
        self._create_figure()
        self._render()

    def close(self):
        """Release all resources: close the matplotlib figure and free every
        cached thumbnail and full-res channel. Call before discarding the
        explorer (re-running the cell does this automatically)."""
        if self._fig is not None:
            plt.close(self._fig)
            self._fig = self._ax = self._bg_img = self._overlay_img = None
        self._chan_cache.clear()
        evict_channel_cache(self.channels)
        gc.collect()


# Tracks the most recently created explorer so re-running the notebook cell can
# tear the previous one down (figure + caches) instead of leaking it.
_ACTIVE_EXPLORER: Optional["SuperpixelExplorer"] = None


def create_superpixel_explorer(
    channels: Union[dict, OMETiffChannels],
    display_target: int = 512,
    initial_size: int = 64,
    min_size: int = 8,
    max_size: int = 512,
    step: int = 8,
) -> SuperpixelExplorer:
    """Build, display, and return a :class:`SuperpixelExplorer`.

    Re-running this closes the previously created explorer first, so figures and
    cached channel data don't accumulate across cell re-runs.

    Args:
        channels: Dict-like of marker_name -> 2D array (e.g. OMETiffChannels).
        display_target: Longest-edge size of the preview thumbnails.
        initial_size: Initial superpixel square size (px).
        min_size / max_size / step: Bounds/step of the size slider.

    Returns:
        The live explorer object; call ``.export_features(...)`` after tuning.
    """
    global _ACTIVE_EXPLORER
    if _ACTIVE_EXPLORER is not None:
        try:
            _ACTIVE_EXPLORER.close()
        except Exception:
            pass
        _ACTIVE_EXPLORER = None

    explorer = SuperpixelExplorer(
        channels,
        display_target=display_target,
        initial_size=initial_size,
        min_size=min_size,
        max_size=max_size,
        step=step,
    )
    explorer.display()
    _ACTIVE_EXPLORER = explorer
    return explorer
