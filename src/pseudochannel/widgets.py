"""Interactive ipywidgets interface for pseudochannel tuning."""

import time
from pathlib import Path
from typing import Optional, Union

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from IPython.display import display, Javascript
from matplotlib.widgets import RectangleSelector

from .core import compute_pseudochannel
from .io import load_channel_folder, load_ome_tiff, OMETiffChannels, validate_channels
from .preview import create_preview_stack, downsample_image
from .segmentation import (
    CellposeConfig,
    _check_cellpose,
    create_cellpose_model,
    run_segmentation,
    extract_mask_contours,
    overlay_contours_on_axes,
)


def create_weight_sliders(
    channel_names: list[str],
    initial_weights: Optional[dict[str, float]] = None,
) -> dict[str, widgets.FloatSlider]:
    """Create ipywidgets FloatSlider for each channel.

    Args:
        channel_names: List of channel names
        initial_weights: Optional dict of initial weight values

    Returns:
        Dict of channel_name -> FloatSlider widget
    """
    sliders = {}

    for name in sorted(channel_names):
        initial = 0.0
        if initial_weights and name in initial_weights:
            initial = initial_weights[name]

        slider = widgets.FloatSlider(
            value=initial,
            min=0.0,
            max=1.0,
            step=0.01,
            description=name[:15],
            description_tooltip=name,
            continuous_update=False,
            orientation="horizontal",
            readout=True,
            readout_format=".2f",
            style={"description_width": "100px"},
            layout=widgets.Layout(width="280px"),
        )
        sliders[name] = slider

    return sliders


def get_weights_from_sliders(sliders: dict[str, widgets.FloatSlider]) -> dict[str, float]:
    """Extract current weight values from sliders.

    Args:
        sliders: Dict of channel_name -> FloatSlider

    Returns:
        Dict of channel_name -> weight value
    """
    return {name: slider.value for name, slider in sliders.items()}


class PseudochannelExplorer:
    """Interactive widget for pseudochannel weight tuning."""

    def __init__(
        self,
        channels: Union[str, Path, dict, OMETiffChannels],
        marker_file: Optional[Union[str, Path]] = None,
        marker_column: Optional[Union[int, str]] = None,
        preview_size: int = 512,
        debounce_ms: int = 100,
        exclude_channels: set[str] | list[str] | None = None,
        nuclear_marker_path: Optional[Union[str, Path]] = None,
        max_zoom_size: int = 1024,
        mcmicro_markers: bool = False,
    ):
        """Initialize the explorer.

        Args:
            channels: One of:
                - Path to folder containing individual channel TIFFs
                - Path to OME-TIFF file (requires marker_file)
                - Pre-loaded dict of channel_name -> array
                - OMETiffChannels instance
            marker_file: Path to marker names file (required for OME-TIFF)
            marker_column: Column for CSV/TSV marker file
            preview_size: Size for preview images
            debounce_ms: Debounce time for slider updates in milliseconds
            exclude_channels: Channel names to exclude (case-insensitive).
                If None, uses DEFAULT_EXCLUDED_CHANNELS.
                Pass empty set/list to include all channels.
            nuclear_marker_path: Path to nuclear marker TIFF file (e.g., DAPI).
                If provided, enables nuclear overlay toggle. For OME-TIFF input,
                the DAPI channel will be auto-detected if not specified.
            max_zoom_size: Maximum size for zoom region. Larger regions will be
                downsampled for faster display.
            mcmicro_markers: If True, parse marker_file as MCMICRO format
                (with marker_name column and remove column for filtering).
        """
        self.preview_size = preview_size
        self.debounce_ms = debounce_ms
        self.exclude_channels = exclude_channels
        self.max_zoom_size = max_zoom_size

        self._last_update_time = 0
        self._pending_update = False

        # Zoom feature state
        self._zoom_region = None  # (x1, y1, x2, y2) in preview coords
        self._zoom_output = None  # Output widget for zoomed view
        self._rect_selector = None  # RectangleSelector instance

        # Figure references (created once, reused)
        self._preview_fig = None
        self._preview_ax = None
        self._preview_img = None  # AxesImage object for set_data()
        self._preview_is_rgb = False  # Track if current image is RGB

        self._zoom_fig = None
        self._zoom_ax = None
        self._zoom_img = None
        self._zoom_is_rgb = False

        # Nuclear marker overlay state
        self.nuclear_marker = None  # Full-res nuclear marker array
        self.nuclear_preview = None  # Downsampled nuclear marker

        # Segmentation state
        self._cellpose_model = None
        self._cellpose_config = CellposeConfig()
        self._segmentation_masks = None  # Cached (H,W) int mask
        self._mask_contour_lines = []  # matplotlib Line2D artists
        self._mask_contours_data = []  # Cached contour coordinate lists
        self._seg_zoom_region = None  # Zoom region when masks were computed
        self._seg_weights_hash = None  # Hash of weights when masks were computed

        # Store original channels input and marker info for OME-TIFF auto-detection
        self._original_channels_input = channels
        self._marker_file = marker_file
        self._marker_column = marker_column
        self._mcmicro_markers = mcmicro_markers

        self.channels = self._load_channels(
            channels, marker_file, marker_column, exclude_channels, mcmicro_markers
        )
        validate_channels(self.channels)

        self.previews = create_preview_stack(self.channels, preview_size)
        self.channel_names = list(self.channels.keys())

        # Load nuclear marker
        self._load_nuclear_marker(nuclear_marker_path)

        self._setup_widgets()

    def _load_channels(
        self,
        channels: Union[str, Path, dict, OMETiffChannels],
        marker_file: Optional[Union[str, Path]],
        marker_column: Optional[Union[int, str]],
        exclude_channels: set[str] | list[str] | None,
        mcmicro_markers: bool = False,
    ) -> Union[dict, OMETiffChannels]:
        """Load channels from various input types."""
        if isinstance(channels, (dict, OMETiffChannels)):
            return channels

        path = Path(channels)

        if path.is_dir():
            return load_channel_folder(path, exclude_channels=exclude_channels)

        if path.is_file():
            if marker_file is None:
                raise ValueError(
                    "marker_file is required when loading from OME-TIFF"
                )
            # Use OMETiffChannels for lazy loading and nuclear marker auto-detection
            return OMETiffChannels(
                path,
                marker_file,
                marker_column=marker_column,
                exclude_channels=exclude_channels,
                mcmicro_markers=mcmicro_markers,
            )

        raise ValueError(f"Invalid channels input: {channels}")

    def _load_nuclear_marker(
        self, nuclear_marker_path: Optional[Union[str, Path]]
    ) -> None:
        """Load nuclear marker for overlay.

        Args:
            nuclear_marker_path: Path to nuclear marker TIFF file.
                If None, attempts to auto-detect DAPI from OME-TIFF.
        """
        # Option A: Load from explicit path
        if nuclear_marker_path is not None:
            path = Path(nuclear_marker_path)
            if path.is_file():
                self.nuclear_marker = tifffile.imread(str(path))
                self.nuclear_preview = downsample_image(
                    self.nuclear_marker, self.preview_size
                )
                return

        # Option B: Auto-detect DAPI from OME-TIFF
        if isinstance(self.channels, OMETiffChannels):
            self._load_nuclear_from_ome_tiff()

    def _load_nuclear_from_ome_tiff(self) -> None:
        """Try to load DAPI channel from OME-TIFF source."""
        if not isinstance(self.channels, OMETiffChannels):
            return

        # Look for DAPI in the original (unfiltered) marker names
        dapi_names = {"dapi", "hoechst", "nuclear", "nuclei"}
        for i, name in enumerate(self.channels._all_marker_names):
            if name.lower() in dapi_names or "dapi" in name.lower():
                # Load channel by original index (bypasses exclusion filtering)
                self.nuclear_marker = self.channels.get_channel_by_index(i)
                self.nuclear_preview = downsample_image(
                    self.nuclear_marker, self.preview_size
                )
                return

    def _normalize_nuclear(self, arr: np.ndarray) -> np.ndarray:
        """Normalize nuclear marker to 0-1 range using percentile stretch.

        Args:
            arr: Nuclear marker array

        Returns:
            Normalized array in 0-1 range
        """
        arr = arr.astype(np.float32)
        vmin, vmax = np.percentile(arr, [1, 99])
        if vmax > vmin:
            return np.clip((arr - vmin) / (vmax - vmin), 0, 1)
        return np.zeros_like(arr)

    def _setup_widgets(self):
        """Create all widget components with responsive layout."""
        self.sliders = create_weight_sliders(self.channel_names)

        self.output = widgets.Output()

        self.reset_button = widgets.Button(
            description="Reset Weights",
            button_style="warning",
            icon="refresh",
        )
        self.reset_button.on_click(self._on_reset)

        self.normalize_dropdown = widgets.Dropdown(
            options=["minmax", "percentile", "none"],
            value="minmax",
            description="Normalize:",
            style={"description_width": "80px"},
        )
        self.normalize_dropdown.observe(self._on_slider_change, names="value")

        self.cmap_dropdown = widgets.Dropdown(
            options=["gray", "viridis", "magma", "inferno", "plasma"],
            value="gray",
            description="Colormap:",
            style={"description_width": "80px"},
        )
        self.cmap_dropdown.observe(self._on_slider_change, names="value")

        for slider in self.sliders.values():
            slider.observe(self._on_slider_change, names="value")

        # Setup zoom widgets and nuclear toggle
        self._setup_extra_widgets()

        # Controls bar with flex wrap for narrow screens
        controls = widgets.Box(
            [
                self.reset_button,
                self.normalize_dropdown,
                self.cmap_dropdown,
                self.nuclear_toggle,
                self.reset_zoom_button,
                widgets.HTML("<span style='color:#888; margin: 0 5px;'>|</span>"),
                self.columns_dropdown,
                self.layout_dropdown,
                widgets.HTML("<span style='color:#888; margin: 0 5px;'>|</span>"),
                self.segment_button,
                self.show_masks_toggle,
                self.seg_status_label,
            ],
            layout=widgets.Layout(
                display="flex",
                flex_flow="row wrap",
                gap="10px",
                align_items="center",
                margin="0 0 10px 0",
            ),
        )

        # Create slider container (will be reorganized dynamically)
        self._slider_container = widgets.Box(
            layout=widgets.Layout(
                display="flex",
                flex_flow="row wrap",
                gap="5px",
                min_width="200px",
                flex="1 1 auto",
            )
        )
        self._update_slider_layout()

        # Preview/zoom container
        self._preview_container = widgets.Box(
            [self.output, self._zoom_output],
            layout=widgets.Layout(
                display="flex",
                flex_flow="row wrap",
                gap="10px",
                justify_content="flex-start",
                align_items="flex-start",
                flex="0 0 auto",
            ),
        )

        # Main content area with responsive flex layout
        # Sliders and preview will wrap to stack vertically when narrow
        self._main_content = widgets.Box(
            [self._slider_container, self._preview_container],
            layout=widgets.Layout(
                display="flex",
                flex_flow="row wrap",
                gap="20px",
                align_items="flex-start",
                width="100%",
            ),
        )

        # Width tracker for responsive updates
        self._width_tracker = widgets.HTML(
            value="",
            layout=widgets.Layout(display="none"),
        )

        self.main_widget = widgets.VBox(
            [
                widgets.HTML("<h3>Pseudochannel Weight Tuning</h3>"),
                controls,
                self._main_content,
                self._width_tracker,
            ],
            layout=widgets.Layout(width="100%"),
        )

    def _on_slider_change(self, change):
        """Handle slider value changes with debouncing."""
        current_time = time.time() * 1000

        if current_time - self._last_update_time > self.debounce_ms:
            self._update_preview()
            self._last_update_time = current_time

    def _on_reset(self, button):
        """Reset all sliders to zero."""
        for slider in self.sliders.values():
            slider.value = 0.0
        self._update_preview()

    def _update_slider_layout(self, n_cols: int = None):
        """Update slider layout with specified number of columns.

        Args:
            n_cols: Number of columns. If None, auto-detect based on slider count.
        """
        sliders = list(self.sliders.values())
        n_sliders = len(sliders)

        # Auto-detect columns if not specified
        if n_cols is None:
            if n_sliders >= 30:
                n_cols = 4
            elif n_sliders >= 20:
                n_cols = 3
            elif n_sliders >= 10:
                n_cols = 2
            else:
                n_cols = 1

        self._current_slider_cols = n_cols

        # Use fixed slider width - ipywidgets doesn't handle percentage widths well
        slider_width = "280px"
        for slider in sliders:
            slider.layout.width = slider_width

        # Distribute sliders evenly across columns
        sliders_per_col = (n_sliders + n_cols - 1) // n_cols
        columns = []
        for i in range(n_cols):
            start = i * sliders_per_col
            end = min(start + sliders_per_col, n_sliders)
            col_sliders = sliders[start:end]
            if col_sliders:
                col = widgets.VBox(
                    col_sliders,
                    layout=widgets.Layout(padding="5px"),
                )
                columns.append(col)

        # Update the slider container
        self._slider_container.children = columns

    def set_layout_columns(self, n_cols: int):
        """Manually set the number of slider columns.

        Args:
            n_cols: Number of columns (1-6)
        """
        n_cols = max(1, min(6, n_cols))
        self._update_slider_layout(n_cols)

    def set_layout_mode(self, mode: str):
        """Set the layout mode.

        Args:
            mode: One of:
                - "horizontal": Sliders beside preview (default for wide screens)
                - "vertical": Sliders above preview (better for narrow screens)
                - "auto": Automatically choose based on content
        """
        if mode == "vertical":
            self._main_content.layout.flex_flow = "column"
            self._slider_container.layout.width = "100%"
            self._preview_container.layout.width = "100%"
        elif mode == "horizontal":
            self._main_content.layout.flex_flow = "row wrap"
            self._slider_container.layout.width = "auto"
            self._preview_container.layout.width = "auto"
        else:  # auto
            self._main_content.layout.flex_flow = "row wrap"
            self._slider_container.layout.width = "auto"
            self._preview_container.layout.width = "auto"

    def _setup_extra_widgets(self):
        """Create zoom output widget, reset button, nuclear toggle, and layout controls."""
        self._zoom_output = widgets.Output()

        self.reset_zoom_button = widgets.Button(
            description="Reset Zoom",
            button_style="info",
            icon="search-minus",
        )
        self.reset_zoom_button.on_click(self._on_reset_zoom)

        # Nuclear marker overlay toggle
        has_nuclear = self.nuclear_marker is not None
        self.nuclear_toggle = widgets.Checkbox(
            value=False,
            description="Show Nuclear (DAPI)",
            indent=False,
            disabled=not has_nuclear,
            layout=widgets.Layout(width="180px"),
        )
        if not has_nuclear:
            self.nuclear_toggle.description = "Nuclear N/A"
        self.nuclear_toggle.observe(self._on_slider_change, names="value")

        # Layout control widgets
        n_sliders = len(self.channel_names)
        max_cols = min(6, max(1, n_sliders // 5))
        default_cols = 3 if n_sliders >= 20 else (2 if n_sliders >= 10 else 1)

        self.columns_dropdown = widgets.Dropdown(
            options=[(str(i), i) for i in range(1, max_cols + 1)],
            value=default_cols,
            description="Columns:",
            style={"description_width": "60px"},
            layout=widgets.Layout(width="120px"),
        )
        self.columns_dropdown.observe(self._on_columns_change, names="value")

        self.layout_dropdown = widgets.Dropdown(
            options=[
                ("Auto", "auto"),
                ("Side by side", "horizontal"),
                ("Stacked", "vertical"),
            ],
            value="auto",
            description="Layout:",
            style={"description_width": "50px"},
            layout=widgets.Layout(width="140px"),
        )
        self.layout_dropdown.observe(self._on_layout_change, names="value")

        # Segmentation widgets
        self.segment_button = widgets.Button(
            description="Segment",
            button_style="success",
            icon="crosshairs",
            disabled=True,
        )
        self.segment_button.on_click(self._on_segment_click)

        self.show_masks_toggle = widgets.Checkbox(
            value=False,
            description="Show Masks",
            indent=False,
            disabled=True,
            layout=widgets.Layout(width="140px"),
        )
        self.show_masks_toggle.observe(self._on_toggle_masks, names="value")

        self.seg_status_label = widgets.HTML(
            value="",
            layout=widgets.Layout(width="200px"),
        )

    def _on_columns_change(self, change):
        """Handle column count change."""
        self._update_slider_layout(change["new"])

    def _on_layout_change(self, change):
        """Handle layout mode change."""
        self.set_layout_mode(change["new"])

    def _on_rectangle_select(self, eclick, erelease):
        """Handle rectangle selection callback from RectangleSelector."""
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata

        # Ensure coordinates are in correct order
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)

        self._zoom_region = (x1, y1, x2, y2)
        self.segment_button.disabled = False
        self._update_zoom_view()

    def _preview_to_full_coords(self, preview_coords: tuple) -> tuple:
        """Map preview coordinates to full-resolution coordinates.

        Args:
            preview_coords: (x1, y1, x2, y2) in preview coordinate space

        Returns:
            (x1, y1, x2, y2) in full-resolution coordinate space
        """
        x1, y1, x2, y2 = preview_coords

        # Get original image dimensions from first channel
        first_channel = next(iter(self.channels.values()))
        orig_height, orig_width = first_channel.shape[:2]

        # Get preview dimensions
        preview_shape = next(iter(self.previews.values())).shape
        preview_height, preview_width = preview_shape[:2]

        # Calculate scale factors
        scale_x = orig_width / preview_width
        scale_y = orig_height / preview_height

        # Map coordinates
        full_x1 = int(x1 * scale_x)
        full_y1 = int(y1 * scale_y)
        full_x2 = int(x2 * scale_x)
        full_y2 = int(y2 * scale_y)

        # Clamp to image bounds
        full_x1 = max(0, min(full_x1, orig_width))
        full_x2 = max(0, min(full_x2, orig_width))
        full_y1 = max(0, min(full_y1, orig_height))
        full_y2 = max(0, min(full_y2, orig_height))

        return (full_x1, full_y1, full_x2, full_y2)

    def _create_zoom_figure(self):
        """Create the zoom figure once (lazily on first zoom)."""
        with self._zoom_output:
            self._zoom_fig, self._zoom_ax = plt.subplots(figsize=(6, 6))
            self._zoom_fig.canvas.header_visible = False
            self._zoom_ax.axis("off")
            plt.tight_layout()
            plt.show()

    def _get_zoom_segmentation_inputs(self):
        """Compute pseudochannel and nuclear arrays for the current zoom region.

        Returns:
            Tuple of (pseudochannel, nuclear_or_None, step) where
            pseudochannel is (H, W) float32 in [0, 1],
            nuclear is (H, W) float32 in [0, 1] or None,
            and step is the downsample stride used.
        """
        full_coords = self._preview_to_full_coords(self._zoom_region)
        x1, y1, x2, y2 = full_coords

        region_height = y2 - y1
        region_width = x2 - x1
        max_dim = max(region_height, region_width)

        if max_dim > self.max_zoom_size:
            scale = self.max_zoom_size / max_dim
            step = max(1, int(1 / scale))
        else:
            step = 1

        weights = get_weights_from_sliders(self.sliders)
        active_weights = {k: v for k, v in weights.items() if v > 0}

        zoom_channels = {}
        for name in active_weights:
            if name in self.channels:
                region = self.channels[name][y1:y2:step, x1:x2:step]
                zoom_channels[name] = region.astype(np.float32)

        normalize = self.normalize_dropdown.value

        if zoom_channels:
            pseudochannel = compute_pseudochannel(
                zoom_channels,
                active_weights,
                normalize=normalize,
            )
        else:
            pseudochannel = np.zeros(
                ((y2 - y1) // step or 1, (x2 - x1) // step or 1),
                dtype=np.float32,
            )

        nuclear = None
        if self.nuclear_marker is not None:
            nuclear = self._normalize_nuclear(
                self.nuclear_marker[y1:y2:step, x1:x2:step]
            )

        return pseudochannel, nuclear, step

    def _compute_weights_hash(self) -> int:
        """Hash of active weights + normalization for cache invalidation."""
        weights = get_weights_from_sliders(self.sliders)
        active = tuple(sorted((k, v) for k, v in weights.items() if v > 0))
        return hash((active, self.normalize_dropdown.value))

    def _invalidate_segmentation(self):
        """Clear all cached segmentation data."""
        self._segmentation_masks = None
        self._mask_contours_data = []
        self._clear_mask_contours()
        self._seg_zoom_region = None
        self._seg_weights_hash = None
        self.show_masks_toggle.value = False
        self.show_masks_toggle.disabled = True
        self.seg_status_label.value = ""

    def _update_zoom_view(self):
        """Compute and display zoomed region with optimizations."""
        if self._zoom_region is None:
            return

        # Create zoom figure if it doesn't exist
        if self._zoom_fig is None:
            self._create_zoom_figure()

        pseudochannel, nuclear, step = self._get_zoom_segmentation_inputs()

        # Check if cached segmentation is still valid
        current_hash = self._compute_weights_hash()
        cache_valid = (
            self._segmentation_masks is not None
            and self._seg_zoom_region == self._zoom_region
            and self._seg_weights_hash == current_hash
        )
        if not cache_valid and self._segmentation_masks is not None:
            self._invalidate_segmentation()

        # Determine if we need RGB or grayscale
        use_rgb = (
            self.nuclear_toggle.value
            and self.nuclear_preview is not None
        )

        if use_rgb:
            pseudo_norm = pseudochannel
            nuclear_norm = nuclear
            data = np.stack([pseudo_norm, pseudo_norm, nuclear_norm], axis=-1)
        else:
            data = pseudochannel

        # Always recreate image when data shape changes (new rectangle)
        # This ensures clean updates
        self._zoom_ax.clear()
        self._zoom_ax.axis("off")
        if use_rgb:
            self._zoom_img = self._zoom_ax.imshow(data)
        else:
            self._zoom_img = self._zoom_ax.imshow(
                data, cmap=self.cmap_dropdown.value, vmin=0, vmax=1
            )
        self._zoom_is_rgb = use_rgb

        # Re-overlay cached contours if still valid (ax.clear() removed them)
        if cache_valid and self.show_masks_toggle.value:
            self._mask_contour_lines = overlay_contours_on_axes(
                self._zoom_ax, self._mask_contours_data
            )

        self._zoom_fig.canvas.draw_idle()

    def _on_reset_zoom(self, button):
        """Clear zoom region and reset view."""
        self._zoom_region = None
        self.segment_button.disabled = True
        self._invalidate_segmentation()
        if self._zoom_fig is not None:
            self._zoom_ax.clear()
            self._zoom_ax.axis("off")
            self._zoom_img = None
            self._zoom_fig.canvas.draw_idle()

    def _create_preview_figure(self):
        """Create the preview figure once."""
        with self.output:
            self._preview_fig, self._preview_ax = plt.subplots(figsize=(6, 6))
            self._preview_fig.canvas.header_visible = False
            self._preview_ax.axis("off")

            # Create initial blank image
            preview_shape = next(iter(self.previews.values())).shape
            blank = np.zeros(preview_shape, dtype=np.float32)
            self._preview_img = self._preview_ax.imshow(
                blank, cmap=self.cmap_dropdown.value, vmin=0, vmax=1
            )
            self._preview_is_rgb = False

            # Add RectangleSelector for zoom functionality (only once)
            self._rect_selector = RectangleSelector(
                self._preview_ax,
                self._on_rectangle_select,
                useblit=True,
                button=[1],  # Left click only
                minspanx=5,
                minspany=5,
                interactive=True,
                props=dict(facecolor="cyan", alpha=0.3),
            )

            plt.tight_layout()
            plt.show()

    def _update_preview(self):
        """Update the preview image based on current slider values."""
        weights = get_weights_from_sliders(self.sliders)
        normalize = self.normalize_dropdown.value

        pseudochannel = compute_pseudochannel(
            self.previews,
            weights,
            normalize=normalize,
        )

        # Determine if we need RGB or grayscale
        use_rgb = (
            self.nuclear_toggle.value
            and self.nuclear_preview is not None
        )

        if use_rgb:
            # Create RGB composite: pseudo in red+green (yellow), nuclear in blue
            pseudo_norm = pseudochannel
            nuclear_norm = self._normalize_nuclear(self.nuclear_preview)
            data = np.stack([pseudo_norm, pseudo_norm, nuclear_norm], axis=-1)
        else:
            data = pseudochannel

        # Check if we need to recreate the image (RGB <-> grayscale switch)
        if self._preview_img is None or use_rgb != self._preview_is_rgb:
            # Need to recreate the image
            self._preview_ax.clear()
            self._preview_ax.axis("off")
            if use_rgb:
                self._preview_img = self._preview_ax.imshow(data)
            else:
                self._preview_img = self._preview_ax.imshow(
                    data, cmap=self.cmap_dropdown.value, vmin=0, vmax=1
                )
            self._preview_is_rgb = use_rgb

            # Recreate RectangleSelector after clearing axes
            self._rect_selector = RectangleSelector(
                self._preview_ax,
                self._on_rectangle_select,
                useblit=True,
                button=[1],
                minspanx=5,
                minspany=5,
                interactive=True,
                props=dict(facecolor="cyan", alpha=0.3),
            )
        else:
            # Just update the data
            self._preview_img.set_data(data)
            if not use_rgb:
                self._preview_img.set_cmap(self.cmap_dropdown.value)

        # Refresh the canvas
        self._preview_fig.canvas.draw_idle()

        # Update zoom view if region is selected
        if self._zoom_region is not None:
            self._update_zoom_view()

    # -- Segmentation methods --------------------------------------------------

    def _on_segment_click(self, button):
        """Handle Segment button click."""
        if self._zoom_region is None:
            self.seg_status_label.value = (
                "<span style='color:orange'>Select a zoom region first</span>"
            )
            return

        if not _check_cellpose():
            self.seg_status_label.value = (
                "<span style='color:red'>Cellpose not installed. "
                "Run: pip install cellpose</span>"
            )
            return

        self.segment_button.disabled = True
        self.seg_status_label.value = (
            "<span style='color:#888'>Segmenting...</span>"
        )

        try:
            # Lazy-init model
            if self._cellpose_model is None:
                self._cellpose_model = create_cellpose_model(self._cellpose_config)

            pseudochannel, nuclear, step = self._get_zoom_segmentation_inputs()

            masks = run_segmentation(
                self._cellpose_model, pseudochannel, nuclear, self._cellpose_config
            )
            contours = extract_mask_contours(masks)

            # Cache results
            self._segmentation_masks = masks
            self._mask_contours_data = contours
            self._seg_zoom_region = self._zoom_region
            self._seg_weights_hash = self._compute_weights_hash()

            # Enable and activate mask toggle
            self.show_masks_toggle.disabled = False
            self.show_masks_toggle.value = True

            self._draw_mask_contours()

            n_cells = masks.max()
            self.seg_status_label.value = (
                f"<span style='color:green'>{n_cells} cell{'s' if n_cells != 1 else ''} found</span>"
            )
        except Exception as e:
            self.seg_status_label.value = (
                f"<span style='color:red'>Error: {e}</span>"
            )
        finally:
            self.segment_button.disabled = False

    def _draw_mask_contours(self):
        """Draw cached contours on the zoom axes."""
        self._clear_mask_contours()
        if (
            self.show_masks_toggle.value
            and self._mask_contours_data
            and self._zoom_ax is not None
        ):
            self._mask_contour_lines = overlay_contours_on_axes(
                self._zoom_ax, self._mask_contours_data
            )
            self._zoom_fig.canvas.draw_idle()

    def _clear_mask_contours(self):
        """Remove all contour Line2D artists from the zoom axes."""
        for line in self._mask_contour_lines:
            try:
                line.remove()
            except ValueError:
                pass
        self._mask_contour_lines = []

    def _on_toggle_masks(self, change):
        """Show or hide mask contours."""
        if change["new"]:
            self._draw_mask_contours()
        else:
            self._clear_mask_contours()
            if self._zoom_fig is not None:
                self._zoom_fig.canvas.draw_idle()

    def display(self):
        """Display the interactive widget."""
        display(self.main_widget)
        self._create_preview_figure()
        self._update_preview()

    def get_weights(self) -> dict[str, float]:
        """Get current weight configuration."""
        return get_weights_from_sliders(self.sliders)

    def set_weights(self, weights: dict[str, float]):
        """Set slider values from a weight dict."""
        for name, value in weights.items():
            if name in self.sliders:
                self.sliders[name].value = value
        self._update_preview()

    def close(self):
        """Close figures and clean up resources."""
        if self._preview_fig is not None:
            plt.close(self._preview_fig)
            self._preview_fig = None
            self._preview_ax = None
            self._preview_img = None
        if self._zoom_fig is not None:
            plt.close(self._zoom_fig)
            self._zoom_fig = None
            self._zoom_ax = None
            self._zoom_img = None
        # Clean up segmentation resources
        self._cellpose_model = None
        self._segmentation_masks = None
        self._mask_contour_lines = []
        self._mask_contours_data = []


def create_interactive_explorer(
    channels: Union[str, Path, dict, OMETiffChannels],
    marker_file: Optional[Union[str, Path]] = None,
    marker_column: Optional[Union[int, str]] = None,
    preview_size: int = 512,
    exclude_channels: set[str] | list[str] | None = None,
    nuclear_marker_path: Optional[Union[str, Path]] = None,
    max_zoom_size: int = 1024,
    mcmicro_markers: bool = False,
) -> PseudochannelExplorer:
    """Main function to create and display the interactive explorer.

    Args:
        channels: One of:
            - Path to folder containing individual channel TIFFs
            - Path to OME-TIFF file (requires marker_file)
            - Pre-loaded dict of channel_name -> array
            - OMETiffChannels instance
        marker_file: Path to marker names file (required for OME-TIFF)
        marker_column: Column for CSV/TSV marker file
        preview_size: Size for preview images
        exclude_channels: Channel names to exclude (case-insensitive).
            If None, uses DEFAULT_EXCLUDED_CHANNELS.
            Pass empty set/list to include all channels.
        nuclear_marker_path: Path to nuclear marker TIFF file (e.g., DAPI).
            If provided, enables nuclear overlay toggle. For OME-TIFF input,
            the DAPI channel will be auto-detected if not specified.
        max_zoom_size: Maximum size for zoom region display. Larger regions
            will be downsampled for faster rendering.
        mcmicro_markers: If True, parse marker_file as MCMICRO format
            (with marker_name column and remove column for filtering).

    Returns:
        PseudochannelExplorer instance

    Examples:
        # From channel folder
        explorer = create_interactive_explorer("./data/channels/")

        # From OME-TIFF with marker file
        explorer = create_interactive_explorer(
            "./data/image.ome.tiff",
            marker_file="./data/markers.txt"
        )

        # From OME-TIFF with CSV marker file
        explorer = create_interactive_explorer(
            "./data/image.ome.tiff",
            marker_file="./data/panel.csv",
            marker_column="marker_name"
        )

        # Include all channels (no exclusions)
        explorer = create_interactive_explorer(
            "./data/channels/",
            exclude_channels=[]
        )

        # With nuclear marker overlay (folder input)
        explorer = create_interactive_explorer(
            "./data/channels/",
            nuclear_marker_path="./data/DAPI.tif"
        )
    """
    explorer = PseudochannelExplorer(
        channels,
        marker_file=marker_file,
        marker_column=marker_column,
        preview_size=preview_size,
        exclude_channels=exclude_channels,
        nuclear_marker_path=nuclear_marker_path,
        max_zoom_size=max_zoom_size,
        mcmicro_markers=mcmicro_markers,
    )
    explorer.display()
    return explorer
