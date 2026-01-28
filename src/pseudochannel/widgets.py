"""Interactive ipywidgets interface for pseudochannel tuning."""

import time
from pathlib import Path
from typing import Optional, Union

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from IPython.display import display
from matplotlib.widgets import RectangleSelector

from .core import compute_pseudochannel
from .io import load_channel_folder, load_ome_tiff, OMETiffChannels, validate_channels
from .preview import create_preview_stack, downsample_image


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
        """
        self.preview_size = preview_size
        self.debounce_ms = debounce_ms
        self.exclude_channels = exclude_channels

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

        # Store original channels input and marker info for OME-TIFF auto-detection
        self._original_channels_input = channels
        self._marker_file = marker_file
        self._marker_column = marker_column

        self.channels = self._load_channels(
            channels, marker_file, marker_column, exclude_channels
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
            return load_ome_tiff(
                path, marker_file, marker_column, exclude_channels=exclude_channels
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
                # Extract from the underlying data
                if self.channels._data.ndim == 3:
                    self.nuclear_marker = np.array(self.channels._data[i])
                else:
                    self.nuclear_marker = np.array(
                        self.channels._data[i, self.channels._z_slice]
                    )
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
        """Create all widget components."""
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

        # Organize sliders into columns based on count
        slider_box = self._create_slider_columns(list(self.sliders.values()))

        # Setup zoom widgets and nuclear toggle
        self._setup_extra_widgets()

        controls = widgets.HBox([
            self.reset_button,
            self.normalize_dropdown,
            self.cmap_dropdown,
            self.nuclear_toggle,
            self.reset_zoom_button,
        ])

        # Main row: sliders | preview | zoom (aligned at top)
        main_row = widgets.HBox(
            [slider_box, self.output, self._zoom_output],
            layout=widgets.Layout(align_items="flex-start"),
        )

        self.main_widget = widgets.VBox([
            widgets.HTML("<h3>Pseudochannel Weight Tuning</h3>"),
            controls,
            main_row,
        ])

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

    def _create_slider_columns(self, sliders: list) -> widgets.HBox:
        """Distribute sliders across multiple columns based on count.

        Args:
            sliders: List of slider widgets

        Returns:
            HBox containing VBox columns of sliders
        """
        n_sliders = len(sliders)

        # Determine number of columns based on slider count
        if n_sliders >= 20:
            n_cols = 3
        elif n_sliders >= 10:
            n_cols = 2
        else:
            n_cols = 1

        # Adjust slider width for multi-column layout
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
            columns.append(
                widgets.VBox(col_sliders, layout=widgets.Layout(padding="5px"))
            )

        return widgets.HBox(columns, layout=widgets.Layout(padding="10px"))

    def _setup_extra_widgets(self):
        """Create zoom output widget, reset button, and nuclear toggle."""
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

    def _on_rectangle_select(self, eclick, erelease):
        """Handle rectangle selection callback from RectangleSelector."""
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata

        # Ensure coordinates are in correct order
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)

        self._zoom_region = (x1, y1, x2, y2)
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
            self._zoom_ax.axis("off")
            plt.tight_layout()
            plt.show()

    def _update_zoom_view(self):
        """Compute and display zoomed region at full resolution."""
        if self._zoom_region is None:
            return

        # Create zoom figure if it doesn't exist
        if self._zoom_fig is None:
            self._create_zoom_figure()

        # Get full-resolution coordinates
        full_coords = self._preview_to_full_coords(self._zoom_region)
        x1, y1, x2, y2 = full_coords

        # Extract full-resolution region from each channel
        zoom_channels = {}
        for name, channel in self.channels.items():
            zoom_channels[name] = channel[y1:y2, x1:x2]

        # Compute pseudochannel for zoomed region
        weights = get_weights_from_sliders(self.sliders)
        normalize = self.normalize_dropdown.value

        zoomed_pseudochannel = compute_pseudochannel(
            zoom_channels,
            weights,
            normalize=normalize,
        )

        # Determine if we need RGB or grayscale
        use_rgb = (
            self.nuclear_toggle.value
            and self.nuclear_marker is not None
        )

        if use_rgb:
            nuclear_region = self.nuclear_marker[y1:y2, x1:x2]
            pseudo_norm = zoomed_pseudochannel
            nuclear_norm = self._normalize_nuclear(nuclear_region)
            data = np.stack([pseudo_norm, pseudo_norm, nuclear_norm], axis=-1)
        else:
            data = zoomed_pseudochannel

        # Check if we need to recreate the image (RGB <-> grayscale switch or first time)
        if self._zoom_img is None or use_rgb != self._zoom_is_rgb:
            self._zoom_ax.clear()
            self._zoom_ax.axis("off")
            if use_rgb:
                self._zoom_img = self._zoom_ax.imshow(data)
            else:
                self._zoom_img = self._zoom_ax.imshow(
                    data, cmap=self.cmap_dropdown.value, vmin=0, vmax=1
                )
            self._zoom_is_rgb = use_rgb
        else:
            self._zoom_img.set_data(data)
            if not use_rgb:
                self._zoom_img.set_cmap(self.cmap_dropdown.value)
            # Update extent for new region size
            self._zoom_img.set_extent([0, data.shape[1], data.shape[0], 0])

        self._zoom_fig.canvas.draw_idle()

    def _on_reset_zoom(self, button):
        """Clear zoom region and reset view."""
        self._zoom_region = None
        if self._zoom_fig is not None:
            self._zoom_ax.clear()
            self._zoom_ax.axis("off")
            self._zoom_img = None
            self._zoom_fig.canvas.draw_idle()

    def _create_preview_figure(self):
        """Create the preview figure once."""
        with self.output:
            self._preview_fig, self._preview_ax = plt.subplots(figsize=(6, 6))
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


def create_interactive_explorer(
    channels: Union[str, Path, dict, OMETiffChannels],
    marker_file: Optional[Union[str, Path]] = None,
    marker_column: Optional[Union[int, str]] = None,
    preview_size: int = 512,
    exclude_channels: set[str] | list[str] | None = None,
    nuclear_marker_path: Optional[Union[str, Path]] = None,
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
    )
    explorer.display()
    return explorer
