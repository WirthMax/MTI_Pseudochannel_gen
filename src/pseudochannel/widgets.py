"""Interactive ipywidgets interface for pseudochannel tuning."""

import time
from pathlib import Path
from typing import Optional, Union

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from matplotlib.widgets import RectangleSelector

from .core import compute_pseudochannel
from .io import load_channel_folder, load_ome_tiff, OMETiffChannels, validate_channels
from .preview import create_preview_stack


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
            style={"description_width": "120px"},
            layout=widgets.Layout(width="400px"),
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
        self._fig = None  # Figure reference
        self._ax = None  # Axes reference

        self.channels = self._load_channels(
            channels, marker_file, marker_column, exclude_channels
        )
        validate_channels(self.channels)

        self.previews = create_preview_stack(self.channels, preview_size)
        self.channel_names = list(self.channels.keys())

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

        # Setup zoom widgets
        self._setup_zoom_widgets()

        controls = widgets.HBox([
            self.reset_button,
            self.normalize_dropdown,
            self.cmap_dropdown,
            self.reset_zoom_button,
        ])

        self.main_widget = widgets.VBox([
            widgets.HTML("<h3>Pseudochannel Weight Tuning</h3>"),
            widgets.HBox([
                slider_box,
                self.output,
            ]),
            controls,
            self._zoom_label,
            self._zoom_output,
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
        if n_sliders >= 30:
            n_cols = 3
        elif n_sliders >= 15:
            n_cols = 2
        else:
            n_cols = 1

        # Adjust slider width for multi-column layout
        slider_width = "250px" if n_cols > 1 else "400px"
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

    def _setup_zoom_widgets(self):
        """Create zoom output widget, reset button, and info label."""
        self._zoom_output = widgets.Output()
        self._zoom_label = widgets.HTML(
            value="<h4>Zoom View</h4><i>Draw a rectangle on the preview to zoom</i>"
        )

        self.reset_zoom_button = widgets.Button(
            description="Reset Zoom",
            button_style="info",
            icon="search-minus",
        )
        self.reset_zoom_button.on_click(self._on_reset_zoom)

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

    def _update_zoom_view(self):
        """Compute and display zoomed region at full resolution."""
        if self._zoom_region is None:
            return

        # Get full-resolution coordinates
        full_coords = self._preview_to_full_coords(self._zoom_region)
        x1, y1, x2, y2 = full_coords

        # Update label with region info
        width = x2 - x1
        height = y2 - y1
        self._zoom_label.value = (
            f"<h4>Zoom View</h4>"
            f"Zoomed Region: ({x1},{y1}) to ({x2},{y2}) - {width}x{height} px"
        )

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

        # Display zoomed region
        with self._zoom_output:
            self._zoom_output.clear_output(wait=True)

            # Determine figure size based on aspect ratio
            aspect = width / height if height > 0 else 1
            fig_height = 6
            fig_width = min(12, fig_height * aspect)

            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            ax.imshow(zoomed_pseudochannel, cmap=self.cmap_dropdown.value)
            ax.set_title(f"Full Resolution Zoom ({width}x{height} px)")
            ax.axis("off")
            plt.tight_layout()
            plt.show()

    def _on_reset_zoom(self, button):
        """Clear zoom region and reset view."""
        self._zoom_region = None
        self._zoom_label.value = (
            "<h4>Zoom View</h4><i>Draw a rectangle on the preview to zoom</i>"
        )
        with self._zoom_output:
            self._zoom_output.clear_output()

    def _update_preview(self):
        """Update the preview image based on current slider values."""
        weights = get_weights_from_sliders(self.sliders)
        normalize = self.normalize_dropdown.value

        pseudochannel = compute_pseudochannel(
            self.previews,
            weights,
            normalize=normalize,
        )

        with self.output:
            self.output.clear_output(wait=True)

            self._fig, self._ax = plt.subplots(figsize=(6, 6))
            self._ax.imshow(pseudochannel, cmap=self.cmap_dropdown.value)
            self._ax.set_title("Pseudochannel Preview (draw rectangle to zoom)")
            self._ax.axis("off")

            # Add RectangleSelector for zoom functionality
            self._rect_selector = RectangleSelector(
                self._ax,
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

        # Update zoom view if region is selected
        if self._zoom_region is not None:
            self._update_zoom_view()

    def display(self):
        """Display the interactive widget."""
        self._update_preview()
        display(self.main_widget)

    def get_weights(self) -> dict[str, float]:
        """Get current weight configuration."""
        return get_weights_from_sliders(self.sliders)

    def set_weights(self, weights: dict[str, float]):
        """Set slider values from a weight dict."""
        for name, value in weights.items():
            if name in self.sliders:
                self.sliders[name].value = value
        self._update_preview()


def create_interactive_explorer(
    channels: Union[str, Path, dict, OMETiffChannels],
    marker_file: Optional[Union[str, Path]] = None,
    marker_column: Optional[Union[int, str]] = None,
    preview_size: int = 512,
    exclude_channels: set[str] | list[str] | None = None,
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
    """
    explorer = PseudochannelExplorer(
        channels,
        marker_file=marker_file,
        marker_column=marker_column,
        preview_size=preview_size,
        exclude_channels=exclude_channels,
    )
    explorer.display()
    return explorer
