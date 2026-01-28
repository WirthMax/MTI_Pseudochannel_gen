"""Interactive ipywidgets interface for pseudochannel tuning."""

import time
from pathlib import Path
from typing import Optional, Union

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display

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

        slider_box = widgets.VBox(
            list(self.sliders.values()),
            layout=widgets.Layout(padding="10px"),
        )

        controls = widgets.HBox([
            self.reset_button,
            self.normalize_dropdown,
            self.cmap_dropdown,
        ])

        self.main_widget = widgets.VBox([
            widgets.HTML("<h3>Pseudochannel Weight Tuning</h3>"),
            widgets.HBox([
                slider_box,
                self.output,
            ]),
            controls,
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

            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(pseudochannel, cmap=self.cmap_dropdown.value)
            ax.set_title("Pseudochannel Preview")
            ax.axis("off")
            plt.tight_layout()
            plt.show()

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
