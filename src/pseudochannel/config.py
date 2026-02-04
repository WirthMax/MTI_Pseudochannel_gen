"""Configuration save/load utilities for pseudochannel weights."""

from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import yaml


def save_config(
    weights: dict[str, float],
    output_path: Union[str, Path],
    name: Optional[str] = None,
    description: Optional[str] = None,
    normalization: str = "minmax",
    extra_sections: Optional[dict] = None,
) -> Path:
    """Save weight configuration to YAML file.

    Args:
        weights: Dict of channel_name -> weight
        output_path: Path for output YAML file
        name: Optional name for the configuration
        description: Optional description
        normalization: Normalization method used
        extra_sections: Optional dict of extra top-level sections to merge
            into the config (e.g. ``{"cellpose": {...}}``).

    Returns:
        Path to saved config file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    active_weights = {k: v for k, v in weights.items() if v != 0}

    config = {
        "name": name or output_path.stem,
        "description": description or "",
        "channels": active_weights,
        "normalization": normalization,
        "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    if extra_sections:
        config.update(extra_sections)

    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    return output_path


def load_config(config_path: Union[str, Path]) -> dict:
    """Load weight configuration from YAML file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Dict with config data including:
        - name: Config name
        - description: Config description
        - channels: Dict of channel weights
        - normalization: Normalization method
        - created: Creation timestamp

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Invalid config format in {config_path}")

    if "channels" not in config:
        raise ValueError(f"Config missing 'channels' key: {config_path}")

    return config


def get_weights_from_config(config: dict) -> dict[str, float]:
    """Extract weights dict from loaded config.

    Args:
        config: Config dict from load_config

    Returns:
        Dict of channel_name -> weight
    """
    return config.get("channels", {})


def get_normalization_from_config(config: dict) -> str:
    """Extract normalization method from loaded config.

    Args:
        config: Config dict from load_config

    Returns:
        Normalization method string
    """
    return config.get("normalization", "minmax")


def list_configs(config_dir: Union[str, Path]) -> list[dict]:
    """List all config files in a directory with their metadata.

    Args:
        config_dir: Directory containing config files

    Returns:
        List of dicts with config info (path, name, description, created)
    """
    config_dir = Path(config_dir)

    if not config_dir.exists():
        return []

    configs = []

    for yaml_file in sorted(config_dir.glob("*.yaml")):
        try:
            config = load_config(yaml_file)
            configs.append({
                "path": yaml_file,
                "name": config.get("name", yaml_file.stem),
                "description": config.get("description", ""),
                "created": config.get("created", ""),
                "num_channels": len(config.get("channels", {})),
            })
        except (ValueError, yaml.YAMLError):
            continue

    return configs


def validate_config_channels(
    config: dict,
    available_channels: list[str],
) -> tuple[bool, list[str]]:
    """Check if config channels match available channels.

    Args:
        config: Config dict from load_config
        available_channels: List of available channel names

    Returns:
        Tuple of (is_valid, missing_channels)
    """
    config_channels = set(config.get("channels", {}).keys())
    available = set(available_channels)

    missing = config_channels - available

    return len(missing) == 0, list(missing)


def load_cellpose_config(config: dict) -> Optional[dict]:
    """Extract the cellpose section from a loaded config.

    Args:
        config: Config dict from load_config().

    Returns:
        Cellpose parameter dict, or None if not present.
    """
    return config.get("cellpose")


def save_cellpose_config(config: dict, cellpose_params: dict) -> dict:
    """Merge a cellpose section into a config dict (in-memory).

    Args:
        config: Existing config dict.
        cellpose_params: Cellpose parameter dict to merge.

    Returns:
        Updated config dict (same object, mutated).
    """
    config["cellpose"] = cellpose_params
    return config
