# Configuration Reference

## YAML Config Format

Pseudochannel weights and Cellpose parameters are stored in YAML files:

```yaml
name: membrane
description: CD45 + CD3 blend for immune cell boundaries

channels:
  CD45: 0.3
  CD3: 0.2
  CD8: 0.15
  pan-CK: 0.25

normalization: minmax
created: "2026-01-28"

cellpose:
  model_type: cyto3
  diameter: 35
  flow_threshold: 0.4
  cellprob_threshold: 0.0
  min_size: 15
```

### Fields

| Field | Required | Description |
|-------|----------|-------------|
| `name` | No | Human-readable name for the config |
| `description` | No | What this config is for |
| `channels` | Yes | Map of marker name → weight (0.0–1.0) |
| `normalization` | No | `"minmax"` or `"percentile"` (default: `"percentile"`) |
| `created` | No | Auto-set by `save_config()` |
| `cellpose` | No | Cellpose parameters (see below) |

## Saving and Loading

```python
from pseudochannel import save_config, load_config

# Save
save_config(
    weights=explorer.get_weights(),
    output_path="configs/membrane_weights.yaml",
    name="membrane",
    description="CD45 + CD3 + pan-CK blend",
)

# Load
config = load_config("configs/membrane_weights.yaml")
weights = config["channels"]
```

## CellposeConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_type` | str | `"cyto3"` | Cellpose model. Options: `"cyto3"`, `"cyto2"`, `"cyto"`, `"nuclei"` |
| `diameter` | float \| None | `None` | Expected cell diameter in pixels. `None` = auto-estimate |
| `flow_threshold` | float | `0.4` | Flow error threshold. Lower = stricter, fewer fragmented cells (0–1) |
| `cellprob_threshold` | float | `0.0` | Cell probability threshold. Higher = fewer cells, high-confidence only (-6 to 6) |
| `gpu` | bool \| None | `None` | Use GPU. `None` = auto-detect CUDA |
| `min_size` | int | `15` | Minimum cell size in pixels. Smaller cells are removed |

### Creating a CellposeConfig

```python
from pseudochannel import CellposeConfig

config = CellposeConfig(
    model_type="cyto3",
    diameter=35,
    flow_threshold=0.4,
    cellprob_threshold=0.0,
    min_size=15,
)
```

### From the Widget

```python
config = explorer.get_cellpose_config()
```

### From a YAML File

When you pass a YAML file to `segment_mcmicro_batch()`, the `cellpose` section is extracted automatically:

```python
seg_outputs = segment_mcmicro_batch(
    root_path="/data/CRC/",
    config="configs/membrane_weights.yaml",  # Reads cellpose section
)
```

## Example Config

A minimal example is included at `configs/example_config.yaml`:

```yaml
name: membrane_composite
description: Membrane markers for cell segmentation.
channels:
  CD45: 0.3
  E-cadherin: 0.5
  Pan-CK: 0.2
normalization: minmax
created: "2026-01-28"
```
