# Batch Processing

Apply tuned pseudochannel weights and segmentation parameters to entire datasets.

## MCMICRO Batch Processing

### Discovering Experiments

`find_mcmicro_experiments()` recursively finds folders with a `background/` directory containing an OME-TIFF and a sibling `markers.csv`:

```python
from pseudochannel import find_mcmicro_experiments

experiments = find_mcmicro_experiments("/data/CRC/")
print(f"Found {len(experiments)} experiments")
```

### Generating Pseudochannels

```python
from pseudochannel import process_mcmicro_batch

output_paths = process_mcmicro_batch(
    root_path="/data/CRC/",
    config_path="configs/membrane_weights.yaml",
    mcmicro_markers=True,
)
```

Output per experiment:
```
experiment/
├── markers.csv
├── background/
│   └── image.ome.tiff
└── pseudochannel/           # Created
    └── pseudochannel.tif
```

### Batch Segmentation

After generating pseudochannels:

```python
from pseudochannel import segment_mcmicro_batch

seg_outputs = segment_mcmicro_batch(
    root_path="/data/CRC/",
    config=explorer.get_cellpose_config(),
    mcmicro_markers=True,
)
```

Or do both in one call:

```python
from pseudochannel import process_and_segment_mcmicro_batch

pseudo_paths, seg_paths = process_and_segment_mcmicro_batch(
    root_path="/data/CRC/",
    config_path="configs/membrane_weights.yaml",
    mcmicro_markers=True,
)
```

Output after segmentation:
```
experiment/
├── pseudochannel/
│   └── pseudochannel.tif
└── segmentation/            # Created
    ├── seg_mask.tif         # uint32 label mask
    └── seg_flows.pkl        # Cellpose flows
```

## Config Sources for Segmentation

| Source | Usage |
|--------|-------|
| Widget explorer | `config=explorer.get_cellpose_config()` |
| YAML file | `config="path/to/config.yaml"` (extracts `cellpose` section) |
| Direct | `config=CellposeConfig(diameter=30, flow_threshold=0.4)` |
| Defaults | `config=None` (auto GPU, cyto3 model) |

## Skip-Existing Behavior

Both `process_mcmicro_batch()` and `segment_mcmicro_batch()` skip already-processed experiments by default. Use `overwrite=True` to recompute.

## General Batch Processing

For non-MCMICRO folder structures:

```python
from pseudochannel import batch_process_directory

batch_process_directory(
    root_path="/data/experiment/",
    config_path="configs/membrane_weights.yaml",
    output_folder="/data/experiment/pseudochannels/"
)
```
