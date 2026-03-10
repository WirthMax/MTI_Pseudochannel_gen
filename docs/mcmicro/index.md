# MCMICRO Pipeline

!!! info "Scripts live on the MCMICRO branches"

    The MCMICRO pipeline integration scripts (ASHLAR registration, DAPI swap, batch shell scripts) are maintained on dedicated branches:

    - `MCMICRO_run_scripts_Claassenlab` — shell scripts for running MCMICRO on MACSima data on DENBI servers

    These branches add `src/run_MCMICRO_general.sh`, `src/batch_swap_dapi.sh`, and `src/swap_dapi_channel.py`.

    Switch to the corresponding version of this documentation (when available) to see full MCMICRO pipeline docs.

## MCMICRO Format Support on `main`

The `main` branch supports **reading and writing** MCMICRO-style folder structures for pseudochannel generation and segmentation:

```
experiment/
├── markers.csv          # MCMICRO format (marker_name, channel_number, cycle_number, remove)
├── background/          # backsub output
│   └── *.ome.tiff
├── pseudochannel/       # Created by process_mcmicro_batch()
│   └── pseudochannel.tif
└── segmentation/        # Created by segment_mcmicro_batch()
    ├── seg_mask.tif
    └── seg_flows.pkl
```

See [Batch Processing](../user-guide/batch.md) for usage details.
