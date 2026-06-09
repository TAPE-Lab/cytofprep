# cytofprep

A Python toolkit for CyTOF debarcoding, implementing the Zunder-lab SCD (Single-Cell Debarcoder) algorithm.

## Features

- FCS 3.0/3.1 file reader and writer (no external FCS library required)
- Zunder-style barcode separation and Mahalanobis-distance debarcoding
- Per-well FCS file export compatible with Cytobank
- Diagnostic plots: well counts, separation histogram, yield-vs-cutoff, per-well biaxials and event scatter

## Installation

```bash
# From GitHub (before PyPI release)
pip install git+https://github.com/haoqichen20/cytofprep.git

# From PyPI (once published)
pip install cytofprep
```

## Quick start

```python
from cytofprep import analyse_scd_run, generate_all_plots
from pathlib import Path

scd, counts, written = analyse_scd_run(
    fcs_path="sample.fcs",
    key_path="barcode_key.csv",
    output_dir="results/",
    basename="my_run",
    sep_cutoff=0.1,
    mahal_cutoff=float("inf"),
    write_fcs=True,
    include_unassigned=True,
    memory_map=True,
)

counts.to_csv("results/counts.csv", index=False)

generate_all_plots(scd, "results/plots/", wells=list(range(5)), color_by="mahal", dpi=300)
```

### Barcode key format

The key CSV has no header. Row 0 contains the barcode mass channels; rows 1+ are wells.
Column 0 holds well labels; columns 1+ are the binary barcode matrix (0/1).

```
,89,113,115,139,141,142
A1,1,0,1,0,1,0
A2,0,1,0,1,0,1
...
```

## Requirements

- Python >= 3.10
- numpy >= 1.24
- pandas >= 2.0
- matplotlib >= 3.7

## License

MIT
