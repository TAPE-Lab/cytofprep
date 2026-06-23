# cytofprep

A Python toolkit for CyTOF debarcoding (reimplementation of the Zunder-lab SCD (Single-Cell Debarcoder) algorithm) and downstream data preprocessing (in development).

## **Beta version. Use with care.**

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

## Batch mode (parallel processing)

When you have many FCS files, debarcode them in parallel across CPU cores. Run this as a **`.py` script** (not from a Jupyter notebook) by following these steps: 

**1. Saving the code** as `run_batch.py`.

**2. Prepare the mapping CSV (`fcs_key_map.csv`).** One row per FCS file, with at least an `fcs_file` and a `key_file` column. Paths are relative to `data_dir`:

```
fcs_file,key_file
fcs/experiment1.fcs,keys/key1.csv
fcs/experiment2.fcs,keys/key2.csv
fcs/experiment3.fcs,keys/key3.csv
```

**3. Create `run_batch.py`** with the code above and edit the paths in the `__main__` block (`data_dir`, `out_dir`) and the cutoffs to match your experiment.

**4. Run it from the terminal** as a module:

```bash
python -m run_batch
```

Each input is debarcoded into its own subfolder under `out_dir`, containing the per-well FCS files, a `counts.csv`, and a `plots/` directory.


```python
import os
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from pathlib import Path

import pandas as pd

from cytofprep import analyse_scd_run, generate_all_plots


def debarcode_one(fcs_file, key_file, output_dir, sep_cutoff, mahal_cutoff, generate_plot=True):
    """Debarcode a single FCS file and write per-well FCS files + a counts table."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scd, counts, written = analyse_scd_run(
        fcs_path=fcs_file,
        key_path=key_file,
        output_dir=output_dir,
        basename="",
        sep_cutoff=sep_cutoff,
        mahal_cutoff=mahal_cutoff,
        write_fcs=True,
        include_unassigned=True,
        memory_map=True,
    )
    counts.to_csv(output_dir / "counts.csv", index=False)

    if generate_plot:
        generate_all_plots(scd, output_dir / "plots", wells=list(range(5)), color_by="mahal", dpi=300)

    return output_dir.name, counts


if __name__ == "__main__":
    # ---- Update these paths for your data ----
    data_dir = Path("data")
    out_dir = Path("results")
    sep_cutoff = 0.1
    mahal_cutoff = float("inf")

    mapping = pd.read_csv(data_dir / "fcs_key_map.csv")
    fcs_files = [data_dir / p for p in mapping["fcs_file"]]
    key_files = [data_dir / p for p in mapping["key_file"]]
    # Each input gets its own output subfolder so per-well file names never collide.
    output_dirs = [out_dir / Path(p).stem for p in mapping["fcs_file"]]

    max_workers = min(len(fcs_files), os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for basename, counts in executor.map(
            debarcode_one,
            fcs_files,
            key_files,
            output_dirs,
            repeat(sep_cutoff),
            repeat(mahal_cutoff),
        ):
            print(f"Done: {basename} ({len(counts)} wells)")
```


## Controlling per-well plots with `wells`

`generate_all_plots` always writes 3 run-level plots (`well_counts`, `separation_histogram`, `yield_vs_cutoff`). On top of that, it writes **2 plots per well** (an event scatter and an all-biaxials grid). The `wells` argument decides *which* wells get those per-well plots, so it directly controls how many figures are rendered — and rendering biaxials at `dpi=300` is the slowest part of a batch run.

`wells` accepts:

| Value | Meaning |
| --- | --- |
| `"first"` (default) | Only the first well — 2 per-well plots. |
| `"all"` | Every well — 2 × number-of-barcodes plots. |
| `int` | A single well by 0-based index, e.g. `0`. |
| `str` | A single well by label, e.g. `"A1"`. |
| `list[int \| str]` | A specific subset, e.g. `list(range(5))` or `["A1", "B2"]`. |

The example uses `wells=list(range(5))` (the first 5 wells). Be careful with `"all"`: a 96- or 120-barcode key produces ~190–240 per-well figures **per file**, which dominates runtime. For routine batch QC, a small subset (or `"first"`) is usually enough; switch to `"all"` only when you need to inspect every well. To skip per-well plots entirely, pass `generate_plot=False` to `debarcode_one`.

## Barcode key format

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
