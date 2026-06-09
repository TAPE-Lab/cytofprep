from __future__ import annotations
import re
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from .scd_zunder_python import SCD, bmtrans

ColorBy = Literal["mahal", "separation"]
GUI_GREEN = (0.0, 0.5, 0.4)
BAR_EDGE = (0.15, 0.25, 0.2)  # dark green; or use "black"
BAR_EDGEWIDTH = 0.5

def _asinh_ticks(cofactor: float = 10.0) -> tuple[np.ndarray, list[str]]:
    raw = np.array([0.0, 10.0, 100.0, 1000.0, 10000.0], dtype=float)
    positions = bmtrans(raw, cofactor, out_dtype=np.float64)
    labels = ["0", "10", "100", "1k", "10k"]
    return positions, labels


def _asinh_xlim(cofactor: float = 10.0) -> tuple[float, float]:
    lo, hi = bmtrans(np.array([-10.0, 10000.0]), cofactor, out_dtype=np.float64)
    return float(lo), float(hi)


def _channel_legend(scd: SCD) -> list[str]:
    if scd.bc_cols is None or scd.c is None:
        return list(scd.masses)
    labels: list[str] = []
    for col in scd.bc_cols:
        marker = ""
        if scd.m is not None and col < len(scd.m):
            marker = str(scd.m[col]).strip()
        if marker:
            labels.append(marker)
        else:
            labels.append(str(scd.c[col]))
    return labels


def _resolve_well(scd: SCD, well: int | str) -> int:
    if isinstance(well, int):
        if well < 0 or well >= scd.num_codes:
            raise ValueError(f"Well index {well} out of range [0, {scd.num_codes - 1}]")
        return well
    if well in scd.well_labels:
        return scd.well_labels.index(well)
    raise ValueError(f"No barcode with label {well!r}")


def _well_mask(scd: SCD, well_idx: int) -> np.ndarray:
    if scd.bcind is None or scd.deltas is None or scd.mahal is None:
        raise ValueError("Run debarcoding and Mahalanobis calculation before plotting.")
    return (
        (scd.bcind == (well_idx + 1))
        & (scd.mahal < scd.mahal_cutoff_val)
        & (scd.deltas > scd.sep_cutoff)
    )


def _well_yields(scd: SCD) -> np.ndarray:
    if scd.bcind is None or scd.deltas is None or scd.mahal is None:
        raise ValueError("Run debarcoding and Mahalanobis calculation before plotting.")
    within = (scd.mahal < scd.mahal_cutoff_val) & (scd.deltas > scd.sep_cutoff)
    yields = np.zeros(scd.num_codes, dtype=float)
    for i in range(scd.num_codes):
        yields[i] = scd.sample_ratio * np.count_nonzero((scd.bcind == (i + 1)) & within)
    return yields


def _resolve_wells(scd: SCD, wells: str | int | list[int | str]) -> list[int]:
    if wells == "first":
        return [0] if scd.num_codes else []
    if wells == "all":
        return list(range(scd.num_codes))
    if isinstance(wells, int):
        return [_resolve_well(scd, wells)]
    if isinstance(wells, str):
        return [_resolve_well(scd, wells)]
    return [_resolve_well(scd, w) for w in wells]


def _validate_color_by(color_by: str) -> ColorBy:
    if color_by not in ("mahal", "separation"):
        raise ValueError(f"color_by must be 'mahal' or 'separation', got {color_by!r}")
    return color_by  # type: ignore[return-value]


def _color_values(scd: SCD, mask: np.ndarray, color_by: ColorBy) -> np.ndarray:
    if color_by == "mahal":
        return scd.mahal[mask]
    return scd.deltas[mask]


def _colorbar_label(color_by: ColorBy) -> str:
    return "Mahalanobis Distance" if color_by == "mahal" else "Separation"


def _biaxial_norm_and_cmap(
    scd: SCD,
    color_by: ColorBy,
    colors: np.ndarray,
) -> tuple[Normalize, object]:
    if color_by == "mahal":
        cmap = plt.cm.jet_r
        vmax = scd.mahal_p99
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = float(max(np.nanmax(colors), 1.0)) if colors.size else 1.0
        return Normalize(0.0, vmax), cmap
    return Normalize(scd.sep_cutoff, 1.0), plt.cm.jet


def plot_well_counts(scd: SCD) -> tuple[Figure, Axes]:
    yields = _well_yields(scd)
    n = scd.num_codes
    # ~0.35–0.45 in per well is usually enough for short barcode labels at 315°
    inches_per_well = 0.3
    fig_width = max(10.0, n * inches_per_well)
    fig_height = 0.2 * fig_width
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    x = np.arange(1, n + 1)
    ax.bar(x, yields, color=GUI_GREEN, edgecolor=BAR_EDGE, linewidth=BAR_EDGEWIDTH)
    ax.set_ylabel("Cell count")
    ax.set_xlim(0, n + 1)
    ax.set_xticks([])
    yl = ax.get_ylim()
    label_y = -yl[1] / 15.0
    for i, label in enumerate(scd.well_labels):
        ax.text(i + 1, label_y, label, rotation=315, ha="left", va="top", fontsize=8)
    total_events = scd.x.shape[0] if scd.x is not None else 0
    pct = round(100 * yields.sum() / total_events) if total_events else 0
    ax.set_title(f"Barcode Yields with Current Filters: {pct}% assigned")
    # Extra bottom space for rotated labels
    fig.subplots_adjust(bottom=0.22)
    return fig, ax


def plot_separation_histogram(scd: SCD) -> tuple[Figure, Axes]:
    if scd.deltas is None:
        raise ValueError("Separation distances are not available.")

    fig, ax = plt.subplots(figsize=(8, 4))
    counts, edges = np.histogram(scd.deltas, bins=100, range=(0.0, 1.0))
    centers = 0.5 * (edges[:-1] + edges[1:])
    ax.bar(centers, 
           scd.sample_ratio * counts, 
           width=edges[1] - edges[0], 
           color=GUI_GREEN,
           edgecolor=BAR_EDGE,
           linewidth=BAR_EDGEWIDTH,
    )
    ax.set_xlabel("Barcode separation")
    ax.set_ylabel("Event count")
    ax.set_xlim(0, 1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig, ax


def plot_yield_vs_cutoff(scd: SCD) -> tuple[Figure, Axes]:
    if scd.seprange is None or scd.clust_size is None:
        raise ValueError("Run compute_well_abundances() before plotting yield vs cutoff.")

    fig, ax = plt.subplots(figsize=(7, 5))
    cmap = plt.cm.jet_r(np.linspace(0, 1, min(20, scd.num_codes)))
    yields = scd.sample_ratio * scd.clust_size

    for j in range(scd.num_codes):
        ax.plot(
            scd.seprange,
            yields[:, j],
            color=cmap[j % len(cmap)],
            linewidth=1.0,
            label=scd.well_labels[j],
        )

    ax.set_xlabel("Barcode separation threshold")
    ax.set_ylabel("Event yield after debarcoding")
    ax.set_xticks(np.arange(0.0, 1.0 + 1e-9, 0.1))
    ax.set_xlim(0, 1)
    ax.axvline(x=scd.sep_cutoff, color="red", linestyle="--", linewidth=1.5)
    if scd.num_codes <= 20:
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig, ax


def plot_event_scatter(scd: SCD, well: int | str = 0) -> tuple[Figure, np.ndarray]:
    well_idx = _resolve_well(scd, well)
    mask = _well_mask(scd, well_idx)
    if scd.bcs is None or scd.normbcs is None:
        raise ValueError("Barcode data must be loaded before plotting.")

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    legend_labels = _channel_legend(scd)
    tick_pos, tick_labels = _asinh_ticks(scd.default_cofactor)

    for ax, data, ylabel in zip(
        axes,
        (scd.bcs[mask, :], scd.normbcs[mask, :]),
        ("Barcode intensities", "Rescaled values"),
    ):
        if data.size == 0:
            ax.set_ylabel(ylabel)
            continue
        event_idx = np.arange(1, data.shape[0] + 1)
        for ch in range(data.shape[1]):
            ax.plot(event_idx, data[:, ch], ".", markersize=2, label=legend_labels[ch])
        if ylabel == "Barcode intensities":
            ax.set_yticks(tick_pos)
            ax.set_yticklabels(tick_labels)
        ax.set_ylabel(ylabel)
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)

    axes[-1].set_xlabel("Event")
    title = " ".join(str(v) for v in scd.key[well_idx, :])
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    return fig, axes



def plot_all_biaxials(
    scd: SCD,
    well: int | str = 0,
    color_by: ColorBy = "mahal",
) -> tuple[Figure, np.ndarray]:
    color_by = _validate_color_by(color_by)
    well_idx = _resolve_well(scd, well)
    mask = _well_mask(scd, well_idx)
    if scd.bcs is None:
        raise ValueError("Barcode data must be loaded before plotting.")

    n = scd.num_masses
    grid_n = n + 1
    fig = plt.figure(figsize=(2.2 * grid_n, 2.2 * grid_n))
    gs = GridSpec(
        grid_n,
        grid_n,
        figure=fig,
        wspace=0.08,
        hspace=0.08,
        left=0.05,
        right=0.86,
        top=0.95,
        bottom=0.05,
    )

    xlim = _asinh_xlim(scd.default_cofactor)
    data = scd.bcs[mask, :]
    colors = _color_values(scd, mask, color_by) if data.size else np.array([])
    norm, cmap = _biaxial_norm_and_cmap(scd, color_by, colors)

    for i in range(grid_n):
        for j in range(grid_n):
            ax = fig.add_subplot(gs[i, j])

            if j == 0 and i != n:
                ax.axis("off")
                ax.text(
                    0.5,
                    0.5,
                    scd.masses[i],
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
            elif j <= i and i < n and j > 0:
                if data.size:
                    ax.scatter(
                        data[:, j - 1],
                        data[:, i],
                        s=2,
                        c=colors,
                        cmap=cmap,
                        norm=norm,
                    )
                ax.set_xlim(xlim)
                ax.set_ylim(xlim)
                ax.set_xticks([])
                ax.set_yticks([])
            elif j > 0 and i == j - 1:
                if data.size and np.any(mask):
                    ax.hist(
                        data[:, j - 1],
                        bins=100,
                        color=GUI_GREEN,
                        edgecolor="none",
                    )
                ax.set_xlim(xlim)
                ax.set_xticks([])
                ax.set_yticks([])
            elif i == n and j > 0:
                ax.axis("off")
                ax.text(
                    0.5,
                    0.5,
                    scd.masses[j - 1],
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
            else:
                ax.axis("off")

    if data.size:
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cax = fig.add_axes([0.89, 0.35, 0.02, 0.4])
        cb = fig.colorbar(sm, cax=cax)
        cb.set_label(_colorbar_label(color_by), fontsize=12)

    title = " ".join(str(v) for v in scd.key[well_idx, :])
    fig.suptitle(title, fontweight="bold")
    return fig, gs


def _save_figure(fig: Figure, path: Path, dpi: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def generate_all_plots(
    scd: SCD,
    outdir: str | Path,
    *,
    wells: str | int | list[int | str] = "first",
    color_by: ColorBy = "mahal",
    dpi: int = 150,
) -> list[Path]:
    color_by = _validate_color_by(color_by)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    run_level = [
        ("well_counts", lambda: plot_well_counts(scd)),
        ("separation_histogram", lambda: plot_separation_histogram(scd)),
        ("yield_vs_cutoff", lambda: plot_yield_vs_cutoff(scd)),
    ]
    for name, plot_fn in run_level:
        fig, _ = plot_fn()
        written.append(_save_figure(fig, outdir / f"{name}.png", dpi))

    well_indices = _resolve_wells(scd, wells)
    for well_idx in well_indices:
        label = scd.well_labels[well_idx]
        safe_label = re.sub(r'[\\/:*?"<>|]+', "_", str(label))
        prefix = f"well_{well_idx:03d}_{safe_label}"

        fig, _ = plot_event_scatter(scd, well_idx)
        written.append(_save_figure(fig, outdir / f"{prefix}_event_scatter.png", dpi))

        fig, _ = plot_all_biaxials(scd, well_idx, color_by=color_by)
        written.append(_save_figure(fig, outdir / f"{prefix}_all_biaxials.png", dpi))

    return written