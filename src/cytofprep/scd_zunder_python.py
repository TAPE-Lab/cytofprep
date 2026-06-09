from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence
import argparse
import math
import re

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Minimal FCS 3.0/3.1 reader/writer for CyTOF-style float data
# -----------------------------------------------------------------------------

def _parse_fcs_text(text: bytes) -> dict[str, str]:
    delim = chr(text[0])
    parts = text[1:].decode("latin1").split(delim)
    return {parts[i]: parts[i + 1] for i in range(0, len(parts) - 1, 2)}


def read_fcs_header(filename: str | Path) -> dict:
    filename = Path(filename)
    with filename.open("rb") as f:
        first = f.read(58).decode("ascii", errors="replace")
        fcstype = first[:6]
        if fcstype not in {"FCS3.0", "FCS3.1"}:
            raise ValueError(f"Unsupported FCS version {fcstype!r}")

        text_start = int(first[10:18])
        text_stop = int(first[18:26])
        data_start_header = int(first[26:34])
        data_stop_header = int(first[34:42])

        f.seek(text_start)
        text = f.read(text_stop - text_start + 1)

    hdr = _parse_fcs_text(text)
    n_events = int(hdr["$TOT"])
    n_par = int(hdr["$PAR"])

    data_start = data_start_header or int(hdr.get("$BEGINDATA", 0))
    data_stop = data_stop_header or int(hdr.get("$ENDDATA", 0))

    datatype = hdr.get("$DATATYPE", "F")
    byteorder = hdr.get("$BYTEORD", "1,2,3,4")

    if datatype == "F":
        dtype = np.dtype("<f4" if byteorder == "1,2,3,4" else ">f4")
    elif datatype == "D":
        dtype = np.dtype("<f8" if byteorder == "1,2,3,4" else ">f8")
    else:
        raise NotImplementedError(f"FCS datatype {datatype!r} is not implemented")

    channel_names = [hdr.get(f"$P{i}N", "") for i in range(1, n_par + 1)]
    marker_names = [hdr.get(f"$P{i}S", "") for i in range(1, n_par + 1)]

    return {
        "filename": filename,
        "hdr": hdr,
        "n_events": n_events,
        "n_par": n_par,
        "data_start": data_start,
        "data_stop": data_stop,
        "dtype": dtype,
        "channel_names": channel_names,
        "marker_names": marker_names,
    }


def read_fcs_matrix(
    filename: str | Path,
    *,
    allow_truncated: bool = False,
    dtype: str | np.dtype | None = "float32",
    memory_map: bool = False,
):
    info = read_fcs_header(filename)
    filename = info["filename"]
    n_events = info["n_events"]
    n_par = info["n_par"]
    source_dtype = info["dtype"]
    data_start = info["data_start"]

    expected_values = n_events * n_par
    readable_values = max(0, (filename.stat().st_size - data_start) // source_dtype.itemsize)

    if readable_values < expected_values:
        complete_events = readable_values // n_par
        msg = (
            f"FCS data segment is incomplete: header says {n_events} events x {n_par} parameters, "
            f"but only {complete_events} complete events are readable."
        )
        if not allow_truncated:
            raise ValueError(msg)
        n_events = complete_events
        expected_values = n_events * n_par

    if memory_map:
        data = np.memmap(
            filename,
            dtype=source_dtype,
            mode="r",
            offset=data_start,
            shape=(expected_values,),
        ).reshape((n_events, n_par))
    else:
        data = np.fromfile(
            filename,
            dtype=source_dtype,
            count=expected_values,
            offset=data_start,
        ).reshape((n_events, n_par))
        if dtype is not None:
            data = data.astype(dtype, copy=False)

    return data, info["channel_names"], info["marker_names"], info["hdr"]


def _safe_range_value(col: np.ndarray) -> int:
    if col.size == 0:
        return 1
    mx = np.nanmax(col)
    if not np.isfinite(mx):
        return 1
    return max(1, int(math.ceil(float(mx))))


def _build_fcs_text(
    n_events: int,
    channel_names: list[str],
    marker_names: list[str],
    ranges: list[int],
    filename: str,
    source_hdr: dict | None,
    data_start: int,
    data_end: int,
) -> str:
    parts: list[str] = []

    def add(k, v):
        parts.extend([str(k), str(v)])

    add("$BEGINANALYSIS", 0)
    add("$ENDANALYSIS", 0)
    add("$BEGINSTEXT", 0)
    add("$ENDSTEXT", 0)
    add("$NEXTDATA", 0)
    add("$TOT", n_events)
    add("$PAR", len(channel_names))
    add("FCSversion", 3)
    add("CREATOR", "Zunder SCD Python writer")
    add("FILENAME", Path(filename).name)
    add("GUID", "1.fcs")
    add("ORIGINALGUID", "1.fcs")
    add("$BYTEORD", "4,3,2,1")
    add("$DATATYPE", "F")
    add("$MODE", "L")

    if source_hdr:
        for key in ("$BTIM", "$ETIM", "$CYT", "$DATE", "$CYTSN"):
            value = source_hdr.get(key, "")
            if value != "":
                add(key, value)
        plate = source_hdr.get("PLATE NAME", "")
        if plate != "":
            add("PLATE NAME", plate)
    else:
        add("$CYT", "DVSSCIENCES-CYTOF")

    for i, (chan, mark, rng) in enumerate(zip(channel_names, marker_names, ranges), start=1):
        add(f"$P{i}B", 32)
        add(f"$P{i}N", chan)
        if mark != "":
            add(f"$P{i}S", mark)
        add(f"$P{i}R", rng)
        add(f"$P{i}E", "0,0")

    add("$BEGINDATA", data_start)
    add("$ENDDATA", data_end)

    delim = "\\"
    return delim + delim.join(str(x).replace(delim, " ") for x in parts) + delim


def _fcs_first_line(text_start: int, text_stop: int, data_start: int, data_end: int) -> str:
    ds, de = (data_start, data_end) if data_end <= 99_999_999 else (0, 0)
    return f"FCS3.0    {text_start:>8}{text_stop:>8}{ds:>8}{de:>8}{0:>8}{0:>8}"


def write_fcs_cytobank(
    filename: str | Path,
    data: np.ndarray | None,
    marker_names: list[str],
    channel_names: list[str],
    source_hdr: dict | None = None,
    *,
    row_iter: Iterable[np.ndarray] | None = None,
    n_events: int | None = None,
    ranges: list[int] | None = None,
):
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)

    if data is not None:
        arr = np.asarray(data, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError("data must be events x channels")
        n_events = int(arr.shape[0])
        n_channels = int(arr.shape[1])
        ranges = [_safe_range_value(arr[:, i]) for i in range(n_channels)]
    else:
        if row_iter is None or n_events is None or ranges is None:
            raise ValueError("streaming write requires row_iter, n_events, and ranges")
        n_events = int(n_events)
        n_channels = len(channel_names)

    if len(marker_names) != n_channels or len(channel_names) != n_channels:
        raise ValueError("marker/channel names must match data width")

    text_start = 100
    data_start = 0
    data_end = 0

    for _ in range(30):
        text = _build_fcs_text(
            n_events,
            channel_names,
            marker_names,
            ranges,
            str(filename),
            source_hdr,
            data_start,
            data_end,
        )
        text_stop = text_start + len(text.encode("latin1")) + 100 - 1
        new_data_start = text_stop
        new_data_end = new_data_start + n_events * n_channels * 4
        if new_data_start == data_start and new_data_end == data_end:
            break
        data_start, data_end = new_data_start, new_data_end

    first_line = _fcs_first_line(text_start, text_stop, data_start, data_end)
    header = first_line + (" " * (text_start - len(first_line))) + text
    header = header + (" " * (text_stop - len(header)))

    with filename.open("wb") as f:
        f.write(header.encode("latin1"))
        if data is not None:
            f.write(np.asarray(arr, dtype=">f4").T.tobytes(order="F"))
        else:
            for chunk in row_iter:
                chunk = np.asarray(chunk, dtype=np.float32)
                if chunk.size == 0:
                    continue
                if chunk.ndim != 2 or chunk.shape[1] != n_channels:
                    raise ValueError(f"chunk shape {chunk.shape} does not match (*, {n_channels})")
                f.write(chunk.astype(">f4", copy=False).T.tobytes(order="F"))


# -----------------------------------------------------------------------------
# Zunder MATLAB-style debarcoder
# -----------------------------------------------------------------------------

def bmtrans(x, c=10.0, out_dtype=np.float32):
    y = np.arcsinh(np.asarray(x, dtype=np.float64) / c)
    return y.astype(out_dtype, copy=False) if out_dtype is not None else y


def matlab_prctile(a, q, axis=0):
    # MATLAB prctile compatibility is version-dependent. Hazen usually matches older
    # MATLAB behaviour better than NumPy's default linear method for small populations.
    return np.percentile(a, q, axis=axis, method="hazen")


@dataclass
class SCD:
    key_filename: str | Path
    default_cofactor: float = 10.0
    sep_cutoff: float = 0.0
    mahal_cutoff_val: float = np.inf
    work_dtype: str | np.dtype = "float32"

    masses: list[str] = field(init=False)
    well_labels: list[str] = field(init=False)
    key: np.ndarray = field(init=False)

    x: np.ndarray | None = None
    c: list[str] | None = None
    m: list[str] | None = None
    source_hdr: dict | None = None

    bc_cols: np.ndarray | None = None
    bcs: np.ndarray | None = None
    normbcs: np.ndarray | None = None
    deltas: np.ndarray | None = None
    bcind: np.ndarray | None = None
    mahal: np.ndarray | None = None
    mahal_max: float = np.inf
    mahal_p95: float = np.inf
    mahal_p99: float = np.inf
    sample_ratio: float = 1.0
    well_yield: np.ndarray | None = None
    seprange: np.ndarray | None = None
    clust_size: np.ndarray | None = None

    def __post_init__(self):
        self.work_dtype = np.dtype(self.work_dtype)
        self._load_key(self.key_filename)

    @property
    def num_masses(self) -> int:
        return len(self.masses)

    @property
    def num_codes(self) -> int:
        return len(self.well_labels)

    def _load_key(self, key_filename: str | Path) -> None:
        key_filename = Path(key_filename)
        if key_filename.suffix.lower() != ".csv":
            raise ValueError("Barcode key must be a csv file.")
        if not key_filename.exists():
            raise FileNotFoundError(f"Barcode key filename not found: {key_filename}")

        raw = pd.read_csv(key_filename, header=None, dtype=str).fillna("")
        if raw.shape[0] < 2 or raw.shape[1] < 2:
            raise ValueError("Barcode key CSV must contain masses, well labels, and key matrix.")

        # MATLAB importdata reads first numeric row as masses. Preserve digits but avoid
        # spaces such as ' 89'.
        self.masses = [str(v).strip() for v in raw.iloc[0, 1:].tolist()]
        self.well_labels = [str(v) for v in raw.iloc[1:, 0].tolist()]
        self.key = raw.iloc[1:, 1:].astype(float).astype(int).to_numpy()
        self.well_yield = np.zeros(len(self.well_labels), dtype=float)

    def load_fcs_files(
        self,
        filenames: str | Path | Sequence[str | Path],
        *,
        memory_map: bool = False,
        allow_truncated: bool = False,
    ) -> "SCD":
        if isinstance(filenames, (list, tuple)):
            mats = []
            c = m = hdr = None
            for fn in filenames:
                dat, c, m, hdr = read_fcs_matrix(
                    fn,
                    allow_truncated=allow_truncated,
                    dtype=self.work_dtype,
                    memory_map=False,
                )
                mats.append(dat)
            self.x = np.vstack(mats)
            self.c, self.m, self.source_hdr = c, m, hdr
        else:
            self.x, self.c, self.m, self.source_hdr = read_fcs_matrix(
                filenames,
                allow_truncated=allow_truncated,
                dtype=self.work_dtype,
                memory_map=memory_map,
            )

        self.bc_cols = None
        self.bcs = None
        self.normbcs = None
        self.deltas = None
        self.bcind = None
        self.mahal = None
        self.sample_ratio = 1.0
        self.well_yield = np.zeros(self.num_codes, dtype=float)
        self.seprange = None
        self.clust_size = None
        return self

    def find_bc_cols_by_mass(self) -> "SCD":
        # Deliberately matches the Zunder MATLAB regexp-on-short-channel-name logic:
        # col_i = find(~cellfun(@isempty, regexp(obj.c, obj.masses(i))))
        if self.x is None or self.c is None:
            raise ValueError("An FCS file must be opened before assigning barcode columns.")

        cols: list[int] = []
        for mass in self.masses:
            hits = [i for i, name in enumerate(self.c) if re.search(re.escape(str(mass)), str(name))]
            if len(hits) != 1:
                raise ValueError(f"not all barcode channels found for mass {mass}: hits={hits}")
            cols.append(hits[0])

        self.bc_cols = np.array(cols, dtype=np.int32)
        return self

    def load_bcs(self, sample_size: int | None = None, random_state: int | None = None) -> "SCD":
        if self.x is None:
            raise ValueError("An FCS file must be opened before loading BCs.")
        if self.bc_cols is None:
            raise ValueError("Barcode columns must be found before loading BCs.")

        n = self.x.shape[0]
        if sample_size is not None and n > sample_size:
            rng = np.random.default_rng(random_state)
            idx = rng.choice(n, size=sample_size, replace=False)
            raw = np.asarray(self.x[np.ix_(idx, self.bc_cols)], dtype=self.work_dtype)
            self.sample_ratio = n / sample_size
        else:
            raw = np.asarray(self.x[:, self.bc_cols], dtype=self.work_dtype)
            self.sample_ratio = 1.0

        self.bcs = bmtrans(raw, self.default_cofactor, out_dtype=self.work_dtype)
        return self

    def normalize_by_pop(self, fieldname: str = "bcs") -> "SCD":
        data = getattr(self, fieldname)
        if data is None:
            raise ValueError("Barcodes must be loaded before normalizing.")
        if self.bcind is None:
            raise ValueError("Preliminary assignments are required before normalize_by_pop.")

        normed = np.zeros_like(data, dtype=self.work_dtype)
        for i in range(self.num_codes):
            inbc = self.bcind == (i + 1)
            if np.count_nonzero(inbc) > 1:
                pos_bcs = data[np.ix_(inbc, self.key[i, :] == 1)]
                norm_val = matlab_prctile(pos_bcs.ravel(), 95)
                if np.isfinite(norm_val) and norm_val != 0:
                    normed[inbc, :] = data[inbc, :] / norm_val

        self.normbcs = normed
        return self

    def normalize_bcs(self, fieldname: str = "bcs") -> "SCD":
        data = getattr(self, fieldname)
        if data is None:
            raise ValueError("Barcodes must be loaded before normalizing.")

        percs = matlab_prctile(data, [1, 99], axis=0)
        ranges = percs[1, :] - percs[0, :]
        ranges[ranges == 0] = 1.0
        self.normbcs = ((data - percs[0, :]) / ranges).astype(self.work_dtype, copy=False)
        return self

    def compute_debarcoding(self, fieldname: str = "normbcs") -> "SCD":
        data = getattr(self, fieldname)
        if data is None:
            raise ValueError("Barcodes must be loaded before debarcoding.")
        if self.bcs is None:
            raise ValueError("Raw barcode data must be loaded before debarcoding.")

        N = data.shape[0]
        cutoff = 0.0

        if len(np.unique(self.key.sum(axis=1))) == 1:
            ix = np.argsort(-data, axis=1, kind="stable")
            sorted_data = np.take_along_axis(data, ix, axis=1)
            numdf = int(self.key[0, :].sum())

            lowests = sorted_data[:, numdf - 1].astype(self.work_dtype, copy=True)
            lowest_pos_cols = ix[:, numdf - 1]
            lowests[self.bcs[np.arange(N), lowest_pos_cols] < cutoff] = np.nan
            deltas = sorted_data[:, numdf - 1] - sorted_data[:, numdf]
        else:
            ix = np.argsort(data, axis=1, kind="stable")
            sorted_data = np.take_along_axis(data, ix, axis=1)
            seps = np.diff(sorted_data, axis=1)
            locs = np.argsort(-seps, axis=1, kind="stable")

            betws = ix[np.arange(N), locs[:, 0] + 1]
            lowests = data[np.arange(N), betws].astype(self.work_dtype, copy=True)

            betws_low = ix[np.arange(N), locs[:, 0]]
            nextlowests = data[np.arange(N), betws_low].astype(self.work_dtype, copy=True)

            toolow = np.where(self.bcs[np.arange(N), betws_low] < cutoff)[0]
            if len(toolow):
                if locs.shape[1] < 2:
                    lowests[toolow] = np.nan
                    nextlowests[toolow] = np.nan
                else:
                    betws2 = ix[toolow, locs[toolow, 1] + 1]
                    lowests_next = data[toolow, betws2]
                    highernow = self.bcs[toolow, betws2] > cutoff

                    lowests[toolow[highernow]] = lowests_next[highernow]
                    lowests[toolow[~highernow]] = np.nan

                    betws3 = ix[toolow, locs[toolow, 1]]
                    modified_next = data[toolow, betws3]
                    nextlowests[toolow[highernow]] = modified_next[highernow]
                    nextlowests[toolow[~highernow]] = np.nan

            deltas = lowests - nextlowests

        self.deltas = deltas.astype(np.float32, copy=False)

        code_assign = data >= lowests[:, None]
        bcind = np.zeros(N, dtype=np.int32)

        for i in range(self.num_codes):
            clust = np.ones(N, dtype=bool)
            for j in range(self.num_masses):
                clust &= code_assign[:, j] == bool(self.key[i, j])
            bcind[clust] = i + 1

        self.bcind = bcind
        return self

    def compute_mahal(self) -> "SCD":
        # Zunder MATLAB equivalent:
        # in_bc = (obj.bcind==i) & (obj.deltas > obj.sep_cutoff)
        # obj.mahal(in_bc) = mahal(bci,bci)
        # obj.mahal_cutoff_val = max(max(obj.mahal)) + 1
        if self.bcs is None or self.bcind is None or self.deltas is None:
            raise ValueError("Need loaded barcodes and debarcoding before Mahalanobis distances.")

        mahal = np.zeros_like(self.deltas, dtype=np.float32)
        well_yield = np.zeros(self.num_codes, dtype=float)

        for i in range(self.num_codes):
            in_bc = (self.bcind == (i + 1)) & (self.deltas > self.sep_cutoff)
            bci = self.bcs[in_bc, :]

            if bci.shape[0] > self.num_codes:
                x = bci.astype(np.float64, copy=False)
                mu = x.mean(axis=0)
                cov = np.cov(x, rowvar=False, bias=False)
                cov = np.atleast_2d(cov)
                invcov = np.linalg.pinv(cov)
                dif = x - mu
                d2 = np.einsum("ij,jk,ik->i", dif, invcov, dif)
                mahal[in_bc] = np.maximum(d2, 0).astype(np.float32)

            well_yield[i] = self.sample_ratio * np.count_nonzero(
                in_bc & (mahal < self.mahal_cutoff_val)
            )

        self.mahal = mahal
        self.well_yield = well_yield

        # MATLAB GUI resets the displayed mahal cutoff to max + 1 after calculation.
        # For export equivalence, set the manual GUI threshold AFTER compute_mahal().
        finite = mahal[np.isfinite(mahal)]
        self.mahal_max = float(np.max(finite))
        self.mahal_p95 = float(np.percentile(finite, 95))
        self.mahal_p99 = float(np.percentile(finite, 99))
        self.mahal_cutoff_val = float(np.max(finite) + 1) if finite.size else np.inf
        return self

    def compute_well_abundances(
        self,
        numseps: int = 50,
        minsep: float = 0.0,
        maxsep: float = 1.0,
    ) -> "SCD":
        if self.bcind is None or self.deltas is None:
            raise ValueError("Barcodes must be computed before computing well abundances.")

        self.seprange = np.linspace(minsep, maxsep, numseps)
        self.clust_size = np.zeros((numseps, self.num_codes), dtype=float)

        for i, s in enumerate(self.seprange):
            for j in range(self.num_codes):
                self.clust_size[i, j] = np.count_nonzero(
                    (self.bcind == (j + 1)) & (self.deltas > s)
                )
        return self

    def run_zunder_gui_equivalent(
        self,
        fcs_file: str | Path,
        *,
        sep_cutoff_for_mahal_model: float = 0.1,
        manual_mahal_cutoff: float = 10.0,
        memory_map: bool = False,
        allow_truncated: bool = False,
        sample_size: int | None = None,
        random_state: int | None = None,
    ) -> "SCD":
        # This matches your described GUI workflow:
        # 1. debarcode
        # 2. set/use separation threshold of 0.1 before mahal model/QC
        # 3. choose mahal threshold 10 for final export/counts
        self.sep_cutoff = float(sep_cutoff_for_mahal_model)
        self.mahal_cutoff_val = np.inf

        self.load_fcs_files(fcs_file, memory_map=memory_map, allow_truncated=allow_truncated)
        self.find_bc_cols_by_mass()
        self.load_bcs(sample_size=sample_size, random_state=random_state)
        self.compute_debarcoding("bcs")
        self.normalize_by_pop("bcs")
        self.compute_debarcoding("normbcs")
        self.compute_mahal()
        self.compute_well_abundances()

        self.mahal_cutoff_val = float(manual_mahal_cutoff)
        return self

    def final_mask(self) -> np.ndarray:
        if self.bcind is None or self.deltas is None or self.mahal is None:
            raise ValueError("Run debarcoding and Mahalanobis calculation first.")
        return (
            (self.bcind > 0)
            & (self.mahal < self.mahal_cutoff_val)
            & (self.deltas > self.sep_cutoff)
        )

    def assignment_counts(self) -> tuple[pd.DataFrame, int]:
        keep = self.final_mask()
        rows = []
        for i, label in enumerate(self.well_labels):
            mask = keep & (self.bcind == (i + 1))
            rows.append((label, int(np.count_nonzero(mask))))
        unassigned = int(np.count_nonzero(~keep))
        return pd.DataFrame(rows, columns=["well_label", "count"]), unassigned

    def assignment_dataframe(self) -> pd.DataFrame:
        keep = self.final_mask()
        labels = np.array(["unassigned"] * len(self.bcind), dtype=object)
        for i, label in enumerate(self.well_labels):
            labels[keep & (self.bcind == (i + 1))] = label
        return pd.DataFrame(
            {
                "event_index": np.arange(len(self.bcind)),
                "barcode_index": np.where(keep, self.bcind, 0),
                "barcode_label": labels,
                "delta": self.deltas,
                "mahal": self.mahal,
                "kept": keep,
            }
        )

    def write_debarcoded_fcs_files(
        self,
        outdir: str | Path,
        basename: str,
        *,
        include_unassigned: bool = True,
        chunk_size: int = 250_000,
    ) -> pd.DataFrame:
        if self.x is None or self.c is None or self.m is None:
            raise ValueError("No FCS data loaded.")
        if self.bcind is None or self.deltas is None or self.mahal is None:
            raise ValueError("Run debarcoding before writing FCS files.")

        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)

        new_marker_names = list(self.m) + ["bc_separation_dist", "mahalanobis_dist"]
        new_channel_names = list(self.c) + ["bc_separation_dist", "mahalanobis_dist"]
        n_channels = self.x.shape[1] + 2

        def compute_ranges(indices: np.ndarray) -> list[int]:
            if indices.size == 0:
                return [1] * n_channels
            ranges = [_safe_range_value(np.asarray(self.x[indices, col])) for col in range(self.x.shape[1])]
            ranges.append(_safe_range_value(self.deltas[indices]))
            ranges.append(_safe_range_value(self.mahal[indices]))
            return ranges

        def rows_for(indices: np.ndarray):
            for start in range(0, indices.size, chunk_size):
                idx = indices[start:start + chunk_size]
                yield np.column_stack((self.x[idx, :], self.deltas[idx], self.mahal[idx]))

        written = []
        not_in_a_well = np.ones(self.x.shape[0], dtype=bool)

        for i, label in enumerate(self.well_labels):
            mask = (
                (self.bcind == (i + 1))
                & (self.mahal < self.mahal_cutoff_val)
                & (self.deltas > self.sep_cutoff)
            )
            idx = np.flatnonzero(mask)
            not_in_a_well[idx] = False

            if idx.size == 0:
                continue

            safe_label = re.sub(r"[\\/:*?\"<>|]+", "_", str(label))
            outfile = outdir / f"{basename}_{safe_label}.fcs"
            write_fcs_cytobank(
                outfile,
                None,
                new_marker_names,
                new_channel_names,
                self.source_hdr,
                row_iter=rows_for(idx),
                n_events=int(idx.size),
                ranges=compute_ranges(idx),
            )
            written.append({"well_label": label, "events": int(idx.size), "file": str(outfile)})

        if include_unassigned:
            idx = np.flatnonzero(not_in_a_well)
            outfile = outdir / f"{basename}_unassigned.fcs"
            write_fcs_cytobank(
                outfile,
                None,
                new_marker_names,
                new_channel_names,
                self.source_hdr,
                row_iter=rows_for(idx),
                n_events=int(idx.size),
                ranges=compute_ranges(idx),
            )
            written.append({"well_label": "unassigned", "events": int(idx.size), "file": str(outfile)})

        return pd.DataFrame(written)


# -----------------------------------------------------------------------------
# Analysis runner
# -----------------------------------------------------------------------------

def analyse_scd_run(
    fcs_path: str | Path,
    key_path: str | Path,
    output_dir: str | Path | None = None,
    *,
    basename: str | None = None,
    sep_cutoff: float = 0.1,
    mahal_cutoff: float = 10.0,
    write_fcs: bool = False,
    include_unassigned: bool = True,
    memory_map: bool = False,
    allow_truncated: bool = False,
) -> tuple[SCD, pd.DataFrame, pd.DataFrame | None]:
    fcs_path = Path(fcs_path)
    basename = fcs_path.stem if basename is None else basename

    scd = SCD(
        key_path,
        default_cofactor=10.0,
        sep_cutoff=sep_cutoff,
        mahal_cutoff_val=np.inf,
    )
    scd.run_zunder_gui_equivalent(
        fcs_path,
        sep_cutoff_for_mahal_model=sep_cutoff,
        manual_mahal_cutoff=mahal_cutoff,
        memory_map=memory_map,
        allow_truncated=allow_truncated,
    )

    counts, unassigned = scd.assignment_counts()
    counts.loc[len(counts)] = ["unassigned", unassigned]

    written = None
    if write_fcs:
        if output_dir is None:
            raise ValueError("output_dir is required when write_fcs=True")
        written = scd.write_debarcoded_fcs_files(
            output_dir,
            basename,
            include_unassigned=include_unassigned,
        )

    return scd, counts, written


# def main() -> None:
#     parser = argparse.ArgumentParser(description="Run Zunder/MATLAB-style SCD debarcoding on a CyTOF FCS file.")
#     parser.add_argument("--fcs", required=True, help="Input FCS file")
#     parser.add_argument("--key", required=True, help="Barcode key CSV")
#     parser.add_argument("--outdir", default=None, help="Output directory for CSV and optional FCS files")
#     parser.add_argument("--basename", default=None, help="Output basename; defaults to input FCS stem")
#     parser.add_argument("--sep", type=float, default=0.1, help="GUI barcode separation threshold")
#     parser.add_argument("--mahal", type=float, default=10.0, help="GUI Mahalanobis threshold")
#     parser.add_argument("--write-fcs", action="store_true", help="Write per-well FCS files")
#     parser.add_argument("--include-unassigned", action="store_true", default=True, help="Write unassigned FCS when --write-fcs is used")
#     parser.add_argument("--memory-map", action="store_true", help="Memory-map input FCS matrix")
#     parser.add_argument("--allow-truncated", action="store_true", help="Allow truncated FCS data segments")
#     args = parser.parse_args()

#     outdir = Path(args.outdir) if args.outdir else Path.cwd()
#     outdir.mkdir(parents=True, exist_ok=True)
#     basename = Path(args.fcs).stem if args.basename is None else args.basename

#     scd, counts, written = analyse_scd_run(
#         args.fcs,
#         args.key,
#         outdir,
#         basename=basename,
#         sep_cutoff=args.sep,
#         mahal_cutoff=args.mahal,
#         write_fcs=args.write_fcs,
#         include_unassigned=args.include_unassigned,
#         memory_map=args.memory_map,
#         allow_truncated=args.allow_truncated,
#     )

#     counts_path = outdir / f"{basename}_scd_counts.csv"
#     assignments_path = outdir / f"{basename}_scd_assignments.csv"
#     counts.to_csv(counts_path, index=False)
#     scd.assignment_dataframe().to_csv(assignments_path, index=False)

#     print(counts.to_string(index=False))
#     print(f"\nCounts written to: {counts_path}")
#     print(f"Assignments written to: {assignments_path}")

#     if written is not None:
#         written_path = outdir / f"{basename}_written_fcs.csv"
#         written.to_csv(written_path, index=False)
#         print(f"FCS write manifest written to: {written_path}")


# if __name__ == "__main__":
#     main()