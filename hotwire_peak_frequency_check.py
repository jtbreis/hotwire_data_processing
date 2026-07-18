# %% [markdown]
# # Hot-wire peak-frequency contamination check
#
# Overlay Welch PSDs from every pitot-calibrated run under
# ``data/hotwire/tnti_aligned_with_gravity``. Narrow spikes at the **same**
# frequency across wall positions (while $\bar U$ changes) indicate fixed-frequency
# contamination rather than a flow-physical scale.

# %%
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.ndimage import median_filter

from mcflow_plotting import load_all_calibrated_hotwire_velocities
from mcflow_plotting.style.colors import OVERLAY_COLORS
from mcflow_plotting.style.figure import (
    analysis_plots_dir,
    save_figure,
    use_lab_matplotlib_style,
)

# %matplotlib inline  # uncomment when running in Jupyter

# %%
DATA_DIR = Path(
    "/workspaces/hotwire_data_processing/data/hotwire/tnti_aligned_with_gravity"
)
CALIBRATED_GLOB = "TNTI_aligned_with_g_*_pitot_calibrated_hotwire.h5"

NPERSEG = 65_536
F_MIN_HZ = 3.0
# Mark candidate fixed contamination lines (Hz).
FIXED_FREQ_MARKERS_HZ = (1491.0, 3183.0)

FIGSIZE = (10.0, 8.5)
PLOTS_ROOT = Path(__file__).resolve().parent / "plots"
PLOTS_DIR = analysis_plots_dir(PLOTS_ROOT, DATA_DIR, "peak-frequency")
SHOW_PLOTS = True

_RUN_Y_MM_RE = re.compile(r"(\d+(?:\.\d+)?)\s*mm", re.IGNORECASE)
_RUN_GEO_RE = re.compile(r"Run-([\d.p]+in)", re.IGNORECASE)
_RUN_ANGLE_RE = re.compile(r"(\d+)\s*deg", re.IGNORECASE)


@dataclass(frozen=True)
class RunRecord:
    h5_stem: str
    run_name: str
    geometry: str
    angle_deg: int | None
    y_mm: float
    u_mean: float
    fs_hz: float
    f: np.ndarray
    log_p_db: np.ndarray
    excess_db: np.ndarray


def parse_y_mm(run_name: str) -> float:
    match = _RUN_Y_MM_RE.search(run_name)
    if match is None:
        raise ValueError(f"Could not parse wall distance from {run_name!r}.")
    return float(match.group(1))


def parse_geometry(run_name: str) -> str:
    match = _RUN_GEO_RE.search(run_name)
    if match is None:
        raise ValueError(f"Could not parse geometry from {run_name!r}.")
    return match.group(1)


def parse_angle_deg(run_name: str) -> int | None:
    match = _RUN_ANGLE_RE.search(run_name)
    return int(match.group(1)) if match else None


def file_range_tag(h5_stem: str) -> str:
    return h5_stem.replace("TNTI_aligned_with_g_", "").replace(
        "_pitot_calibrated_hotwire", ""
    )


def welch_log_psd_and_excess(
    u: np.ndarray,
    fs_hz: float,
    *,
    nperseg: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u = np.asarray(u, dtype=float)
    u = u - float(np.mean(u))
    f, pxx = signal.welch(
        u,
        fs=fs_hz,
        nperseg=min(nperseg, u.size),
        window="hann",
    )
    log_p = np.log10(pxx + 1e-30)
    bg = median_filter(log_p, size=201, mode="nearest")
    excess_db = 10.0 * (log_p - bg)
    return f, 10.0 * log_p, excess_db


def load_all_runs(data_dir: Path) -> list[RunRecord]:
    records: list[RunRecord] = []
    for h5_path in sorted(data_dir.glob(CALIBRATED_GLOB)):
        for run_name, cv in load_all_calibrated_hotwire_velocities(h5_path):
            u = cv.u.astype(float)
            u_mean = float(np.mean(u))
            f, log_p_db, excess_db = welch_log_psd_and_excess(
                u, float(cv.fs_hz), nperseg=NPERSEG
            )
            records.append(
                RunRecord(
                    h5_stem=h5_path.stem,
                    run_name=run_name,
                    geometry=parse_geometry(run_name),
                    angle_deg=parse_angle_deg(run_name),
                    y_mm=parse_y_mm(run_name),
                    u_mean=u_mean,
                    fs_hz=float(cv.fs_hz),
                    f=f,
                    log_p_db=log_p_db,
                    excess_db=excess_db,
                )
            )
    records.sort(key=lambda r: (r.geometry, r.y_mm, r.angle_deg or -1, r.h5_stem))
    return records


def peak_in_band(
    f: np.ndarray, excess_db: np.ndarray, *, lo: float, hi: float
) -> float:
    band = (f >= lo) & (f <= hi)
    if not np.any(band):
        return float("nan")
    return float(f[band][np.argmax(excess_db[band])])


def disambiguate_labels(records: list[RunRecord]) -> list[str]:
    """Label runs by geometry, wall distance, and optional angle / file-range tag."""
    key_counts: dict[tuple[str, float, int | None], int] = {}
    for rec in records:
        key = (rec.geometry, rec.y_mm, rec.angle_deg)
        key_counts[key] = key_counts.get(key, 0) + 1

    labels: list[str] = []
    for rec in records:
        parts = [rec.geometry, f"{rec.y_mm:.0f} mm"]
        if rec.angle_deg is not None:
            parts.append(f"{rec.angle_deg}°")
        base = " ".join(parts)
        key = (rec.geometry, rec.y_mm, rec.angle_deg)
        if key_counts[key] == 1:
            labels.append(base)
        else:
            labels.append(f"{base} ({file_range_tag(rec.h5_stem)})")
    return labels


def plot_geometry_panels(
    records: list[RunRecord],
    labels: list[str],
    *,
    fs_hz: float,
) -> plt.Figure:
    geometries = sorted({rec.geometry for rec in records})
    fig, axes = plt.subplots(
        len(geometries),
        2,
        figsize=FIGSIZE,
        sharex="col",
        squeeze=False,
    )

    for row, geometry in enumerate(geometries):
        subset = [
            (rec, lbl)
            for rec, lbl in zip(records, labels)
            if rec.geometry == geometry
        ]
        ax_psd = axes[row, 0]
        ax_excess = axes[row, 1]
        for idx, (rec, lbl) in enumerate(subset):
            color = OVERLAY_COLORS[idx % len(OVERLAY_COLORS)]
            ax_psd.semilogx(
                rec.f[1:],
                rec.log_p_db[1:],
                color=color,
                lw=0.7,
                alpha=0.85,
                label=lbl,
            )
            ax_excess.semilogx(
                rec.f,
                rec.excess_db,
                color=color,
                lw=0.8,
                alpha=0.85,
                label=lbl,
            )

        ax_psd.set_ylabel(r"$10\log_{10} P_{uu}$ (dB)")
        ax_psd.set_title(f"{geometry} diffuser — Welch PSD")
        ax_psd.legend(fontsize=6, ncol=2, loc="upper right")
        ax_psd.grid(True, which="both", alpha=0.25)

        ax_excess.set_ylabel("Excess over broadband\nbackground (dB)")
        ax_excess.set_title(f"{geometry} diffuser — narrow-band spikes")
        ax_excess.legend(fontsize=6, ncol=2, loc="upper right")
        ax_excess.grid(True, which="both", alpha=0.25)
        for fx in FIXED_FREQ_MARKERS_HZ:
            ax_excess.axvline(fx, color="k", ls="--", lw=1.0, alpha=0.65)

    axes[-1, 0].set_xlabel("Frequency (Hz)")
    axes[-1, 1].set_xlabel("Frequency (Hz)")
    for ax in axes.ravel():
        ax.set_xlim(F_MIN_HZ, fs_hz / 2.0)

    fig.suptitle(
        "Hot-wire peak-frequency contamination check — all calibrated TNTI runs",
        fontsize=12,
        y=1.01,
    )
    fig.tight_layout()
    return fig


# %%
records = load_all_runs(DATA_DIR)
if not records:
    raise ValueError(f"No calibrated runs found under {DATA_DIR}.")

labels = disambiguate_labels(records)
fs_hz = records[0].fs_hz
geometries = sorted({rec.geometry for rec in records})

print(f"Loaded {len(records)} runs from {len(set(rec.h5_stem for rec in records))} HDF5 files")
print(f"Geometries: {', '.join(geometries)}")
print(
    f"{'label':36s} {'U_mean(m/s)':>11s} {'peak 2.5-3.5kHz':>16s} {'peak 0.8-1.6kHz':>16s}"
)
for rec, lbl in zip(records, labels):
    p3 = peak_in_band(rec.f, rec.excess_db, lo=2500.0, hi=3500.0)
    p1 = peak_in_band(rec.f, rec.excess_db, lo=800.0, hi=1600.0)
    print(f"{lbl:36s} {rec.u_mean:11.4f} {p3:16.2f} {p1:16.2f}")

# %%
use_lab_matplotlib_style()
fig = plot_geometry_panels(records, labels, fs_hz=fs_hz)

pdf_path = PLOTS_DIR / "all-runs.pdf"
save_figure(fig, pdf_path)
print(f"Wrote {pdf_path.resolve()}")

if SHOW_PLOTS:
    plt.show()
else:
    plt.close(fig)
