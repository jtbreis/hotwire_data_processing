# %% [markdown]
# # Hot-wire continuous wavelet transform (CWT) scalogram
#
# Load pitot-calibrated HDF5 velocity, mean-remove fluctuations, and plot a
# time–frequency scalogram from a Morlet CWT (``PyWavelets``).

# %%
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mcflow_plotting import (
    load_all_calibrated_hotwire_velocities,
    spectrum_window_indices,
    velocity_cwt_scalogram,
)
from mcflow_plotting.plots.hotwire import plot_velocity_wavelet_scalogram
from mcflow_plotting.style.figure import (
    analysis_plots_dir,
    save_figure,
    use_lab_matplotlib_style,
)


def _jupyter_matplotlib_inline() -> None:
    """Show figures inside the notebook / Interactive Window (no-op in plain ``python``)."""
    try:
        from IPython import get_ipython

        shell = get_ipython()
        if shell is not None:
            shell.run_line_magic("matplotlib", "inline")
    except (ImportError, AttributeError):
        pass


_jupyter_matplotlib_inline()

# %%
H5_PATH = Path(
    "/workspaces/hotwire_data_processing/data/hotwire/tnti_aligned_with_gravity/"
    "TNTI_aligned_with_g_6in_200-400mm_pitot_calibrated_hotwire.h5"
)
# Analysis window inside each run (None = full record).
# Default: first 10 s — full 120 s at 10 kHz is slow for CWT.
SPECTRUM_T_START_S: float | None = 0.0
SPECTRUM_T_END_S: float | None = 10.0
# Scalogram plot zoom (seconds, relative to the analysis window). None = full window.
PLOT_T_START_S: float | None = None
PLOT_T_END_S: float | None = None

WAVELET = "morl"
F_MIN_HZ: float | None = 20.0
F_MAX_HZ: float | None = 2000.0
N_SCALES = 96
POWER_SCALE = "db"  # "db", "log10", or "linear"

FIGSIZE = (9.0, 5.0)
PLOTS_ROOT = Path(__file__).resolve().parent / "plots"
SHOW_PLOTS = False

# %%


def _safe_plot_stem(name: str, *, max_len: int = 72) -> str:
    out: list[str] = []
    for c in name.lower():
        if c.isalnum() or c in "-_":
            out.append(c)
        else:
            out.append("-")
    s = "".join(out).strip("-")
    while "--" in s:
        s = s.replace("--", "-")
    return s[:max_len] if s else "hotwire-run"


def wavelet_scalogram_pdf_path(
    plots_dir: Path,
    h5: Path,
    *,
    run_name: str,
    t_spec0: float,
    t_spec1: float,
    t_plot0: float,
    t_plot1: float,
    fs_hz: float,
    wavelet: str,
) -> Path:
    plots_dir.mkdir(parents=True, exist_ok=True)
    stem = _safe_plot_stem(h5.stem)
    run_stem = _safe_plot_stem(run_name)
    s0 = f"{float(t_spec0):.4g}".replace(".", "p")
    s1 = f"{float(t_spec1):.4g}".replace(".", "p")
    p0 = f"{float(t_plot0):.4g}".replace(".", "p")
    p1 = f"{float(t_plot1):.4g}".replace(".", "p")
    fs_i = int(round(float(fs_hz)))
    wv = _safe_plot_stem(wavelet)
    return plots_dir / (
        f"{stem}_{run_stem}_wavelet_{wv}_"
        f"specWin{s0}-{s1}s_plot{p0}-{p1}s_fs{fs_i}Hz.pdf"
    )


def relative_plot_slice(
    n: int,
    fs_hz: float,
    *,
    t0_s: float | None,
    t1_s: float | None,
) -> tuple[int, int]:
    """Half-open indices within an already windowed segment."""
    i0 = 0 if t0_s is None else max(0, int(np.floor(t0_s * fs_hz)))
    i1 = n if t1_s is None else min(n, int(np.ceil(t1_s * fs_hz)))
    if i1 <= i0:
        raise ValueError(f"Empty plot window: indices {i0}..{i1}.")
    return i0, i1


# %%
use_lab_matplotlib_style()

for run_name, cv in load_all_calibrated_hotwire_velocities(H5_PATH):
    u_all = cv.u.astype(float)
    fs = float(cv.fs_hz)
    n = u_all.size

    s0, s1 = spectrum_window_indices(
        n, fs, t_start_s=SPECTRUM_T_START_S, t_end_s=SPECTRUM_T_END_S
    )
    u_spec = u_all[s0:s1]
    t_base = float(s0) / fs

    t_s, f_hz, power, u_mean, meta = velocity_cwt_scalogram(
        u_spec,
        fs,
        wavelet=WAVELET,
        f_min_hz=F_MIN_HZ,
        f_max_hz=F_MAX_HZ,
        n_scales=N_SCALES,
    )

    rel0, rel1 = relative_plot_slice(
        int(t_s.size),
        fs,
        t0_s=PLOT_T_START_S,
        t1_s=PLOT_T_END_S,
    )
    t_plot = t_s[rel0:rel1] + t_base
    power_plot = power[:, rel0:rel1]

    pdf_path = wavelet_scalogram_pdf_path(
        analysis_plots_dir(PLOTS_ROOT, H5_PATH, "wavelet"),
        H5_PATH,
        run_name=run_name,
        t_spec0=float(t_base),
        t_spec1=float(t_base + t_s[-1]),
        t_plot0=float(t_plot[0]),
        t_plot1=float(t_plot[-1]),
        fs_hz=fs,
        wavelet=WAVELET,
    )

    fig, ax = plot_velocity_wavelet_scalogram(
        t_s=t_plot,
        f_hz=f_hz,
        power=power_plot,
        u_mean=u_mean,
        run_name=run_name,
        wavelet=WAVELET,
        power_scale=POWER_SCALE,
        figsize=FIGSIZE,
    )

    save_figure(fig, pdf_path)
    print(
        f"{run_name}: CWT {meta['n_scales']} scales, "
        f"f ∈ [{meta['f_min_hz']:.2g}, {meta['f_max_hz']:.2g}] Hz, "
        f"σ_u'={float(meta['variance']) ** 0.5:.4f} m/s"
    )
    print(f"Wrote {pdf_path.resolve()}")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)
