# %% [markdown]
# # Hot-wire raw frequency spectrum
#
# Load a pitot-calibrated HDF5 record (same path / time window as **``hotwire_turbulence_indicator.py``**),
# then plot **\(|\hat{u}(f)|^2\)** from a one-sided FFT of raw \(u(t)\) vs frequency \(f\).
# No PSD density scaling (\(1/f_s N\)), one-sided doubling, mean removal, or windowing.

# %%
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mcflow_plotting import load_all_calibrated_hotwire_velocities
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
    "Hotwire_and_Pitot_TNTI_aligned_with_g_2026-07-02T15-25-42.206797_pitot_calibrated_hotwire.h5"
)
# Spectrum uses this slice of the file (None = start / end of record).
SPECTRUM_T_START_S: float | None = None
SPECTRUM_T_END_S: float | None = None

FIGSIZE = (7.0, 4.5)
PLOTS_ROOT = Path(__file__).resolve().parent / "plots"
SHOW_PLOTS = True

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


def frequency_spectrum_pdf_path(
    plots_dir: Path,
    h5: Path,
    *,
    run_name: str,
    t_spec0: float,
    t_spec1: float,
    fs_hz: float,
) -> Path:
    plots_dir.mkdir(parents=True, exist_ok=True)
    stem = _safe_plot_stem(h5.stem, max_len=48)
    run_stem = _safe_plot_stem(run_name, max_len=40)
    s0 = f"{float(t_spec0):.4g}".replace(".", "p")
    s1 = f"{float(t_spec1):.4g}".replace(".", "p")
    fs_i = int(round(float(fs_hz)))
    return plots_dir / (
        f"{stem}_{run_stem}_fft_specWin{s0}-{s1}s_fs{fs_i}Hz.pdf"
    )


def velocity_fft_power(
    u: np.ndarray,
    fs_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    One-sided **\(|\hat{u}(f)|^2\)** from ``np.fft.rfft(u)`` with no extra scaling.

    Raw squared DFT coefficients (not divided by ``N`` or ``f_s``, no one-sided factor of 2).
    """
    u = np.asarray(u, dtype=float)
    n = int(u.size)
    if n < 2:
        raise ValueError("Need at least two samples for a spectrum.")
    u_hat = np.fft.rfft(u)
    f_hz = np.fft.rfftfreq(n, d=1.0 / float(fs_hz))
    power = np.abs(u_hat) ** 2
    return f_hz, power


def spectrum_window_indices(
    n: int,
    fs_hz: float,
    *,
    t_start_s: float | None,
    t_end_s: float | None,
) -> tuple[int, int]:
    s0 = 0 if t_start_s is None else max(0, int(np.floor(t_start_s * fs_hz)))
    s1 = n if t_end_s is None else min(n, int(np.ceil(t_end_s * fs_hz)))
    if s1 <= s0:
        raise ValueError(f"Empty spectrum window: indices {s0}..{s1}.")
    return s0, s1


# %%
use_lab_matplotlib_style()

for run_name, cv in load_all_calibrated_hotwire_velocities(H5_PATH):
    u_all = cv.u.astype(float)
    fs = float(cv.fs_hz)
    n = u_all.size
    t_all = np.arange(n, dtype=float) / fs

    s0, s1 = spectrum_window_indices(
        n, fs, t_start_s=SPECTRUM_T_START_S, t_end_s=SPECTRUM_T_END_S
    )
    u_spec = u_all[s0:s1]
    t_spec = t_all[s0:s1]
    u_mean_spec = float(np.mean(u_spec))

    f_hz, power = velocity_fft_power(u_spec, fs)
    f_nyq = 0.5 * fs
    f_plot = f_hz[1:]
    p_plot = power[1:]

    pdf_path = frequency_spectrum_pdf_path(
        analysis_plots_dir(PLOTS_ROOT, H5_PATH, "frequency-spectrum"),
        H5_PATH,
        run_name=run_name,
        t_spec0=float(t_spec[0]),
        t_spec1=float(t_spec[-1]),
        fs_hz=fs,
    )

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.semilogx(f_plot, p_plot, color="k", lw=0.7)
    ax.set_xlim(float(f_plot[0]), f_nyq)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(r"$|\hat{u}(f)|^2$")
    ax.set_title(
        rf"FFT power of $u(t)$ (unnormalized), $\bar{{u}}={u_mean_spec:.4f}$ m s$^{{-1}}$"
        f"\n{run_name}",
        fontsize=11,
    )
    ax.grid(True, which="both", alpha=0.25)

    save_figure(fig, pdf_path)
    print(f"Wrote {pdf_path.resolve()}")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)
