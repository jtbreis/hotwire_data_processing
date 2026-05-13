# %% [markdown]
# # Hot-wire velocity PDF
#
# Histogram (and optional KDE) of calibrated streamwise velocity from a pitot-calibrated HDF5.
# Data loading: **``mcflow_plotting.hotwire``**; plotting: **``mcflow_plotting.plots``**.

# %%
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from mcflow_plotting import load_calibrated_hotwire_velocity
from mcflow_plotting.plots.hotwire import plot_hotwire_velocity_pdf
from mcflow_plotting.style.figure import save_figure

# %matplotlib inline  # uncomment when running in Jupyter

# %%
H5_PATH = Path(
    "/workspace/data/hotwire/tti_no_gravity/"
    "hotwire_10kHz_TTI_no_gravity_non_turbulent_side_pitot_calibrated_hotwire.h5"
)
HIST_BINS = 120
USE_KDE = False
KDE_MAX_SAMPLES = 200_000
RNG_SEED = 0
SAVE_FIG: Path | None = None

# %%
cv = load_calibrated_hotwire_velocity(H5_PATH)
fig, ax = plot_hotwire_velocity_pdf(
    u=cv.u,
    fs_hz=cv.fs_hz,
    h5_name=H5_PATH.name,
    dataset_path=cv.dataset_path,
    hist_bins=HIST_BINS,
    use_kde=USE_KDE,
    kde_max_samples=KDE_MAX_SAMPLES,
    rng_seed=RNG_SEED,
)
if SAVE_FIG is not None:
    save_figure(fig, SAVE_FIG)
    print(f"Wrote {SAVE_FIG}")
else:
    plt.show()
