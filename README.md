# Hot-wire data processing

Hot-wire calibration, velocity PDFs, longitudinal TKE spectrum (Welch / FFT), and Taylor-microscale analysis. Core logic lives in the **`mcflow_plotting`** package (including **`mcflow_plotting.hotwire`**); root `.py` files are short example drivers.

## Open in Google Colab

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jtbreis/hotwire_data_processing/blob/main/notebooks/Colab_Getting_Started.ipynb)

The notebook clones this repository, installs dependencies, and verifies imports. **Sample HDF5 paths** in the scripts point at `data/` in this repo; mount [Google Drive](https://colab.research.google.com/notebooks/io.ipynb) or copy those files into the Colab runtime if you need to run analyses on real data.

## Local setup

```bash
git clone https://github.com/jtbreis/hotwire_data_processing.git
cd hotwire_data_processing
pip install -r requirements-colab.txt
pip install -e ./mcflow-plotting --no-deps
```

If you need Voronoi helpers under `mcflow_plotting.inertial_particles`, install the full dependency set instead (may require a Conda environment for `freud-analysis`):

```bash
pip install -e ./mcflow-plotting
```

## Repository layout

| Path | Role |
|------|------|
| `mcflow-plotting/` | Installable `mcflow_plotting` package (style, plots, `hotwire` numerics) |
| `hotwire_calibration.py`, `tke_spectrum.py`, … | Example scripts (paths and options at top) |
| `data/` | Example datasets (ignored by git if large; adjust `.gitignore` as needed) |

## Colab URL (copy-paste)

If the badge does not open the correct branch or file, use:

`https://colab.research.google.com/github/jtbreis/hotwire_data_processing/blob/main/notebooks/Colab_Getting_Started.ipynb`

Replace `main` with your branch name if different.
