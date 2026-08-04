# Spatial Proteomics Threshold Tuner

Interactive tool for tuning per-marker intensity thresholds used in cell type prediction from multiplexed imaging data (zarr format).

## Setup

```bash
conda env create -f environment.yml
conda activate sp_threshold_tuner
```

## Run

```bash
streamlit run streamlit_app.py
```

## Usage

1. **Enter path** — Paste the full path to your `.zarr` dataset in the text input at the top.
2. **Check thresholds** — The app loads initial thresholds from `config.yml` (quantile-based). If a saved CSV exists for this sample in `threshold_csvs/`, it loads those instead and shows *Threshold found*.
3. **Adjust sliders** — Each marker has its own threshold slider above its 4-panel detail image. Changing a slider updates only that marker (the page does not reload).
4. **Run Pipeline** — Click **Run Pipeline** to apply all threshold values. A progress bar shows rendering status. After execution the button changes to **Re-run Pipeline**.
5. **Save** — Click **Save Threshold & Log** at the bottom to export the current thresholds to `threshold_csvs/` and per-marker statistics to `logs/`.
6. **Low Data Mode** — Toggle on to render images at half resolution (faster on low-config devices).
7. **Zoom** — Expand the *Zoom* section below any marker to explore with synchronized zoom/pan across raw, thresholded, and predicted panels.

## Files

| File | Purpose |
|------|---------|
| `streamlit_app.py` | Main application |
| `config.yml` | Marker-to-celltype mapping, colors, initial quantile thresholds |
| `environment.yml` | Conda environment specification |
| `threshold_csvs/` | Saved threshold CSV files |
| `logs/` | Per-sample log files with marker statistics |

## Configuration

Edit `config.yml` to customize:
- `ct_marker_dict` — marker → cell type label
- `all_celltype_colors` / `all_marker_colors` — hex colors
- `marker_threshold_dict` — initial quantile fraction (0–1) for each marker
- `threshold_csv_dir` — output directory for saved thresholds
