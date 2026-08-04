# Spatial Proteomics Threshold Tuner

Interactive tool for tuning per-marker intensity thresholds used in cell type prediction from multiplexed imaging data (`.zarr` format).

---

## 💾 Setup

1. **Clone the repository:**
   ```bash
   git clone -b v1.2 https://github.com/jubayer98/spatial-proteomics-threshold-tuner.git
   cd spatial-proteomics-threshold-tuner
   ```

2. **Configure the environment:**
   ```bash
   conda update -n base -c conda-forge conda
   conda env create -f environment.yml
   conda activate sp_threshold_tuner
   ```

3. **Apply patch (OS-specific):**
   * **Linux:**
     ```bash
     python patch.py
     ```
   * **macOS:**
     ```bash
     python3 patch.py
     ```
   * **Windows (PowerShell):**
     ```powershell
     python patch.py
     ```

---

## 🚀 Run

```bash
streamlit run streamlit_app.py
```

---

## 🛠️ Usage

1. **Enter path** — Paste the full path to your `.zarr` dataset in the text input at the top.
2. **Check thresholds** — The app loads initial thresholds from `config.yml` (quantile-based). If a saved CSV exists for this sample in `threshold_csvs/`, it loads those instead and displays `Threshold found`.
3. **Adjust sliders** — Each marker has an independent threshold slider above its 4-panel detail image. Updating a slider adjusts only that marker without reloading the entire page.
4. **Run Pipeline** — Click **Run Pipeline** to apply all threshold values (a progress bar tracks rendering). Once complete, the button context shifts to **Re-run Pipeline**.
5. **Save** — Click **Save Threshold & Log** at the bottom to export current thresholds to `threshold_csvs/` and per-marker statistics to `logs/`.
6. **Low Data Mode** — Enable this toggle to render images at half resolution for faster execution on low-spec hardware.
7. **Zoom** — Expand the **Zoom** section beneath any marker to explore with synchronized pan and zoom across raw, thresholded, and predicted panels.

---

## 📁 File Structure

| File / Folder | Purpose |
| :--- | :--- |
| `streamlit_app.py` | Main application entry point |
| `config.yml` | Marker-to-celltype mappings, color schemes, and initial quantile thresholds |
| `environment.yml` | Conda environment configuration |
| `threshold_csvs/` | Directory for exported threshold CSV files |
| `logs/` | Directory for per-sample log files with marker statistics |

---

## ⚙️ Configuration

Edit `config.yml` to customize key parameters:

* `ct_marker_dict` — Maps markers to their corresponding cell type labels.
* `all_celltype_colors` / `all_marker_colors` — Defines hex color codes for visualization.
* `marker_threshold_dict` — Sets initial quantile fractions (`0.0`–`1.0`) for each marker.
* `threshold_csv_dir` — Specifies the output path for saved threshold files.
