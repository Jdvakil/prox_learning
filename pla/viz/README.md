# `pla/viz/` — figures, videos, dataset plots

## Purpose

Everything the paper needs as visual output. The four paper figures land
here:

  1. **System overview** with tensor shapes (`composite.system_overview`)
  2. **ToF heatmap sequence** — far / mid / near / pre-grasp
     (`heatmap.tof_sequence`)
  3. **Sensor importance heatmap on FR3 body** (`heatmap.sensor_importance_heatmap`)
  4. **Composite trajectory video** with proximity readings overlaid
     (`composite.trajectory_video`)

## Files

| file                | what's in it                                        |
|---------------------|-----------------------------------------------------|
| `heatmap.py`        | ToF heatmap sequences + sensor-importance heatmaps  |
| `composite.py`      | composite trajectory videos (RGB + ToF overlay)     |
| `pointcloud.py`     | sensor pointcloud reconstruction (3D)               |
| `dataset_plots.py`  | depth histograms, coverage stats, per-sensor stats  |
| `cvae_plots.py`     | (legacy) CVAE pretrain plots                        |

## Outputs land in...

  * `reports/figures/` — paper-bound PDF figures
  * `reports/videos/` — composite trajectory MP4s for supplementary
  * `reports/checks/` — diagnostic plots from sanity checks

## Run

```bash
# 1. Paper figures (Day 12-13).
python -m pla.viz.heatmap --tof-h5 data/raw/near_contact/episode_000005.h5 \
    --out reports/figures/tof_sequence.pdf
python -m pla.viz.heatmap --importance-json reports/tables/sensor_importance.json \
    --out reports/figures/sensor_importance.pdf

# 2. Trajectory video (supplementary).
python -m pla.viz.composite --episode data/raw/near_contact/episode_000005.h5 \
    --out reports/videos/episode_5.mp4

# 3. Dataset coverage plots.
python -m pla.viz.dataset_plots --data-dir data/raw/near_contact \
    --out reports/figures/dataset_coverage.pdf
```

## Style requirements (paper)

  * minimum font size 8 pt at final embed scale
  * colorblind-safe palette (default: cmocean ``thermal`` or matplotlib ``viridis``)
  * self-contained captions — every axis labeled with units (mm for depth)
  * no embedded raster bitmaps for vector content (heatmaps may be raster
    inside a PDF wrapper)

## Sanity-check checklist (Day 13)

- [ ] All four headline figures render without `[?]` or missing fonts
- [ ] No emoji or non-ASCII characters appear in figure captions
- [ ] Compressed PDF output < 4 MB total
- [ ] Sensor importance heatmap colors match the colorbar legend
      (failure mode: the palette inverts when min > max)
