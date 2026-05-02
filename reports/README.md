# `reports/` — paper-bound artifacts

```
reports/
├── figures/   PDF figures for the paper (system overview, heatmaps, etc.)
├── tables/    JSON results, Latex result tables, sensor importance JSON
├── videos/    Composite trajectory videos for supplementary
├── logs/      Stdout logs from collection / training / eval (gitignored)
├── eval/      Per (method, task) eval JSON output (gitignored)
└── checks/    Sanity-check JSON/CSV outputs (gitignored)
```

What's tracked vs gitignored is in `.gitignore`. The rule of thumb:
**figures and the headline results table are tracked; everything else is
regenerated**.

## Headline files (Day 13-14 output)

| file                                       | what's in it                                       |
|--------------------------------------------|----------------------------------------------------|
| `figures/system_overview.pdf`              | tensor-shape diagram, paper §3                     |
| `figures/tof_sequence.pdf`                 | far/mid/near/pre-grasp ToF heatmap                 |
| `figures/sensor_importance.pdf`            | per-sensor delta heatmap on FR3 body               |
| `figures/results_table.tex`                | LaTeX table with bootstrap CIs + paired p-values   |
| `tables/sensor_importance.json`            | post-hoc sensor masking sweep results              |
| `videos/episode_*.mp4`                     | one composite video per qualitative example        |

## How to regenerate

```bash
# Figures
python -m pla.viz.heatmap --tof-h5 data/raw/near_contact/episode_000005.h5 \
    --out reports/figures/tof_sequence.pdf
python -m pla.viz.heatmap --importance-json reports/tables/sensor_importance.json \
    --out reports/figures/sensor_importance.pdf

# Results table (after running eval_all.sh)
python -m pla.eval.run_eval --print-table reports/eval/ \
    | tee reports/tables/results.txt
```
