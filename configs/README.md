# `configs/` — YAML experiment configs

Three categories:

```
configs/
├── data/    Task definitions for collection
├── train/   Training configs (PLA + baseline + ablations)
└── eval/    Evaluation configs (per-task seed bases, episode counts)
```

## `train/`

| file                              | role                                   |
|-----------------------------------|----------------------------------------|
| `pla.yaml`                        | full PLA — headline run                |
| `act_baseline.yaml`               | VLM-only ACT — `vlm_only: true`         |
| `ablation_wrist_only.yaml`        | mask non-wrist sensors                 |
| `ablation_handcrafted.yaml`       | `encoder_type: handcrafted`            |
| `ablation_conv2d.yaml`            | `encoder_type: conv2d`                 |
| `ablation_cross_attn.yaml`        | `fusion_type: cross_attn`              |
| `cvae.yaml`                       | (legacy) CVAE pretrain proof            |

Every config consumed by `pla.train.train` is **flat** (no nested keys).
The full schema is documented inline in each file. The crucial flags:

  * `vlm_only: false` — set `true` for the baseline run; **do not change
    anything else**.
  * `encoder_type: shared_mlp` — set `handcrafted` / `conv2d` for ablations.
  * `fusion_type: concat` — set `cross_attn` for the cross-attn ablation.
  * `sensor_mask: null` — list of sensor indices to zero (wrist-only).

## `data/`

| file                       | role                                        |
|----------------------------|---------------------------------------------|
| `near_contact.yaml`        | obstacle 5-8 cm from expert path (PRIMARY)  |
| `standard_pnp.yaml`        | open workspace                              |

These configs feed `pla.data.collect`. `task_name` controls the output
subdirectory under `data/raw/`.

## `eval/`

| file              | role                              |
|-------------------|-----------------------------------|
| `default.yaml`    | `seed_base=42`, `n_episodes=100`  |

## Conventions

* All paths are relative to the repo root.
* Stats path is `stats.json` at repo root by default; pin to a per-run
  path if you collect multiple datasets.
* `output_dir` defaults to `runs/<run_name>` if omitted.
