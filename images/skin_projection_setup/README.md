# Skin and sensor projection figure

`figure.png` and `figure.pdf` are the unlabeled two-panel figure. `skin_sensors.png`, `scene_projections.png`, and `scene_clean.png` are separate panels.

Reproduce with `/opt/conda/envs/mlspaces/bin/python scripts/render_skin_projection_setup.py`; add `--rebuild` to restore the source scene and recompute measurements.

The original v12 fumehood and collection objects are restored using the existing scene loader. The arm uses the existing validated visualization pose (link 6 lowered by 16 cm from the initial dataset pose), not a recorded observation. See `source.json`. All 40 sensors use 64 pixel-center rays each. Endpoints are actual MuJoCo ray intersections; native 8 × 8 depth renders are also saved in `measurements.npz`. Ray data are not reconstructed from an RGB image.

The outward-facing sensor orientations are preserved: rays that face the outer room remain in the export but are visually faint. The arm is excluded from sensing using the configured geometry mask. All rays are added to the render, with normal camera framing and occlusion.

Rear view with original scene materials and lighting: `scene_projections_rear.png` / `.pdf`; matching view without the sensor overlay: `scene_clean_rear.png`. Reproduce with `--rear`. The same model state and 2,560 measurements are used; camera parameters are saved in `rear_view.json`.
