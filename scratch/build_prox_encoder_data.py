"""Build the proximity-encoder training set (paper PACT front-end).

For every (sensor, timestep) where the pickup object is VISIBLE in that sensor and the
gripper has NOT yet grasped it, pair the 8x8 depth patch with the object's 3-D position in
that sensor's local (OpenCV camera) frame:  p_sensor = extrinsic_cv @ [obj_world; 1].
Saves X (N,8,8), Y (N,3), and the per-sample sensor index. Train an encoder depth->p_sensor.
"""
import glob, json
import numpy as np
import h5py

SRC = "/root/prox_learning/assets/datagen/fridge_two_level_v2_exoblind"
OUT = "/root/prox_learning/scratch/prox_encoder_data.npz"
FAR = 4.5

files = sorted(glob.glob(f"{SRC}/**/trajectories_batch_*.h5", recursive=True))
print(f"# h5: {len(files)}")

def grasped_mask(grp, T):
    """True once the object is grasped (decode the json grasp_state)."""
    out = np.zeros(T, bool)
    ds = grp.get("obs/extra/grasp_state_pickup_obj")
    if ds is None:
        return out
    for t in range(T):
        raw = bytes(ds[t]).split(b"\x00", 1)[0]
        if not raw:
            continue
        try:
            d = json.loads(raw)
            v = d.get("is_grasped", d.get("grasped", d.get("grasp", 0)))
            out[t] = bool(v)
        except Exception:
            pass
    return out

X, Y, S = [], [], []
sensor_list = None
for fp in files:
    with h5py.File(fp, "r") as f:
        for tk in f.keys():
            g = f[tk]
            if "obs/proximity" not in g:
                continue
            sensors = sorted(g["obs/proximity"].keys())
            if sensor_list is None:
                sensor_list = sensors
            T = g[f"obs/proximity/{sensors[0]}"].shape[0]
            objw = g["obs/extra/obj_start"][:, :3]                     # (T,3) world
            grasped = grasped_mask(g, T)
            oip = "obs/extra/object_image_points/pickup_obj"
            for si, s in enumerate(sensors):
                depth = g[f"obs/proximity/{s}"][:].mean(axis=1)        # (T,8,8)
                extr = g[f"obs/sensor_param/{s}/extrinsic_cv"][:]      # (T,3,4) world->cam
                npts = g[f"{oip}/{s}/num_points"][:].reshape(-1)       # (T,)
                for t in range(T):
                    if npts[t] <= 0 or grasped[t]:
                        continue
                    p = extr[t, :, :3] @ objw[t] + extr[t, :, 3]       # object in sensor frame
                    if not np.isfinite(p).all() or np.linalg.norm(p) > FAR:
                        continue
                    X.append(depth[t].astype(np.float32))
                    Y.append(p.astype(np.float32))
                    S.append(si)

X = np.asarray(X); Y = np.asarray(Y); S = np.asarray(S, np.int16)
print(f"samples: {len(X)}  | depth {X.shape} label {Y.shape}")
d = np.linalg.norm(Y, axis=1)
print(f"object-in-sensor dist: min={d.min()*100:.1f} med={np.median(d)*100:.1f} "
      f"max={d.max()*100:.1f} cm   (sanity: should be ~5-50cm)")
print(f"label z (depth-axis) range: [{Y[:,2].min():.3f}, {Y[:,2].max():.3f}] m")
print(f"depth value range: [{X.min():.2f}, {X.max():.2f}] (526~=no-hit sentinel)")
np.savez_compressed(OUT, X=X, Y=Y, S=S, sensors=np.array(sensor_list, dtype=object))
print(f"saved {OUT}")
