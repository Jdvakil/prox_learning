"""Go/no-go probe gate: is 'the planner deflected around a hazard bar' linearly
decodable from the frozen Safety-CVAE trunk feature of the skin, at approach time?

Run this BEFORE spending GPU hours on PACT training. If a frozen 256-d trunk
feature of the proximity skin cannot even linearly separate deflect episodes from
free/straight episodes during the inbound approach, the conditioning token cannot
be steering the policy and the training run is not worth launching.
PASS gate: trunk-feature episode-level AUC >= 0.8 (deflect vs free).

LABEL SOURCE INVESTIGATION (2026-07-03, documented so the choice is auditable):
  (a) The converted ACT episodes in act_style_data/obstacle_prox_v1 carry NO
      labels: root attrs are just {'sim': True} and there is no bar/deflect
      dataset. So h5-only labeling is impossible on v1 (--label-source h5 exists
      for future converter versions that stamp label attrs, and errors clearly
      on v1).
  (b) The ORIGINAL datagen output that produced obstacle_prox_v1 is still on
      disk: assets/datagen/hybrid_obstacle_v1/FrankaSkinHybridObstacleConfig/
      20260612_183855. Each traj group holds an `obs_scene` JSON blob with
      `behavior_class` in {"deflect", "free"} (set by the planner,
      ObstacleAwarePickPlannerPolicy) and `scene_params.protrusion_present`
      (bar in the scene or not). That run has 125 trajs of which exactly 100
      are successes — and convert_obstacle_to_act.py (default only_success)
      wrote exactly the 100 episodes of obstacle_prox_v1 from it. This is
      planner GROUND TRUTH, so it is the default label source.

  Episode mapping: the converter iterates sorted(house_*/trajectories*.h5),
  traj keys sorted numerically, skipping fail[-1] trajs, assigning contiguous
  episode indices. We replay that exact iteration here and then VERIFY every
  single episode by decoding three source qpos JSON rows (t = 0, T//2, T-1)
  and requiring an exact match against the converted /observations/qpos. A
  mismatch aborts — no silently mislabeled episodes.

  Label: y=1 behavior_class == "deflect" (planner inserted avoidance waypoints
  around the bar), y=0 "free" (straight plan; includes both bar-present
  close-pass episodes and no-bar episodes). A secondary bar-present label is
  also probed for context (it is the easier problem) but the gate is deflect.

APPROACH TIMESTEPS (the window that is featurized):
  datagen-dir mode: steps with policy_phase in {2, 3} (pregrasp, grasp) BEFORE
  the first phase 4 (gripper-close) — the inbound approach where the arm
  threads past the bar, i.e. exactly where the planner's deflection waypoints
  live (same window as analyze_obstacle_dataset.py). Retry episodes therefore
  contribute only their first approach.
  h5/heuristic mode (no phase stream): steps before the sustained flip of the
  gripper command /action[:, 7] away from its open value (robust to the
  spurious 255 many episodes carry on the very first row) — verified on this
  dataset to coincide with the first gripper-close phase onset.

HEURISTIC LABEL FALLBACK (--label-source heuristic — clearly flagged, noisy):
  y=1 iff the joint-space bow of the 7-dof arm qpos path over the approach
  window (max perpendicular deviation from the straight start->end chord in
  R^7) >= --bow-thresh. Calibrated against ground truth on obstacle_prox_v1:
  deflect bows are 0.101-0.30 rad (median 0.142), free bows reach 0.137
  (median 0.114) — the classes OVERLAP, so no threshold is clean. The default
  0.135 gets 88% label agreement (TP 24, FP 4, FN 8 of 32 true deflects) on
  v1. Use only when no datagen dir survives.

PROBES (both use grouped 5-fold CV split BY EPISODE — a timestep never appears
in both train and test of a fold, and every episode gets one out-of-fold score):
  1. trunk: frozen Safety-CVAE decoder-trunk feature (256-d) from
     submodules/act/prox_cvae.py ProxCVAEEncoder (ckpt assets/safety/cvae_v3).
  2. raw-skin: per-sensor peak closeness (40-d) = max over the 8x8 pixels of
     clip(1 - d/D_MAX, 0, 1) with dead pixels zeroed (same constants as the
     CVAE input), i.e. the min valid depth per sensor. If this beats the trunk,
     the frozen feature is destroying skin information.
  Classifier: L2 logistic regression, class-balanced, features standardized on
  each fold's train split (no sklearn in mlspaces; scipy L-BFGS with a plain
  numpy gradient-descent fallback).

Metrics: timestep-level and episode-level AUC + accuracy (episode score = mean
predicted probability over its approach steps). Exit code 0 on a clean run with
a clear PROBE GATE: PASS/FAIL line (errors exit nonzero).

Usage (repo root, mlspaces env):
    python scripts/probe_prox_decodability.py
    python scripts/probe_prox_decodability.py --label-source heuristic
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "submodules" / "act"))

from prox_cvae import D_MAX, DEAD_PIXEL_M, ProxCVAEEncoder  # noqa: E402

DEFAULT_ACT_DIR = REPO / "act_style_data" / "obstacle_prox_v1"
DEFAULT_DATAGEN_DIR = (
    REPO / "assets" / "datagen" / "hybrid_obstacle_v1"
    / "FrankaSkinHybridObstacleConfig" / "20260612_183855"
)
DEFAULT_CKPT = REPO / "assets" / "safety" / "cvae_v3"

# policy_phase codes (see analyze_obstacle_dataset.py PHASE_NAMES)
APPROACH_PHASES = (2, 3)   # pregrasp, grasp
GRIP_CLOSE = 4

QPOS_ATOL = 1e-5           # converted qpos is the float32 of the source JSON — exact


# --------------------------------------------------------------------- source decode
def _decode_qpos_row(blob: np.ndarray) -> np.ndarray:
    """Decode one (2000,) uint8 JSON row {"arm":[7],"gripper":[2]} -> (9,) float32.

    Same convention as convert_obstacle_to_act.py:_decode_qpos_qvel (vendored to
    keep this gate runnable without importing the converter's cv2 dependency).
    """
    out = np.zeros(9, dtype=np.float32)
    raw = bytes(np.asarray(blob)).split(b"\x00", 1)[0]
    if not raw:
        return out
    d = json.loads(raw.decode("utf-8"))
    arm = d.get("arm") or []
    grip = d.get("gripper") or []
    out[: min(len(arm), 7)] = arm[:7]
    out[7 : 7 + min(len(grip), 2)] = grip[:2]
    return out


def _find_source_h5s(datagen_dir: Path) -> list[Path]:
    files = sorted(datagen_dir.glob("house_*/trajectories*.h5"))
    if not files:
        files = sorted(datagen_dir.glob("trajectories*.h5"))
    if not files:
        raise SystemExit(f"[gate] no trajectories*.h5 under {datagen_dir}")
    return files


def map_episodes_to_source(datagen_dir: Path, n_episodes: int) -> list[dict]:
    """Replay convert_obstacle_to_act.py's iteration to map episode_g -> source traj.

    Returns one record per converted episode, in episode order:
      {house, traj, behavior_class, bar, phase (np.ndarray), qpos_rows (fn)}
    Aborts if the success count does not equal the converted episode count.
    """
    recs = []
    for fp in _find_source_h5s(datagen_dir):
        with h5py.File(fp, "r") as f:
            keys = sorted(f.keys(), key=lambda k: int(k.split("_", 1)[1]))
            for key in keys:
                grp = f[key]
                failed = bool(np.asarray(grp["fail"])[-1]) if "fail" in grp else False
                if failed:
                    continue  # converter default only_success drops these
                scene = json.loads(np.asarray(grp["obs_scene"]).item())
                recs.append(
                    dict(
                        h5=fp,
                        house=fp.parent.name,
                        traj=key,
                        behavior_class=scene.get("behavior_class", "?"),
                        bar=bool(scene.get("scene_params", {}).get("protrusion_present", False)),
                        phase=np.asarray(grp["obs/extra/policy_phase"][:], dtype=int),
                    )
                )
    if len(recs) != n_episodes:
        raise SystemExit(
            f"[gate] mapping FAILED: {len(recs)} source successes but {n_episodes} "
            f"converted episodes. {datagen_dir} is not the run that produced this "
            f"dataset (or the converter used --keep_failures/--max_episodes)."
        )
    return recs


def verify_mapping(recs: list[dict], act_dir: Path) -> None:
    """Prove episode_g == recs[g] by exact qpos row matches at t = 0, T//2, T-1."""
    for g, rec in enumerate(recs):
        with h5py.File(act_dir / f"episode_{g}.hdf5", "r") as f:
            qpos = f["observations/qpos"][:]
        T = qpos.shape[0]
        with h5py.File(rec["h5"], "r") as f:
            src_q = f[rec["traj"]]["obs/agent/qpos"]
            for t in (0, T // 2, T - 1):
                src_row = _decode_qpos_row(src_q[t])
                if not np.allclose(qpos[t], src_row, atol=QPOS_ATOL):
                    raise SystemExit(
                        f"[gate] mapping FAILED at episode_{g} <-> "
                        f"{rec['house']}/{rec['traj']} t={t}: qpos mismatch "
                        f"(max err {np.abs(qpos[t] - src_row).max():.2e}). Labels "
                        f"would be misaligned — refusing to continue."
                    )
    print(f"[gate] mapping verified: all {len(recs)} episodes matched source qpos "
          f"at 3 timesteps each (atol {QPOS_ATOL})")


# ----------------------------------------------------------------- approach windows
def approach_mask_from_phase(phase: np.ndarray, T: int) -> np.ndarray:
    """Inbound approach: phase in {pregrasp, grasp} before the first gripper-close."""
    phase = phase[:T]
    close = np.where(phase == GRIP_CLOSE)[0]
    grip_start = int(close[0]) if close.size else T
    mask = np.isin(phase, APPROACH_PHASES) & (np.arange(T) < grip_start)
    return mask


def approach_mask_from_action(action: np.ndarray) -> np.ndarray:
    """No phase stream: everything before the sustained gripper-close command.

    The gripper channel has boundary glitches (the very first row is often a
    spurious 255 before the open command settles), so the open value is taken as
    the majority of steps 1..7 and the close event is the first t >= 1 that
    starts a 3-step run of the non-open value. Verified against policy_phase:
    this lands on the first gripper-close onset.
    """
    grip = action[:, 7]
    T = action.shape[0]
    head = grip[1 : min(8, T)]
    vals, counts = np.unique(head, return_counts=True)
    open_val = vals[np.argmax(counts)] if head.size else grip[0]
    end = T
    for t in range(1, T):
        run = grip[t : min(t + 3, T)]
        if run.size and (run != open_val).all():
            end = t
            break
    mask = np.zeros(T, dtype=bool)
    mask[:end] = True
    return mask


def chord_bow(path: np.ndarray) -> float:
    """Max perpendicular deviation of an (N, D) path from its start->end chord."""
    if path.shape[0] < 3:
        return 0.0
    a, b = path[0], path[-1]
    ab = b - a
    L = float(np.linalg.norm(ab))
    if L < 1e-8:
        return float(np.linalg.norm(path - a, axis=1).max())
    proj = (path - a) @ ab / (L * L)
    perp = (path - a) - proj[:, None] * ab[None, :]
    return float(np.linalg.norm(perp, axis=1).max())


# ------------------------------------------------------------------------- features
def raw_skin_feature(prox: np.ndarray) -> np.ndarray:
    """(N, 40, 8, 8) depths (m) -> (N, 40) per-sensor peak closeness (= min depth).

    Same D_MAX / dead-pixel constants as the CVAE input so the two probes see the
    same physical quantity, just unaggregated vs trunk-encoded.
    """
    d = prox.astype(np.float32)
    c = np.clip(1.0 - d / D_MAX, 0.0, 1.0)
    c[d < DEAD_PIXEL_M] = 0.0
    return c.reshape(len(c), 40, 64).max(axis=2)


def trunk_feature(encoder: ProxCVAEEncoder, prox: np.ndarray, chunk: int = 2048) -> np.ndarray:
    """(N, 40, 8, 8) depths -> (N, feat_dim) frozen CVAE feature (trunk or delta)."""
    outs = []
    with torch.no_grad():
        for i in range(0, len(prox), chunk):
            x = torch.from_numpy(prox[i : i + chunk]).float()
            outs.append(encoder(x).squeeze(1).cpu().numpy())
    return np.concatenate(outs, axis=0)


# --------------------------------------------------------------- logistic regression
def _logreg_loss_grad(wb: np.ndarray, X: np.ndarray, y: np.ndarray,
                      sw: np.ndarray, lam: float):
    """Class-weighted logistic NLL + L2 (bias unregularized). Returns (loss, grad)."""
    w, b = wb[:-1], wb[-1]
    z = X @ w + b
    # stable log(1 + exp(-y*z))
    m = -y * z
    loss = float(np.sum(sw * np.logaddexp(0.0, m)) + 0.5 * lam * w @ w)
    s = sw * (-y) * (1.0 / (1.0 + np.exp(-m)))   # d/dz
    grad = np.empty_like(wb)
    grad[:-1] = X.T @ s + lam * w
    grad[-1] = s.sum()
    return loss, grad


def fit_logreg(X: np.ndarray, y01: np.ndarray, lam: float = 1e-2) -> np.ndarray:
    """Fit balanced L2 logistic regression; returns (D+1,) weights+bias.

    X must already be standardized. scipy L-BFGS when available, else plain
    numpy gradient descent with momentum (same loss).
    """
    y = np.where(y01 > 0, 1.0, -1.0)
    n_pos = max(int((y > 0).sum()), 1)
    n_neg = max(int((y < 0).sum()), 1)
    n = len(y)
    sw = np.where(y > 0, n / (2.0 * n_pos), n / (2.0 * n_neg)) / n
    x0 = np.zeros(X.shape[1] + 1)
    try:
        from scipy.optimize import minimize
        res = minimize(_logreg_loss_grad, x0, args=(X, y, sw, lam),
                       jac=True, method="L-BFGS-B", options=dict(maxiter=500))
        return res.x
    except ImportError:
        wb, vel = x0, np.zeros_like(x0)
        lr, mom = 0.5, 0.9
        for _ in range(3000):
            _, g = _logreg_loss_grad(wb, X, y, sw, lam)
            vel = mom * vel - lr * g
            wb = wb + vel
        return wb


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def auc_score(y01: np.ndarray, score: np.ndarray) -> float:
    """Mann-Whitney AUC with tie-averaged ranks (pure numpy)."""
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score))
    s = score[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    n1 = int((y01 == 1).sum())
    n0 = int((y01 == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[y01 == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


# --------------------------------------------------------------------- grouped CV
def stratified_group_folds(ep_labels: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Assign each episode to one of k folds, class-stratified. Returns (n_ep,) ints."""
    rng = np.random.default_rng(seed)
    fold = np.zeros(len(ep_labels), dtype=int)
    for cls in (0, 1):
        idx = np.where(ep_labels == cls)[0]
        rng.shuffle(idx)
        fold[idx] = np.arange(len(idx)) % k
    return fold


def run_probe(feat: np.ndarray, ts_ep: np.ndarray, ep_labels: np.ndarray,
              k: int, seed: int, lam: float) -> dict:
    """Grouped k-fold CV probe. feat: (N_ts, D); ts_ep: (N_ts,) episode index of
    each timestep; ep_labels: (n_ep,) 0/1. Returns pooled out-of-fold metrics."""
    n_ep = len(ep_labels)
    y_ts = ep_labels[ts_ep]
    fold = stratified_group_folds(ep_labels, k, seed)
    ts_prob = np.full(len(y_ts), np.nan)
    for f in range(k):
        test_eps = np.where(fold == f)[0]
        tr = ~np.isin(ts_ep, test_eps)
        te = ~tr
        mu = feat[tr].mean(axis=0)
        sd = feat[tr].std(axis=0)
        sd[sd < 1e-6] = 1.0
        wb = fit_logreg((feat[tr] - mu) / sd, y_ts[tr], lam=lam)
        z = ((feat[te] - mu) / sd) @ wb[:-1] + wb[-1]
        ts_prob[te] = sigmoid(z)
    assert not np.isnan(ts_prob).any()
    ep_prob = np.array([ts_prob[ts_ep == e].mean() for e in range(n_ep)])
    return dict(
        ts_auc=auc_score(y_ts, ts_prob),
        ts_acc=float(((ts_prob >= 0.5) == (y_ts == 1)).mean()),
        ep_auc=auc_score(ep_labels, ep_prob),
        ep_acc=float(((ep_prob >= 0.5) == (ep_labels == 1)).mean()),
        ep_prob=ep_prob,
    )


# ------------------------------------------------------------------------------ main
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--act-dir", type=Path, default=DEFAULT_ACT_DIR,
                   help="converted ACT dataset with /observations/proximity")
    p.add_argument("--datagen-dir", type=Path, default=DEFAULT_DATAGEN_DIR,
                   help="original datagen run (label source: obs_scene behavior_class)")
    p.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT,
                   help="frozen Safety-CVAE checkpoint dir")
    p.add_argument("--feature", choices=("trunk", "delta"), default="trunk",
                   help="which frozen CVAE tap to probe (gate is defined on trunk)")
    p.add_argument("--label-source", choices=("datagen-dir", "h5", "heuristic"),
                   default="datagen-dir",
                   help="datagen-dir = planner ground truth (default); h5 = label "
                        "attrs stamped in the episode files (v1 has none); "
                        "heuristic = joint-space bow threshold (NOISY)")
    p.add_argument("--bow-thresh", type=float, default=0.135,
                   help="heuristic mode: joint-space approach bow (rad) >= this => deflect")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--l2", type=float, default=1e-2, help="probe L2 strength")
    p.add_argument("--auc-gate", type=float, default=0.8)
    p.add_argument("--gate-label", choices=("deflect", "bar"), default="deflect",
                   help="which label the PASS/FAIL gate is scored on. 'deflect' = the "
                        "original v1 gate. 'bar' = bar-present vs no-bar, the operative "
                        "gate for the invisible-bar v2 dataset (there the policy only "
                        "needs the skin to carry bar PRESENCE; the Gate-A diagnostic "
                        "showed deflect is undecodable frame-wise even from qpos)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    ep_files = sorted(args.act_dir.glob("episode_*.hdf5"),
                      key=lambda q: int(q.stem.split("_")[1]))
    n_ep = len(ep_files)
    if n_ep == 0:
        raise SystemExit(f"[gate] no episode_*.hdf5 under {args.act_dir}")
    print(f"[gate] dataset: {args.act_dir} ({n_ep} episodes)")

    # ---------------- labels + approach windows
    heuristic_used = False
    recs = None
    if args.label_source == "datagen-dir":
        recs = map_episodes_to_source(args.datagen_dir, n_ep)
        verify_mapping(recs, args.act_dir)
        ep_labels = np.array([1 if r["behavior_class"] == "deflect" else 0 for r in recs])
        bar_labels = np.array([1 if r["bar"] else 0 for r in recs])
        print(f"[gate] labels: planner ground truth from {args.datagen_dir.name} "
              f"(deflect={int(ep_labels.sum())}, free={int((1 - ep_labels).sum())}; "
              f"bar-present={int(bar_labels.sum())})")
    elif args.label_source == "h5":
        # future converters may stamp these; obstacle_prox_v1 does not (verified).
        labels = []
        for fp in ep_files:
            with h5py.File(fp, "r") as f:
                if "behavior_class" not in f.attrs:
                    raise SystemExit(
                        f"[gate] --label-source h5: {fp.name} has no 'behavior_class' "
                        f"attr (attrs: {dict(f.attrs)}). obstacle_prox_v1 carries no "
                        f"labels — use --label-source datagen-dir (default) or heuristic."
                    )
                labels.append(1 if f.attrs["behavior_class"] == "deflect" else 0)
        ep_labels = np.array(labels)
        bar_labels = None
        print(f"[gate] labels: h5 attrs (deflect={int(ep_labels.sum())})")
    else:
        heuristic_used = True
        bar_labels = None
        ep_labels = None  # filled after bows are computed below

    # ---------------- load proximity at approach timesteps
    feats_raw, ts_ep_list, bows = [], [], []
    prox_all = []
    for g, fp in enumerate(ep_files):
        with h5py.File(fp, "r") as f:
            action = f["action"][:]
            qpos = f["observations/qpos"][:]
            T = action.shape[0]
            if recs is not None:
                mask = approach_mask_from_phase(recs[g]["phase"], T)
            else:
                mask = approach_mask_from_action(action)
            if mask.sum() < 3:  # degenerate window — fall back to the first 15 steps
                print(f"[gate] WARNING {fp.name}: approach window has {int(mask.sum())} "
                      f"steps; falling back to first 15")
                mask = np.zeros(T, dtype=bool)
                mask[: min(15, T)] = True
            prox = f["observations/proximity"][:][mask]  # (K, 40, 8, 8)
        bows.append(chord_bow(qpos[mask, :7]))
        prox_all.append(prox)
        ts_ep_list.append(np.full(len(prox), g, dtype=int))
    prox_all = np.concatenate(prox_all, axis=0)
    ts_ep = np.concatenate(ts_ep_list)
    print(f"[gate] approach timesteps: {len(ts_ep)} total "
          f"(mean {len(ts_ep) / n_ep:.1f}/episode)")

    if heuristic_used:
        bows = np.asarray(bows)
        ep_labels = (bows >= args.bow_thresh).astype(int)
        print(f"[gate] *** HEURISTIC LABELS *** joint-space approach bow >= "
              f"{args.bow_thresh} rad => deflect ({int(ep_labels.sum())} of {n_ep}). "
              f"Classes overlap in bow on v1 ground truth — treat results as "
              f"approximate, prefer --label-source datagen-dir.")
    if ep_labels.sum() == 0 or ep_labels.sum() == n_ep:
        raise SystemExit("[gate] degenerate labels (single class) — cannot probe")

    # ---------------- features
    encoder = ProxCVAEEncoder(args.ckpt, feature=args.feature, device=args.device)
    feat_trunk = trunk_feature(encoder, prox_all)
    feat_skin = raw_skin_feature(prox_all)
    print(f"[gate] features: {args.feature} {feat_trunk.shape}, raw-skin {feat_skin.shape}")

    # ---------------- probes
    def show(name, feat, labels):
        r = run_probe(feat, ts_ep, labels, args.folds, args.seed, args.l2)
        print(f"  {name:<22s} dims={feat.shape[1]:<4d} "
              f"ts-AUC={r['ts_auc']:.3f} ts-acc={r['ts_acc']:.3f} | "
              f"ep-AUC={r['ep_auc']:.3f} ep-acc={r['ep_acc']:.3f}")
        return r

    print(f"\n[gate] PRIMARY label: deflect vs free "
          f"(pos={int(ep_labels.sum())}, neg={int(n_ep - ep_labels.sum())}); "
          f"{args.folds}-fold grouped CV by episode, seed {args.seed}")
    r_trunk = show(f"frozen-CVAE {args.feature}", feat_trunk, ep_labels)
    r_skin = show("raw per-sensor skin", feat_skin, ep_labels)

    r_trunk_bar = None
    if bar_labels is not None and 0 < bar_labels.sum() < n_ep:
        print(f"\n[gate] SECONDARY label: bar-present vs no-bar "
              f"(pos={int(bar_labels.sum())}, neg={int(n_ep - bar_labels.sum())})")
        r_trunk_bar = show(f"frozen-CVAE {args.feature}", feat_trunk, bar_labels)
        show("raw per-sensor skin", feat_skin, bar_labels)

    better = "raw skin BEATS frozen trunk" if r_skin["ep_auc"] > r_trunk["ep_auc"] \
        else "frozen trunk >= raw skin"
    print(f"\n[gate] comparison: {better} "
          f"(ep-AUC {r_trunk['ep_auc']:.3f} vs {r_skin['ep_auc']:.3f})")
    if heuristic_used:
        print("[gate] NOTE: numbers above are against HEURISTIC labels, not ground truth")

    if args.gate_label == "bar":
        if r_trunk_bar is None:
            raise SystemExit("[gate] --gate-label bar requires bar-present labels with "
                             "both classes present (need --label-source datagen-dir)")
        gate_auc, gate_name = r_trunk_bar["ep_auc"], "bar-present"
    else:
        gate_auc, gate_name = r_trunk["ep_auc"], "deflect"
    ok = gate_auc >= args.auc_gate
    print(f"\nPROBE GATE: {'PASS' if ok else 'FAIL'} — frozen-CVAE {args.feature} "
          f"episode AUC {gate_auc:.3f} on '{gate_name}' (>= {args.auc_gate:.3f} required)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
