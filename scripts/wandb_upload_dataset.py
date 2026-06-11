"""Push proximity-skin dataset VIDEOS + PLOTS + metrics to a Weights & Biases run.

The assets/ folder is too big to push to HuggingFace via git; this logs a curated, viewable
slice to wandb instead: the engineer plot suite, the per-batch report figures, sample RGB
videos (exo + wrist) chosen to cover each behavior class, episode tables, and the headline
metrics (success / decorrelation / skin-engagement) per scene.

Usage:
  python scripts/wandb_upload_dataset.py --project prox-skin-dataset
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import re
from pathlib import Path

import wandb

REPO = Path("/home/jaydv/code/prox_learning")
DIAG = REPO / "diagnostics_output"
DG = REPO / "assets" / "datagen"

# scene -> (datagen run dir glob, per-batch report dir, label)
SCENES = {
    "fumehood": dict(run="fumehood_smoke/*/2026*", report="*_enclosure_fumehood3"),
    "panel": dict(run="panel_slalom_smoke/*/2026*", report="*_enclosure_panel3"),
    "cubby": dict(run="cubby_smoke/*/2026*", report="*_enclosure_cubby3"),
    "house_fumehood": dict(run="house_fumehood_smoke/*/2026*", report=None),
    "house_panel": dict(run="house_panel_smoke/*/2026*", report=None),
    "house_cubby": dict(run="house_cubby_smoke/*/2026*", report=None),
}
PLOT_DIR = DIAG / "20260610_plots_v3"
DASH_DIR = DIAG / "20260610_foxglove_dashboard"


def latest(globpat: str) -> Path | None:
    hits = sorted(glob.glob(str(DG / globpat)))
    return Path(hits[-1]) if hits else None


def latest_report(pat: str) -> Path | None:
    hits = sorted(DIAG.glob(pat))
    return hits[-1] if hits else None


def parse_probes(report: Path) -> dict:
    """Pull success rate / max|corr| verdict / probe-4 fraction out of probes.txt."""
    out = {}
    p = report / "probes.txt"
    if not p.exists():
        return out
    txt = p.read_text()
    m = re.search(r"max \|corr\| = ([\d.]+).*?VERDICT: (\w+)", txt)
    if m:
        out["decorr_maxcorr"] = float(m.group(1))
        out["decorr_pass"] = m.group(2) == "PASS"
    m = re.search(r"any zone < 8cm = ([\d.]+)", txt)
    if m:
        out["skin_engage_frac"] = float(m.group(1))
    return out


def read_csv_rows(p: Path):
    with open(p) as f:
        return list(csv.DictReader(f))


def pick_videos(run_dir: Path, report: Path | None, max_per_behavior=6, cams=("exo_camera_1",)):
    """Return [(path, caption)] of RGB videos (EXO only by default — wrist is uninformative),
    covering each behavior class, prioritizing the interesting behaviors (deflect/abort)."""
    vids = []
    beh = {}
    if report and (report / "episodes.csv").exists():
        for r in read_csv_rows(report / "episodes.csv"):
            beh[(r["house"], r["traj"])] = (r["behavior"], r["success"], r["clearance_cm"])
    by_b: dict[str, list] = {}
    for (house, traj), (b, succ, clr) in beh.items():
        by_b.setdefault(b, []).append((house, traj, succ, clr))
    chosen = []
    if by_b:
        # interesting behaviors first so they aren't truncated away
        for b in ("deflect", "abort", "free", *[k for k in by_b if k not in ("deflect", "abort", "free")]):
            for house, traj, succ, clr in by_b.get(b, [])[:max_per_behavior]:
                chosen.append((house, traj, b, succ, clr))
    else:
        # no report (house runs): take the first episode of every house (room context)
        for hp in sorted(run_dir.glob("house_*")):
            chosen.append((hp.name, "traj_0", "?", "?", "?"))
    for house, traj, b, succ, clr in chosen:
        idx = int(traj.split("_")[1]) if "_" in traj else 0
        for cam in cams:
            hits = [h for h in (run_dir / house).glob(f"episode_{idx:08d}_{cam}_batch_*.mp4")
                    if "depth" not in h.name]
            if hits:
                vids.append((str(hits[0]),
                             f"{house}/{traj} [{b}] succ={succ} clr={clr}cm — exo"))
    return vids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="prox-skin-dataset")
    ap.add_argument("--name", default="v3_dataset_viz")
    ap.add_argument("--max-videos-per-scene", type=int, default=24)
    ap.add_argument("--videos-only", action="store_true",
                    help="upload only EXO videos (advisor view); skip plots/tables/artifacts")
    args = ap.parse_args()

    run = wandb.init(project=args.project, name=args.name, job_type="dataset-viz",
                     config={"note": "v3 proximity-skin runs: standalone + in-house"})

    # 1) engineer plot suite
    plot_imgs = []
    if PLOT_DIR.exists() and not args.videos_only:
        for png in sorted(PLOT_DIR.glob("*.png")):
            plot_imgs.append(wandb.Image(str(png), caption=png.stem))
    if plot_imgs:
        run.log({"plots/engineer_suite": plot_imgs})

    # 2) per-batch report figures + metrics
    metric_rows = []
    for scene, cfg in SCENES.items():
        run_dir = latest(cfg["run"])
        report = latest_report(cfg["report"]) if cfg["report"] else None
        if report and not args.videos_only:
            rep_imgs = [wandb.Image(str(p), caption=f"{scene}:{p.stem}")
                        for p in sorted(report.glob("*.png"))]
            if rep_imgs:
                run.log({f"reports/{scene}": rep_imgs})
            m = parse_probes(report)
            if (report / "episodes.csv").exists():
                rows = read_csv_rows(report / "episodes.csv")
                n = len(rows)
                nsucc = sum(1 for r in rows if r["success"] == "True")
                m["success_rate"] = nsucc / n if n else 0.0
                m["n_episodes"] = n
                tbl = wandb.Table(columns=list(rows[0].keys()),
                                  data=[list(r.values()) for r in rows])
                run.log({f"episodes/{scene}": tbl})
            metric_rows.append([scene, m.get("success_rate"), m.get("decorr_maxcorr"),
                                m.get("decorr_pass"), m.get("skin_engage_frac"), m.get("n_episodes")])
        # 3) sample videos (every scene with a run dir)
        if run_dir:
            vids = pick_videos(run_dir, report)[: args.max_videos_per_scene]
            for path, cap in vids:
                run.log({f"videos/{scene}": wandb.Video(path, caption=cap, format="mp4")})

    if metric_rows and not args.videos_only:
        run.log({"metrics/summary": wandb.Table(
            columns=["scene", "success_rate", "decorr_maxcorr", "decorr_pass",
                     "skin_engage_frac", "n_episodes"], data=metric_rows)})

    # 4) Foxglove dashboards as an artifact (interactive, not viewable inline)
    if DASH_DIR.exists() and not args.videos_only:
        mcaps = list(DASH_DIR.glob("*.mcap"))
        if mcaps:
            art = wandb.Artifact("foxglove_dashboards", type="visualization")
            for m in mcaps:
                art.add_file(str(m))
            layout = DASH_DIR / "foxglove_dashboard_layout.json"
            if layout.exists():
                art.add_file(str(layout))
            run.log_artifact(art)

    print(f"wandb run: {run.url}")
    run.finish()


if __name__ == "__main__":
    main()
