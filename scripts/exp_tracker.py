#!/usr/bin/env python
"""Experiment ledger. Scans disk, never edits a run.

    python scripts/exp_tracker.py                 # print both tables, write the CSVs
    python scripts/exp_tracker.py --task v1011d   # filter by substring of run / ckpt name
    python scripts/exp_tracker.py --archive       # copy finished eval summaries to reports/eval_summaries/
    python scripts/exp_tracker.py --all           # include scratch dirs (names starting with "_")

Writes ``reports/experiments_evals.csv`` and ``reports/experiments_ckpts.csv``.
Source of truth is ``eval_output/*/episodes.jsonl`` (counts are recomputed, the
summary JSON is only read for protocol fields) and ``submodules/act/ckpts``.

Eval status:
    DONE      records == num_rollouts, hygiene OK
    PARTIAL   fewer records than num_rollouts (do not cite; no resume -> rerun into a new dir)
    DIRTY     mixed seed bases or duplicate episodes (do not cite)
    LEGACY    records have no seed field (pre-frozen evaluators)
    EMPTY     directory without records
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL_ROOT = ROOT / "eval_output"
CKPT_ROOT = ROOT / "submodules" / "act" / "ckpts"
ARCHIVE = ROOT / "reports" / "eval_summaries"
REPORTS = ROOT / "reports"

MIN_CITABLE_N = 20  # below this a run is a wiring smoke, not a rate

_ARM_RE = re.compile(r"_(PACT_READOUT|PACT_RAW|ACT)_s(\d+)")


def _arm_seed(name: str) -> tuple[str, str]:
    m = _ARM_RE.search(name)
    if m:
        return m.group(1), m.group(2)
    low = name.lower()
    for key, arm in (("readout", "PACT_READOUT"), ("raw", "PACT_RAW"), ("vanilla", "ACT"),
                     (r"(?<![a-z])act_", "ACT")):
        if re.search(key, low):
            s = re.search(r"_s(\d+)", low)
            return arm, s.group(1) if s else "?"
    return "?", "?"


def _records(jsonl: Path) -> list[dict]:
    if not jsonl.is_file():
        return []
    return [json.loads(line) for line in jsonl.read_text().splitlines() if line.strip()]


def _hygiene(recs: list[dict]) -> str:
    if not recs:
        return "-"
    if any("seed" not in r for r in recs):
        return "no-seed"
    offsets = Counter(int(r["seed"]) - int(r["episode_idx"]) for r in recs)
    dups = len(recs) - len({(r["episode_idx"], r["seed"]) for r in recs})
    if len(offsets) > 1:
        return f"mixed-seed-base{sorted(offsets)}"
    if dups:
        return f"duplicates={dups}"
    return "ok"


def _count(recs: list[dict], key: str, fallback: str | None = None) -> int:
    return sum(int(bool(r.get(key, r.get(fallback, 0) if fallback else 0))) for r in recs)


def scan_evals(include_scratch: bool) -> list[dict]:
    rows = []
    for d in sorted(p for p in EVAL_ROOT.iterdir() if p.is_dir()):
        if d.name.startswith("_") and not include_scratch:
            continue
        recs = _records(d / "episodes.jsonl")
        summ_path = d / "eval_summary.json"
        summ = json.loads(summ_path.read_text()) if summ_path.is_file() else {}
        proto = summ.get("protocol")
        if not isinstance(proto, dict):  # legacy evaluators store a protocol name string
            proto = {"history_mode": str(proto)} if proto else {}
        ckpt_name = Path(str(summ.get("ckpt_dir") or "")).name
        arm, seed = _arm_seed(ckpt_name or d.name)
        want = summ.get("num_rollouts")
        hyg = _hygiene(recs)
        if not recs:
            status = "EMPTY"
        elif hyg == "no-seed":
            status = "LEGACY"
        elif hyg != "ok":
            status = "DIRTY"
        elif want and len(recs) < int(want):
            status = "PARTIAL"
        else:
            status = "DONE"
        seeds = [int(r["seed"]) for r in recs if "seed" in r]
        lazy = summ.get("lazy_prox_cameras")
        mtime = (d / "episodes.jsonl").stat().st_mtime if recs else d.stat().st_mtime
        rows.append({
            "run": d.name,
            "status": status,
            "task": summ.get("task") or "",
            "arm": arm,
            "train_seed": seed,
            "n": len(recs),
            "n_req": want if want is not None else "",
            "place": _count(recs, "success"),
            "ever": _count(recs, "ever_success", "success"),
            "bar": _count(recs, "hit_bar"),
            "free": _count(recs, "collision_free"),
            "strict": _count(recs, "collision_free_task_success"),
            "seeds": f"{min(seeds)}-{max(seeds)}" if seeds else "",
            "history": proto.get("history_mode") or proto.get("history") or "",
            "clutter_scale": proto.get("clutter_xy_scale", ""),
            "keep_frac": proto.get("sensor_keep_frac", ""),
            "cameras": "+".join(proto.get("cameras") or summ.get("camera_names") or []),
            "horizon": summ.get("task_horizon", ""),
            "path": {True: "fast", False: "original", None: "original(pre-patch)"}.get(lazy, "?") if summ else "",
            "hygiene": hyg,
            "ckpt": ckpt_name,
            "ckpt_sha8": str(summ.get("ckpt_sha256") or "")[:8],
            "script": Path(str(summ.get("script") or "")).name,
            "script_sha8": str(summ.get("script_sha256") or "")[:8],
            "archived": (ARCHIVE / f"{d.name}.json").is_file(),
            "date": time.strftime("%Y-%m-%d %H:%M", time.localtime(mtime)),
        })
    return rows


def _best_epoch(run: Path) -> tuple[str, str]:
    for log in sorted(run.glob("wandb/*/files/output.log"), reverse=True):
        hits = re.findall(r"Best ckpt, val loss ([0-9.]+) @ epoch(\d+)", log.read_text(errors="ignore"))
        if hits:
            return hits[-1][1], hits[-1][0]
    return "", ""


def scan_ckpts(eval_rows: list[dict]) -> list[dict]:
    evaluated: dict[str, list[str]] = {}
    for r in eval_rows:
        if r["ckpt"]:
            evaluated.setdefault(r["ckpt"], []).append(f"{r['run']}[{r['status']} {r['n']}]")
    rows = []
    for task_dir in sorted(p for p in CKPT_ROOT.iterdir() if p.is_dir()):
        for run in sorted(p for p in task_dir.iterdir() if p.is_dir()):
            best = run / "policy_best.ckpt"
            arm, seed = _arm_seed(run.name)
            epoch, val = _best_epoch(run) if best.is_file() else ("", "")
            evals = evaluated.get(run.name, [])
            done = [e for e in evals if re.search(r"\[(DONE|LEGACY) (\d+)\]", e)
                    and int(re.search(r" (\d+)\]", e).group(1)) >= MIN_CITABLE_N]
            rows.append({
                "task": task_dir.name,
                "run": run.name,
                "arm": arm,
                "train_seed": seed,
                "trained": best.is_file(),
                "best_epoch": epoch,
                "best_val": val,
                "date": time.strftime("%Y-%m-%d", time.localtime(best.stat().st_mtime)) if best.is_file() else "",
                "n_done_evals": len(done),
                "evals": "; ".join(evals),
                "todo": "" if done else ("EVAL" if best.is_file() else "no ckpt (aborted?)"),
            })
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _table(rows: list[dict], cols: list[str]) -> str:
    if not rows:
        return "(none)"
    widths = [max(len(c), *(len(str(r[c])) for r in rows)) for c in cols]
    line = lambda vals: "  ".join(str(v).ljust(w) for v, w in zip(vals, widths))
    return "\n".join([line(cols), line("-" * w for w in widths), *(line(r[c] for c in cols) for r in rows)])


def archive(rows: list[dict]) -> None:
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    for r in rows:
        if r["status"] != "DONE" or r["archived"] or r["run"].startswith("_") or r["n"] < MIN_CITABLE_N:
            continue
        shutil.copy2(EVAL_ROOT / r["run"] / "eval_summary.json", ARCHIVE / f"{r['run']}.json")
        r["archived"] = True
        print(f"archived {r['run']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="", help="substring filter on run / ckpt / task name")
    ap.add_argument("--all", action="store_true", help='include scratch dirs named "_*"')
    ap.add_argument("--archive", action="store_true", help="copy DONE summaries into reports/eval_summaries/")
    ap.add_argument("--todo", action="store_true", help="only show what still needs attention")
    args = ap.parse_args()

    evals = scan_evals(args.all)
    ckpts = scan_ckpts(evals)
    if args.archive:
        archive(evals)
    _write_csv(REPORTS / "experiments_evals.csv", evals)
    _write_csv(REPORTS / "experiments_ckpts.csv", ckpts)

    key = args.task.lower()
    ev = [r for r in evals if key in (r["run"] + r["task"] + r["ckpt"]).lower()]
    ck = [r for r in ckpts if key in (r["run"] + r["task"]).lower()]
    if args.todo:
        ev = [r for r in ev if (r["status"] in ("PARTIAL", "DIRTY") and (r["n_req"] or 0) >= MIN_CITABLE_N)
              or (r["status"] == "DONE" and r["n"] >= MIN_CITABLE_N and not r["archived"])]
        ck = [r for r in ck if r["todo"]]

    print(f"\nEVALS ({len(ev)})  place / ever / bar / free / strict are episode counts\n")
    print(_table(ev, ["status", "run", "arm", "train_seed", "n", "n_req", "place", "ever", "bar", "free",
                      "strict", "seeds", "history", "clutter_scale", "keep_frac", "path", "hygiene",
                      "archived", "date"]))
    print(f"\nCHECKPOINTS ({len(ck)})\n")
    print(_table(ck, ["task", "arm", "train_seed", "trained", "best_epoch", "best_val", "n_done_evals",
                      "todo", "run"]))
    bad = [r for r in evals if r["status"] in ("PARTIAL", "DIRTY") and (r["n_req"] or 0) >= MIN_CITABLE_N]
    if bad:
        print("\nDO NOT CITE: " + ", ".join(f"{r['run']} ({r['status']} {r['n']}/{r['n_req']})" for r in bad))
    print(f"\nwrote {REPORTS / 'experiments_evals.csv'}\nwrote {REPORTS / 'experiments_ckpts.csv'}")


if __name__ == "__main__":
    main()
