"""W&B progress logging for frozen closed-loop evals.

Default on. Skip if ``--no_wandb`` or wandb is missing. Do not send videos.
Do not change place / bar / free protocol.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_COLLISION_FRAME_KEYS = (
    "hazard_bar",
    "other_environment",
    "clutter",
    "mounted_fixture",
)

try:
    import wandb

    _WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    _WANDB_AVAILABLE = False


def add_cli_flags(parser) -> None:
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disable Weights & Biases logging (on by default).",
    )
    parser.add_argument(
        "--wandb_project",
        default="PC_ACT_experiments",
        help="W&B project. Default PC_ACT_experiments (same as train_exp).",
    )
    parser.add_argument(
        "--wandb_run_name",
        default=None,
        help="W&B run name. Default is the output_dir folder name.",
    )
    parser.add_argument(
        "--wandb_log_every",
        type=int,
        default=50,
        help="Log live step metrics every N control steps. 0 = episode-only.",
    )


def frames_from_audit(summary: dict | None) -> tuple[int, int]:
    """Disallowed-contact frames and hazard-bar frames from an audit summary."""
    frames = (summary or {}).get("frames_with_contact") or {}
    collision = sum(int(frames.get(key) or 0) for key in _COLLISION_FRAME_KEYS)
    hazard = int(frames.get("hazard_bar") or 0)
    return collision, hazard


def frames_from_rec(rec: dict) -> tuple[int, int]:
    """Same counts from an episodes.jsonl row."""
    frames = rec.get("frames_with_contact")
    if isinstance(frames, dict) and frames:
        collision = sum(int(frames.get(key) or 0) for key in _COLLISION_FRAME_KEYS)
        return collision, int(frames.get("hazard_bar") or 0)
    bar = int(rec.get("bar_contact_frames") or 0)
    other = int(rec.get("other_environment_frames") or 0)
    clutter = int(rec.get("clutter_frames") or 0)
    return bar + other + clutter, bar


def _mean_frames(records: list[dict]) -> tuple[float, float]:
    if not records:
        return 0.0, 0.0
    collisions = 0
    hazards = 0
    for rec in records:
        collision, hazard = frames_from_rec(rec)
        collisions += collision
        hazards += hazard
    n = len(records)
    return collisions / n, hazards / n


class EvalWandb:
    """Live eval dashboard. W&B step = episode * horizon + ep_step."""

    def __init__(
        self,
        *,
        enabled: bool,
        log_every: int,
        num_rollouts: int,
        horizon: int,
    ) -> None:
        self.enabled = bool(enabled)
        self.log_every = int(log_every)
        self.num_rollouts = max(1, int(num_rollouts))
        self.horizon = max(1, int(horizon))

    def wandb_step(self, episode: int, ep_step: int) -> int:
        return int(episode) * self.horizon + int(ep_step)

    def maybe_log_step(self, episode: int, ep_step: int, task) -> None:
        if not self.enabled or self.log_every <= 0:
            return
        step = int(ep_step)
        last = step + 1 >= self.horizon
        if step != 0 and not last and (step + 1) % self.log_every != 0:
            return
        audit = getattr(task, "_contact_audit_hook", None)
        summary = audit.summary() if audit is not None and hasattr(audit, "summary") else {}
        collision, hazard = frames_from_audit(summary)
        progress = (int(episode) + step / self.horizon) / self.num_rollouts
        wandb.log(
            {
                "eval/episode": int(episode),
                "eval/ep_step": step,
                "eval/horizon": self.horizon,
                "eval/progress": progress,
                "eval/ep_collision_frames": collision,
                "eval/ep_hazard_frames": hazard,
            },
            step=self.wandb_step(episode, step),
        )

    def log_episode(
        self,
        *,
        episode: int,
        ep_step: int,
        rec: dict,
        summary: dict | None,
        records: list[dict],
    ) -> None:
        if not self.enabled:
            return
        collision, hazard = frames_from_rec(rec)
        mean_collision, mean_hazard = _mean_frames(records)
        payload = {
            "eval/episode": int(episode),
            "eval/ep_step": int(ep_step),
            "eval/horizon": self.horizon,
            "eval/progress": min(1.0, (int(episode) + 1) / self.num_rollouts),
            "eval/episodes_done": len(records),
            "eval/collision_frames": collision,
            "eval/hazard_frames": hazard,
            "eval/collision_frames_mean": mean_collision,
            "eval/hazard_frames_mean": mean_hazard,
        }
        if summary:
            payload["eval/success_rate"] = float(summary.get("success_rate") or 0.0)
            payload["eval/strict_success_rate"] = float(
                summary.get("collision_free_task_success_rate") or 0.0
            )
            payload["eval/collision_rate"] = float(summary.get("collision_rate") or 0.0)
            payload["eval/bar_hit_rate"] = float(summary.get("bar_hit_rate") or 0.0)
            payload["eval/ever_success_rate"] = float(
                summary.get("ever_success_rate") or 0.0
            )
        wandb.log(payload, step=self.wandb_step(episode, ep_step))

    def log_running(self, records: list[dict], summary: dict | None) -> None:
        """After jsonl resume so a restarted job is not a blank dashboard."""
        if not self.enabled or not records:
            return
        last = records[-1]
        episode = int(last.get("episode_idx") or (len(records) - 1))
        self.log_episode(
            episode=episode,
            ep_step=self.horizon - 1,
            rec=last,
            summary=summary,
            records=records,
        )

    def finish(self) -> None:
        if not self.enabled:
            return
        wandb.finish()


def start(
    args,
    *,
    output_dir: Path,
    horizon: int,
    extra_config: dict[str, Any] | None = None,
) -> EvalWandb:
    log_every = int(getattr(args, "wandb_log_every", 50) or 0)
    num_rollouts = int(getattr(args, "num_rollouts", 1) or 1)
    disabled = EvalWandb(
        enabled=False,
        log_every=log_every,
        num_rollouts=num_rollouts,
        horizon=horizon,
    )
    if bool(getattr(args, "no_wandb", False)):
        print("[eval-wandb] disabled (--no_wandb)", flush=True)
        return disabled
    if not _WANDB_AVAILABLE:
        print(
            "[wandb] requested but not installed — run `pip install wandb`. Skipping logging.",
            flush=True,
        )
        return disabled
    name = getattr(args, "wandb_run_name", None) or Path(output_dir).name
    project = str(getattr(args, "wandb_project", None) or "PC_ACT_experiments")
    config = dict(extra_config or {})
    config.setdefault("output_dir", str(Path(output_dir).resolve()))
    config.setdefault("num_rollouts", num_rollouts)
    config.setdefault("horizon", int(horizon))
    config.setdefault("wandb_log_every", log_every)
    orig_argv = sys.argv
    # DETR already shields model build. Shield wandb.init from eval CLI
    # (unknown --wandb_log_every would trip wandb's argv parse).
    sys.argv = [orig_argv[0] if orig_argv else "eval"]
    try:
        run = wandb.init(
            project=project,
            name=str(name),
            dir=str(Path(output_dir).resolve()),
            config=config,
        )
    finally:
        sys.argv = orig_argv
    url = getattr(run, "url", None) or ""
    print(f"[eval-wandb] logging run {name!r} (project {project}): {url}", flush=True)
    return EvalWandb(
        enabled=True,
        log_every=log_every,
        num_rollouts=num_rollouts,
        horizon=int(horizon),
    )
