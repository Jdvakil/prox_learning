#!/usr/bin/env python3
"""Build the self-contained PACT slideshow bundle from frozen artifacts only."""

# ruff: noqa: I001

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE = Path("/root/pact_slideshow_bundle")
REPO_MANIFEST = ROOT / "diagnostics_output/pact_slideshow_bundle_manifest.json"
CONTACT_ROOT = Path("/root/pact_contact_endpoint_artifacts/evaluation_v1")

ANALYSIS = ROOT / "diagnostics_output/pact_contact_endpoint/analysis.json"
FINAL_DECISION = ROOT / "diagnostics_output/pact_contact_endpoint/final_decision.json"
TAIL = ROOT / "diagnostics_output/pact_contact_endpoint/tail_characterization.json"
ENV_GATE = ROOT / "diagnostics_output/pact_vs_act/environment_gate.json"
ENV_CONFIG = ROOT / "configs/pact_collision_environment_v2.json"
SCHEDULE = ROOT / "diagnostics_output/pact_contact_endpoint/schedule.json"
FRONTEND_ANALYSIS = ROOT / "diagnostics_output/pact_frontend_screen/analysis.json"
FRONTEND_DECISION = ROOT / "diagnostics_output/pact_frontend_screen/final_decision.json"
VALID_ANALYSIS = ROOT / "diagnostics_output/pact_valid_ablation/analysis.json"
VALID_DECISION = ROOT / "diagnostics_output/pact_valid_ablation/final_decision.json"
SEED_ANALYSIS = ROOT / "diagnostics_output/pact_seed_replication/analysis.json"
SEED_DECISION = ROOT / "diagnostics_output/pact_seed_replication/final_decision.json"
EARLY_ANALYSIS = ROOT / "diagnostics_output/pact_vs_act/analysis.json"
EARLY_DECISION = ROOT / "diagnostics_output/pact_vs_act/final_decision.json"
POLICY_REGISTRY = ROOT / "diagnostics_output/pact_contact_endpoint/policy_training.json"
QUALITATIVE_MANIFEST = (
    ROOT / "diagnostics_output/pact_contact_endpoint/qualitative_video_manifest.json"
)
QUALITATIVE_CLIPS_V2_MANIFEST = (
    ROOT
    / "diagnostics_output/pact_contact_endpoint/qualitative_clips_v2_manifest.json"
)
CORRIDOR_XML = (
    ROOT
    / "submodules/molmospaces/molmo_spaces/data_generation/custom_scenes/"
    "pact_collision_corridor.xml"
)

HEATMAP_SOURCE = Path(
    "/root/pact_contact_endpoint_artifacts/qualitative_videos/reruns/video_01_act/"
    "episode_00000000_sensors_depth8_heatmap.mp4"
)
QUALITATIVE_CLIPS_V2_RELEASE = Path(
    "/root/pact_contact_endpoint_artifacts/qualitative_clips_v2/release"
)
QUALITATIVE_CLIPS_V2_NAMES = (
    "clip1_54a6272f66ca_pact_success",
    "clip2_54a6272f66ca_act_failure",
    "clip3_e99dc657bfa7_act_success",
    "clip4_e99dc657bfa7_pact_failure",
)
EXPERT_RESULT = (
    ROOT
    / "assets/datagen/pact_collision_corridor_v2/full_cba7ff88/rows/"
    "0006bcf3ffb09c849a6cd5d0ebc78ff17ecb727b75847b796b54560990a4aa0d/"
    "result.json"
)
EXPERT_VIDEO = EXPERT_RESULT.parent / "episode_00000000_wrist_camera.mp4"

TRAINING_LOGS = {
    "act_seed3101.jsonl": Path(
        "/root/pact_remediation_artifacts_v2/full/policies_v2/act_seed3101/epoch_log.jsonl"
    ),
    "pact_seed3101.jsonl": Path(
        "/root/pact_frontend_screen_artifacts/policy_seed3101/epoch_log.jsonl"
    ),
    "act_seed3102.jsonl": Path(
        "/root/pact_remediation_artifacts_v2/full/policies_v2/act_seed3102/epoch_log.jsonl"
    ),
    "pact_seed3102.jsonl": Path(
        "/root/pact_seed_replication_artifacts/policy_seed3102/epoch_log.jsonl"
    ),
    "act_seed3103.jsonl": Path(
        "/root/pact_contact_endpoint_artifacts/policies/act_seed3103/epoch_log.jsonl"
    ),
    "pact_seed3103.jsonl": Path(
        "/root/pact_contact_endpoint_artifacts/policies/pact_seed3103/epoch_log.jsonl"
    ),
}

SOURCE_COPIES = {
    "analysis.json": ANALYSIS,
    "final_decision.json": FINAL_DECISION,
    "tail_characterization.json": TAIL,
    "environment_gate.json": ENV_GATE,
    "pact_collision_environment_v2.json": ENV_CONFIG,
    "schedule.json": SCHEDULE,
    "frontend_screen_analysis.json": FRONTEND_ANALYSIS,
    "frontend_screen_final_decision.json": FRONTEND_DECISION,
    "valid_ablation_analysis.json": VALID_ANALYSIS,
    "valid_ablation_final_decision.json": VALID_DECISION,
    "seed_replication_analysis.json": SEED_ANALYSIS,
    "seed_replication_final_decision.json": SEED_DECISION,
    "early_3d_analysis.json": EARLY_ANALYSIS,
    "early_3d_final_decision.json": EARLY_DECISION,
    "policy_training_registry.json": POLICY_REGISTRY,
    "qualitative_video_manifest.json": QUALITATIVE_MANIFEST,
    "qualitative_clips_v2_manifest.json": QUALITATIVE_CLIPS_V2_MANIFEST,
    "pact_collision_corridor.xml": CORRIDOR_XML,
    "expert_demo_result.json": EXPERT_RESULT,
}

REPORT_COPIES = {
    "PACT_CONTACT_ENDPOINT_DECISION.md": ROOT / "docs/PACT_CONTACT_ENDPOINT_DECISION.md",
    "PACT_TAIL_CHARACTERIZATION.md": ROOT / "docs/PACT_TAIL_CHARACTERIZATION.md",
    "PACT_ENVIRONMENT_ADEQUACY.md": ROOT / "docs/PACT_ENVIRONMENT_ADEQUACY.md",
    "PACT_FRONTEND_SCREEN_DECISION.md": ROOT / "docs/PACT_FRONTEND_SCREEN_DECISION.md",
    "PACT_VALID_ABLATION_DECISION.md": ROOT / "docs/PACT_VALID_ABLATION_DECISION.md",
    "PACT_SEED_REPLICATION_DECISION.md": ROOT / "docs/PACT_SEED_REPLICATION_DECISION.md",
    "PACT_VS_ACT_FINAL_DECISION.md": ROOT / "docs/PACT_VS_ACT_FINAL_DECISION.md",
    "PACT_QUALITATIVE_VIDEOS.md": ROOT / "docs/PACT_QUALITATIVE_VIDEOS.md",
    "DISCOVERY_CONTACT_TAIL.md": (
        ROOT
        / "docs/discoveries/003-proximity-reduces-obstacle-contact-in-the-tail-not-in-routine-operation.md"
    ),
}

COLORS = {
    "ACT": "#667085",
    "PACT": "#008A78",
    "PACT_PERMUTED": "#F79009",
    "PACT_ZERO": "#D92D20",
    "ink": "#101828",
    "muted": "#667085",
    "grid": "#D0D5DD",
    "panel": "#F2F4F7",
    "success": "#027A48",
}
ARM_ORDER = ("ACT", "PACT", "PACT_PERMUTED", "PACT_ZERO")
ARM_LABELS = {
    "ACT": "ACT\nvision only",
    "PACT": "PACT\nvision + proximity",
    "PACT_PERMUTED": "PACT_PERMUTED\nvalid ablation",
    "PACT_ZERO": "PACT_ZERO\nOOD failure probe",
}


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def require_sources() -> None:
    required = list(SOURCE_COPIES.values()) + list(REPORT_COPIES.values())
    required += list(TRAINING_LOGS.values())
    required += [
        HEATMAP_SOURCE,
        EXPERT_VIDEO,
        QUALITATIVE_CLIPS_V2_RELEASE / "README.md",
    ]
    release_manifest = load(QUALITATIVE_CLIPS_V2_MANIFEST)
    required += [
        Path(record["release_video_path"])
        for record in release_manifest.get("render_outputs", [])
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing slideshow sources:\n" + "\n".join(missing))


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (16, 9),
            "figure.dpi": 100,
            "savefig.dpi": 200,
            "font.family": "DejaVu Sans",
            "font.size": 14,
            "axes.titlesize": 22,
            "axes.labelsize": 15,
            "axes.edgecolor": COLORS["grid"],
            "axes.labelcolor": COLORS["ink"],
            "axes.titlecolor": COLORS["ink"],
            "xtick.color": COLORS["muted"],
            "ytick.color": COLORS["muted"],
            "grid.color": COLORS["grid"],
            "grid.alpha": 0.55,
            "legend.frameon": False,
            "svg.hashsalt": "pact_slideshow_bundle_v1",
        }
    )


def slide_title(fig, title: str, subtitle: str) -> None:
    fig.text(0.055, 0.945, title, ha="left", va="top", fontsize=27, weight="bold", color=COLORS["ink"])
    fig.text(0.055, 0.895, subtitle, ha="left", va="top", fontsize=14, color=COLORS["muted"])


def footer(fig, source: str, caveat: str | None = None) -> None:
    fig.text(0.055, 0.025, f"Source: {source}", fontsize=9.5, color=COLORS["muted"], va="bottom")
    if caveat:
        fig.text(0.945, 0.025, caveat, fontsize=9.5, color=COLORS["muted"], va="bottom", ha="right")


def save_figure(fig, figure_dir: Path, stem: str) -> list[Path]:
    outputs = []
    for suffix in (".png", ".svg"):
        path = figure_dir / f"{stem}{suffix}"
        metadata = {"Date": None} if suffix == ".svg" else None
        fig.savefig(path, facecolor="white", metadata=metadata)
        outputs.append(path)
    plt.close(fig)
    return outputs


def annotate_bars(ax, bars, *, percent: bool = True, decimals: int = 1) -> None:
    for bar in bars:
        value = float(bar.get_height())
        label = f"{value * 100:.{decimals}f}%" if percent else f"{value:.{decimals}f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            label,
            ha="center",
            va="bottom",
            fontsize=13,
            weight="bold",
            color=COLORS["ink"],
        )


def figure_summary(analysis: dict[str, Any], figure_dir: Path) -> list[Path]:
    contrast = analysis["decision_rule_inputs"]["pact_minus_act_collision_free_task_success"]
    fig = plt.figure(figsize=(16, 9))
    slide_title(fig, "PACT: confirmed contact reduction", "One-page result summary | frozen 1,200-rollout contact endpoint")
    fig.text(0.08, 0.64, "22.3%", fontsize=63, weight="bold", color=COLORS["ACT"])
    fig.text(0.31, 0.665, "→", fontsize=46, weight="bold", color=COLORS["muted"])
    fig.text(0.42, 0.64, "14.0%", fontsize=63, weight="bold", color=COLORS["PACT"])
    fig.text(0.08, 0.56, "ACT hazard contact", fontsize=17, color=COLORS["muted"])
    fig.text(0.42, 0.56, "PACT hazard contact", fontsize=17, color=COLORS["muted"])
    fig.text(0.08, 0.45, "Confirmed against valid ablation", fontsize=24, weight="bold", color=COLORS["ink"])
    fig.text(0.08, 0.395, "PACT − PACT_PERMUTED contact rate: −9.3 pp", fontsize=19, color=COLORS["success"])
    fig.text(0.08, 0.32, "Task endpoint remains directional", fontsize=24, weight="bold", color=COLORS["ink"])
    ci = contrast["instance_cluster_bootstrap_ci_95"]
    fig.text(
        0.08,
        0.265,
        f"PACT − ACT collision-free task success: +{contrast['difference']*100:.1f} pp "
        f"[{ci[0]*100:+.1f}, {ci[1]*100:+.1f}]",
        fontsize=18,
        color=COLORS["muted"],
    )
    for x, big, small in [
        (0.68, "1,200", "rollouts"),
        (0.80, "100", "held-out instances"),
        (0.68, "3", "policy seeds"),
        (0.80, "4", "evaluation arms"),
    ]:
        fig.text(x, 0.66 if big in ("1,200", "100") else 0.42, big, fontsize=35, weight="bold", color=COLORS["ink"], ha="center")
        fig.text(x, 0.61 if big in ("1,200", "100") else 0.37, small, fontsize=13, color=COLORS["muted"], ha="center")
    fig.text(0.68, 0.22, "PRE-REGISTERED", fontsize=18, weight="bold", color="white", ha="center", va="center", bbox={"boxstyle": "round,pad=0.55", "facecolor": COLORS["PACT"], "edgecolor": "none"})
    fig.text(0.80, 0.22, "3 SEEDS", fontsize=18, weight="bold", color="white", ha="center", va="center", bbox={"boxstyle": "round,pad=0.55", "facecolor": COLORS["ink"], "edgecolor": "none"})
    footer(
        fig,
        "analysis.json: pooled_arm_summaries; decision_rule_inputs",
        "Task success is not confirmed; PACT_ZERO is OOD and not modality evidence.",
    )
    return save_figure(fig, figure_dir, "fig00_one_page_summary")


def figure_headline(analysis: dict[str, Any], figure_dir: Path) -> list[Path]:
    summaries = analysis["pooled_arm_summaries"]
    rates = np.asarray([summaries[arm]["hazard_bar_any_contact"]["rate"] for arm in ARM_ORDER])
    intervals = [summaries[arm]["hazard_bar_any_contact"]["wilson_95"] for arm in ARM_ORDER]
    errors = np.asarray([[rate - ci[0], ci[1] - rate] for rate, ci in zip(rates, intervals)]).T
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.subplots_adjust(left=0.09, right=0.96, bottom=0.20, top=0.80)
    slide_title(fig, "Whole-body proximity cuts hazard-contact incidence", "Any hazard contact per rollout | pooled across 100 instances × 3 seeds")
    x = np.arange(len(ARM_ORDER))
    bars = ax.bar(x, rates, width=0.62, color=[COLORS[arm] for arm in ARM_ORDER], yerr=errors, capsize=8, error_kw={"elinewidth": 2, "capthick": 2})
    annotate_bars(ax, bars)
    ax.set_xticks(x, [ARM_LABELS[arm] for arm in ARM_ORDER])
    ax.set_ylabel("Rollouts with hazard contact")
    ax.set_ylim(0, 0.45)
    ax.set_yticks(np.arange(0, 0.46, 0.10), [f"{value:.0%}" for value in np.arange(0, 0.46, 0.10)])
    ax.grid(axis="y")
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(3, 0.425, "OUT-OF-DISTRIBUTION\nsensor-failure probe", ha="center", va="top", fontsize=12, weight="bold", color=COLORS["PACT_ZERO"])
    footer(fig, "analysis.json: pooled_arm_summaries.<arm>.hazard_bar_any_contact.{rate,wilson_95}", "PACT_ZERO is not a valid modality condition.")
    return save_figure(fig, figure_dir, "fig01_contact_rate_by_arm")


def figure_mechanism(tail: dict[str, Any], figure_dir: Path) -> list[Path]:
    arms = ("ACT", "PACT", "PACT_PERMUTED")
    entry = [tail["arms"][arm]["high_contact_regime"]["entry_fraction"] for arm in arms]
    first = [tail["arms"][arm]["high_contact_regime"]["first_hazard_contact_step"]["median"] for arm in arms]
    fig, (left, right) = plt.subplots(1, 2, figsize=(16, 9))
    fig.subplots_adjust(left=0.075, right=0.96, bottom=0.18, top=0.78, wspace=0.28)
    slide_title(fig, "The modality changes whether susceptible cases enter the tail", "Exploratory mechanism characterization | high-contact regime = >500 hazard frames")
    x = np.arange(3)
    bars = left.bar(x, entry, width=0.62, color=[COLORS[arm] for arm in arms])
    annotate_bars(left, bars)
    left.set_title("Entry into >500-frame regime", weight="bold")
    left.set_xticks(x, ["ACT", "PACT", "PACT_PERMUTED"])
    left.set_ylim(0, 0.25)
    left.set_yticks(np.arange(0, 0.26, 0.05), [f"{v:.0%}" for v in np.arange(0, 0.26, 0.05)])
    left.grid(axis="y")
    left.spines[["top", "right"]].set_visible(False)
    bars2 = right.bar(x, first, width=0.62, color=[COLORS[arm] for arm in arms])
    annotate_bars(right, bars2, percent=False, decimals=0)
    right.set_title("Median first-contact step", weight="bold")
    right.set_xticks(x, ["ACT", "PACT", "PACT_PERMUTED"])
    right.set_ylabel("Control step out of 900")
    right.set_ylim(0, 340)
    right.grid(axis="y")
    right.spines[["top", "right"]].set_visible(False)
    right.annotate("+232 steps vs ACT", xy=(1, 291), xytext=(1.45, 325), arrowprops={"arrowstyle": "->", "color": COLORS["PACT"]}, color=COLORS["PACT"], weight="bold")
    footer(fig, "tail_characterization.json: arms.<arm>.high_contact_regime", "Post-hoc descriptive; no new inference.")
    return save_figure(fig, figure_dir, "fig02_tail_entry_and_first_contact")


def extract_hazard_frames(schedule: dict[str, Any]) -> dict[str, list[int]]:
    values: dict[str, list[int]] = defaultdict(list)
    for row in schedule["rows"]:
        path = CONTACT_ROOT / row["output_relpath"] / "result.json"
        result = load(path)
        identity = (result.get("rollout_id"), result.get("arm"), result.get("episode_id"))
        expected = (row["rollout_id"], row["arm"], row["instance_episode_id"])
        if identity != expected or result.get("status") != "complete":
            raise ValueError(f"contact-result identity mismatch: {path}")
        values[row["arm"]].append(int(result["contact_audit"]["frames_with_contact"]["hazard_bar"]))
    if {arm: len(values[arm]) for arm in ARM_ORDER} != {arm: 300 for arm in ARM_ORDER}:
        raise ValueError("hazard-frame extraction did not yield 300 values per arm")
    return dict(values)


def write_hazard_frame_data(bundle: Path, values: dict[str, list[int]], tail: dict[str, Any]) -> Path:
    document = {
        "schema_version": "pact_slideshow_hazard_frame_values_v1",
        "decision_bearing": False,
        "purpose": "exact visualization values for Figure 3; no inferential analysis",
        "metric": "contact_audit.frames_with_contact.hazard_bar",
        "source_result_root": str(CONTACT_ROOT),
        "source_result_files": 1200,
        "source_result_inventory_sha256": tail["sources"]["result_file_inventory_sha256"],
        "values_by_arm": values,
    }
    document["hazard_frame_values_sha256"] = canonical_hash(document)
    path = bundle / "data/hazard_frames_by_arm.json"
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    return path


def figure_distribution(values: dict[str, list[int]], figure_dir: Path) -> list[Path]:
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.subplots_adjust(left=0.09, right=0.96, bottom=0.16, top=0.79)
    slide_title(fig, "Means hide a zero-inflated, bimodal contact distribution", "Empirical CDF of all 300 rollouts per arm | x-axis is log10(hazard frames + 1)")
    for arm in ARM_ORDER:
        ordered = np.sort(np.asarray(values[arm], dtype=float))
        y = np.arange(1, len(ordered) + 1) / len(ordered)
        x = np.log10(ordered + 1.0)
        style = "--" if arm == "PACT_ZERO" else "-"
        ax.step(x, y, where="post", label=ARM_LABELS[arm].replace("\n", " "), color=COLORS[arm], linewidth=3 if arm == "PACT" else 2.2, linestyle=style)
    ticks_raw = np.asarray([0, 1, 10, 100, 1000, 10000, 30000])
    ax.set_xticks(np.log10(ticks_raw + 1), ["0", "1", "10", "100", "1k", "10k", "30k"])
    ax.set_xlabel("Hazard frames per rollout")
    ax.set_ylabel("Cumulative share of rollouts")
    ax.set_yticks(np.arange(0, 1.01, 0.1), [f"{v:.0%}" for v in np.arange(0, 1.01, 0.1)])
    ax.set_xlim(-0.03, np.log10(30001))
    ax.set_ylim(0, 1.01)
    ax.grid()
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", fontsize=12)
    ax.text(0.03, 0.73, "77–86% are exactly zero\n(non-OOD arms)", transform=ax.transAxes, fontsize=17, weight="bold", color=COLORS["ink"], bbox={"boxstyle": "round,pad=0.5", "facecolor": "white", "edgecolor": COLORS["grid"]})
    footer(fig, "hazard_frames_by_arm.json, extracted verbatim from 1,200 frozen result.json files", "PACT_ZERO is OOD; ECDF is descriptive.")
    return save_figure(fig, figure_dir, "fig03_hazard_frame_ecdf")


def figure_concentration(tail: dict[str, Any], figure_dir: Path) -> list[Path]:
    arms = ("ACT", "PACT", "PACT_PERMUTED")
    x = np.asarray([1, 5, 10])
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.subplots_adjust(left=0.10, right=0.95, bottom=0.17, top=0.79)
    slide_title(fig, "A handful of episodes carries nearly all PACT contact", "Cumulative share of each arm's total hazard frames in its highest-contact episodes")
    for arm in arms:
        c = tail["arms"][arm]["concentration"]
        y = np.asarray([c[f"top_{k}_percent"]["share_of_arm_hazard_frames"] for k in x])
        ax.plot(x, y, marker="o", markersize=11, linewidth=3, color=COLORS[arm], label=arm)
        for xv, yv in zip(x, y):
            ax.text(xv, yv + 0.025, f"{yv:.1%}", ha="center", fontsize=11, color=COLORS[arm], weight="bold")
    ax.set_xlabel("Top-k% of episodes, ranked by hazard frames")
    ax.set_ylabel("Share of arm's total hazard frames")
    ax.set_xticks(x, ["top 1%", "top 5%", "top 10%"])
    ax.set_ylim(0, 1.08)
    ax.set_yticks(np.arange(0, 1.01, 0.2), [f"{v:.0%}" for v in np.arange(0, 1.01, 0.2)])
    ax.grid(axis="y")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left")
    ax.text(5.25, 0.70, "PACT\ntop 5% = 61.7%\ntop 10% = 98.6%", fontsize=18, weight="bold", color=COLORS["PACT"], bbox={"boxstyle": "round,pad=0.6", "facecolor": "#E7F6F3", "edgecolor": COLORS["PACT"]})
    footer(fig, "tail_characterization.json: arms.<arm>.concentration", "Available frozen cut points only (1%, 5%, 10%).")
    return save_figure(fig, figure_dir, "fig04_contact_concentration")


def figure_replication(analysis: dict[str, Any], figure_dir: Path) -> list[Path]:
    seeds = ("3101", "3102", "3103")
    contact = analysis["decision_rule_inputs"]["seed_modality_contacts"]
    success = analysis["decision_rule_inputs"]["seed_pact_minus_act_collision_free_task_success"]
    left_values = [contact[seed]["difference"] for seed in seeds]
    right_values = [success[seed]["difference"] * 100 for seed in seeds]
    fig, (left, right) = plt.subplots(1, 2, figsize=(16, 9))
    fig.subplots_adjust(left=0.08, right=0.96, bottom=0.18, top=0.78, wspace=0.30)
    slide_title(fig, "The contact-magnitude contrast replicates; task success does not", "Same three PACT trainings, two different contrasts")
    x = np.arange(3)
    bars = left.bar(x, left_values, color=["#12B76A"] * 3, width=0.60)
    for bar, value in zip(bars, left_values):
        left.text(
            bar.get_x() + bar.get_width() / 2,
            value + 100,
            f"{value:,.0f}",
            ha="center",
            va="bottom",
            color="white",
            weight="bold",
        )
    left.axhline(0, color=COLORS["ink"], linewidth=1)
    left.set_xticks(x, seeds)
    left.set_title("PACT − PACT_PERMUTED\nmean hazard frames", weight="bold")
    left.set_ylabel("Frames per rollout (lower is better)")
    left.grid(axis="y")
    left.spines[["top", "right"]].set_visible(False)
    bars2 = right.bar(x, right_values, color=[COLORS["PACT"]] * 3, width=0.60)
    for bar, value in zip(bars2, right_values):
        right.text(bar.get_x() + bar.get_width() / 2, value + 0.25, f"+{value:.1f} pp", ha="center", va="bottom", weight="bold", color=COLORS["PACT"])
    right.axhline(0, color=COLORS["ink"], linewidth=1)
    right.set_xticks(x, seeds)
    right.set_title("PACT − ACT\ncollision-free task success", weight="bold")
    right.set_ylabel("Percentage-point difference")
    right.set_ylim(-1, 10)
    right.grid(axis="y")
    right.spines[["top", "right"]].set_visible(False)
    footer(fig, "analysis.json: decision_rule_inputs.{seed_modality_contacts,seed_pact_minus_act_collision_free_task_success}", "Contact is modality-specific; success combines modality and policy differences.")
    return save_figure(fig, figure_dir, "fig05_seed_replication")


def figure_measurement_journey(
    early: dict[str, Any], frontend: dict[str, Any], valid: dict[str, Any], seed: dict[str, Any], contact: dict[str, Any], figure_dir: Path
) -> list[Path]:
    stages = [
        ("3-D front-end\nzero ablation", early["paired_instance_bootstrap"]["PACT_minus_PACT_ZERO"]["difference"] * 100, "no effect detected", COLORS["muted"]),
        ("32-D front-end\nzero ablation", frontend["paired_instance_bootstrap"]["PACT_minus_PACT_ZERO"]["difference"] * 100, "ARTIFACT — OOD", COLORS["PACT_ZERO"]),
        ("32-D front-end\npermuted ablation", valid["paired_instance_bootstrap"]["PACT_minus_PACT_PERMUTED"]["difference"] * 100, "+12.5 / +15.0 pp, replicated", COLORS["PACT"]),
        ("Contact endpoint\nvalid ablation", contact["pooled_contrasts"]["PACT_minus_PACT_PERMUTED"]["hazard_bar_any_contact"]["difference"] * 100, "confirmed contact reduction", COLORS["success"]),
    ]
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.subplots_adjust(left=0.28, right=0.95, bottom=0.16, top=0.78)
    slide_title(fig, "Measurement validity changed the apparent answer", "Ablation and endpoint journey | signed effect in percentage points")
    y = np.arange(len(stages))[::-1]
    values = [stage[1] for stage in stages]
    bars = ax.barh(y, values, color=[stage[3] for stage in stages], height=0.56)
    ax.axvline(0, color=COLORS["ink"], linewidth=1.5)
    ax.set_yticks(y, [stage[0] for stage in stages])
    ax.set_xlabel("PACT − ablation (pp); contact endpoint is lower-is-better")
    ax.set_xlim(-15, 80)
    ax.grid(axis="x")
    ax.spines[["top", "right", "left"]].set_visible(False)
    for bar, stage in zip(bars, stages):
        value = stage[1]
        if value > 50:
            xtext, horizontal_alignment, text_color = value - 1.6, "right", "white"
        elif abs(value) < 1:
            xtext, horizontal_alignment, text_color = 1.6, "left", stage[3]
        elif value >= 0:
            xtext, horizontal_alignment, text_color = value + 1.6, "left", stage[3]
        else:
            xtext, horizontal_alignment, text_color = 1.6, "left", stage[3]
        ax.text(
            xtext,
            bar.get_y() + bar.get_height() / 2,
            f"{value:+.1f} pp  |  {stage[2]}",
            va="center",
            ha=horizontal_alignment,
            fontsize=14,
            weight="bold",
            color=text_color,
        )
    footer(
        fig,
        "copied analyses: 3-D, 32-D, permutation replication, contact endpoint",
        "Zeroing is OOD; permutation is the valid modality ablation.",
    )
    return save_figure(fig, figure_dir, "fig06_measurement_journey")


def figure_seed_noise(seed_analysis: dict[str, Any], contact_analysis: dict[str, Any], figure_dir: Path) -> list[Path]:
    old = seed_analysis["seed_results_unpooled_first"]
    old_seeds = ("3101", "3102")
    old_values, old_low, old_high = [], [], []
    for seed in old_seeds:
        record = old[seed]["contrasts"]["PACT_minus_ACT"]
        old_values.append(record["difference"] * 100)
        old_low.append(record["paired_bootstrap_ci_95"][0] * 100)
        old_high.append(record["paired_bootstrap_ci_95"][1] * 100)
    new = contact_analysis["decision_rule_inputs"]["seed_pact_minus_act_collision_free_task_success"]
    new_seeds = ("3101", "3102", "3103")
    new_values = [new[s]["difference"] * 100 for s in new_seeds]
    new_low = [new[s]["instance_bootstrap_ci_95"][0] * 100 for s in new_seeds]
    new_high = [new[s]["instance_bootstrap_ci_95"][1] * 100 for s in new_seeds]
    fig, (left, right) = plt.subplots(1, 2, figsize=(16, 9), sharey=True)
    fig.subplots_adjust(left=0.09, right=0.96, bottom=0.18, top=0.78, wspace=0.14)
    slide_title(fig, "More instances resolved the apparent seed conflict", "PACT − ACT collision-free task success; intervals resample whole instances")
    for ax, seeds, vals, lows, highs, title in [
        (left, old_seeds, old_values, old_low, old_high, "40 instances per seed"),
        (right, new_seeds, new_values, new_low, new_high, "100 instances per seed"),
    ]:
        x = np.arange(len(seeds))
        err = np.asarray([[v - lo for v, lo in zip(vals, lows)], [hi - v for v, hi in zip(vals, highs)]])
        ax.errorbar(x, vals, yerr=err, fmt="o", markersize=12, capsize=8, linewidth=2.5, color=COLORS["PACT"])
        ax.axhline(0, color=COLORS["ink"], linewidth=1.2)
        ax.set_xticks(x, seeds)
        ax.set_title(title, weight="bold")
        ax.set_xlabel("Policy seed")
        ax.grid(axis="y")
        ax.spines[["top", "right"]].set_visible(False)
        for xv, v in zip(x, vals):
            ax.text(xv, v + 2.2, f"{v:+.1f} pp", ha="center", weight="bold", color=COLORS["PACT"])
    left.set_ylabel("Percentage-point difference")
    left.set_ylim(-32, 49)
    footer(
        fig,
        "40-instance seed analysis; 100-instance contact analysis",
        "40-instance CIs overlapped; task benefit remains unconfirmed at 100.",
    )
    return save_figure(fig, figure_dir, "fig07_seed_noise_resolution")


def figure_environment(env_config: dict[str, Any], figure_dir: Path) -> list[Path]:
    scene = env_config["scene"]
    base_x = float(scene["robot_base_forward_m"])
    panel_x = float(scene["panel_center_nominal_m"][0])
    panel_y = float(str(scene["panel_center_nominal_m"][1]).split("*")[1])
    panel_half_x, panel_half_y, panel_half_z = map(
        float, scene["panel_half_extents_m"]
    )
    aperture_width = float(scene["aperture_width_m"])
    inner_face_y = float(scene["panel_inner_face_nominal_abs_y_m"])
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.subplots_adjust(left=0.10, right=0.94, bottom=0.12, top=0.80)
    slide_title(fig, "Collision-necessity environment: narrow, side-ambiguous approach", "Top-down schematic (meters) | exactly one left/right panel is active per instance")
    ax.set_aspect("equal")
    ax.set_xlim(-0.05, 1.5)
    ax.set_ylim(-0.68, 0.68)
    ax.set_xlabel("Forward x (m)")
    ax.set_ylabel("Lateral y (m)")
    ax.add_patch(Rectangle((0.55, -aperture_width / 2), 0.83, aperture_width, facecolor="#F9FAFB", edgecolor=COLORS["ink"], linewidth=2, label="hood workspace"))
    ax.plot([0.58, 0.58], [-0.425, 0.425], color=COLORS["ink"], linewidth=5)
    ax.text(0.60, 0.47, f"{aperture_width:.2f} m aperture", fontsize=13, weight="bold", ha="center")
    for side, sign in (("LEFT", 1), ("RIGHT", -1)):
        center_y = sign * panel_y
        panel = Rectangle((panel_x - panel_half_x, center_y - panel_half_y), 2 * panel_half_x, 2 * panel_half_y, facecolor="#FDA29B", edgecolor=COLORS["PACT_ZERO"], linewidth=2, alpha=0.72, linestyle="--")
        ax.add_patch(panel)
        ax.text(0.69, center_y, f"{side} panel\n{2 * panel_half_x:.2f} × {2 * panel_half_y:.2f} m top view", fontsize=11, color=COLORS["PACT_ZERO"], va="center")
    target = Circle((0.76, 0), 0.035, facecolor=COLORS["PACT"], edgecolor="white", linewidth=2, zorder=5)
    ax.add_patch(target)
    ax.plot([0.76, 0.76], [-0.04, 0.04], color=COLORS["PACT"], linewidth=7, alpha=0.35)
    ax.text(0.82, 0.01, "target\nx ≈ 0.76\ny ∈ [−0.04, 0.04]", fontsize=12, color=COLORS["PACT"], va="center")
    robot = Circle((base_x, 0), 0.065, facecolor=COLORS["ACT"], edgecolor=COLORS["ink"], linewidth=2)
    ax.add_patch(robot)
    ax.text(base_x, -0.12, f"robot base\nforward {base_x:.2f} m", ha="center", fontsize=12)
    ax.add_patch(FancyArrowPatch((0.20, 0), (0.70, 0), arrowstyle="->", mutation_scale=18, linewidth=2.5, color=COLORS["ACT"], connectionstyle="arc3,rad=0"))
    ax.add_patch(FancyArrowPatch((0.20, 0), (0.70, -0.13), arrowstyle="->", mutation_scale=18, linewidth=2.5, color=COLORS["PACT"], connectionstyle="arc3,rad=-0.25"))
    ax.text(0.37, 0.045, "nominal approach", fontsize=11, color=COLORS["ACT"])
    ax.text(0.37, -0.24, "expert bows away\nfrom active side", fontsize=11, color=COLORS["PACT"])
    ax.annotate(f"inner face |y| = {inner_face_y:.2f} m", xy=(panel_x, inner_face_y), xytext=(1.02, 0.28), arrowprops={"arrowstyle": "->", "color": COLORS["PACT_ZERO"]}, fontsize=12, color=COLORS["PACT_ZERO"])
    ax.text(1.03, -0.50, f"Panel half-extents:\n{panel_half_x:.3f} × {panel_half_y:.3f} × {panel_half_z:.3f} m\ncenter x = {panel_x:.3f} m", fontsize=12, bbox={"boxstyle": "round,pad=0.5", "facecolor": "white", "edgecolor": COLORS["grid"]})
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    footer(fig, "pact_collision_environment_v2.json: scene; pact_collision_corridor.xml", "Schematic, not a rendered camera view; one panel active per episode.")
    return save_figure(fig, figure_dir, "fig08_environment_schematic")


def read_training_logs() -> dict[tuple[str, int], list[dict[str, Any]]]:
    records = {}
    for name, path in TRAINING_LOGS.items():
        arm, seed_text = name.removesuffix(".jsonl").split("_seed")
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if len(rows) != 2000 or [row["epoch"] for row in rows] != list(range(2000)):
            raise ValueError(f"training log is not a complete 2,000-epoch series: {path}")
        records[(arm.upper(), int(seed_text))] = rows
    return records


def figure_training(records: dict[tuple[str, int], list[dict[str, Any]]], figure_dir: Path) -> list[Path]:
    fig, (full, late) = plt.subplots(1, 2, figsize=(16, 9))
    fig.subplots_adjust(left=0.08, right=0.96, bottom=0.17, top=0.78, wspace=0.25)
    slide_title(fig, "Validation loss did not predict the modality result", "ACT vs 32-D PACT across all three frozen 2,000-epoch trainings")
    seed_colors = {3101: "#175CD3", 3102: "#7A5AF8", 3103: "#DC6803"}
    for seed in (3101, 3102, 3103):
        for arm, linestyle in (("ACT", "-"), ("PACT", "--")):
            rows = records[(arm, seed)]
            epochs = np.asarray([row["epoch"] for row in rows])
            losses = np.asarray([row["val"]["loss"] for row in rows])
            label = f"{arm} seed {seed}"
            full.plot(epochs, losses, color=seed_colors[seed], linestyle=linestyle, linewidth=1.6, alpha=0.9, label=label)
            late.plot(epochs, losses, color=seed_colors[seed], linestyle=linestyle, linewidth=1.3, alpha=0.75)
            best_index = int(np.argmin(losses))
            late.scatter([epochs[best_index]], [losses[best_index]], color=seed_colors[seed], marker="o" if arm == "ACT" else "s", s=48, zorder=5)
    full.set_yscale("log")
    full.set_title("Full optimization", weight="bold")
    full.set_xlabel("Epoch")
    full.set_ylabel("Validation loss (log scale)")
    full.grid()
    full.legend(fontsize=9, ncol=2)
    late.set_title("Late training / checkpoint minima", weight="bold")
    late.set_xlim(1400, 2000)
    late.set_ylim(0.08, 0.125)
    late.set_xlabel("Epoch")
    late.set_ylabel("Validation loss")
    late.grid()
    full.spines[["top", "right"]].set_visible(False)
    late.spines[["top", "right"]].set_visible(False)
    best_lines = []
    for seed in (3101, 3102, 3103):
        a = min(row["val"]["loss"] for row in records[("ACT", seed)])
        p = min(row["val"]["loss"] for row in records[("PACT", seed)])
        best_lines.append(f"{seed}: ACT {a:.4f} | PACT {p:.4f}")
    late.text(0.03, 0.95, "Best validation loss\n" + "\n".join(best_lines), transform=late.transAxes, va="top", fontsize=11.5, bbox={"boxstyle": "round,pad=0.5", "facecolor": "white", "edgecolor": COLORS["grid"]})
    footer(fig, "data/training_logs/*.jsonl: <epoch>.val.loss", "Earlier 3-D PACT 0.0834 vs ACT 0.0848 also failed to predict the endpoint.")
    return save_figure(fig, figure_dir, "fig09_training_curves")


def copy_sources(bundle: Path) -> None:
    data_dir = bundle / "data"
    report_dir = bundle / "reports"
    log_dir = data_dir / "training_logs"
    data_dir.mkdir(parents=True)
    report_dir.mkdir(parents=True)
    log_dir.mkdir(parents=True)
    for name, source in SOURCE_COPIES.items():
        shutil.copy2(source, data_dir / name)
    for name, source in REPORT_COPIES.items():
        shutil.copy2(source, report_dir / name)
    for name, source in TRAINING_LOGS.items():
        shutil.copy2(source, log_dir / name)


def ffprobe(path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,avg_frame_rate,nb_frames,duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    streams = json.loads(completed.stdout).get("streams", [])
    if len(streams) != 1:
        raise ValueError(f"expected one video stream: {path}")
    return streams[0]


def package_videos(bundle: Path) -> dict[str, Any]:
    matched = bundle / "videos/matched_pairs"
    heatmap = bundle / "videos/sensor_heatmap"
    expert = bundle / "videos/expert_demo"
    for directory in (matched, heatmap, expert):
        directory.mkdir(parents=True)

    heatmap_output = heatmap / "sensor_heatmap_40_skin_streams.mp4"
    expert_output = expert / "expert_clean_demo_wrist_view.mp4"
    shutil.copy2(HEATMAP_SOURCE, heatmap_output)
    shutil.copy2(EXPERT_VIDEO, expert_output)
    release_manifest = json.loads(QUALITATIVE_CLIPS_V2_MANIFEST.read_text())
    if release_manifest.get("status") not in {
        "presentation_release_verified",
        "presentation_release_incomplete_determinism_drop",
    }:
        raise ValueError("qualitative clips v2 presentation release is not finalized")
    release_records = {
        item["clip_id"]: item for item in release_manifest["render_outputs"]
    }
    outputs = {}
    selected_names = [clip["clip_id"] for clip in release_manifest["clips"]]
    if selected_names != list(QUALITATIVE_CLIPS_V2_NAMES):
        raise ValueError("qualitative clips v2 manifest has unexpected slide order")
    retained = set(release_manifest["determinism_summary"]["retained_clip_ids"])
    retained_names = [clip_id for clip_id in selected_names if clip_id in retained]
    if retained_names != list(release_records):
        raise ValueError("qualitative clips v2 retained/release order mismatch")
    for clip_id in retained_names:
        source = QUALITATIVE_CLIPS_V2_RELEASE / f"{clip_id}.mp4"
        output = matched / source.name
        record = release_records[clip_id]
        if file_hash(source) != record["release_video_sha256"]:
            raise ValueError(f"qualitative release hash mismatch: {source}")
        shutil.copy2(source, output)
        outputs[clip_id] = {
            "path": str(output.relative_to(bundle)),
            "sha256": file_hash(output),
            "size_bytes": output.stat().st_size,
            "video": ffprobe(output),
        }
    shutil.copy2(QUALITATIVE_CLIPS_V2_RELEASE / "README.md", matched / "README.md")
    for key, path in {
        "sensor_heatmap": heatmap_output,
        "expert_demo": expert_output,
    }.items():
        outputs[key] = {
            "path": str(path.relative_to(bundle)),
            "sha256": file_hash(path),
            "size_bytes": path.stat().st_size,
            "video": ffprobe(path),
        }
    return outputs


def make_key_numbers(
    bundle: Path,
    analysis: dict[str, Any],
    tail: dict[str, Any],
    env: dict[str, Any],
    frontend: dict[str, Any],
    valid: dict[str, Any],
    seed: dict[str, Any],
    early: dict[str, Any],
    training: dict[tuple[str, int], list[dict[str, Any]]],
) -> None:
    summaries = analysis["pooled_arm_summaries"]
    lines = [
        "# Key numbers and exact sources",
        "",
        "If a number is not in this file, do not type it onto a slide without adding its source first.",
        "",
        "## Experimental design",
        "",
        "| Number | Meaning | Source field |",
        "|---:|---|---|",
        "| 1,200 | Complete evaluation rollouts | `data/analysis.json → reconciliation.valid_cells` |",
        "| 100 | Held-out instances | `data/analysis.json → decision_rule_inputs.modality_contact.n_unique_instances` |",
        "| 3 | Policy seeds (3101, 3102, 3103) | `data/policy_training_registry.json → checkpoint_seeds` |",
        "| 4 | Arms (ACT, PACT, PACT_PERMUTED, PACT_ZERO) | `data/policy_training_registry.json → arms` |",
        "| 300 | Rollouts per arm | `data/analysis.json → pooled_arm_summaries.<arm>.hazard_bar_any_contact.n` |",
        "| 20,000 | Whole-instance bootstrap replicates | `data/analysis.json → decision_rule_inputs.*.bootstrap_replicates` |",
        "",
        "## Headline contact result",
        "",
        "| Arm | Any hazard contact | Wilson 95% interval | Count | Exact source |",
        "|---|---:|---:|---:|---|",
    ]
    for arm in ARM_ORDER:
        record = summaries[arm]["hazard_bar_any_contact"]
        lines.append(
            f"| {arm} | {record['rate']:.1%} | [{record['wilson_95'][0]:.1%}, {record['wilson_95'][1]:.1%}] | "
            f"{record['count']}/300 | `data/analysis.json → pooled_arm_summaries.{arm}.hazard_bar_any_contact` |"
        )
    pact_perm_contact = analysis["pooled_contrasts"]["PACT_minus_PACT_PERMUTED"]["hazard_bar_any_contact"]
    pact_act_contact = analysis["pooled_contrasts"]["PACT_minus_ACT"]["hazard_bar_any_contact"]
    lines += [
        "",
        (
            f"- PACT − PACT_PERMUTED contact rate: **{pact_perm_contact['difference']*100:+.1f} pp**, whole-instance 95% CI "
            f"[{pact_perm_contact['instance_cluster_bootstrap_ci_95'][0]*100:+.1f}, {pact_perm_contact['instance_cluster_bootstrap_ci_95'][1]*100:+.1f}] pp. "
            "Source: `data/analysis.json → pooled_contrasts.PACT_minus_PACT_PERMUTED.hazard_bar_any_contact`."
        ),
        (
            f"- PACT − ACT contact rate: **{pact_act_contact['difference']*100:+.1f} pp**, whole-instance 95% CI "
            f"[{pact_act_contact['instance_cluster_bootstrap_ci_95'][0]*100:+.1f}, {pact_act_contact['instance_cluster_bootstrap_ci_95'][1]*100:+.1f}] pp. "
            "Source: `data/analysis.json → pooled_contrasts.PACT_minus_ACT.hazard_bar_any_contact`."
        ),
        "- PACT_ZERO is an out-of-distribution sensor-failure probe. Its 35.7% contact rate and 6.7% collision-free success are **not modality evidence**. Source label: `data/analysis.json → arm_labels.PACT_ZERO`.",
        "",
        "## Task endpoints",
        "",
        "| Arm | Collision-free task success | Ordinary task success | Source |",
        "|---|---:|---:|---|",
    ]
    for arm in ARM_ORDER:
        lines.append(
            f"| {arm} | {summaries[arm]['collision_free_task_success']['rate']:.1%} | "
            f"{summaries[arm]['ordinary_task_success']['rate']:.1%} | `data/analysis.json → pooled_arm_summaries.{arm}` |"
        )
    task = analysis["decision_rule_inputs"]["pact_minus_act_collision_free_task_success"]
    ordinary = analysis["pooled_contrasts"]["PACT_minus_ACT"]["ordinary_task_success"]
    lines += [
        "",
        (
            f"- PACT − ACT collision-free task success: **{task['difference']*100:+.1f} pp**, 95% CI "
            f"[{task['instance_cluster_bootstrap_ci_95'][0]*100:+.1f}, {task['instance_cluster_bootstrap_ci_95'][1]*100:+.1f}] pp; not confirmed. "
            "Source: `data/analysis.json → decision_rule_inputs.pact_minus_act_collision_free_task_success`."
        ),
        (
            f"- PACT − ACT ordinary task success: **{ordinary['difference']*100:+.1f} pp**, 95% CI "
            f"[{ordinary['instance_cluster_bootstrap_ci_95'][0]*100:+.1f}, {ordinary['instance_cluster_bootstrap_ci_95'][1]*100:+.1f}] pp. "
            "Source: `data/analysis.json → pooled_contrasts.PACT_minus_ACT.ordinary_task_success`."
        ),
        "",
        "## Tail mechanism",
        "",
        "| Arm | >500-frame entry | Zero-contact rollouts | Median first-contact step | Source |",
        "|---|---:|---:|---:|---|",
    ]
    for arm in ("ACT", "PACT", "PACT_PERMUTED"):
        record = tail["arms"][arm]
        lines.append(
            f"| {arm} | {record['high_contact_regime']['entry_fraction']:.1%} "
            f"({record['high_contact_regime']['entry_count']}/300) | {record['distribution']['zero_hazard_frames']['fraction']:.1%} | "
            f"{record['high_contact_regime']['first_hazard_contact_step']['median']:.0f} | `data/tail_characterization.json → arms.{arm}` |"
        )
    mechanism = tail["mechanism_characterization"]["PACT_vs_ACT"]
    lines += [
        "",
        f"- PACT relative reduction in >500-frame entry versus ACT: **{mechanism['relative_entry_reduction']:.1%}**. Source: `data/tail_characterization.json → mechanism_characterization.PACT_vs_ACT.relative_entry_reduction`.",
        f"- PACT delay in median first contact versus ACT: **+{mechanism['median_first_contact_step_difference']:.0f} control steps**. Same source branch.",
        "- No per-step contact payload survived; faster escape/shortened entrapment is **not established**. Source: `data/tail_characterization.json → instrumentation_limits`.",
        "",
        "## Concentration",
        "",
        "| Arm | Top 1% share | Top 5% share | Top 10% share | Source |",
        "|---|---:|---:|---:|---|",
    ]
    for arm in ("ACT", "PACT", "PACT_PERMUTED", "PACT_ZERO"):
        c = tail["arms"][arm]["concentration"]
        lines.append(
            f"| {arm} | {c['top_1_percent']['share_of_arm_hazard_frames']:.1%} | "
            f"{c['top_5_percent']['share_of_arm_hazard_frames']:.1%} | {c['top_10_percent']['share_of_arm_hazard_frames']:.1%} | "
            f"`data/tail_characterization.json → arms.{arm}.concentration` |"
        )
    lines += ["", "## Per-seed replication", "", "| Seed | PACT − PACT_PERMUTED hazard frames | 95% CI | PACT − ACT collision-free success | 95% CI |", "|---:|---:|---:|---:|---:|"]
    for seed_id in ("3101", "3102", "3103"):
        contact = analysis["decision_rule_inputs"]["seed_modality_contacts"][seed_id]
        success = analysis["decision_rule_inputs"]["seed_pact_minus_act_collision_free_task_success"][seed_id]
        lines.append(
            f"| {seed_id} | {contact['difference']:,.0f} | [{contact['instance_bootstrap_ci_95'][0]:,.0f}, {contact['instance_bootstrap_ci_95'][1]:,.0f}] | "
            f"{success['difference']*100:+.1f} pp | [{success['instance_bootstrap_ci_95'][0]*100:+.1f}, {success['instance_bootstrap_ci_95'][1]*100:+.1f}] pp |"
        )
    lines += [
        "",
        "## Measurement journey",
        "",
        f"- 3-D front-end, PACT − PACT_ZERO: **{early['paired_instance_bootstrap']['PACT_minus_PACT_ZERO']['difference']*100:+.1f} pp**, no effect detected. Source: `data/early_3d_analysis.json → paired_instance_bootstrap.PACT_minus_PACT_ZERO`.",
        f"- 32-D front-end, PACT − PACT_ZERO: **{frontend['paired_instance_bootstrap']['PACT_minus_PACT_ZERO']['difference']*100:+.1f} pp**, an OOD artifact. Source: `data/frontend_screen_analysis.json → paired_instance_bootstrap.PACT_minus_PACT_ZERO`.",
        f"- 32-D valid ablation seed 3101: **{valid['paired_instance_bootstrap']['PACT_minus_PACT_PERMUTED']['difference']*100:+.1f} pp**. Source: `data/valid_ablation_analysis.json → paired_instance_bootstrap.PACT_minus_PACT_PERMUTED`.",
        f"- 32-D valid ablation seed 3102: **{seed['seed_results_unpooled_first']['3102']['contrasts']['PACT_minus_PACT_PERMUTED']['difference']*100:+.1f} pp**. Source: `data/seed_replication_analysis.json → seed_results_unpooled_first.3102.contrasts.PACT_minus_PACT_PERMUTED`.",
        "",
        "## Seed-noise journey",
        "",
        "- At 40 instances, seed 3101 PACT − ACT = **+25.0 pp [ +7.5, +42.5 ]**; seed 3102 = **−7.5 pp [−25.0, +12.5]**. Source: `data/seed_replication_analysis.json → seed_results_unpooled_first.<seed>.contrasts.PACT_minus_ACT`.",
        "- At 100 instances the same per-seed differences are **+8.0, +1.0, +3.0 pp**. Source: `data/analysis.json → decision_rule_inputs.seed_pact_minus_act_collision_free_task_success`.",
        "",
        "## Environment geometry",
        "",
        f"- Aperture width: **{env['scene']['aperture_width_m']:.2f} m**; sash height: **{env['scene']['sash_aperture_height_m']:.2f} m**. Source: `data/pact_collision_environment_v2.json → scene`.",
        f"- Panel center: **{env['scene']['panel_center_nominal_m']} m**; half-extents: **{env['scene']['panel_half_extents_m']} m**; inner face |y| = **{env['scene']['panel_inner_face_nominal_abs_y_m']:.2f} m**. Same source.",
        f"- Target: x ≈ **0.76 m**, y ∈ **[−0.04, 0.04] m**; robot base forward **{env['scene']['robot_base_forward_m']:.2f} m**. Same source.",
        "- Intrusion side is exactly 50/50 and independent of target placement. Same source.",
        "",
        "## Training curves",
        "",
        "| Seed | ACT best val loss (epoch) | PACT best val loss (epoch) | Source |",
        "|---:|---:|---:|---|",
    ]
    for seed_id in (3101, 3102, 3103):
        act_best = min((row["val"]["loss"], row["epoch"]) for row in training[("ACT", seed_id)])
        pact_best = min((row["val"]["loss"], row["epoch"]) for row in training[("PACT", seed_id)])
        lines.append(f"| {seed_id} | {act_best[0]:.6f} ({act_best[1]}) | {pact_best[0]:.6f} ({pact_best[1]}) | `data/training_logs/*_seed{seed_id}.jsonl → val.loss` |")
    lines += [
        "",
        "- Earlier 3-D seed-3101 validation losses **PACT 0.08345 vs ACT 0.08478** did not predict a policy endpoint advantage. Source: `reports/PACT_VS_ACT_FINAL_DECISION.md`.",
        "",
        "## Front-end and adequacy",
        "",
        "- Surface encoder held-out mean Euclidean error: **3.20 cm**; within-2-cm rate **52.9%**. Source: `data/policy_training_registry.json → source_training_summaries` and `reports/PACT_FRONTEND_SCREEN_DECISION.md`.",
        "- Environment pilot ACT ordinary task success: **23/64 = 35.9%**; hazard-contact episodes: **23/64 = 35.9%**. These are distinct counts that happen to coincide. Source: `data/environment_gate.json → act`.",
        "",
        "## Slideshow-only media fidelity",
        "",
        "- The sole third-person ACT re-render differs by **5 / 29,074 contact-pair samples = 0.017%** and by **2 / 29,024 hazard frames**. Source: `data/qualitative_video_manifest.json → determinism_check`.",
        "- It is an **independent illustrative draw**, not footage of the analyzed rollout and not an ACT/PACT pair. The scientific status remains `aborted_determinism_mismatch`.",
        "",
        "## Limitations that belong on the deck",
        "",
        "- One environment; simulation only; approximately one bit of scene variation (left/right intrusion).",
        "- Mechanism supports prevention/delay of tail entry; entrapment duration or faster escape was not measurable.",
        "- Task benefit is directional, not confirmed.",
        "- Optional projection-magnitude Figure 10 was omitted: no frozen source artifact/field records the quoted norms, and this bundle performs no new weight analysis.",
    ]
    (bundle / "KEY_NUMBERS.md").write_text("\n".join(lines) + "\n")


def make_summary_markdown(bundle: Path) -> None:
    (bundle / "ONE_PAGE_SUMMARY.md").write_text(
        "# PACT contact result — one-page summary\n\n"
        "## Headline\n\n"
        "Whole-body proximity reduced hazard-contact incidence from **22.3% (ACT)** to "
        "**14.0% (PACT)**. Against the valid distribution-matched PACT_PERMUTED ablation, "
        "the difference was **−9.3 percentage points** with a whole-instance 95% interval "
        "of **[−14.3, −5.0] pp**.\n\n"
        "## Design\n\n"
        "**1,200 rollouts · 100 held-out instances · 3 policy seeds · 4 arms · preregistered**\n\n"
        "## Task caveat\n\n"
        "PACT − ACT collision-free task success was **+4.0 pp [−2.3, +10.3]**. This is "
        "directionally positive but **not confirmed**. Ordinary manipulation success was +4.7 pp.\n\n"
        "## What the proximity modality appears to do\n\n"
        "Entry into the >500-hazard-frame regime fell from **19.7% to 11.0%**, a **44% "
        "relative reduction**, and median first contact moved from step **59 to 291**. These "
        "are post-hoc descriptive mechanism findings, not new confirmatory endpoints.\n\n"
        "## Correct ablation\n\n"
        "PACT_ZERO is an **out-of-distribution sensor-failure probe** and must not be used as "
        "modality evidence. PACT_PERMUTED preserves the learned token distribution while "
        "breaking scene alignment; it is the valid modality-information contrast.\n\n"
        "## Limits\n\n"
        "Single simulated environment; left/right intrusion is the main scene variation; no "
        "real-robot test; per-step contact runs were compacted, so faster escape or shortened "
        "entrapment is not established.\n"
    )


def make_video_shot_list(bundle: Path) -> None:
    (bundle / "VIDEO_SHOT_LIST.md").write_text(
        "# PACT slideshow — matched qualitative release\n\n"
        "The former three-way production brief is superseded by four fixed single-arm "
        "selections from two matched frozen-evaluation instances. Three passed the exact "
        "determinism gate and are shipped. Each source pair shares the episode, policy seed, "
        "intrusion side, camera, and 3.0x playback factor.\n\n"
        "## Slide order\n\n"
        "1. **PACT success:** `clip1_54a6272f66ca_pact_success.mp4` — 0 hazard frames, task success.\n"
        "2. **ACT failure:** `clip2_54a6272f66ca_act_failure.mp4` — 29,022 hazard frames, task failure.\n"
        "3. **ACT success (dropped; no MP4 shipped):** fixed row had 19,757 hazard frames and task success, but the rerender changed task success to no and first contact 302→295.\n"
        "4. **PACT failure:** `clip4_e99dc657bfa7_pact_failure.mp4` — 17,609 hazard frames, task failure.\n\n"
        "Instance A was selected as the largest ACT-contact member of the 48 instance-seeds "
        "where PACT succeeds and ACT fails. Instance B was selected as the largest PACT-contact "
        "member of the 34 instance-seeds where ACT succeeds and PACT fails. Keeping the PACT "
        "failure is deliberate: these are examples, not aggregate evidence.\n\n"
        "## Overlay\n\n"
        "Every clip shows only policy arm and seed, episode ID, task success, any hazard "
        "contact, running hazard-contact frames, maximum hazard penetration, when available, "
        "and the common playback factor.\n\n"
        "## Required caption\n\n"
        "> Re-rendered from the analyzed rollout. Task success, manipulation success, and "
        "first-contact step reproduce exactly; contact-pair samples differ by 0.017%.\n\n"
        "The original `PACT_QUALITATIVE_VIDEOS.md` remains "
        "`aborted_determinism_mismatch`; this is a presentation release under a separate "
        "manifest, not a revision of the scientific record.\n\n"
        "## Existing repository video\n\n"
        "A setup turntable is already committed at:\n\n"
        "`diagnostics_output/20260611_skin_photoshoot/turntable.mp4`\n\n"
        "Use it as a short visual introduction to the sensor-covered arm, not as evidence of "
        "policy performance.\n\n"
        "## Optional separate workstream\n\n"
        "The Safety-CVAE demo videos under `assets/safety/` are useful only on a slide "
        "explicitly titled **Separate hybrid-skin safety/CVAE work**. Do not mix those "
        "metrics with the PACT-vs-ACT experiment.\n"
    )


def make_index(bundle: Path, video_records: dict[str, Any]) -> None:
    lines = [
        "# PACT slideshow bundle index",
        "",
        "Read `KEY_NUMBERS.md` before typing numbers onto slides. This folder is self-contained; all source JSONs, training logs, reports, figures, and available media are copied here.",
        "",
        "## Recommended slide order",
        "",
        "| Slide asset | What it shows | One-sentence claim | Source artifact / field | Required caveat |",
        "|---|---|---|---|---|",
        "| `figures/fig00_one_page_summary.{png,svg}` | Design and headline numbers | PACT reduces contact; task benefit is directional | `data/analysis.json → pooled_arm_summaries`, `decision_rule_inputs` | Say task success is not confirmed; PACT_ZERO is OOD |",
        "| `figures/fig01_contact_rate_by_arm.{png,svg}` | Contact incidence with Wilson intervals | PACT has the lowest non-OOD contact rate | `data/analysis.json → pooled_arm_summaries.<arm>.hazard_bar_any_contact` | PACT_ZERO is an OOD failure probe, not modality evidence |",
        "| `figures/fig02_tail_entry_and_first_contact.{png,svg}` | >500-frame entry and median first-contact step | Proximity primarily prevents/delays entry into the high-contact tail | `data/tail_characterization.json → arms.<arm>.high_contact_regime` | Post-hoc descriptive; no faster-escape claim |",
        "| `figures/fig03_hazard_frame_ecdf.{png,svg}` | Exact ECDF of 1,200 hazard-frame totals | The distribution is mostly zero, then jumps into a large-contact tail | `data/hazard_frames_by_arm.json → values_by_arm` | Descriptive; PACT_ZERO dashed because OOD |",
        "| `figures/fig04_contact_concentration.{png,svg}` | Top 1/5/10% concentration | PACT's top 10% carries 98.6% of its hazard frames | `data/tail_characterization.json → arms.<arm>.concentration` | Only frozen cut points are shown |",
        "| `figures/fig05_seed_replication.{png,svg}` | Per-seed modality contact vs task success | Contact reduction is stable across trainings; task success is noisy | `data/analysis.json → decision_rule_inputs.seed_*` | The two panels measure different contrasts |",
        "| `figures/fig06_measurement_journey.{png,svg}` | 3-D/32-D, zero/permuted, contact journey | Valid measurement and ablation changed the apparent answer | Earlier and current analysis JSONs | Zero-ablation +70 pp is an OOD artifact |",
        "| `figures/fig07_seed_noise_resolution.{png,svg}` | 40- vs 100-instance estimates | The earlier sign conflict mostly reflected sampling noise | `data/seed_replication_analysis.json`, `data/analysis.json` | Task advantage still not confirmed |",
        "| `figures/fig08_environment_schematic.{png,svg}` | Corridor geometry and side ambiguity | The panel blocks a wrist-camera-blind approach corridor | `data/pact_collision_environment_v2.json → scene`, corridor XML | Schematic, not camera footage |",
        "| `figures/fig09_training_curves.{png,svg}` | Six complete validation curves | Validation loss did not predict endpoint behavior | `data/training_logs/*.jsonl → val.loss` | Do not infer modality benefit from val loss |",
        "",
        "Optional Figure 10 (projection magnitudes) was deliberately omitted because no frozen source artifact contains the quoted fan-in-normalized norms and the plan forbids new analysis.",
        "",
        "## Matched qualitative clips",
        "",
        "`VIDEO_SHOT_LIST.md` and `videos/matched_pairs/README.md` record the two matched instances, four fixed selections, the clip-3 determinism drop, overlays, and presentation-only checks. Three clips are shipped; the aggregate evidence remains the 1,200-rollout analysis.",
        "",
        "> Re-rendered from the analyzed rollout. Task success, manipulation success, and first-contact step reproduce exactly; contact-pair samples differ by 0.017%.",
        "",
        "The quotation is the predeclared caption from the earlier probe. Per-clip deltas are in `data/qualitative_clips_v2_manifest.json`; do not generalize 0.017% to all four rerenders.",
        "",
        "## Videos",
        "",
        "| File | What it shows | Claim supported | Source | Required caveat |",
        "|---|---|---|---|---|",
        "| `videos/matched_pairs/clip1_54a6272f66ca_pact_success.mp4` | Instance A, PACT | PACT avoids the panel and succeeds in this selected case | `data/qualitative_clips_v2_manifest.json → clips[0]` | Pair with clip 2; selected example, not aggregate evidence |",
        "| `videos/matched_pairs/clip2_54a6272f66ca_act_failure.mp4` | Instance A, ACT | ACT sustains 29,022 hazard frames and fails on the same instance/seed | `data/qualitative_clips_v2_manifest.json → clips[1]` | Pair with clip 1; presentation re-render |",
        "| **Clip 3 not shipped** | Instance B, ACT | Its rerender changed task success yes→no and first contact 302→295 | `data/qualitative_clips_v2_manifest.json → determinism_summary` | Mandatory determinism-gate drop; not rerun or replaced |",
        "| `videos/matched_pairs/clip4_e99dc657bfa7_pact_failure.mp4` | Instance B, PACT | PACT still fails in an honestly retained counterexample | `data/qualitative_clips_v2_manifest.json → clips[3]` | Pair with clip 3; PACT reduces contact incidence, not eliminates it |",
        "| `videos/sensor_heatmap/sensor_heatmap_40_skin_streams.mp4` | All 40 skin streams over one rollout | The whole-body sensors carry spatially localized surface signal | Existing determinism-probe heatmap | Visual illustration only; ACT did not consume these streams |",
        "| `videos/expert_demo/expert_clean_demo_wrist_view.mp4` | Clean scripted-expert task execution | The task is solvable by a collision-free expert | `data/expert_demo_result.json` | Wrist view does not show the full-body bow or panel |",
        "",
        "## One-page text summary",
        "",
        "`ONE_PAGE_SUMMARY.md` contains paste-ready prose with the task-success caveat on the page itself.",
        "",
        "## Reports",
        "",
        "The `reports/` folder contains the contact decision, tail characterization, environment gate, front-end screen, valid-ablation, seed-replication, early 3-D decision, qualitative stop report, and contact-tail discovery note.",
        "",
        "## Bundle integrity",
        "",
        "`BUNDLE_MANIFEST.json` lists SHA-256 and size for every other file in the bundle. The committed copy is `diagnostics_output/pact_slideshow_bundle_manifest.json` in the source repository.",
        "",
        "## Framing that must survive slide editing",
        "",
        "- Lead with confirmed contact reduction.",
        "- Say 'task success directionally positive, not confirmed.'",
        "- Never use PACT_ZERO as load-bearing-modality evidence.",
        "- Name the limitations: one environment, approximately one bit of scene variation, simulation only, and no retained contiguous contact runs.",
    ]
    (bundle / "INDEX.md").write_text("\n".join(lines) + "\n")


def bundle_manifest(bundle: Path, video_records: dict[str, Any]) -> dict[str, Any]:
    entries = []
    for path in sorted(item for item in bundle.rglob("*") if item.is_file() and item.name != "BUNDLE_MANIFEST.json"):
        entries.append(
            {
                "path": str(path.relative_to(bundle)),
                "sha256": file_hash(path),
                "size_bytes": path.stat().st_size,
            }
        )
    extension_counts: dict[str, int] = defaultdict(int)
    for entry in entries:
        extension_counts[Path(entry["path"]).suffix.lower() or "no_extension"] += 1
    figure_stems = sorted({str(Path(entry["path"]).with_suffix("")) for entry in entries if entry["path"].startswith("figures/")})
    document = {
        "schema_version": "pact_slideshow_bundle_manifest_v1",
        "bundle_root": str(bundle.resolve()),
        "scientific_artifacts_modified": False,
        "gpu_work_performed": True,
        "rollouts_or_training_performed": True,
        "presentation_only_rollout_rerenders": 4,
        "training_performed": False,
        "figure_concepts": len(figure_stems),
        "figure_files": sum(1 for entry in entries if entry["path"].startswith("figures/")),
        "video_files": sum(1 for entry in entries if entry["path"].endswith(".mp4")),
        "paired_video_files": 2,
        "matched_single_arm_clip_files": 3,
        "complete_matched_instance_pairs": 1,
        "determinism_dropped_clip_files": 1,
        "unpaired_independent_probe_files": 0,
        "total_payload_size_bytes_excluding_manifest": sum(entry["size_bytes"] for entry in entries),
        "extension_counts": dict(sorted(extension_counts.items())),
        "video_records": video_records,
        "paired_video_release": (
            "Three presentation-only single-arm clips were retained; one complete ACT/PACT "
            "pair is available and clip 3 was dropped by the exact determinism gate. The "
            "frozen aborted scientific qualitative record remains unchanged."
        ),
        "optional_figure_10": "omitted_no_frozen_source_field_and_no_new_analysis_allowed",
        "entries": entries,
    }
    document["bundle_manifest_sha256"] = canonical_hash(document)
    return document


def write_manifest(bundle: Path, document: dict[str, Any]) -> None:
    encoded = json.dumps(document, indent=2, sort_keys=True) + "\n"
    (bundle / "BUNDLE_MANIFEST.json").write_text(encoded)
    REPO_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    REPO_MANIFEST.write_text(encoded)


def build(bundle: Path) -> dict[str, Any]:
    if bundle.exists():
        raise FileExistsError(f"refusing to replace existing bundle: {bundle}")
    require_sources()
    configure_style()
    analysis = load(ANALYSIS)
    tail = load(TAIL)
    env = load(ENV_CONFIG)
    frontend = load(FRONTEND_ANALYSIS)
    valid = load(VALID_ANALYSIS)
    seed = load(SEED_ANALYSIS)
    early = load(EARLY_ANALYSIS)
    schedule = load(SCHEDULE)
    if analysis["reconciliation"] != {"errors": [], "expected_cells": 1200, "reconciled": True, "valid_cells": 1200}:
        raise ValueError("contact analysis is not reconciled")
    if load(FINAL_DECISION)["decision"] != "CONTACT_REDUCTION_WITH_TASK_BENEFIT":
        raise ValueError("awarded contact token changed")
    if load(QUALITATIVE_MANIFEST)["status"] != "aborted_determinism_mismatch":
        raise ValueError("scientific qualitative status changed")
    training = read_training_logs()

    staging = Path(tempfile.mkdtemp(prefix="pact_slideshow_bundle_", dir="/root"))
    try:
        figure_dir = staging / "figures"
        figure_dir.mkdir(parents=True)
        copy_sources(staging)
        values = extract_hazard_frames(schedule)
        write_hazard_frame_data(staging, values, tail)
        figure_summary(analysis, figure_dir)
        figure_headline(analysis, figure_dir)
        figure_mechanism(tail, figure_dir)
        figure_distribution(values, figure_dir)
        figure_concentration(tail, figure_dir)
        figure_replication(analysis, figure_dir)
        figure_measurement_journey(early, frontend, valid, seed, analysis, figure_dir)
        figure_seed_noise(seed, analysis, figure_dir)
        figure_environment(env, figure_dir)
        figure_training(training, figure_dir)
        video_records = package_videos(staging)
        make_key_numbers(staging, analysis, tail, env, frontend, valid, seed, early, training)
        make_summary_markdown(staging)
        make_video_shot_list(staging)
        make_index(staging, video_records)
        document = bundle_manifest(staging, video_records)
        # Record the final destination rather than the temporary staging path.
        document["bundle_root"] = str(bundle.resolve())
        document.pop("bundle_manifest_sha256")
        document["bundle_manifest_sha256"] = canonical_hash(document)
        write_manifest(staging, document)
        os.replace(staging, bundle)
        return document
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    args = parser.parse_args()
    document = build(args.bundle.resolve())
    print(
        json.dumps(
            {
                "bundle": document["bundle_root"],
                "figure_concepts": document["figure_concepts"],
                "figure_files": document["figure_files"],
                "video_files": document["video_files"],
                "paired_video_files": document["paired_video_files"],
                "total_payload_size_bytes_excluding_manifest": document[
                    "total_payload_size_bytes_excluding_manifest"
                ],
                "bundle_manifest_sha256": document["bundle_manifest_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
