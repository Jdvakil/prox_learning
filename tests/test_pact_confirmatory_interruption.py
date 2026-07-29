from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "reconcile_pact_confirmatory_interruption",
        ROOT / "scripts" / "reconcile_pact_confirmatory_interruption.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


reconcile = _load()


def _load_finalizer():
    spec = importlib.util.spec_from_file_location(
        "finalize_pact_confirmatory_interruption",
        ROOT / "scripts" / "finalize_pact_confirmatory_interruption.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


finalize = _load_finalizer()


def _row(index: int) -> dict:
    return {
        "schedule_index": index,
        "rollout_id": f"rollout-{index}",
        "schedule_row_sha256": f"row-{index}",
        "instance_episode_id": f"episode-{index}",
        "arm": "ACT",
        "checkpoint_sha256": "checkpoint",
        "output_relpath": f"rows/{index:03d}",
    }


def _boundary(row: dict) -> dict:
    return {
        "initial_observation_accepted": True,
        "rollout_id": row["rollout_id"],
        "schedule_row_sha256": row["schedule_row_sha256"],
        "episode_id": row["instance_episode_id"],
        "checkpoint_sha256": row["checkpoint_sha256"],
    }


def test_reconciliation_never_launches_untouched_rows(tmp_path, monkeypatch):
    rows = [_row(index) for index in range(3)]
    schedule = {"rows": rows}

    complete_dir = tmp_path / rows[0]["output_relpath"]
    complete_dir.mkdir(parents=True)
    boundary = _boundary(rows[0])
    (complete_dir / "initial_observation_accepted.json").write_text(
        json.dumps(boundary)
    )
    (complete_dir / "result.json").write_text(
        json.dumps(
            {
                **boundary,
                "status": "complete",
                "arm": "ACT",
            }
        )
    )

    interrupted_dir = tmp_path / rows[1]["output_relpath"]
    interrupted_dir.mkdir(parents=True)
    (interrupted_dir / "initial_observation_accepted.json").write_text(
        json.dumps(_boundary(rows[1]))
    )

    def forbidden_subprocess(*args, **kwargs):
        raise AssertionError("reconciliation attempted to launch a subprocess")

    monkeypatch.setattr(reconcile.runner.subprocess, "run", forbidden_subprocess)
    terminal, not_started = reconcile.reconcile_rows(
        schedule,
        output_root=tmp_path,
    )

    assert [item["driver"]["status"] for item in terminal] == [
        "complete",
        "post_boundary_failure",
    ]
    assert [row["rollout_id"] for row in not_started] == ["rollout-2"]
    assert not (tmp_path / rows[2]["output_relpath"]).exists()
    assert json.loads(
        (interrupted_dir / "driver_result.json").read_text()
    )["resume_action"] == "reconciled_boundary_marker_without_rerun"


def test_interruption_report_preserves_exact_final_token():
    report = "\n".join(
        [
            "# Final",
            "",
            "## Decision",
            "",
            "PACT_EXPERIMENT_INCOMPLETE",
            "",
        ]
    )
    incident = {
        "incident_sha256": "incident",
        "interruption": {
            "post_boundary_terminal_count": 8,
            "never_started_count": 951,
        },
    }
    analysis = {
        "reconciliation": {
            "valid": 1,
            "missing": list(range(959)),
        }
    }
    rendered = finalize.insert_interruption_section(
        report,
        incident=incident,
        analysis=analysis,
    )
    assert "8 terminal post-boundary rows" in rendered
    assert "951 never-started rows" in rendered
    assert rendered.rstrip().splitlines()[-1] == "PACT_EXPERIMENT_INCOMPLETE"
