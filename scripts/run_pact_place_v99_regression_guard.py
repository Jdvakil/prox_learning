#!/usr/bin/env python3
"""Fresh V9.5 eight-row regression guard for V9.9 stage 1.

Replays the frozen raw-smoke rows. Does not insert a pendant. Success is 6/8
clean with all eight outcomes matching the authoritative raw smoke.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "scripts", ROOT / "submodules" / "molmospaces"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pact_place_v99_pendant_contract import empty_authorization  # noqa: E402
from reconstruct_pact_place_v99_baseline import write_immutable  # noqa: E402
from replay_pact_place_v95_smoke import (  # noqa: E402
    DEFAULT_OUTPUT_ROOT,
    SOURCE_SUMMARY,
    main as replay_main,
)

DEFAULT_OUTPUT = ROOT / "diagnostics_output/pact_place_v99_regression_guard"


def stamp_authorization(output_root: Path) -> None:
    guard_path = output_root / "guard.json"
    if not guard_path.is_file():
        return
    document = json.loads(guard_path.read_text())
    document.update(empty_authorization())
    document["stage"] = "v99_regression_guard"
    document["authorizes_paired_screen"] = bool(
        document.get("passed") and int(document.get("clean_rows") or 0) == 6
    )
    write_immutable(guard_path, document)


if __name__ == "__main__":
    if "--output-root" not in sys.argv:
        sys.argv.extend(["--output-root", str(DEFAULT_OUTPUT)])
    code = replay_main()
    output = Path(sys.argv[sys.argv.index("--output-root") + 1])
    stamp_authorization(output)
    raise SystemExit(code)
