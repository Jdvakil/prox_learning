# Superseded V10.9 training preflight, attempt 01

Retained, not deleted. **Not** operative. `ready: false`, so no training ran.

The preflight refused for one reason: the ACT submodule working tree was dirty.
The untracked file was `eval_pact_place_v109_row.py`, the V10.9 evaluator this
task itself adds. It was committed to the submodule as `a22d02b`, making it a
tracked pure addition alongside the three earlier evaluation scripts, and the
preflight was re-run.

Everything else already passed in this attempt and is unchanged in the operative
preflight: the parsed ACT/PACT command diff was identical except `--ckpt_dir`
and the five allowed PACT flags, disk was sufficient, and no training, model, or
loader source differed from the V5 training commit.
