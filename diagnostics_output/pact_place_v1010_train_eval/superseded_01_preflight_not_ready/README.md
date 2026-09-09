# Superseded V10.10 training preflight, attempt 01

Retained, not deleted. `ready: false`, so no training ran.

Two blockers, both genuine:

1. The ACT submodule was dirty — `eval_pact_place_v1010_row.py` and the V10.9R
   files were untracked. Committed as `f4f59d7`.

2. The disk check demanded 18 GiB on the assumption that both checkpoint trees
   coexist. They no longer do: the training runner prunes each arm's
   intermediate checkpoints when it finishes, so the peak is one full tree
   (~7.6 GiB) plus the previous arm's retained best/last (~0.9 GiB). The
   threshold now derives from that retention policy instead of a fixed number.
   This is a correction to a stale assumption, not a relaxation to force a pass:
   the measured peak is what changed.
