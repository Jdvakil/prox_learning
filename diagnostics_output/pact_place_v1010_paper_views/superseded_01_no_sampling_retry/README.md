# Superseded V10.10 paper-views manifest, attempt 01

Retained. 8 of 12 episodes rendered; 48 of 72 clips.

The renderer sampled each scene once, while the collection retries up to 12
times. Four episodes therefore failed before any frame was drawn:
  - 3x "IK failed for pregrasp pose"
  - 1x "settled clutter overlaps target"

Both are ordinary sampler rejections, not render faults. The ledger does not
record which retry produced an accepted episode, so the loop must be replayed;
retry seeds come from pact_place_v108_contract.cell_seed because that is what
the V10.10 collection actually called.

The 8 episodes that did render all passed the fidelity check (step counts within
one and success flags matching the ledger), which confirms they were retry-0
draws and that the re-simulation reproduces the recorded episodes.
