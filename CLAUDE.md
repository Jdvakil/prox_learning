# Protocol: Caveman (ALWAYS)

Speak primitive. Nouns and verbs. Drop filler words (the, is, are, of, a). Short
words. Save tokens. Blunt. Simple. Applies to all chat replies in this project.

Carve-outs (stay normal English — caveman breaks these):
- Code, code comments, docstrings.
- Commit messages, PR titles/bodies.
- File content written for tools/configs.

# Workflow constraint

Prepare code edits, then tell the user the exact commands to run. The user runs them. Unless user requests to do so. Fan out subagents which will help me with my experiments. 

ALWAYS USE `/effort` TO ULTRACODE ALL THE TIME. HARD CONSTRAINT. NEVER USE ANYTHING LESS.
ALWAYS DOCUMENT AND REFER TO THE README IN /home/jaydv/code/prox_learning/README.md NO NEW README FILES OR MARKDOWN FILES.

## Datagen run recipe
```
conda activate mlspaces
cd submodules/molmospaces
python -m molmo_spaces.data_generation.main <ConfigName>
```

## Training and evaluation instructions

Use the root [README.md](README.md), starting at “Start here: dataset to results
with the wrapper,” as the canonical user runbook. Convert / prepare / train for
registered `v12`, `v1011d` and `hallway` use `python scripts/pact.py` from the
repository root. **In-env experiment eval is repo-root `eval_act.py`.** Do not
give `pact.py eval` / `verify` as the default; that wrapper eval is parked
(v12 construction failures; different success/history protocol). Explain the
dataset → prepared manifest → named run relationship for training. The default
full PACT train arm is `readout`; `raw` and `act` are baselines.

When explaining an older instruction, show the original command first and its
wrapper equivalent second, clearly marked as alternatives. Distinguish actual
behavioral differences (split, normalization, environment, rendering, checkpoint
pairing) from syntactic changes. Do not send users to a historical evaluator
(`eval_act_place_corridor.py`, `eval_pact.py`) as the default. If `eval_act.py`
does not cover a `data/` dump yet (v12 overlay, v107, mixed, table_smoke), say
so; do not invent `--task` names.

Reuse completed conversion/preparation across runs. Use unique training run
names, then `offline` / `check` if wanted. Closed-loop numbers come from
`eval_act.py`. Never describe parity or offline loss as certification of task
success. Consult README §4.22–4.23 for train flags. Update README and the
CURSOR session log when these instructions change. Documentation-only work
must not launch or disturb training/evaluation.
