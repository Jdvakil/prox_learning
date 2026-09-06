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
with the wrapper,” as the canonical user runbook. For registered `v12`, `v1011d`
and `hallway` experiments, give `python scripts/pact.py` commands from the repository
root. Explain the dataset → prepared manifest → named run relationship and the
purpose of each requested step. The default full PACT arm is `readout`, jointly
finetuning the surface encoder; `raw` and `act` are baselines.

When explaining an older instruction, show the original command first and its
wrapper equivalent second, clearly marked as alternatives. Distinguish actual
behavioral differences (split, normalization, environment, rendering, checkpoint
pairing) from syntactic changes. Do not send users to a historical evaluator as
the default for a registered run. If the wrapper does not cover a task, say so
and link the original README recipe; do not invent wrapper flags.

Reuse completed conversion/preparation/setup across runs. Use unique training
run names, then offline/check, checkpoint-specific parity verification and staged
smoke/dev/test evaluation. Never describe parity or offline loss as certification
of task success. Consult README §4.22–4.23 for flags, artifacts, batching and
limitations. Update README and the CURSOR session log when these instructions
change. Documentation-only work must not launch or disturb training/evaluation.
