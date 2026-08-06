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
